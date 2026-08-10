"""
GPU-resident CUB dataset with torchvision v2 transforms.

Preloads all images into GPU memory as uint8 tensors at initialization,
then applies transforms on-device each epoch — eliminating PIL conversion,
disk I/O, and CPU→GPU transfer from the training loop.

Classes:
    CUB_GPU_Dataset: GPU-resident dataset for CUB images.
    GPUBatchIterator: Custom iterator yielding batches from GPU-resident data.

Functions:
    get_CUB_gpu_dataloaders: Build GPU-resident datasets for train/val/test splits.
"""

import os
import pickle
import numpy as np
from PIL import Image

import torch
from torchvision.transforms import v2


class CUB_GPU_Dataset:
    """
    Loads the entire CUB split into GPU memory at init time.

    All images are resized to a uniform resolution (LOAD_RESOLUTION x LOAD_RESOLUTION)
    and stored as uint8 tensors (N, C, H, W) on GPU. This enables batched transforms
    without any per-image Python loops.

    Labels and concepts are stored as GPU tensors.
    Transforms are torchvision v2 and run on the full batch on-device.
    """

    LOAD_RESOLUTION = 500  # Pre-resize all images to this uniform size

    def __init__(self, data_pkl, transform, device):
        self.transform = transform
        self.device = device
        self.num_samples = len(data_pkl)
        res = self.LOAD_RESOLUTION

        # Pre-resize transform (runs on CPU during loading)
        pil_resize = Image.Resampling.BILINEAR

        # Load all images, resize to uniform size, convert to tensor
        print(f"  Loading {self.num_samples} images from disk (resizing to {res}x{res})...")
        self.images = torch.zeros(
            (self.num_samples, 3, res, res), dtype=torch.uint8, device=device
        )
        labels = []
        concepts = []

        for i, item in enumerate(data_pkl):
            img = Image.open(item["img_path"]).convert("RGB")
            img = img.resize((res, res), pil_resize)
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # HWC -> CHW uint8
            self.images[i] = img_tensor.to(device)
            labels.append(item["class_label"])
            concepts.append(np.array(item["attribute_label"], dtype=np.float32))

        # Store labels and concepts on GPU
        self.labels = torch.tensor(labels, dtype=torch.long, device=device)
        self.concepts = torch.tensor(np.stack(concepts), dtype=torch.float32, device=device)

        mem_mb = self.images.element_size() * self.images.nelement() / (1024 * 1024)
        print(f"  GPU preload complete: {mem_mb:.0f} MB for images, "
              f"{self.num_samples} samples on {device}")

    def __len__(self):
        return self.num_samples

    def get_batch(self, indices):
        """
        Get a batch with transforms applied to the full batch tensor at once.

        Args:
            indices: 1D tensor of sample indices

        Returns:
            dict with keys: img_code, labels, features, concepts
        """
        # Single GPU index operation: (B, 3, 500, 500) uint8
        batch_imgs = self.images[indices]
        # Apply v2 transforms to entire batch at once — no Python loop
        batch_imgs = self.transform(batch_imgs)

        return {
            "img_code": indices,
            "labels": self.labels[indices],
            "features": batch_imgs,
            "concepts": self.concepts[indices],
        }


class GPUBatchIterator:
    """
    Drop-in replacement for DataLoader when data is GPU-resident.

    Shuffles indices each epoch (if shuffle=True), yields batch dicts
    compatible with the existing training loop. No CPU workers needed.
    """

    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, generator=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator

    def __iter__(self):
        n = len(self.dataset)
        if self.shuffle:
            indices = torch.randperm(n, generator=self.generator)
        else:
            indices = torch.arange(n)

        # Yield batches
        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            if end > n and self.drop_last:
                break
            batch_indices = indices[start:min(end, n)]
            yield self.dataset.get_batch(batch_indices)

    def __len__(self):
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


def _make_train_transform_v2():
    """Training transforms using torchvision v2 (GPU-compatible)."""
    resol = 299
    return v2.Compose([
        v2.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
        v2.RandomResizedCrop(resol, antialias=True),
        v2.Resize(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),  # uint8 -> float32, divide by 255
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _make_test_transform_v2():
    """Test/val transforms using torchvision v2 (GPU-compatible)."""
    resol = 299
    return v2.Compose([
        v2.CenterCrop(resol),
        v2.Resize(size=(224, 224), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def train_test_split_CUB(root_dir):
    """Performs train-validation-test split for the CUB dataset (same as original)."""
    data_train = []
    data_val = []
    data_test = []
    data_train.extend(
        pickle.load(open(
            os.path.join(root_dir, "CUB/CUB_processed/class_attr_data_10/train.pkl"), "rb",
        ))
    )
    data_val.extend(
        pickle.load(open(
            os.path.join(root_dir, "CUB/CUB_processed/class_attr_data_10/val.pkl"), "rb",
        ))
    )
    data_test.extend(
        pickle.load(open(
            os.path.join(root_dir, "CUB/CUB_processed/class_attr_data_10/test.pkl"), "rb",
        ))
    )
    for dataset in [data_train, data_val, data_test]:
        for i in range(len(dataset)):
            parts = dataset[i]["img_path"].split("/")
            index = parts.index("images")
            end_path = "/".join(parts[index:])
            dataset[i]["img_path"] = os.path.join(root_dir, "CUB/CUB_200_2011/", end_path)

    return data_train, data_val, data_test


def get_CUB_gpu_dataloaders(config, device):
    """
    Returns GPU-resident CUB datasets for train/val, and raw test pkl data.

    Only train and val are preloaded to GPU — test is evaluated once, so it
    uses the regular CPU DataLoader to avoid wasting VRAM.

    Returns:
        tuple of (train_dataset, val_dataset, test_imgs)
        where train/val are CUB_GPU_Dataset and test_imgs is raw pkl data.
    """
    print("Loading CUB dataset to GPU...")
    train_imgs, val_imgs, test_imgs = train_test_split_CUB(root_dir=config.data_path)

    train_transform = _make_train_transform_v2()
    test_transform = _make_test_transform_v2()

    print("  Preloading training set...")
    train_dataset = CUB_GPU_Dataset(train_imgs, train_transform, device)
    print("  Preloading validation set...")
    val_dataset = CUB_GPU_Dataset(val_imgs, test_transform, device)

    return train_dataset, val_dataset, test_imgs
