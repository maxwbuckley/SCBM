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

    Images are stored as uint8 tensors (CHW) padded to max dimensions.
    Labels and concepts are stored as GPU tensors.
    Transforms are torchvision v2 and run on-device.
    """

    def __init__(self, data_pkl, transform, device):
        self.transform = transform
        self.device = device
        self.num_samples = len(data_pkl)

        # First pass: load all images and find max dimensions
        print(f"  Loading {self.num_samples} images from disk...")
        pil_images = []
        labels = []
        concepts = []
        max_h, max_w = 0, 0

        for item in data_pkl:
            img = Image.open(item["img_path"]).convert("RGB")
            w, h = img.size
            max_h = max(max_h, h)
            max_w = max(max_w, w)
            # Convert to tensor immediately, skip PIL for future use
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # HWC -> CHW uint8
            pil_images.append((img_tensor, h, w))
            labels.append(item["class_label"])
            concepts.append(np.array(item["attribute_label"], dtype=np.float32))

        # Allocate GPU tensors
        print(f"  Allocating GPU memory ({max_h}x{max_w} max dims)...")
        self.images = torch.zeros(
            (self.num_samples, 3, max_h, max_w), dtype=torch.uint8, device=device
        )
        self.heights = torch.zeros(self.num_samples, dtype=torch.int32, device="cpu")
        self.widths = torch.zeros(self.num_samples, dtype=torch.int32, device="cpu")

        # Copy images into the padded GPU tensor
        for i, (img_tensor, h, w) in enumerate(pil_images):
            self.images[i, :, :h, :w] = img_tensor
            self.heights[i] = h
            self.widths[i] = w

        # Store labels and concepts on GPU
        self.labels = torch.tensor(labels, dtype=torch.long, device=device)
        self.concepts = torch.tensor(np.stack(concepts), dtype=torch.float32, device=device)

        # Free the intermediate list
        del pil_images

        mem_mb = self.images.element_size() * self.images.nelement() / (1024 * 1024)
        print(f"  GPU preload complete: {mem_mb:.0f} MB for images, "
              f"{self.num_samples} samples on {device}")

    def __len__(self):
        return self.num_samples

    def get_transformed_sample(self, idx):
        """Get a single sample with transforms applied on GPU."""
        h = self.heights[idx].item()
        w = self.widths[idx].item()
        # Extract unpadded image (already on GPU)
        img = self.images[idx, :, :h, :w]
        # Apply v2 transforms (runs on GPU since img is a CUDA tensor)
        img = self.transform(img)
        return img

    def get_batch(self, indices):
        """
        Get a batch of transformed samples.

        Args:
            indices: 1D tensor of sample indices

        Returns:
            dict with keys: img_code, labels, features, concepts
        """
        batch_imgs = []
        for idx in indices:
            idx_val = idx.item()
            batch_imgs.append(self.get_transformed_sample(idx_val))

        return {
            "img_code": indices,
            "labels": self.labels[indices],
            "features": torch.stack(batch_imgs),
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
    Returns GPU-resident CUB datasets wrapped in GPUBatchIterators.

    Returns:
        tuple of (train_loader, val_loader, test_loader) as GPUBatchIterators
    """
    print("Loading CUB dataset to GPU...")
    train_imgs, val_imgs, test_imgs = train_test_split_CUB(root_dir=config.data_path)

    train_transform = _make_train_transform_v2()
    test_transform = _make_test_transform_v2()

    print("  Preloading training set...")
    train_dataset = CUB_GPU_Dataset(train_imgs, train_transform, device)
    print("  Preloading validation set...")
    val_dataset = CUB_GPU_Dataset(val_imgs, test_transform, device)
    print("  Preloading test set...")
    test_dataset = CUB_GPU_Dataset(test_imgs, test_transform, device)

    return train_dataset, val_dataset, test_dataset
