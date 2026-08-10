"""
Profile GPU preload training to identify bottlenecks.
Runs 2 warmup epochs + 3 measured epochs, timing each stage per batch.
"""
import time
import torch
import hydra
from omegaconf import DictConfig
from pathlib import Path

from models.models import create_model
from models.losses import create_loss
from utils.data import get_data
from utils.utils import reset_random_seeds


def profile_epoch(train_loader, model, optimizer, loss_fn, device, epoch, is_gpu_preload):
    """Run one epoch with per-stage timing."""
    model.train()

    times = {"data": [], "to_device": [], "forward": [], "loss": [], "backward": [], "optim": []}

    torch.cuda.synchronize()
    t_epoch_start = time.perf_counter()
    t_prev = t_epoch_start

    for k, batch in enumerate(train_loader):
        torch.cuda.synchronize()
        t_data = time.perf_counter()
        times["data"].append(t_data - t_prev)

        # Move to device (no-op if gpu_preload since already on GPU)
        batch_features = batch["features"].to(device)
        concepts_true = batch["concepts"].to(device)
        target_true = batch["labels"].to(device)
        torch.cuda.synchronize()
        t_to_device = time.perf_counter()
        times["to_device"].append(t_to_device - t_data)

        # Forward
        concepts_mcmc_probs, triang_cov, target_pred_logits = model(
            batch_features, epoch, c_true=concepts_true
        )
        torch.cuda.synchronize()
        t_forward = time.perf_counter()
        times["forward"].append(t_forward - t_to_device)

        # Loss
        optimizer.zero_grad()
        target_loss, concepts_loss, prec_loss, total_loss = loss_fn(
            concepts_mcmc_probs, concepts_true, target_pred_logits, target_true,
            triang_cov,
        )
        torch.cuda.synchronize()
        t_loss = time.perf_counter()
        times["loss"].append(t_loss - t_forward)

        # Backward
        total_loss.backward()
        torch.cuda.synchronize()
        t_backward = time.perf_counter()
        times["backward"].append(t_backward - t_loss)

        # Optimizer step
        optimizer.step()
        torch.cuda.synchronize()
        t_optim = time.perf_counter()
        times["optim"].append(t_optim - t_backward)

        t_prev = t_optim

    torch.cuda.synchronize()
    t_epoch_end = time.perf_counter()

    return times, t_epoch_end - t_epoch_start


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    print(f"Profiling with gpu_preload={config.gpu_preload}")
    gen = reset_random_seeds(config.seed)
    device = torch.device("cuda")

    train_loader, val_loader, test_loader = get_data(config, config.data, gen)
    model = create_model(config).to(device)
    loss_fn = create_loss(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.model.learning_rate)

    # Warmup
    print("Warmup (2 epochs)...")
    for e in range(2):
        profile_epoch(train_loader, model, optimizer, loss_fn, device, e, config.gpu_preload)
    print()

    # Measured epochs
    all_times = {"data": [], "to_device": [], "forward": [], "loss": [], "backward": [], "optim": []}
    epoch_times = []

    print("Profiling (3 epochs)...")
    for e in range(3):
        times, epoch_time = profile_epoch(train_loader, model, optimizer, loss_fn, device, e + 2, config.gpu_preload)
        epoch_times.append(epoch_time)
        for k in all_times:
            all_times[k].extend(times[k])
        print(f"  Epoch {e+1}: {epoch_time:.2f}s")

    # Summary
    total = sum(sum(v) for v in all_times.values())
    print(f"\n{'='*60}")
    print(f"Profile Summary (3 epochs, {len(all_times['data'])} batches)")
    print(f"{'='*60}")
    print(f"{'Stage':<15} {'Total (s)':>10} {'Per-batch (ms)':>15} {'% of time':>10}")
    print(f"{'-'*60}")
    for stage in ["data", "to_device", "forward", "loss", "backward", "optim"]:
        vals = all_times[stage]
        s = sum(vals)
        avg_ms = (s / len(vals)) * 1000
        pct = (s / total) * 100
        print(f"{stage:<15} {s:>10.3f} {avg_ms:>15.2f} {pct:>9.1f}%")
    print(f"{'-'*60}")
    print(f"{'TOTAL':<15} {total:>10.3f}")
    print(f"\nAvg epoch: {sum(epoch_times)/len(epoch_times):.2f}s")
    print(f"Avg batch: {(total / len(all_times['data'])) * 1000:.2f}ms")
    print(f"Throughput: {len(all_times['data']) / total * 64:.0f} samples/s")


if __name__ == "__main__":
    main()
