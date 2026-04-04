# Claude's Speedup Estimates for GPU Preload Prototype

## Context (from the article)

| Baseline | Runtime | Notes |
|---|---|---|
| Original | ~87 min | 300 epochs, RTX 2080, 2 workers |
| Fully optimized (caching + vectorized losses + torch.compile) | ~13 min | 6.7x speedup already achieved |

Profiling showed **100/145 seconds in DataLoaders** during a 5-epoch run (~69% of
wall time was data loading). The existing shared-memory cache eliminated repeated
disk I/O after epoch 1, but still does: shared-memory lookup -> numpy unpad ->
convert to PIL -> CPU transforms (ColorJitter, RandomResizedCrop, Resize, Flip,
ToTensor, Normalize) -> pin_memory -> PCIe transfer to GPU. Every batch, every
epoch.

## What GPU preload eliminates

- PIL Image creation (~6K per epoch)
- All CPU-side transform computation (the resize/crop kernels are the expensive part)
- Pin-memory copy + PCIe CPU->GPU transfer per batch (~93 batches/epoch x 300 epochs)
- DataLoader worker process overhead and IPC serialization
- `persistent_workers` keepalive cost

## What GPU preload adds

- One-time startup cost: ~10-20 seconds to load all images from disk and transfer to GPU
- ~2.5-4.5 GB additional GPU VRAM (depending on image sizes in the split)
- Per-image Python loop in `get_batch()` (kernel launch overhead per sample, not batched)
- GPU transform compute (fast individually, but serial per image due to variable sizes)

## Estimates

### Over the already-optimized 13-minute baseline

The 13-min version already has caching, so disk I/O is gone. The remaining data
loading cost is PIL conversion + CPU transforms + transfer. I estimate this is
roughly **30-40% of the 13 minutes** (~4-5 min), since the vectorized losses and
torch.compile already shrank the compute portion significantly.

GPU-side transforms should cut that data portion by roughly **60-70%** (transforms
are faster on GPU, but the per-image Python loop and kernel launch overhead prevent
a full 10x on the data path alone).

| Metric | Estimate |
|---|---|
| Data loading time (before) | ~4-5 min of 13 min |
| Data loading time (after) | ~1-2 min |
| Startup overhead | ~15 sec (one-time) |
| **New total** | **~9-10 min** |
| **Speedup over optimized baseline** | **~1.3-1.5x** |

### Over the original 87-minute baseline (combined with all other optimizations)

| Metric | Estimate |
|---|---|
| **New total** | **~9-10 min** |
| **Speedup over original** | **~9-10x** |

### Where the remaining time goes

At ~9-10 minutes for 300 epochs, the bottleneck shifts decisively to **model
forward/backward passes** (ResNet18 encoder + MC sampling + loss computation).
Further speedups would need:

- Batched GPU transforms (requires pre-resizing images to uniform size at load time)
- `torch.compile` on the v2 transform pipeline
- Mixed precision training (AMP) for the model itself
- Larger batch sizes to improve GPU utilization

### Confidence

**Medium confidence.** The direction is certain (this will be faster), but the
magnitude depends heavily on:

1. How much of the 13-min runtime is actually data loading vs. compute (I'm
   estimating 30-40%, but profiling would nail this down)
2. The per-image Python loop overhead in `get_batch()` -- if this is significant,
   batched transforms would unlock more
3. GPU memory pressure -- if the preloaded images compete with model activations
   for VRAM, it could cause spillover and slow down training

The honest answer: **profile it.** Run 5 epochs with `gpu_preload=True` vs
`gpu_preload=False` on the same GPU and compare wall times. The prototype is
structured to make this a one-flag comparison.
