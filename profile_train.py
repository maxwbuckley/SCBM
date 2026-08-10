"""
Kernel-level profiler for one SCBM training step.

Runs the exact same step as train_one_epoch_scbm, but instrumented three ways:

  1. Sync detection  -- flags every implicit GPU->CPU synchronisation, with the
                        Python line that caused it.
  2. CUDA-event timing -- GPU-busy time per region vs. wall time, so the gap
                        between them exposes launch/host overhead.
  3. torch.profiler  -- ranked kernel table + chrome trace.

Every region is also wrapped in an NVTX range, so the same script under
`nsys profile` produces a readable timeline.

Usage:
    python profile_train.py +model=SCBM +data=CUB data.data_path=datasets/ \
        gpu_preload=True

    nsys profile -o trace --capture-range=cudaProfilerApi \
        python profile_train.py +model=SCBM +data=CUB data.data_path=datasets/
"""

import itertools
import os
import traceback
import warnings
from collections import Counter

import hydra
import torch
from omegaconf import DictConfig

from models.losses import create_loss
from models.models import create_model
from utils.data import get_data
from utils.training import create_optimizer
from utils.utils import configure_performance, perf_option, reset_random_seeds

WARMUP_STEPS = 20
MEASURE_STEPS = 30
REPO = os.path.dirname(os.path.abspath(__file__))


class nvtx_range:
    """NVTX range that is also a no-op-cheap context manager."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        torch.cuda.nvtx.range_push(self.name)
        return self

    def __exit__(self, *exc):
        torch.cuda.nvtx.range_pop()
        return False


def train_step(batch, model, optimizer, loss_fn, device, epoch):
    """One training step, matching utils/training.py:train_one_epoch_scbm."""
    with nvtx_range("h2d"):
        batch_features = batch["features"].to(device, non_blocking=True)
        target_true = batch["labels"].to(device, non_blocking=True)
        concepts_true = batch["concepts"].to(device, non_blocking=True)

    with nvtx_range("forward"):
        concepts_mcmc_probs, triang_cov, target_pred_logits = model(
            batch_features, epoch, c_true=concepts_true
        )

    with nvtx_range("zero_grad"):
        optimizer.zero_grad()

    with nvtx_range("loss"):
        target_loss, concepts_loss, prec_loss, total_loss = loss_fn(
            concepts_mcmc_probs,
            concepts_true,
            target_pred_logits,
            target_true,
            triang_cov,
        )

    with nvtx_range("backward"):
        total_loss.backward()

    with nvtx_range("optim"):
        optimizer.step()

    return total_loss


# ---------------------------------------------------------------------------
#   1. Sync detection
# ---------------------------------------------------------------------------


def detect_syncs(batches, model, optimizer, loss_fn, device):
    """
    Run a few steps with sync debug mode on, recording the Python frame that
    triggered each implicit device synchronisation.
    """
    hits = Counter()

    def record(message, category, filename, lineno, file=None, line=None):
        # Walk the stack to the deepest frame inside this repo -- that is the
        # line the user actually wrote that caused the sync. Fall back to the
        # profiler's own frame if the op was called straight from train_step.
        repo_frames = [
            f for f in traceback.extract_stack() if f.filename.startswith(REPO)
        ]
        own = [f for f in repo_frames if not f.filename.endswith("profile_train.py")]
        frame = (own or repo_frames or [None])[-1]
        if frame is None:
            origin = f"{filename}:{lineno}"
        else:
            rel = os.path.relpath(frame.filename, REPO)
            origin = f"{rel}:{frame.lineno}  {(frame.line or '').strip()}"
        hits[(str(message).split("(Triggered")[0].strip(), origin)] += 1

    torch.cuda.synchronize()
    # NOTE: deliberately not using warnings.catch_warnings() -- it buffers and
    # replays warnings, which destroys the stack we need for attribution.
    old_showwarning = warnings.showwarning
    old_filters = warnings.filters[:]
    warnings.showwarning = record
    warnings.simplefilter("always")
    torch.cuda.set_sync_debug_mode("warn")
    try:
        for batch in itertools.islice(batches, 3):
            train_step(batch, model, optimizer, loss_fn, device, epoch=0)
    finally:
        torch.cuda.set_sync_debug_mode("default")
        warnings.showwarning = old_showwarning
        warnings.filters[:] = old_filters

    print(f"\n{'=' * 78}")
    print("1. IMPLICIT GPU->CPU SYNCHRONISATIONS  (3 steps)")
    print(f"{'=' * 78}")
    if not hits:
        print("None detected.")
    else:
        total = sum(hits.values())
        print(f"{total} sync(s) over 3 steps = {total / 3:.1f} per step\n")
        for (msg, origin), n in hits.most_common():
            print(f"  [{n:>2}x] {origin}")
            print(f"        {msg[:150]}")
    return hits


# ---------------------------------------------------------------------------
#   2. CUDA-event timing: GPU busy vs wall clock
# ---------------------------------------------------------------------------


def time_regions(batches, model, optimizer, loss_fn, device):
    """
    Measure whole-step GPU time with CUDA events and compare against wall time.

    A large wall-vs-GPU gap means the GPU is idle waiting on the host: either
    kernel-launch overhead or a synchronisation stall.
    """
    import time

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(MEASURE_STEPS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(MEASURE_STEPS)]

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i, batch in enumerate(itertools.islice(batches, MEASURE_STEPS)):
        starts[i].record()
        train_step(batch, model, optimizer, loss_fn, device, epoch=0)
        ends[i].record()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    gpu_ms = sum(s.elapsed_time(e) for s, e in zip(starts, ends))
    wall_ms = wall * 1000

    print(f"\n{'=' * 78}")
    print(f"2. GPU-BUSY vs WALL CLOCK  ({MEASURE_STEPS} steps)")
    print(f"{'=' * 78}")
    print(f"  wall time        {wall_ms:>9.2f} ms   ({wall_ms / MEASURE_STEPS:.2f} ms/step)")
    print(f"  GPU busy         {gpu_ms:>9.2f} ms   ({gpu_ms / MEASURE_STEPS:.2f} ms/step)")
    print(f"  idle / overhead  {wall_ms - gpu_ms:>9.2f} ms   "
          f"({100 * (wall_ms - gpu_ms) / wall_ms:.1f}% of wall)")
    print(f"  throughput       {MEASURE_STEPS / wall:>9.2f} it/s")
    return wall_ms, gpu_ms


# ---------------------------------------------------------------------------
#   3. torch.profiler kernel table
# ---------------------------------------------------------------------------


def kernel_table(batches, model, optimizer, loss_fn, device, trace_path):
    from torch.profiler import ProfilerActivity, profile

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for batch in itertools.islice(batches, 10):
            train_step(batch, model, optimizer, loss_fn, device, epoch=0)
        torch.cuda.synchronize()

    # torch renamed self_cuda_time_total -> self_device_time_total
    try:
        table = prof.key_averages().table(
            sort_by="self_device_time_total", row_limit=30
        )
    except (KeyError, AssertionError):
        table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30)

    # Layout-conversion tax: cuDNN's fast conv kernels are NHWC, so an NCHW
    # model pays a transpose in and out of every convolution.
    ev = prof.key_averages()
    total = sum(e.self_device_time_total for e in ev)
    transpose = sum(
        e.self_device_time_total
        for e in ev
        if "nchwToNhwc" in e.key or "nhwcToNchw" in e.key
    )
    n_transpose = sum(
        e.count for e in ev if "nchwToNhwc" in e.key or "nhwcToNchw" in e.key
    )

    print(f"\n{'=' * 78}")
    print("3. TOP KERNELS BY SELF GPU TIME  (10 steps)")
    print(f"{'=' * 78}")
    print(f"  layout transposes: {transpose / 1000:.2f} ms over {n_transpose} launches "
          f"({100 * transpose / total:.1f}% of GPU time)")
    print(table)

    prof.export_chrome_trace(trace_path)
    print(f"\nChrome trace written to: {trace_path}")
    print("  View at chrome://tracing or https://ui.perfetto.dev")


# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    device = torch.device("cuda")

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"torch:  {torch.__version__}")
    print(f"gpu_preload={config.gpu_preload}  batch={config.model.train_batch_size}  "
          f"mc={config.model.num_monte_carlo}")

    gen = reset_random_seeds(config.seed)
    configure_performance(config)  # must follow reset_random_seeds, which it overrides

    print(f"channels_last={perf_option(config, 'channels_last', False)}  "
          f"cudnn.benchmark={torch.backends.cudnn.benchmark}  "
          f"deterministic={torch.are_deterministic_algorithms_enabled()}  "
          f"fill_uninit={torch.utils.deterministic.fill_uninitialized_memory}")

    train_loader, _, _ = get_data(config, config.data, gen)
    model = create_model(config).to(device)
    loss_fn = create_loss(config)
    optimizer = create_optimizer(config.model, model)
    model.train()

    # Endless stream of batches so each phase can just take what it needs.
    batches = (b for _ in itertools.count() for b in train_loader)

    print(f"\nWarmup ({WARMUP_STEPS} steps)...")
    for batch in itertools.islice(batches, WARMUP_STEPS):
        train_step(batch, model, optimizer, loss_fn, device, epoch=0)
    torch.cuda.synchronize()

    detect_syncs(batches, model, optimizer, loss_fn, device)
    time_regions(batches, model, optimizer, loss_fn, device)

    # Only the measured region goes into an nsys capture range.
    torch.cuda.cudart().cudaProfilerStart()
    trace_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    kernel_table(batches, model, optimizer, loss_fn, device,
                 f"{trace_path}/trace.json")
    torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
