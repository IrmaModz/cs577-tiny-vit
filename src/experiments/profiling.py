"""
Basic efficiency profile: time, peak memory, params and FLOPs for patch 4 and patch 8.

Paper reference: ViT Section 4.4 / Figure 5 (compute vs performance), project requirement 4.7
Author: <name>
Adapted from: none
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from configs.config import ViTConfig

if TYPE_CHECKING:
    import pandas as pd


def profile_model(
    model_cfg: ViTConfig,
    batch_size: int = 128,
    device: str = "auto",
    warmup_iters: int = 10,
    iters_per_run: int = 20,
    runs: int = 5,
) -> dict:
    """
    Measure one model config. How we measure (write same in report):
        - input is random tensor (batch_size, 3, 32, 32), so we measure only model, not data loading
        - two modes: "inference" (eval + no_grad) and "train" (forward + backward + AdamW step)
        - first warmup_iters not timed, then `runs` times `iters_per_run` iterations
        - torch.cuda.synchronize() before reading time, use time.perf_counter
        - peak memory: reset_peak_memory_stats -> max_memory_allocated ("n/a" on CPU)
        - params count, and FLOPs per image with torch.utils.flop_counter.FlopCounterMode
        - GPU utilization we dont report (nvidia-smi not reliable for short runs),
          params + FLOPs are our proxy, like we wrote in proposal

    Returns:
        dict with results for both modes (mean and std of ms per batch, images/s,
        peak memory MB, params, GFLOPs per image, device, gpu name)
    """
    # TODO
    raise NotImplementedError


def run_profiling(
    experiments: list[str] | None = None, out_dir: str = "results/profiling"
) -> "pd.DataFrame":
    """
    Run profile_model for each experiment (default ["baseline_p4", "ablation_p8"]).

    Saves:
        - results/profiling/profile.csv with columns:
          experiment, patch_size, num_tokens, device, gpu_name, mode, batch_size,
          warmup_iters, runs, ms_per_batch_mean, ms_per_batch_std, images_per_s,
          peak_mem_mb, params, gflops_per_image
        - docs/figures/profile_table.tex (table ready for report)
    """
    # TODO
    raise NotImplementedError
