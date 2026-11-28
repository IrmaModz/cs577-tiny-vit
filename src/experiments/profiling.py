"""
Basic efficiency profile: time, peak memory, params and FLOPs for patch 4 and patch 8.

Paper reference: ViT Section 4.4 / Figure 5 (compute vs performance), project requirement 4.7
Author: <name>
Adapted from: none
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import torch
from torch import nn

from configs.config import ViTConfig, load_experiment
from src.models.vit import ViT, count_parameters
from src.reproducibility import get_device, get_env_info, set_seed

if TYPE_CHECKING:
    import pandas as pd

DEFAULT_EXPERIMENTS = ["baseline_p4", "ablation_p8"]

CSV_COLUMNS = [
    "experiment", "patch_size", "num_tokens", "device", "gpu_name", "mode", "batch_size",
    "warmup_iters", "runs", "ms_per_batch_mean", "ms_per_batch_std", "images_per_s",
    "peak_mem_mb", "params", "gflops_per_image",
]


def _sync(device: torch.device) -> None:
    """GPU works async, so before reading time we wait until all GPU work is finished."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_step(
    step: Callable[[], None], device: torch.device, warmup_iters: int, iters_per_run: int, runs: int
) -> tuple[float, float]:
    """Run `step` many times. Returns (mean, std) of milliseconds per batch over `runs`."""
    for _ in range(warmup_iters):  # not timed: first iterations are slow (memory, cuDNN setup)
        step()
    _sync(device)

    ms_per_run = []
    for _ in range(runs):
        _sync(device)
        start = time.perf_counter()
        for _ in range(iters_per_run):
            step()
        _sync(device)
        ms_per_run.append((time.perf_counter() - start) * 1000 / iters_per_run)

    std = statistics.stdev(ms_per_run) if runs > 1 else 0.0
    return statistics.mean(ms_per_run), std


def _peak_mem_mb(device: torch.device) -> float | str:
    """Peak GPU memory since last reset, in MiB (1 MiB = 2**20 bytes). On CPU we can not measure it like this."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**20
    return "n/a"


def _reset_peak_mem(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


@torch.no_grad()
def _gflops_per_image(model: nn.Module, model_cfg: ViTConfig, device: torch.device) -> float | None:
    """FLOPs of one forward for ONE image, in GFLOPs. None if this torch has no FlopCounterMode."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None
    was_training = model.training
    model.eval()
    x = torch.randn(1, model_cfg.in_channels, model_cfg.image_size, model_cfg.image_size, device=device)
    counter = FlopCounterMode(display=False)
    with counter:
        model(x)
    model.train(was_training)
    return counter.get_total_flops() / 1e9


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
    dev = get_device(device)
    set_seed(0)  # same init weights every time (speed does not depend on it, but cleaner)

    model = ViT(model_cfg).to(dev)
    x = torch.randn(batch_size, model_cfg.in_channels, model_cfg.image_size, model_cfg.image_size, device=dev)
    y = torch.randint(0, model_cfg.num_classes, (batch_size,), device=dev)

    result = {
        "device": dev.type,
        "gpu_name": get_env_info(dev)["device_name"],  # name of the device really used
        "batch_size": batch_size,
        "warmup_iters": warmup_iters,
        "runs": runs,
        "params": count_parameters(model),
        "gflops_per_image": _gflops_per_image(model, model_cfg, dev),
    }

    # --- inference: eval mode, no gradients -------------------------------
    model.eval()

    def inference_step() -> None:
        with torch.no_grad():
            model(x)

    _reset_peak_mem(dev)
    mean_ms, std_ms = _time_step(inference_step, dev, warmup_iters, iters_per_run, runs)
    result["inference"] = {
        "ms_per_batch_mean": mean_ms,
        "ms_per_batch_std": std_ms,
        "images_per_s": batch_size / (mean_ms / 1000),
        "peak_mem_mb": _peak_mem_mb(dev),
    }

    # --- train: forward + backward + AdamW step ----------------------------
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    def train_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    _reset_peak_mem(dev)
    mean_ms, std_ms = _time_step(train_step, dev, warmup_iters, iters_per_run, runs)
    result["train"] = {
        "ms_per_batch_mean": mean_ms,
        "ms_per_batch_std": std_ms,
        "images_per_s": batch_size / (mean_ms / 1000),
        "peak_mem_mb": _peak_mem_mb(dev),
    }
    return result


def _fmt(value, digits: int = 1) -> str:
    """Number -> short text for LaTeX table. Text (like "n/a") stays same."""
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    return f"{value:,.{digits}f}"


def _write_latex_table(df: "pd.DataFrame", path: Path) -> None:
    """Small LaTeX table for report. We write it by hand, so no extra package is needed."""
    lines = [
        "% made by src/experiments/profiling.py, dont edit by hand",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Experiment & Mode & Tokens & ms/batch & images/s & Peak mem (MiB) & Params & GFLOPs/img \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        name = str(r["experiment"]).replace("_", r"\_")
        lines.append(
            f"{name} & {r['mode']} & {r['num_tokens']} & "
            f"{_fmt(r['ms_per_batch_mean'])} $\\pm$ {_fmt(r['ms_per_batch_std'])} & "
            f"{_fmt(r['images_per_s'], 0)} & {_fmt(r['peak_mem_mb'], 0)} & "
            f"{int(r['params']):,} & {_fmt(r['gflops_per_image'], 3)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


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
    import pandas as pd

    experiments = experiments or DEFAULT_EXPERIMENTS
    rows = []
    for name in experiments:
        model_cfg, train_cfg = load_experiment(name)
        # batch size and device from config, so we dont hard-code numbers here
        r = profile_model(model_cfg, batch_size=train_cfg.batch_size, device=train_cfg.device)
        for mode in ("inference", "train"):
            rows.append({
                "experiment": name,
                "patch_size": model_cfg.patch_size,
                "num_tokens": model_cfg.num_tokens,
                "device": r["device"],
                "gpu_name": r["gpu_name"],
                "mode": mode,
                "batch_size": r["batch_size"],
                "warmup_iters": r["warmup_iters"],
                "runs": r["runs"],
                **r[mode],  # ms_per_batch_mean, ms_per_batch_std, images_per_s, peak_mem_mb
                "params": r["params"],
                "gflops_per_image": r["gflops_per_image"],
            })

    df = pd.DataFrame(rows, columns=CSV_COLUMNS)

    out_path = Path(out_dir) / "profile.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    _write_latex_table(df, Path("docs/figures/profile_table.tex"))
    return df