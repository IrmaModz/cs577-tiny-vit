"""
Patch-size ablation: train patch 4 vs patch 8 with 3 seeds each, then summary table and plots.

Paper reference: ViT Section 4 (patch size vs compute trade-off, e.g. ViT-B/16 vs B/32)
Author: <name>
Adapted from: none
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def run_ablation(
    experiments: list[str] | None = None,
    seeds: list[int] | None = None,
    out_dir: str = "results/ablation",
    resume: bool = True,
) -> "pd.DataFrame":
    """
    For each experiment and each seed: load_experiment(name) -> train(...).

    - default experiments: ["baseline_p4", "ablation_p8"]
    - default seeds: train_cfg.seeds from config
    - resume=True: skip run if its JSON already exists (Colab disconnects a lot!)
    - skip "_description" key in experiments.json, it is not experiment
    - at end call summarize() and plot_results()

    Returns:
        summary table (same as summary.csv)
    """
    # TODO
    raise NotImplementedError


def summarize(out_dir: str = "results/ablation") -> "pd.DataFrame":
    """
    Read all run JSONs and make results/ablation/summary.csv.

    Columns:
        experiment, patch_size, num_tokens, n_seeds, test_acc_mean, test_acc_std,
        val_acc_final_mean, train_time_mean_s, num_params

    Also: difference acc(p4) - acc(p8) for each seed (same seeds -> paired), print or save it.
    """
    # TODO
    raise NotImplementedError


def plot_results(out_dir: str = "results/ablation", fig_dir: str = "docs/figures") -> None:
    """
    Make figures for report and slides:
        - ablation_acc.pdf: test accuracy, point for each seed + mean ± std, both patch sizes
        - training_curves.pdf: val accuracy per epoch, mean ± std band, both patch sizes
    """
    # TODO
    raise NotImplementedError
