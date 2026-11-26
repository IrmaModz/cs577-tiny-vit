"""
Patch-size ablation: train patch 4 vs patch 8 with 3 seeds each, then summary table and plots.

Paper reference: ViT Section 4 (patch size vs compute trade-off, e.g. ViT-B/16 vs B/32)
Author: <name>, Chatgpt
Adapted from: none
"""

from __future__ import annotations

#from typing import TYPE_CHECKING
#
#if TYPE_CHECKING:
#    import pandas as pd
import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from configs.config import load_experiment
from src.training import train


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
    if experiments is None:
        experiments = ["baseline_p4", "ablation_p8"]

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for experiment in experiments:

        # Skip comment keys
        if experiment.startswith("_"):
            continue

        model_cfg, train_cfg = load_experiment(experiment)

        run_seeds = train_cfg.seeds if seeds is None else seeds

        for seed in run_seeds:

            result_path = (
                output_dir / f"{experiment}_seed{seed}.json"
            )

            if resume and result_path.exists():
                print(f"Skipping {experiment}, seed={seed}")
                continue

            print(f"Running {experiment}, seed={seed}")

            train(
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                seed=seed,
                experiment=experiment,
                out_dir=out_dir
            )

    summary = summarize(out_dir)
    plot_results(out_dir)

    return summary


def summarize(out_dir: str = "results/ablation") -> "pd.DataFrame":
    """
    Read all run JSONs and make results/ablation/summary.csv.

    Columns:
        experiment, patch_size, num_tokens, n_seeds, test_acc_mean, test_acc_std,
        val_acc_final_mean, train_time_mean_s, num_params

    Also: difference acc(p4) - acc(p8) for each seed (same seeds -> paired), print or save it.
    """
    runs = load_runs(out_dir)
    rows = []

    for run in runs:

        model_cfg = run["config"]["model"]

        patch_size = model_cfg["patch_size"]
        image_size = model_cfg["image_size"]

        num_tokens = (image_size // patch_size) ** 2 + 1

        rows.append({
            "experiment": run["experiment"],
            "seed": run["seed"],
            "patch_size": patch_size,
            "num_tokens": num_tokens,
            "test_acc": run["test"]["acc"],
            "val_acc_final": run["history"][-1]["val_acc"],
            "train_time_s": run["train_time_s"],
            "num_params": run["num_params"]
        })

    df = pd.DataFrame(rows)

    summary = df.groupby(
        ["experiment", "patch_size", "num_tokens"],
        as_index=False
    ).agg(
        n_seeds=("seed", "nunique"),
        test_acc_mean=("test_acc", "mean"),
        test_acc_std=("test_acc", "std"),
        val_acc_final_mean=("val_acc_final", "mean"),
        train_time_mean_s=("train_time_s", "mean"),
        num_params=("num_params", "first")
    )

    summary = summary.sort_values("patch_size")

    # Save summary
    summary.to_csv(
        Path(out_dir) / "summary.csv",
        index=False
    )

    print(summary.to_string(index=False))

    # Paired comparison
    paired = df.pivot(
        index="seed",
        columns="experiment",
        values="test_acc"
    )

    if {"baseline_p4", "ablation_p8"}.issubset(paired.columns):

        paired = paired[
            ["baseline_p4", "ablation_p8"]
        ].dropna().copy()

        paired["acc_diff"] = (
            paired["baseline_p4"] - paired["ablation_p8"]
        )

        paired.to_csv(
            Path(out_dir) / "paired_difference.csv"
        )

        print("\nPaired accuracy differences:")
        print(paired)

        print("Mean difference:", paired["acc_diff"].mean())
        print("Std difference:", paired["acc_diff"].std())

    return summary


def plot_results(out_dir: str = "results/ablation", fig_dir: str = "docs/figures") -> None:
    """
    Make figures for report and slides:
        - ablation_acc.pdf: test accuracy, point for each seed + mean ± std, both patch sizes
        - training_curves.pdf: val accuracy per epoch, mean ± std band, both patch sizes
    """
    runs = load_runs(out_dir)

    figure_dir = Path(fig_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    acc_rows = []
    history_rows = []

    for run in runs:

        experiment = run["experiment"]
        seed = run["seed"]
        patch_size = run["config"]["model"]["patch_size"]

        acc_rows.append({
            "experiment": experiment,
            "seed": seed,
            "patch_size": patch_size,
            "test_acc": run["test"]["acc"]
        })

        for epoch in run["history"]:
            history_rows.append({
                "experiment": experiment,
                "seed": seed,
                "patch_size": patch_size,
                "epoch": epoch["epoch"],
                "val_acc": epoch["val_acc"]
            })

    acc_df = pd.DataFrame(acc_rows)
    history_df = pd.DataFrame(history_rows)

    # ----------------------------------
    # Figure 1: Test accuracy
    # ----------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))

    groups = list(
        acc_df.sort_values("patch_size").groupby(
            ["experiment", "patch_size"], sort=False
        )
    )

    labels = []

    for i, ((experiment, patch_size), group) in enumerate(groups):

        values = group["test_acc"].to_numpy() * 100

        mean = values.mean()
        std = values.std(ddof=1) if len(values) > 1 else 0.0

        # Small horizontal offsets for individual seeds
        offsets = (
            np.linspace(-0.08, 0.08, len(values))
            if len(values) > 1
            else np.zeros(1)
        )

        ax.scatter(
            i + offsets,
            values,
            alpha=0.7
        )

        ax.errorbar(
            i,
            mean,
            yerr=std,
            fmt="D",
            capsize=6
        )

        labels.append(f"Patch {patch_size}\n{experiment}")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)

    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Patch Size Ablation")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(figure_dir / "ablation_acc.pdf")
    plt.close(fig)

    # ----------------------------------
    # Figure 2: Validation accuracy
    # ----------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))

    for (experiment, patch_size), group in history_df.groupby(
        ["experiment", "patch_size"]
    ):

        stats = group.groupby("epoch")["val_acc"].agg(
            ["mean", "std"]
        )

        epochs = stats.index.to_numpy()
        mean = stats["mean"].to_numpy() * 100
        std = stats["std"].fillna(0).to_numpy() * 100

        line, = ax.plot(
            epochs,
            mean,
            label=f"{experiment} (p={patch_size})"
        )

        ax.fill_between(
            epochs,
            mean - std,
            mean + std,
            color=line.get_color(),
            alpha=0.2
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Validation Accuracy over Training")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(figure_dir / "training_curves.pdf")
    plt.close(fig)


def load_runs(out_dir: str) -> list[dict]:

    runs = []

    for path in sorted(Path(out_dir).glob("*_seed*.json")):
        with open(path, encoding="utf-8") as f:
            runs.append(json.load(f))

    if not runs:
        raise ValueError(f"No run results found in {out_dir}")

    return runs