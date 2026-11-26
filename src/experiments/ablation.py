"""
Patch-size ablation: train patch 4 vs patch 8 with 3 seeds each, then summary table and plots.

Paper reference: ViT Section 4 (patch size vs compute trade-off, e.g. ViT-B/16 vs B/32)
Author: <name>, Chatgpt
Adapted from: none
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from configs.config import load_experiment
from src.training.train import train


DEFAULT_EXPERIMENTS = ["baseline_p4", "ablation_p8"]


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
    # Use default experiments if none are specified
    experiments = DEFAULT_EXPERIMENTS if experiments is None else experiments
    output_dir = Path(out_dir)

    for name in experiments:
        # Skip description entries
        if name.startswith("_"):
            continue

        # Load configurations for each experiment
        model_cfg, train_cfg = load_experiment(name)

        # Use seeds from config unless manually specified
        run_seeds = train_cfg.seeds if seeds is None else seeds

        for seed in run_seeds:
            # Check whether the experiment has already been completed
            output_path = output_dir / f"{name}_seed{seed}.json"

            if resume and output_path.exists():
                print(f"Skipping {name}, seed={seed}")
                continue

            # Train the model for the current experiment and seed
            train(model_cfg=model_cfg,
                  train_cfg=train_cfg,
                  seed=seed,
                  experiment=name,
                  out_dir=out_dir,
                  fake=False)

    # Generate summary table and figures
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
    rows = []

    # Extract metrics from each experiment run
    for result in _load_results(out_dir):

        model_cfg = result["config"]["model"]
        patch_size = model_cfg["patch_size"]
        image_size = model_cfg["image_size"]

        # Calculate the number of tokens including the CLS token
        num_tokens = (image_size // patch_size) ** 2 + 1

        rows.append({
            "experiment": result["experiment"],
            "seed": result["seed"],
            "patch_size": patch_size,
            "num_tokens": num_tokens,
            "test_acc": result["test"]["acc"],
            "val_acc_final": result["history"][-1]["val_acc"],
            "train_time_s": result["train_time_s"],
            "num_params": result["num_params"]
        })

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No experiment results found.")

    # Calculate mean and standard deviation for each experiment
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

    # Save the summary table
    summary.to_csv(Path(out_dir) / "summary.csv", index=False)

    # Match experiments using the same seed
    paired = df.pivot(
        index="seed",
        columns="experiment",
        values="test_acc"
    )

    # Calculate paired accuracy differences
    if {"baseline_p4", "ablation_p8"}.issubset(paired.columns):

        paired["acc_difference"] = (
            paired["baseline_p4"] - paired["ablation_p8"]
        )

        print("Paired accuracy differences:")
        print(paired["acc_difference"].dropna())

    return summary


def plot_results(out_dir: str = "results/ablation", fig_dir: str = "docs/figures") -> None:
    """
    Make figures for report and slides:
        - ablation_acc.pdf: test accuracy, point for each seed + mean ± std, both patch sizes
        - training_curves.pdf: val accuracy per epoch, mean ± std band, both patch sizes
    """
    results = _load_results(out_dir)

    # Create the figure directory
    figure_dir = Path(fig_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    # Group runs by experiment
    groups = {}

    for result in results:
        name = result["experiment"]
        groups.setdefault(name, []).append(result)

    # 1. Test accuracy comparison
    plt.figure(figsize=(7, 5))

    for i, (name, runs) in enumerate(groups.items()):

        # Collect test accuracy for each seed
        accs = [r["test"]["acc"] for r in runs]

        # Calculate mean and sample standard deviation
        mean = np.mean(accs)
        std = np.std(accs, ddof=1) if len(accs) > 1 else 0.0

        # Plot individual seed results
        plt.scatter([i] * len(accs), accs, alpha=0.6)

        # Plot mean and standard deviation
        plt.errorbar(
            i, mean, yerr=std,
            fmt="D", capsize=5
        )

    plt.xticks(range(len(groups)), list(groups.keys()))
    plt.ylabel("Test Accuracy")
    plt.title("Patch Size Ablation")
    plt.tight_layout()

    plt.savefig(figure_dir / "ablation_acc.pdf")
    plt.close()

    # 2. Validation accuracy curves
    plt.figure(figsize=(8, 5))

    for name, runs in groups.items():

        # Collect validation accuracy for every epoch and seed
        val_accs = np.array([
            [h["val_acc"] for h in r["history"]]
            for r in runs
        ])

        # Calculate statistics across seeds for each epoch
        mean = val_accs.mean(axis=0)
        std = (
            val_accs.std(axis=0, ddof=1)
            if len(runs) > 1
            else np.zeros_like(mean)
        )

        epochs = np.arange(1, len(mean) + 1)

        # Plot mean validation accuracy
        line, = plt.plot(epochs, mean, label=name)

        # Plot the standard deviation band
        plt.fill_between(
            epochs,
            mean - std,
            mean + std,
            color=line.get_color(),
            alpha=0.2
        )

    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy Curves")
    plt.legend()
    plt.tight_layout()

    plt.savefig(figure_dir / "training_curves.pdf")
    plt.close()


def _load_results(out_dir: str) -> list[dict]:

    results = []

    # Load all completed experiment runs
    for path in sorted(Path(out_dir).glob("*_seed*.json")):
        with path.open("r", encoding="utf-8") as f:
            results.append(json.load(f))

    return results