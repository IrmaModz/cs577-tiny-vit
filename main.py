"""
One entry point for whole project. Every experiment is one command:

    python main.py smoke     [--out-dir DIR]                 tiny run on fake data, CPU, < 1 min
    python main.py train     --experiment NAME --seed S      one full training run
    python main.py ablate    [--experiments ...] [--seeds ...]   patch 4 vs 8, resumable
    python main.py profile   [--experiments ...]             speed / memory / params / FLOPs
    python main.py sanity                                    tiny-batch overfit curve
    python main.py trace     --experiment NAME               print shape after each part

Tests are not here, run them with `pytest -q`.

Paper reference: none
Author: <name>
Adapted from: none
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """
    Make command line parser with subcommands:
        smoke   --out-dir (default "results/sanity")
        train   --experiment (required), --seed (default 0), --out-dir (default "results/ablation")
        ablate  --experiments (list, default baseline_p4 ablation_p8), --seeds (list, default from config)
        profile --experiments (list, default baseline_p4 ablation_p8)
        sanity  (no args)
        trace   --experiment (default baseline_p4)
    """
    # TODO
    raise NotImplementedError


def cmd_smoke(args: argparse.Namespace) -> None:
    """load_experiment("smoke") -> train(..., fake=True). Writes <out_dir>/smoke_seed0.json."""
    # TODO
    raise NotImplementedError


def cmd_train(args: argparse.Namespace) -> None:
    """load_experiment(args.experiment) -> train(...) for one seed."""
    # TODO
    raise NotImplementedError


def cmd_ablate(args: argparse.Namespace) -> None:
    """run_ablation(...) from src/experiments/ablation.py."""
    # TODO
    raise NotImplementedError


def cmd_profile(args: argparse.Namespace) -> None:
    """run_profiling(...) from src/experiments/profiling.py."""
    # TODO
    raise NotImplementedError


def cmd_sanity(args: argparse.Namespace) -> None:
    """
    overfit_batch on 32 fake images, save loss curve:
    results/sanity/overfit_curve.json and docs/figures/overfit.pdf
    """
    # TODO
    raise NotImplementedError


def cmd_trace(args: argparse.Namespace) -> None:
    """trace_shapes for chosen experiment, print table and save results/sanity/shape_trace.csv."""
    # TODO
    raise NotImplementedError


def main() -> None:
    """Parse args and call the right cmd_* function."""
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    main()
