"""
One entry point for whole project. Every experiment is one command:

    python main.py smoke     [--out-dir DIR]                 tiny run on fake data, CPU, < 1 min
    python main.py train     --experiment NAME --seed S      one full training run
    python main.py ablate    [--experiments ...] [--seeds ...]   patch 4 vs 8, resumable
    python main.py profile   [--experiments ...]             speed / memory / params / FLOPs
    python main.py sanity                                    tiny-batch overfit curve
    python main.py trace     --experiment NAME               print shape after each part

Tests are not here, run them with `pytest -q`.

Note: imports of project code are inside each cmd_* function. So one command works
even if code for other commands is not ready yet, and `python main.py --help` is fast.

Paper reference: none
Author: <name>
Adapted from: none
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

DEFAULT_EXPERIMENTS = ["baseline_p4", "ablation_p8"]


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
    parser = argparse.ArgumentParser(description="CS577 Tiny ViT: one command for every experiment.")
    # required=True: running `python main.py` without command shows error and help
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("smoke", help="tiny run on fake data, CPU, < 1 min")
    p.add_argument("--out-dir", default="results/sanity")
    p.set_defaults(func=cmd_smoke)

    p = sub.add_parser("train", help="one full training run")
    p.add_argument("--experiment", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="results/ablation")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("ablate", help="patch 4 vs 8 for all seeds, resumable")
    p.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS)
    # None means: take seeds from TrainConfig
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.set_defaults(func=cmd_ablate)

    p = sub.add_parser("profile", help="speed / memory / params / FLOPs")
    p.add_argument("--experiments", nargs="+", default=DEFAULT_EXPERIMENTS)
    p.set_defaults(func=cmd_profile)

    p = sub.add_parser("sanity", help="tiny-batch overfit curve")
    p.set_defaults(func=cmd_sanity)

    p = sub.add_parser("trace", help="print shape after each part of the model")
    p.add_argument("--experiment", default="baseline_p4")
    p.set_defaults(func=cmd_trace)

    return parser


def cmd_smoke(args: argparse.Namespace) -> None:
    """load_experiment("smoke") -> train(..., fake=True). Writes <out_dir>/smoke_seed0.json."""
    # TODO (Jungwoo): needs train() from src/training/train.py
    raise NotImplementedError


def cmd_train(args: argparse.Namespace) -> None:
    """load_experiment(args.experiment) -> train(...) for one seed."""
    # TODO (Jungwoo): needs train() from src/training/train.py
    raise NotImplementedError


def cmd_ablate(args: argparse.Namespace) -> None:
    """run_ablation(...) from src/experiments/ablation.py."""
    # TODO (Jungwoo): needs src/experiments/ablation.py
    raise NotImplementedError


def cmd_profile(args: argparse.Namespace) -> None:
    """run_profiling(...) from src/experiments/profiling.py."""
    # TODO (Jungwoo): needs src/experiments/profiling.py
    raise NotImplementedError


def cmd_sanity(args: argparse.Namespace) -> None:
    """
    overfit_batch on 32 fake images, save loss curve:
    results/sanity/overfit_curve.json and docs/figures/overfit.pdf
    """
    # TODO (Jungwoo): needs overfit_batch() from src/training/train.py
    raise NotImplementedError


def cmd_trace(args: argparse.Namespace) -> None:
    """trace_shapes for chosen experiment, print table and save results/sanity/shape_trace.csv."""
    import torch

    from configs.config import load_experiment
    from src.models.vit import ViT, trace_shapes
    from src.reproducibility import set_seed

    model_cfg, _ = load_experiment(args.experiment)
    set_seed(0)
    model = ViT(model_cfg)

    # one fake image is enough, we only need shapes. N = 1 in the table
    x = torch.zeros(1, model_cfg.in_channels, model_cfg.image_size, model_cfg.image_size)
    shapes = trace_shapes(model, x)  # [("input", (1, 3, 32, 32)), ("patch_embed", ...), ...]

    # print table
    print(f"Shape trace for {args.experiment} (N = 1)")
    print(f"{'part':<15}shape")
    for name, shape in shapes:
        print(f"{name:<15}{shape}")

    # save csv for report table
    out_path = Path("results/sanity/shape_trace.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "part", "shape"])
        for name, shape in shapes:
            writer.writerow([args.experiment, name, " x ".join(str(s) for s in shape)])
    print(f"saved {out_path}")


def main() -> None:
    """Parse args and call the right cmd_* function."""
    args = build_parser().parse_args()
    args.func(args)  # every subcommand saved its cmd_* function in `func`


if __name__ == "__main__":
    main()