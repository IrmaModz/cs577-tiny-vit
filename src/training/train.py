"""
Training pipeline: optimizer, learning rate schedule, training loop, tiny-batch overfit
helper, and saving one result JSON per run.

Paper reference: ViT Section 4.1 / Appendix B.1 (Adam-type optimizer, warmup, weight decay)
Author: <name>
Adapted from: none
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from configs.config import TrainConfig, ViTConfig


def build_optimizer(model: nn.Module, train_cfg: TrainConfig) -> torch.optim.Optimizer:
    """
    AdamW with lr and weight_decay from config.
    NO weight decay for: biases, LayerNorm weights, cls_token, pos_embed.
    """
    # TODO
    raise NotImplementedError


def build_scheduler(
    optimizer: torch.optim.Optimizer, train_cfg: TrainConfig, steps_per_epoch: int
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Learning rate per step: linear warmup for warmup_epochs, then cosine down to ~0.
    """
    # TODO
    raise NotImplementedError


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    grad_clip: float | None,
) -> dict:
    """
    One pass over train data.
    Each step: forward -> loss -> backward -> (clip grads) -> optimizer step -> scheduler step.

    Returns:
        {"loss": float, "acc": float}  averaged over epoch
    """
    # TODO
    raise NotImplementedError


def overfit_batch(model: nn.Module, x: Tensor, y: Tensor, steps: int, lr: float) -> list[float]:
    """
    Train on ONE small fixed batch again and again (no augmentation, no regularization).
    If pipeline is correct, loss goes almost to 0. Used in test_training_sanity and
    `python main.py sanity`.

    Args:
        x: images, (n, 3, 32, 32), for example n = 32
        y: labels, (n,)
        steps: how many steps
        lr: learning rate

    Returns:
        loss for each step (list of length `steps`)
    """
    # TODO
    raise NotImplementedError


def train(
    model_cfg: ViTConfig,
    train_cfg: TrainConfig,
    seed: int,
    experiment: str,
    out_dir: str,
    fake: bool = False,
) -> dict:
    """
    Full training run for one experiment and one seed.

    Steps:
        1. set_seed(seed), make data loaders and model, move to device
        2. train exactly train_cfg.epochs epochs (NO early stopping)
           after each epoch: evaluate on val, save train/val loss+acc, lr, epoch time
        3. after last epoch: evaluate on test ONE time
        4. save JSON to out_dir/<experiment>_seed<seed>.json with keys:
           experiment, seed, split_seed, config, env, num_params, train_time_s, history, test
        5. checkpoint (optional) goes only to checkpoints/, never to results/

    Args:
        model_cfg:  ViTConfig
        train_cfg:  TrainConfig
        seed:       training seed
        experiment: name, e.g. "baseline_p4"
        out_dir:    folder for result JSON
        fake:       use fake data (for tests and smoke)

    Returns:
        same dict that is saved to JSON
    """
    # TODO
    raise NotImplementedError
