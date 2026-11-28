"""
Training pipeline: optimizer, learning rate schedule, training loop, tiny-batch overfit
helper, and saving one result JSON per run.

Paper reference: ViT Section 4.1 / Appendix B.1 (Adam-type optimizer, warmup, weight decay)
Author: <name>, Chatgpt
Adapted from: none
"""

from __future__ import annotations

import math
import json
import time
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from configs.config import TrainConfig, ViTConfig, config_to_dict

from src.training.evaluate import evaluate
from src.data import get_dataloaders
from src.reproducibility import set_seed, configure_determinism, get_device, get_env_info
from src.models.vit import ViT


def build_optimizer(model: nn.Module, train_cfg: TrainConfig) -> torch.optim.Optimizer:
    """
    AdamW with lr and weight_decay from config.
    NO weight decay for: biases, LayerNorm weights, cls_token, pos_embed.
    """
    decay = []
    no_decay = []

    for name, param in model.named_parameters():

        # Skip non-trainable parameters
        if not param.requires_grad:
            continue

        # Exclude biases, LayerNorm weights, cls_token, pos_embed 
        # param.ndim==1 for bias, layernorm
        if param.ndim == 1 or name.endswith(("cls_token", "pos_embed")):
            no_decay.append(param)
        else:
            decay.append(param)

    param_groups = [
        {"params": decay, "weight_decay": train_cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0}
    ]

    return torch.optim.AdamW(param_groups, lr=train_cfg.lr)


def build_scheduler(
    optimizer: torch.optim.Optimizer, train_cfg: TrainConfig, steps_per_epoch: int
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Learning rate per step: linear warmup for warmup_epochs, then cosine down to ~0.
    """
    total_steps = train_cfg.epochs * steps_per_epoch
    warmup_steps = train_cfg.warmup_epochs * steps_per_epoch

    def lr_lambda(step):

        # Linear warmup
        if step < warmup_steps:
            return step / max(1, warmup_steps)

        # Cosine decay
        progress = ((step - warmup_steps) / max(1, total_steps - warmup_steps))
        progress = min(1.0, max(0.0, progress))

        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


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
    model.train()

    # Loss, TP, and sample count
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    use_amp = (
        scaler is not None
        and scaler.is_enabled()
        and device.type == "cuda")

    for x, y in loader:
        x = x.to(device) # x: (N, 3, 32, 32)
        y = y.to(device) # y: (N,)

        # clear grad
        optimizer.zero_grad()

        # Forward pass
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        # Backward pass and optimizer update
        if use_amp: # if amp used
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # clip calculated loss
        if grad_clip is not None:
            if use_amp:
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Update model parameters and learning rate
        if use_amp:
            old_scale = scaler.get_scale()

            scaler.step(optimizer)
            scaler.update()

            # Advance scheduler only if optimizer step was not skipped
            if scaler.get_scale() >= old_scale:
                scheduler.step()
        else:
            optimizer.step()
            scheduler.step()

        # take batch size for sample count
        N = y.size(0)

        # accumulate
        total_loss += loss.item() * N
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total_samples += N

    return {"loss": total_loss / total_samples,
            "acc": total_correct / total_samples}


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
    # image, labels to device
    device = next(model.parameters()).device
    x = x.to(device)
    y = y.to(device)

    # Disable dropout, stochastic depth, patch dropout
    model.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    losses = []

    for _ in range(steps):
        optimizer.zero_grad()

        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return losses


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
    # 1. set seed
    set_seed(seed)
    configure_determinism(train_cfg.deterministic)
    device = get_device(train_cfg.device)

    # DataLoaders and model, move to device
    loaders = get_dataloaders(train_cfg, seed=seed, fake=fake)
    model = ViT(model_cfg).to(device)

    # number of params for experiment result
    num_params = sum(p.numel() for p in model.parameters())

    # Optimizer, scheduler and Loss function
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg, steps_per_epoch=len(loaders["train"]))
    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg.label_smoothing)

    # enable amp if cuda
    is_cuda = device.type == "cuda"
    use_amp = train_cfg.amp and is_cuda
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # save history for experiment result
    history = []

    # time measure for train
    if is_cuda:
        torch.cuda.synchronize(device)
    train_start = time.perf_counter()

    # 2. Train
    for epoch in range(1, train_cfg.epochs+1):
        # time measure for each epoch
        epoch_start = time.perf_counter()

        # run one epoch
        train_metrics = train_one_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            scaler=scaler,
            grad_clip=train_cfg.grad_clip
        )

        # Validation after every epoch
        val_metrics = evaluate(
            model=model,
            loader=loaders["val"],
            device=device
        )

        # get epoch time taken
        if is_cuda:
            torch.cuda.synchronize(device)
        epoch_time = time.perf_counter() - epoch_start

        lr = optimizer.param_groups[0]["lr"]

        # save train/val loss+acc, lr, epoch time
        epoch_result = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "lr": lr,
            "epoch_time_s": epoch_time
        }

        history.append(epoch_result)
        print(f"Epoch {epoch}/{train_cfg.epochs} done")

    # get total train time
    train_time_s = time.perf_counter() - train_start

    # 3. after last epoch: evaluate on test ONE time
    test_metrics = evaluate(
        model=model,
        loader=loaders["test"],
        device=device
    )

    # 4. save JSON to out_dir/<experiment>_seed<seed>.json with keys:
    #    experiment, seed, split_seed, config, env, num_params, train_time_s, history, test
    results = {
        "experiment": experiment,
        "seed": seed,
        "split_seed": train_cfg.split_seed,
        "config": config_to_dict(model_cfg, train_cfg),
        "env": get_env_info(device),
        "num_params": num_params,
        "train_time_s": train_time_s,
        "history": history,
        "test": test_metrics
    }

    # Save JSON
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (output_dir / f"{experiment}_seed{seed}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    return results
