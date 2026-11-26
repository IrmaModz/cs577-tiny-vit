"""
Sanity tests for training: initial loss, tiny-batch overfit, controlled failure.

Paper reference: none
Author: <name>
Adapted from: none
"""

from __future__ import annotations

import pytest
import json
import torch
from torch import nn

from configs.config import load_experiment
from src.models.vit import ViT
from src.training.train import train, overfit_batch
import numpy as np


def test_initial_loss_close_to_ln10(tiny_cfg, tiny_batch):
    """Known case: at init, cross entropy is about ln(10) = 2.303 (tolerance 0.15)."""
    model = ViT(tiny_cfg)
    model.eval()

    image_batch, label_batch = tiny_batch
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        pred = model(image_batch)
        loss = criterion(pred, label_batch)

    assert loss.item() == pytest.approx(np.log(10), abs=0.15)


def test_overfit_tiny_batch(tiny_cfg):
    """overfit_batch on 32 fixed fake images reaches 100% train acc / loss < 0.05 in 300 steps."""
    model = ViT(tiny_cfg)

    #fake images and labels
    image_batch, label_batch = torch.randn(32, 3, 32, 32), torch.randint(0, 10, (32,))

    #overfit the model, may not pass loss with seed change
    _ = overfit_batch(model, image_batch, label_batch, 500, lr=0.001)

    # evaluate if overfit
    model.eval()
    with torch.no_grad():
        pred = model(image_batch)
        loss = nn.CrossEntropyLoss()(pred, label_batch)
        acc = (pred.argmax(dim=1) == label_batch).float().mean().item()

    assert loss.item() < 0.05
    assert acc == 1.0 # maybe threshold

def test_zero_lr_does_not_learn(tiny_cfg):
    """Controlled failure: with lr = 0, loss does not go down and params dont change."""
    model = ViT(tiny_cfg)

    #fake images and labels
    image_batch, label_batch = torch.rand(32, 3, 32, 32), torch.randint(0, 10, (32,))

    initial_params = [params.detach().clone() for params in model.parameters()]

    #overfit/train the model with 0 lr
    losses = overfit_batch(model, image_batch, label_batch, 5, lr=0)
    expect = [losses[0] for _ in range(len(losses))]

    trained_params = [params.detach().clone() for params in model.parameters()]

    # verify model did not learned
    assert losses == pytest.approx(expect, abs=1e-6)
    for initial, trained in zip(initial_params, model.parameters()):
        assert torch.equal(initial, trained)


def test_train_writes_result_json(tmp_path):
    """train(..., fake=True) with smoke config writes JSON with all required keys."""
    model_cfg, train_cfg = load_experiment("smoke")

    result = train(model_cfg=model_cfg,
                   train_cfg=train_cfg,
                   seed=0,
                   experiment="smoke",
                   out_dir=tmp_path,
                   fake=True)

    # output_path = (output_dir / f"{experiment}_seed{seed}.json")
    # check if file exists
    output_path = tmp_path / "smoke_seed0.json"
    assert output_path.exists()

    # check keys
    with open(output_path, "r", encoding="utf-8") as f:
       saved_results = json.load(f)

    # Verify that all required keys are present
    required_keys = {
        "experiment",
        "seed",
        "split_seed",
        "config",
        "env",
        "num_params",
        "train_time_s",
        "history",
        "test"
    }

    assert required_keys.issubset(saved_results.keys())