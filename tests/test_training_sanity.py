"""
Sanity tests for training: initial loss, tiny-batch overfit, controlled failure.

Paper reference: none
Author: <name>
Adapted from: none
"""

from __future__ import annotations

import pytest


def test_initial_loss_close_to_ln10(tiny_cfg, tiny_batch):
    """Known case: at init, cross entropy is about ln(10) = 2.303 (tolerance 0.15)."""
    pytest.skip("TODO")


def test_overfit_tiny_batch(tiny_cfg):
    """overfit_batch on 32 fixed fake images reaches 100% train acc / loss < 0.05 in 300 steps."""
    pytest.skip("TODO")


def test_zero_lr_does_not_learn(tiny_cfg):
    """Controlled failure: with lr = 0, loss does not go down and params dont change."""
    pytest.skip("TODO")


def test_train_writes_result_json(tmp_path):
    """train(..., fake=True) with smoke config writes JSON with all required keys."""
    pytest.skip("TODO")
