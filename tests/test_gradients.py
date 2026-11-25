"""
Gradient tests. This is evidence for our focused gradient-flow section in report.

Paper reference: ViT Eq. 1-4
Author: <name>
Adapted from: none
"""

from __future__ import annotations

import pytest


def test_all_parameters_get_gradient(tiny_cfg, tiny_batch):
    """After one backward, every trainable param (also cls_token, pos_embed, patch proj) has finite non-zero grad."""
    pytest.skip("TODO")


def test_logits_gradient_matches_formula(tiny_cfg, tiny_batch):
    """Analytic check: dL/dlogits == (softmax(logits) - onehot(y)) / N for mean cross entropy."""
    pytest.skip("TODO")


def test_patch_gradient_only_through_attention(tiny_cfg, tiny_batch):
    """
    Known case: with depth = 0, gradient to patch tokens is exactly 0 (head reads only CLS).
    With depth >= 1, gradient to every patch is non-zero -> it comes only through attention.
    """
    pytest.skip("TODO")


def test_gradcheck_tiny_model():
    """torch.autograd.gradcheck in float64 on very tiny ViT (D=8, h=2, L=1, p=16)."""
    pytest.skip("TODO")


def test_detached_path_is_detected(tiny_cfg, tiny_batch):
    """Controlled failure: if patch_embed output is .detach()-ed, our grad checker must catch it."""
    pytest.skip("TODO")
