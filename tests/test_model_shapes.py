"""
End-to-end shape tests for full ViT, parameter count and config checks.

Paper reference: ViT Eq. 1-4
Author: <name>
Adapted from: none
"""

from __future__ import annotations

import pytest


@pytest.mark.parametrize("patch_size", [4, 8])
@pytest.mark.parametrize("batch_size", [1, 5])
def test_vit_output_shape(patch_size, batch_size):
    """ViT(cfg)(x).shape == (N, 10)."""
    pytest.skip("TODO")


def test_trace_shapes_matches_expected_table():
    """trace_shapes for baseline config must be same as expected shape table (used in report)."""
    pytest.skip("TODO")


@pytest.mark.parametrize("patch_size", [4, 8])
def test_param_count_matches_formula(patch_size):
    """count_parameters(model) == formula we compute by hand in this test."""
    pytest.skip("TODO")


def test_number_of_layernorms(tiny_cfg):
    """Number of nn.LayerNorm must be 2*L + 1 (catches missing or double final LN)."""
    pytest.skip("TODO")


def test_invalid_config_raises():
    """patch_size=5 or embed_dim not divisible by num_heads must give ValueError."""
    pytest.skip("TODO")


def test_same_seed_same_model():
    """Same seed -> same parameters. Different seed -> different parameters."""
    pytest.skip("TODO")


def test_patch_dropout_keeps_logits_shape(tiny_cfg, tiny_batch):
    """With patch_dropout_rate > 0 in train mode, logits are still (N, 10)."""
    pytest.skip("TODO")
