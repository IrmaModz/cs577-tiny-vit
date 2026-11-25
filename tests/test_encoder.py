"""
Tests for attention and transformer encoder.

Paper reference: ViT Eq. 2-3, Appendix A
Author: <name>
Adapted from: none
"""

from __future__ import annotations

import pytest


def test_attention_equals_pytorch_reference():
    """
    Independent implementation: our MHSA == nn.MultiheadAttention(batch_first=True)
    after copying qkv -> in_proj_weight and proj -> out_proj (atol 1e-5).
    """
    pytest.skip("TODO")


def test_attention_weights_shape_and_sum():
    """Attention weights shape (N, h, T, T) and each row sums to 1."""
    pytest.skip("TODO")


def test_encoder_permutation_equivariance(tiny_cfg):
    """Without positional embedding: encoder(x[:, perm]) == encoder(x)[:, perm]."""
    pytest.skip("TODO")


@pytest.mark.parametrize("num_tokens", [17, 65])
def test_encoder_any_sequence_length(tiny_cfg, num_tokens):
    """Same encoder must work for different T (patch dropout changes T)."""
    pytest.skip("TODO")


def test_depth_zero_is_only_layernorm(tiny_cfg):
    """depth = 0 -> encoder returns LN(x)."""
    pytest.skip("TODO")


def test_eval_is_deterministic(tiny_cfg):
    """In eval mode two forward passes give same output (no dropout / drop path active)."""
    pytest.skip("TODO")
