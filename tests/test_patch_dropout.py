"""
Tests for patch dropout.

Paper reference: none (PatchDropout, Liu et al., 2023)
Author: <name>
Adapted from: none
"""

from __future__ import annotations

import pytest


def test_identity_in_eval_and_rate_zero():
    """In eval mode, and when rate == 0, output must be same as input."""
    pytest.skip("TODO")


def test_output_shape_and_cls_kept():
    """In train mode: shape (N, 1+K, D) with K = max(1, round((1-rate)*P)), and out[:, 0] == x[:, 0]."""
    pytest.skip("TODO")


def test_kept_tokens_are_real_input_rows():
    """Every kept row must be exactly one of input rows (we select, not mix)."""
    pytest.skip("TODO")


def test_different_mask_per_sample():
    """With fixed seed, samples in same batch should keep different patches."""
    pytest.skip("TODO")
