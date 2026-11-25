"""
Tests for patch embedding and token embedding (CLS + positions).

Paper reference: ViT Eq. 1
Author: <name>
Adapted from: none
"""

from __future__ import annotations

import pytest


@pytest.mark.parametrize("patch_size", [4, 8])
def test_patch_embedding_shape(patch_size):
    """num_patches == (32 / p) ** 2 and output is (N, P, D)."""
    pytest.skip("TODO")


def test_patch_order_is_row_major():
    """
    Known case: if D = p*p*C and conv weights are identity, row i of output must be
    exactly the manually cut patch i (left to right, top to bottom).
    Catches wrong patch order (column-major or transposed).
    """
    pytest.skip("TODO")


def test_matches_conv2d_reference():
    """
    Independent implementation: our reshape + Linear projection must give same result
    as Conv2d(C, D, kernel=p, stride=p) with same weights (atol 1e-6).
    """
    pytest.skip("TODO")


def test_token_embedding_adds_cls_and_positions():
    """
    out has shape (N, P+1, D); out[:, 0] is same for every sample and == cls + pos[0];
    out[:, 1:] - x == pos[:, 1:].
    """
    pytest.skip("TODO")


def test_wrong_input_size_raises():
    """Image that is not 32x32 must give ValueError."""
    pytest.skip("TODO")
