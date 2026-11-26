"""
Tests for patch dropout.

Paper reference: none (PatchDropout, Liu et al., 2023)
Author: <name>
Adapted from: none
"""

from __future__ import annotations

import torch

from src.models.patch_dropout import PatchDropout

N, P, D = 4, 16, 8


def make_tokens() -> torch.Tensor:
    """Tokens where every row is different: value = sample * 1000 + row number.
    So we can see exactly which input row was kept. Shape (N, 1+P, D)."""
    x = torch.zeros(N, 1 + P, D)
    for n in range(N):
        for i in range(1 + P):
            x[n, i] = n * 1000 + i
    return x


def test_identity_in_eval_and_rate_zero():
    """In eval mode, and when rate == 0, output must be same as input."""
    x = make_tokens()
    assert torch.equal(PatchDropout(rate=0.5).eval()(x), x)  # eval mode
    assert torch.equal(PatchDropout(rate=0.0).train()(x), x)  # rate == 0


def test_output_shape_and_cls_kept():
    """In train mode: shape (N, 1+K, D) with K = max(1, round((1-rate)*P)), and out[:, 0] == x[:, 0]."""
    torch.manual_seed(0)
    x = make_tokens()
    for rate in (0.25, 0.5, 0.99):
        out = PatchDropout(rate=rate).train()(x)
        k = max(1, round((1 - rate) * P))  # 12, 8, 1
        assert out.shape == (N, 1 + k, D)
        assert torch.equal(out[:, 0], x[:, 0])  # CLS kept


def test_kept_tokens_are_real_input_rows():
    """Every kept row must be exactly one of input rows (we select, not mix)."""
    torch.manual_seed(0)
    x = make_tokens()
    out = PatchDropout(rate=0.5).train()(x)
    for n in range(N):
        kept_rows = out[n, 1:]  # (K, D)
        for row in kept_rows:
            # must be equal to exactly one patch row of the SAME sample
            matches = (x[n, 1:] == row).all(dim=1)  # (P,)
            assert matches.sum() == 1
        ids = kept_rows[:, 0].tolist()
        assert len(set(ids)) == len(ids)  # no patch is taken two times


def test_different_mask_per_sample():
    """With fixed seed, samples in same batch should keep different patches."""
    torch.manual_seed(0)
    x = make_tokens()
    out = PatchDropout(rate=0.5).train()(x)
    # row number of kept patch = value % 1000
    kept_sets = [frozenset((out[n, 1:, 0] % 1000).tolist()) for n in range(N)]
    assert len(set(kept_sets)) > 1