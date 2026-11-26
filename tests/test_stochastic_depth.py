"""
Tests for stochastic depth (DropPath).

Paper reference: none (Huang et al., 2016)
Author: <name>
Adapted from: none
"""

from __future__ import annotations

import pytest
import torch
import src.models.stochastic_depth as stochastic_depth

def test_identity_in_eval_and_p_zero():
    """In eval mode, and when p == 0, output == input."""
    N, P, D = 4, 4, 4
    x = torch.randn(N, P, D)

    drop_path = stochastic_depth.DropPath(p=0.5)
    drop_path.eval()
    out = drop_path(x)
    assert torch.equal(out, x)

    drop_path = stochastic_depth.DropPath(p=0.0)
    drop_path.train()
    out = drop_path(x)
    assert torch.equal(out, x)


def test_each_sample_zero_or_scaled():
    """In train mode each sample is either all zero or exactly x / (1 - p)."""
    N, P, D = 4, 4, 4
    p = 0.5
    x = torch.randn(N, P, D)

    drop_path = stochastic_depth.DropPath(p)
    drop_path.train()
    out = drop_path(x)
    for i in range(N):
        is_zero = torch.all(out[i] == 0)
        is_scaled = torch.equal(out[i], x[i]/(1.0-p))
        assert is_zero or is_scaled


def test_expected_value_preserved():
    """Mean over many samples (e.g. 10000) is close to x."""
    N, P, D = 10000, 33, 128
    p = 0.5
    x = torch.randn(N, P, D)

    drop_path = stochastic_depth.DropPath(p)
    drop_path.train()
    out = drop_path(x)

    assert torch.allclose(out.mean(dim=0), x.mean(dim=0), atol=0.05)


def test_drop_path_schedule_linear():
    """drop_path_schedule(0.1, 6) goes linear from 0 to 0.1."""
    test_lst = torch.linspace(0, 0.1, steps=6)
    out = torch.tensor(stochastic_depth.drop_path_schedule(0.1, 6))
    assert torch.allclose(out, test_lst, atol=1e-5)
