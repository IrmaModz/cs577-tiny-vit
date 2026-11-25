"""
Tests for stochastic depth (DropPath).

Paper reference: none (Huang et al., 2016)
Author: <name>
Adapted from: none
"""

from __future__ import annotations

import pytest


def test_identity_in_eval_and_p_zero():
    """In eval mode, and when p == 0, output == input."""
    pytest.skip("TODO")


def test_each_sample_zero_or_scaled():
    """In train mode each sample is either all zero or exactly x / (1 - p)."""
    pytest.skip("TODO")


def test_expected_value_preserved():
    """Mean over many samples (e.g. 10000) is close to x."""
    pytest.skip("TODO")


def test_drop_path_schedule_linear():
    """drop_path_schedule(0.1, 6) goes linear from 0 to 0.1."""
    pytest.skip("TODO")
