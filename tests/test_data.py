"""
Tests for data pipeline: train/val split and transforms. No download needed.

Paper reference: none
Author: <name>
Adapted from: none
"""

from __future__ import annotations

import pytest


def test_split_sizes_and_no_overlap():
    """stratified_split on synthetic labels: 45000 / 5000, train and val have no common index."""
    pytest.skip("TODO")


def test_split_is_stratified():
    """Exactly 500 images of each class in val."""
    pytest.skip("TODO")


def test_split_is_deterministic():
    """Same split_seed -> same indices every time."""
    pytest.skip("TODO")


def test_eval_transform_has_no_augmentation():
    """Same image through eval transform two times -> identical tensors."""
    pytest.skip("TODO")


def test_fake_loaders_shapes():
    """get_dataloaders(fake=True): batches (B, 3, 32, 32), labels in [0, 10)."""
    pytest.skip("TODO")
