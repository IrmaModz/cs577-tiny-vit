"""
Tests for data pipeline: train/val split and transforms. No download needed.

Paper reference: none
Author: <name>, Chatgpt
Adapted from: none
"""

from __future__ import annotations

import pytest
import torch
import numpy as np
from src.data import stratified_split, build_transforms, get_dataloaders
from configs.config import TrainConfig


def test_split_sizes_and_no_overlap():
    """stratified_split on synthetic labels: 45000 / 5000, train and val have no common index."""
    # synthetic data labels 10 class with 5000 samples
    targets = torch.arange(10).repeat_interleave(5000).tolist()
    
    # test stratified split
    train_idx, val_idx = stratified_split(targets, val_size=5000, seed=42)

    # check the size and common index
    assert len(train_idx) == 45000
    assert len(val_idx) == 5000
    assert set(train_idx).isdisjoint(set(val_idx))
    assert set(train_idx) | set(val_idx) == set(range(50000))


def test_split_is_stratified():
    """Exactly 500 images of each class in val."""
    # synthetic data labels 10 class with 5000 samples
    targets = torch.arange(10).repeat_interleave(5000)

    # Check for val set
    _, val_idx = stratified_split(targets.tolist(), val_size=5000, seed=42)

    # count by class
    val_labels = targets[val_idx]
    counts = torch.bincount(val_labels, minlength=10)

    # check equal
    assert torch.equal(counts, torch.full((10,), 500))


def test_split_is_deterministic():
    """Same split_seed -> same indices every time."""
    # synthetic data labels 10 class with 5000 samples
    targets = torch.arange(10).repeat_interleave(5000).tolist()

    # test stratified split with same seed
    train_idx1, val_idx1 = stratified_split(targets, val_size=5000, seed=42)
    train_idx2, val_idx2 = stratified_split(targets, val_size=5000, seed=42)

    # check if equal
    assert train_idx1 == train_idx2
    assert val_idx1 == val_idx2


def test_eval_transform_has_no_augmentation():
    """Same image through eval transform two times -> identical tensors."""
    # create fake img
    image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    # same image transform twice with train=False
    transform = build_transforms(train=False)
    out1 = transform(image)
    out2 = transform(image)

    # check shape and equal
    assert out1.shape == (3, 32, 32)
    assert torch.equal(out1, out2)


def test_fake_loaders_shapes():
    """get_dataloaders(fake=True): batches (B, 3, 32, 32), labels in [0, 10)."""
    
    B = 4
    train_cfg = TrainConfig(batch_size=B, num_workers=0, split_seed=0)

    # fake loader
    loaders = get_dataloaders(train_cfg, seed=42, fake=True)

    # verify if all three keys are present
    assert set(loaders.keys()) == {"train", "val", "test"}

    for loader in loaders.values():
        for images, labels in loader:

            # Check if batch shape is correct
            assert images.shape == (B, 3, 32, 32)
            assert labels.shape == (B,)

            # check label is 0~9
            assert torch.all(labels >= 0)
            assert torch.all(labels < 10)
