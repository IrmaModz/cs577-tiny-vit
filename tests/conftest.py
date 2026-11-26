"""
Shared fixtures for all tests: tiny model config, small fake batch and fixed seed
Everything here is small so whole test suite runs on CPU in less than 2 minutes.

Paper reference: none
Author: <name>
Adapted from: none
"""

from __future__ import annotations

import pytest
import torch

from configs.config import ViTConfig
from src.reproducibility import set_seed


@pytest.fixture(autouse=True)
def fixed_seed() -> None:
    """
    Runs automatically before every test (autouse=True), no need to ask for it.
    Calls set_seed(0), so random numbers are same in every run.
    """
    set_seed(0)

@pytest.fixture
def tiny_cfg() -> ViTConfig:
    """
    Very small model for fast tests: patch 8 -> 16 patches, D=32, 4 heads, 2 blocks.
    All dropout and drop path are 0, so outputs are deterministic.
    """
    return ViTConfig(
        image_size=32,
        patch_size=8,
        embed_dim=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        attn_dropout=0.0,
        drop_path_rate=0.0,
        patch_dropout_rate=0.0,
    )


@pytest.fixture
def tiny_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """
    Small fake batch, always same numbers.
    Returns:
        x: (4, 3, 32, 32) random images
        y: (4,) random labels in [0, 10)
    """
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4, 3, 32, 32, generator=g)
    y = torch.randint(0, 10, (4,), generator=g)
    return x, y