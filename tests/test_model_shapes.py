"""
End-to-end shape tests for full ViT, parameter count and config checks.

Paper reference: ViT Eq. 1-4
Author: <name>
Adapted from: none
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch import nn

from configs.config import ViTConfig
from src.models.vit import ViT, count_parameters, trace_shapes
from src.reproducibility import set_seed


@pytest.mark.parametrize("patch_size", [4, 8])
@pytest.mark.parametrize("batch_size", [1, 5])
def test_vit_output_shape(patch_size, batch_size):
    """ViT(cfg)(x).shape == (N, 10)."""
    model = ViT(ViTConfig(patch_size=patch_size)).eval()
    x = torch.randn(batch_size, 3, 32, 32)  # (N, 3, 32, 32)
    assert model(x).shape == (batch_size, 10)  # (N, 10)


def test_trace_shapes_matches_expected_table():
    """trace_shapes for baseline config must be same as expected shape table (used in report)."""
    model = ViT(ViTConfig())  # baseline: p=4 -> P=64, D=128
    shapes = trace_shapes(model, torch.randn(2, 3, 32, 32))
    assert shapes == [
        ("input", (2, 3, 32, 32)),
        ("patch_embed", (2, 64, 128)),
        ("token_embed", (2, 65, 128)),
        ("patch_dropout", (2, 65, 128)),  # eval mode, so nothing is dropped
        ("encoder", (2, 65, 128)),
        ("head", (2, 10)),
    ]


@pytest.mark.parametrize("patch_size", [4, 8])
def test_param_count_matches_formula(patch_size):
    """count_parameters(model) == formula we compute by hand in this test."""
    cfg = ViTConfig(patch_size=patch_size)
    D, C, p = cfg.embed_dim, cfg.in_channels, cfg.patch_size
    P, L, M = cfg.num_patches, cfg.depth, cfg.mlp_dim

    patch_embed = (p * p * C) * D + D  # Linear(p*p*C, D) with bias
    token_embed = D + (P + 1) * D  # cls_token + pos_embed
    one_block = (
        2 * D  # LayerNorm before attention (weight + bias)
        + D * 3 * D + 3 * D  # qkv Linear(D, 3D)
        + D * D + D  # output projection Linear(D, D)
        + 2 * D  # LayerNorm before MLP
        + D * M + M  # MLP Linear(D, M)
        + M * D + D  # MLP Linear(M, D)
    )
    final_ln = 2 * D
    head = D * cfg.num_classes + cfg.num_classes  # Linear(D, 10)
    # patch dropout and drop path have no parameters

    expected = patch_embed + token_embed + L * one_block + final_ln + head
    assert count_parameters(ViT(cfg)) == expected


def test_number_of_layernorms(tiny_cfg):
    """Number of nn.LayerNorm must be 2*L + 1 (catches missing or double final LN)."""
    model = ViT(tiny_cfg)
    n_ln = sum(isinstance(m, nn.LayerNorm) for m in model.modules())
    assert n_ln == 2 * tiny_cfg.depth + 1  # 2 per block + 1 final


def test_invalid_config_raises():
    """patch_size=5 or embed_dim not divisible by num_heads must give ValueError."""
    with pytest.raises(ValueError):
        ViTConfig(patch_size=5)  # 32 can not be divided by 5
    with pytest.raises(ValueError):
        ViTConfig(embed_dim=130, num_heads=4)  # 130 can not be divided by 4


def test_same_seed_same_model():
    """Same seed -> same parameters. Different seed -> different parameters."""
    cfg = ViTConfig()

    set_seed(0)
    a = ViT(cfg)
    set_seed(0)
    b = ViT(cfg)
    set_seed(1)
    c = ViT(cfg)

    same = all(torch.equal(pa, pb) for pa, pb in zip(a.parameters(), b.parameters()))
    different = any(not torch.equal(pa, pc) for pa, pc in zip(a.parameters(), c.parameters()))
    assert same
    assert different


def test_patch_dropout_keeps_logits_shape(tiny_cfg, tiny_batch):
    """With patch_dropout_rate > 0 in train mode, logits are still (N, 10)."""
    x, _ = tiny_batch  # (4, 3, 32, 32)
    cfg = replace(tiny_cfg, patch_dropout_rate=0.5)  # 16 patches -> keep 8
    model = ViT(cfg).train()
    assert model(x).shape == (x.shape[0], 10)