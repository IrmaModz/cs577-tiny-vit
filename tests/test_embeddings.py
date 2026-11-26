"""
Tests for patch embedding and token embedding (CLS + positions).

Paper reference: ViT Eq. 1
Author: <name>
Adapted from: none
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.models.patch_embedding import PatchEmbedding
from src.models.token_embedding import TokenEmbedding


@pytest.mark.parametrize("patch_size", [4, 8])
def test_patch_embedding_shape(patch_size):
    """num_patches == (32 / p) ** 2 and output is (N, P, D)."""
    N, C, H, W, D = 8, 3, 32, 32, 5
    P = (H // patch_size) ** 2
    test_embed = PatchEmbedding(image_size=H, patch_size=patch_size, in_channels=C, embed_dim=D)
    x = torch.ones(N, C, H, W)
    out = test_embed(x)
    assert test_embed.num_patches == P
    assert out.shape == (N, P, D)


def test_patch_order_is_row_major():
    """
    Known case: if D = p*p*C and conv weights are identity, row i of output must be
    exactly the manually cut patch i (left to right, top to bottom).
    Catches wrong patch order (column-major or transposed).
    """
    N, C, H, W, p = 1, 1, 4, 4, 2
    D = p * p * C
    test_embed = PatchEmbedding(image_size=H, patch_size=p, in_channels=C, embed_dim=D)

    with torch.no_grad():
        test_embed.linear1.weight.copy_(torch.eye(D))
        test_embed.linear1.bias.zero_()

    x = torch.arange(16, dtype=torch.float32).reshape(N, C, H, W)
    out = test_embed(x)  # (1, 4, 4)

    expect = torch.tensor(
        [[[0, 1, 4, 5],
          [2, 3, 6, 7],
          [8, 9, 12, 13],
          [10, 11, 14, 15]]],
        dtype=torch.float32,
    )
    assert torch.equal(out, expect)


def test_matches_conv2d_reference():
    """
    Independent implementation: our reshape + Linear projection must give same result
    as Conv2d(C, D, kernel=p, stride=p) with same weights (atol 1e-6).
    """
    N, C, H, W, p, D = 4, 3, 4, 4, 2, 4
    test_embed = PatchEmbedding(image_size=H, patch_size=p, in_channels=C, embed_dim=D)
    test_conv = nn.Conv2d(in_channels=C, out_channels=D, kernel_size=p, stride=p)

    with torch.no_grad():
        test_conv.weight.copy_(test_embed.linear1.weight.reshape(D, C, p, p))
        test_conv.bias.copy_(test_embed.linear1.bias)

    x = torch.randn(N, C, H, W)
    out1 = test_embed(x)  # (N, P, D)
    out2 = test_conv(x).flatten(2).transpose(1, 2)  # (N, D, 2, 2) -> (N, P, D)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_token_embedding_adds_cls_and_positions():
    """
    out has shape (N, P+1, D); out[:, 0] is same for every sample and == cls + pos[0];
    out[:, 1:] - x == pos[:, 1:].
    """
    N, P, D = 4, 16, 8
    emb = TokenEmbedding(num_patches=P, embed_dim=D, dropout=0.0).eval()

    # with only zeros the checks are too easy, so fill random numbers
    with torch.no_grad():
        nn.init.normal_(emb.cls_token)
        nn.init.normal_(emb.pos_embed)

    x = torch.randn(N, P, D)  # (N, P, D)
    out = emb(x)  # (N, P+1, D)

    assert out.shape == (N, P + 1, D)

    expected_cls = emb.cls_token[0, 0] + emb.pos_embed[0, 0]  # (D,)
    for i in range(N):
        assert torch.allclose(out[i, 0], expected_cls)  # same CLS for every sample

    diff = out[:, 1:] - x  # (N, P, D)
    assert torch.allclose(diff, emb.pos_embed[:, 1:].expand_as(diff), atol=1e-6)


@pytest.mark.parametrize("num_patches", [64, 16])  # p=4 and p=8
def test_token_embedding_param_count_and_zeros(num_patches):
    """Number of parameters == D + (P+1) * D, and cls_token, pos_embed start as zeros."""
    D = 32
    emb = TokenEmbedding(num_patches=num_patches, embed_dim=D)
    assert sum(p.numel() for p in emb.parameters()) == D + (num_patches + 1) * D
    assert torch.count_nonzero(emb.cls_token) == 0
    assert torch.count_nonzero(emb.pos_embed) == 0


def test_wrong_input_size_raises():
    """Image that is not 32x32 must give ValueError."""
    test_embed = PatchEmbedding(image_size=32, patch_size=4, in_channels=3, embed_dim=8)
    with pytest.raises(ValueError):
        test_embed(torch.randn(2, 3, 28, 28))  # 28x28 instead of 32x32