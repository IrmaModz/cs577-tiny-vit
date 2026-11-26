"""
Tests for patch embedding and token embedding (CLS + positions).

Paper reference: ViT Eq. 1
Author: <name>
Adapted from: none
"""

from __future__ import annotations

from src.models.patch_embedding import PatchEmbedding
from torch import nn
import torch
import pytest
import torch
from torch import nn

from src.models.patch_embedding import PatchEmbedding
from src.models.token_embedding import TokenEmbedding


@pytest.mark.parametrize("patch_size", [4, 8])
def test_patch_embedding_shape(patch_size):
    """num_patches == (32 / p) ** 2 and output is (N, P, D)."""
    N, C, H, W, D = 8, 3, 32, 32, 5
    P = (H//patch_size)**2
    test_embed = PatchEmbedding(image_size=H, 
                                patch_size=patch_size, 
                                in_channels=C, 
                                embed_dim=D)
    x = torch.ones(N, C, H, W)
    out = test_embed(x)
    assert test_embed.num_patches == (32 / patch_size) ** 2
    assert out.shape == (N, P, D)


def test_patch_order_is_row_major():
    """
    Known case: if D = p*p*C and conv weights are identity, row i of output must be
    exactly the manually cut patch i (left to right, top to bottom).
    Catches wrong patch order (column-major or transposed).
    """
    N, C, H, W, p = 1, 1, 4, 4, 2
    D = p*p*C
    test_embed = PatchEmbedding(image_size=H, 
                                patch_size=2, 
                                in_channels=C, 
                                embed_dim=D)
    
    with torch.no_grad():
        test_embed.linear1.weight.copy_(torch.eye(D))
        test_embed.linear1.bias.zero_()

    test_lst = torch.tensor([i for i in range(16)], dtype=torch.float32)
    test_lst = test_lst.reshape(N, C, H, W)

    out = test_embed(test_lst)

    expect = torch.tensor([
        [
            [0, 1, 4, 5], 
            [2, 3, 6, 7],
            [8, 9, 12, 13],
            [10, 11, 14, 15]
        ]
    ], dtype=torch.float32)

    assert torch.equal(out, expect)

def test_matches_conv2d_reference():
    """
    Independent implementation: our reshape + Linear projection must give same result
    as Conv2d(C, D, kernel=p, stride=p) with same weights (atol 1e-6).
    """
    N, C, H, W, p, D = 4, 3, 4, 4, 2, 4
    test_embed = PatchEmbedding(image_size=H, 
                                patch_size=2, 
                                in_channels=C, 
                                embed_dim=D)

    # conv 2d
    test_conv = nn.Conv2d(
        in_channels=C,
        out_channels=D,
        kernel_size=p,
        stride=p
    )

    with torch.no_grad():
        test_conv.weight.copy_(
            test_embed.linear1.weight.reshape(D, C, p, p)
        )
        test_conv.bias.copy_(test_embed.linear1.bias)

    x = torch.randn(N, C, H, W)

    out1 = test_embed(x)
    out2 = test_conv(x)
    out2 = out2.flatten(2).transpose(1, 2)

    assert torch.allclose(out1, out2, atol=1e-6)


def test_token_embedding_adds_cls_and_positions():
    """
    out has shape (N, P+1, D); out[:, 0] is same for every sample and == cls + pos[0];
    out[:, 1:] - x == pos[:, 1:].
    """
    N, C, H, W, p, D, P = 4, 3, 4, 4, 2, 4, 2
    test_embed = PatchEmbedding(image_size=H, 
                                patch_size=P, 
                                in_channels=C, 
                                embed_dim=D)



def test_wrong_input_size_raises():
    """Image that is not 32x32 must give ValueError."""
    test_embed = PatchEmbedding(image_size=32, patch_size=4, in_channels=3, embed_dim=8)
    with pytest.raises(ValueError):
        test_embed(torch.randn(2, 3, 28, 28))  # 28x28 instead of 32x32