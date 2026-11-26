"""
Small tests for src/models/token_embedding.py, they check all points from "Done when". 
Run: pytest tests/test_token_embedding.py

"""

import pytest
import torch
from torch import nn

from src.models.token_embedding import TokenEmbedding

D = 32  # small embed size, so tests are fast


def make_embedding(num_patches):
    """TokenEmbedding with random cls_token and pos_embed.
    With only zeros the checks below would pass too easy, so we fill random numbers."""
    emb = TokenEmbedding(num_patches, D, dropout=0.0)
    with torch.no_grad():
        nn.init.normal_(emb.cls_token)
        nn.init.normal_(emb.pos_embed)
    return emb.eval()


@pytest.mark.parametrize("num_patches", [64, 16])  # p=4 and p=8
def test_output_shape(num_patches):
    emb = make_embedding(num_patches)
    x = torch.randn(4, num_patches, D)  # (N, P, D)
    assert emb(x).shape == (4, num_patches + 1, D)  # (N, P+1, D)


@pytest.mark.parametrize("num_patches", [64, 16])
def test_cls_is_same_for_all_and_at_index_0(num_patches):
    emb = make_embedding(num_patches)
    out = emb(torch.randn(4, num_patches, D))
    expected = (emb.cls_token + emb.pos_embed[:, 0:1])[:, 0]  # (1, D)
    for i in range(4):
        assert torch.allclose(out[i, 0], expected[0])


@pytest.mark.parametrize("num_patches", [64, 16])
def test_patches_get_only_positions(num_patches):
    emb = make_embedding(num_patches)
    x = torch.randn(4, num_patches, D)
    out = emb(x)
    diff = out[:, 1:] - x  # (N, P, D)
    assert torch.allclose(diff, emb.pos_embed[:, 1:].expand_as(diff), atol=1e-6)


@pytest.mark.parametrize("num_patches", [64, 16])
def test_number_of_parameters(num_patches):
    emb = TokenEmbedding(num_patches, D)
    n_params = sum(p.numel() for p in emb.parameters())
    assert n_params == D + (num_patches + 1) * D


def test_starts_with_zeros():
    emb = TokenEmbedding(16, D)
    assert torch.count_nonzero(emb.cls_token) == 0
    assert torch.count_nonzero(emb.pos_embed) == 0