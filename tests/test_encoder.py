"""
Tests for attention and transformer encoder.

Paper reference: ViT Eq. 2-3, Appendix A
Author: <name>
Adapted from: none
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from src.models.attention import MultiHeadSelfAttention as mha
from src.models.encoder import TransformerBlock, Encoder
from dataclasses import replace


def test_attention_equals_pytorch_reference():
    """
    Independent implementation: our MHSA == nn.MultiheadAttention(batch_first=True)
    after copying qkv -> in_proj_weight and proj -> out_proj (atol 1e-5).
    """
    N, P, D, num_heads = 2, 10, 10, 2
    x = torch.randn(N, P, D)

    our_mhsa = mha(D, num_heads)
    torch_mhsa = nn.MultiheadAttention(D, num_heads, batch_first=True)

    with torch.no_grad():
        torch_mhsa.in_proj_weight.copy_(our_mhsa.qkv_weights.weight)
        torch_mhsa.in_proj_bias.copy_(our_mhsa.qkv_weights.bias)

        torch_mhsa.out_proj.weight.copy_(our_mhsa.linear1.weight)
        torch_mhsa.out_proj.bias.copy_(our_mhsa.linear1.bias)
    
    our_mhsa.eval()
    torch_mhsa.eval()

    our_out, our_attn = our_mhsa(x, return_attn=True)
    torch_out, torch_attn = torch_mhsa(x, x, x, need_weights=True, average_attn_weights=False)

    assert torch.allclose(our_out, torch_out, atol=1e-5)
    assert torch.allclose(our_attn, torch_attn, atol=1e-5)


def test_attention_weights_shape_and_sum():
    """Attention weights shape (N, h, T, T) and each row sums to 1."""
    N, P, D, num_heads = 2, 10, 10, 2
    x = torch.randn(N, P, D)

    test_mha = mha(D, num_heads)

    test_mha.eval()
    _, attn = test_mha(x, return_attn=True)
    row_sum = torch.sum(attn, dim=-1)

    assert attn.shape == (N, num_heads, P, P)
    assert torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-5)


def test_encoder_permutation_equivariance(tiny_cfg):
    """Without positional embedding: encoder(x[:, perm]) == encoder(x)[:, perm]."""
    N, T, D = 2, 10, tiny_cfg.embed_dim
    x = torch.randn(N, T, D)

    encoder = Encoder(tiny_cfg)
    
    encoder.eval()

    perm = torch.randperm(T) # permute

    out1 = encoder(x)[:, perm, :]
    out2 = encoder(x[:, perm, :])

    assert torch.allclose(out1, out2, atol=1e-5)


@pytest.mark.parametrize("num_tokens", [17, 65])
def test_encoder_any_sequence_length(tiny_cfg, num_tokens):
    """Same encoder must work for different T (patch dropout changes T)."""
    N, T, D = 2, num_tokens, tiny_cfg.embed_dim
    x = torch.randn(N, T, D)

    encoder = Encoder(tiny_cfg)
    encoder.eval()

    out = encoder(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_depth_zero_is_only_layernorm(tiny_cfg):
    """depth = 0 -> encoder returns LN(x)."""
    N, T, D = 2, 10, tiny_cfg.embed_dim
    x = torch.randn(N, T, D)

    new_cfg = replace(tiny_cfg, depth=0)
    encoder = Encoder(new_cfg)

    encoder.eval()

    out = encoder(x)
    expect = encoder.layer_norm(x)

    assert torch.equal(out, expect)


def test_eval_is_deterministic(tiny_cfg):
    """In eval mode two forward passes give same output (no dropout / drop path active)."""
    N, T, D = 2, 10, tiny_cfg.embed_dim
    x = torch.randn(N, T, D)

    encoder = Encoder(tiny_cfg)

    encoder.eval()

    out1 = encoder(x)
    out2 = encoder(x)

    assert torch.equal(out1, out2)