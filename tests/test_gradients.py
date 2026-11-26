"""
Gradient tests. This is evidence for our focused gradient-flow section in report.

Paper reference: ViT Eq. 1-4
Author: <name>
Adapted from: none
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from configs.config import ViTConfig
from src.models.vit import ViT


def params_without_good_grad(model: nn.Module) -> list[str]:
    """
    Our grad checker: names of trainable params where grad is missing,
    not finite (nan / inf), or all zero. Empty list means all is ok.
    """
    bad = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum() == 0:
            bad.append(name)
    return bad


def test_all_parameters_get_gradient(tiny_cfg, tiny_batch):
    """After one backward, every trainable param (also cls_token, pos_embed, patch proj) has finite non-zero grad."""
    x, y = tiny_batch  # (4, 3, 32, 32), (4,)
    model = ViT(tiny_cfg).train()
    F.cross_entropy(model(x), y).backward()
    assert params_without_good_grad(model) == []


def test_logits_gradient_matches_formula(tiny_cfg, tiny_batch):
    """Analytic check: dL/dlogits == (softmax(logits) - onehot(y)) / N for mean cross entropy."""
    x, y = tiny_batch
    n = x.shape[0]
    model = ViT(tiny_cfg).eval()

    logits = model(x)  # (N, 10)
    logits.retain_grad()  # logits is not a leaf, so we ask torch to keep its grad
    F.cross_entropy(logits, y).backward()  # mean over N

    expected = (torch.softmax(logits.detach(), dim=1) - F.one_hot(y, 10).float()) / n  # (N, 10)
    assert torch.allclose(logits.grad, expected, atol=1e-6)


def patch_token_grad(model: ViT, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Forward part by part, so we can keep gradient of tokens right after token_embed.
    Returns gradient for patch tokens only (without CLS), shape (N, P, D).
    """
    tokens = model.token_embed(model.patch_embed(x))  # (N, P+1, D)
    tokens.retain_grad()
    logits = model.head(model.encoder(model.patch_dropout(tokens)))  # (N, 10)
    F.cross_entropy(logits, y).backward()
    return tokens.grad[:, 1:]  # (N, P, D)


def test_patch_gradient_only_through_attention(tiny_cfg, tiny_batch):
    """
    Known case: with depth = 0, gradient to patch tokens is exactly 0 (head reads only CLS).
    With depth >= 1, gradient to every patch is non-zero -> it comes only through attention.
    """
    x, y = tiny_batch

    # depth 0: encoder is only final LayerNorm, it works on each token alone,
    # and head reads only CLS -> patches can not change the loss at all
    model0 = ViT(replace(tiny_cfg, depth=0)).eval()
    g0 = patch_token_grad(model0, x, y)
    assert torch.all(g0 == 0)

    # depth 2: attention mixes tokens, so every patch of every image gets gradient
    model2 = ViT(tiny_cfg).eval()
    g2 = patch_token_grad(model2, x, y)
    per_patch = g2.abs().sum(dim=-1)  # (N, P)
    assert torch.all(per_patch > 0)


def test_gradcheck_tiny_model():
    """torch.autograd.gradcheck in float64 on very tiny ViT (D=8, h=2, L=1, p=16)."""
    cfg = ViTConfig(
        patch_size=16,  # 32 / 16 -> only 4 patches, so it is fast
        embed_dim=8,
        num_heads=2,
        depth=1,
        mlp_ratio=2.0,
        dropout=0.0,
        attn_dropout=0.0,
        drop_path_rate=0.0,
    )
    model = ViT(cfg).double().eval()  # float64 is needed for exact numeric check
    x = torch.randn(1, 3, 32, 32, dtype=torch.float64, requires_grad=True)

    # compares our backward with numeric derivative (small change of input)
    # fast_mode checks random directions, much faster for big input
    assert torch.autograd.gradcheck(model, (x,), eps=1e-6, atol=1e-4, fast_mode=True)


def test_detached_path_is_detected(tiny_cfg, tiny_batch):
    """Controlled failure: if patch_embed output is .detach()-ed, our grad checker must catch it."""
    x, y = tiny_batch
    model = ViT(tiny_cfg).train()

    # break the graph on purpose: hook returns detached output, so no grad goes back to patch_embed
    model.patch_embed.register_forward_hook(lambda module, inputs, output: output.detach())

    F.cross_entropy(model(x), y).backward()
    bad = params_without_good_grad(model)

    patch_params = [n for n, _ in model.named_parameters() if n.startswith("patch_embed.")]
    assert patch_params  # make sure we really have params there
    assert all(n in bad for n in patch_params)  # checker catches all of them
    assert all(n.startswith("patch_embed.") for n in bad)  # and nothing else is broken