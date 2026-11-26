"""
Full Vision Transformer: connects all model parts. Also weight init and helpers
for counting params and tracing shapes (for report table).

Paper reference: ViT Eq. 1-4, Section 3.1, Figure 1
Author: <name>
Adapted from: none
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from configs.config import ViTConfig
from src.models.classification_head import ClassificationHead
from src.models.encoder import Encoder
from src.models.patch_dropout import PatchDropout
from src.models.patch_embedding import PatchEmbedding
from src.models.token_embedding import TokenEmbedding

# names of the 5 parts, in forward order. trace_shapes uses this list too
PART_NAMES = ("patch_embed", "token_embed", "patch_dropout", "encoder", "head")


class ViT(nn.Module):
    """
    Order of forward:
        patch_embed -> token_embed -> patch_dropout -> encoder -> head

    Attribute names must be EXACTLY these, tests and profiling use them:
        self.patch_embed, self.token_embed, self.patch_dropout, self.encoder, self.head

    Args:
        cfg: ViTConfig

    Shapes:
        input:  (N, 3, 32, 32)
        output: (N, 10) logits
    """

    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg

        # image -> patch vectors (matrix E in paper Eq. 1)
        self.patch_embed = PatchEmbedding(
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            in_channels=cfg.in_channels,
            embed_dim=cfg.embed_dim,
        )
        # add CLS token and positions. we take P from config, it is always int
        self.token_embed = TokenEmbedding(
            num_patches=cfg.num_patches,
            embed_dim=cfg.embed_dim,
            dropout=cfg.dropout,
        )
        # only active in train mode and when rate > 0
        self.patch_dropout = PatchDropout(rate=cfg.patch_dropout_rate)
        # L transformer blocks + final LayerNorm
        self.encoder = Encoder(cfg)
        # CLS token -> 10 logits
        self.head = ClassificationHead(embed_dim=cfg.embed_dim, num_classes=cfg.num_classes)

        self._init_weights()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: images, (N, 3, 32, 32)
        Returns:
            logits, (N, 10)
        """
        x = self.patch_embed(x)  # (N, 3, 32, 32) -> (N, P, D)
        x = self.token_embed(x)  # (N, P, D) -> (N, P+1, D)
        x = self.patch_dropout(x)  # (N, P+1, D) -> (N, 1+K, D), K = P in eval mode
        x = self.encoder(x)  # (N, 1+K, D) -> (N, 1+K, D)
        return self.head(x)  # (N, 1+K, D) -> (N, 10)

    def _init_weights(self) -> None:
        """
        All weight init is ONLY here (other files dont do their own init):
            - Linear: trunc_normal std=0.02, bias = 0
            - LayerNorm: weight = 1, bias = 0
            - cls_token and pos_embed: trunc_normal std=0.02
            - patch conv: same like Linear
        """
        # apply() calls _init_module for every submodule (Linear, LayerNorm, ...)
        self.apply(_init_module)
        nn.init.trunc_normal_(self.token_embed.cls_token, std=0.02)
        nn.init.trunc_normal_(self.token_embed.pos_embed, std=0.02)


def _init_module(m: nn.Module) -> None:
    """Init for one module. Our patch projection is Linear, but Conv2d is here too, just in case."""
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Number of parameters in model (only trainable ones by default)."""
    return sum(
        p.numel() for p in model.parameters() if p.requires_grad or not trainable_only
    )


@torch.no_grad()
def trace_shapes(model: ViT, x: Tensor) -> list[tuple[str, tuple[int, ...]]]:
    """
    Run one forward and save output shape after each part (with forward hooks).
    This makes shape table for report, so we dont copy shapes by hand.

    We run in eval mode (so patch dropout does nothing), and after that put model
    back in the mode it was before.

    Returns:
        for example
        [("input", (N, 3, 32, 32)), ("patch_embed", (N, 64, 128)),
         ("token_embed", (N, 65, 128)), ("patch_dropout", (N, 65, 128)),
         ("encoder", (N, 65, 128)), ("head", (N, 10))]
    """
    shapes: list[tuple[str, tuple[int, ...]]] = [("input", tuple(x.shape))]
    handles = []

    def make_hook(name: str):
        # hook runs after the part, `output` is what the part returned
        def hook(module, inputs, output):
            shapes.append((name, tuple(output.shape)))
        return hook

    was_training = model.training
    model.eval()
    try:
        for name in PART_NAMES:
            handles.append(getattr(model, name).register_forward_hook(make_hook(name)))
        model(x)
    finally:
        for h in handles:
            h.remove()  # remove hooks, otherwise they stay and run in every forward
        model.train(was_training)

    return shapes