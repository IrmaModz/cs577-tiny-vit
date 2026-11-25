"""
Full Vision Transformer: connects all model parts. Also weight init and helpers
for counting params and tracing shapes (for report table).

Paper reference: ViT Eq. 1-4, Section 3.1, Figure 1
Author: <name>
Adapted from: none
"""

from __future__ import annotations

from torch import Tensor, nn

from configs.config import ViTConfig


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
        # TODO: create 5 parts from cfg, then call self._init_weights()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: images, (N, 3, 32, 32)
        Returns:
            logits, (N, 10)
        """
        # TODO
        raise NotImplementedError

    def _init_weights(self) -> None:
        """
        All weight init is ONLY here (other files dont do their own init):
            - Linear: trunc_normal std=0.02, bias = 0
            - LayerNorm: weight = 1, bias = 0
            - cls_token and pos_embed: trunc_normal std=0.02
            - patch conv: same like Linear
        """
        # TODO
        raise NotImplementedError


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Number of parameters in model (only trainable ones by default)."""
    # TODO
    raise NotImplementedError


def trace_shapes(model: ViT, x: Tensor) -> list[tuple[str, tuple[int, ...]]]:
    """
    Run one forward and save output shape after each part (with forward hooks).
    This makes shape table for report, so we dont copy shapes by hand.

    Returns:
        for example
        [("input", (N, 3, 32, 32)), ("patch_embed", (N, 64, 128)),
         ("token_embed", (N, 65, 128)), ("patch_dropout", (N, 65, 128)),
         ("encoder", (N, 65, 128)), ("head", (N, 10))]
    """
    # TODO
    raise NotImplementedError
