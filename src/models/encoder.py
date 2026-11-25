"""
Transformer block and full encoder (L blocks + final LayerNorm).

Paper reference: ViT Eq. 2-3 (pre-norm blocks) and Eq. 4 (final LN), Section 3.1
Author: <name>
Adapted from: none
"""

from __future__ import annotations

from torch import Tensor, nn

from configs.config import ViTConfig


class TransformerBlock(nn.Module):
    """
    One pre-norm transformer block:
        x = x + DropPath(Attention(LN(x)))
        x = x + DropPath(MLP(LN(x)))

    Args:
        dim:          D
        num_heads:    h
        mlp_dim:      M
        dropout:      dropout in MLP and after attention projection
        attn_dropout: dropout on attention weights
        drop_path:    stochastic depth rate for this block

    Shapes:
        input:  (N, T, D)
        output: (N, T, D)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        # TODO: 2 LayerNorms, MultiHeadSelfAttention, MLP, DropPath

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: tokens, (N, T, D)
        Returns:
            (N, T, D)
        """
        # TODO
        raise NotImplementedError


class Encoder(nn.Module):
    """
    L transformer blocks, then FINAL LayerNorm on all tokens.
    (we agreed: final LN is here, NOT in classification head)

    - drop path rates come from drop_path_schedule(cfg.drop_path_rate, cfg.depth)
    - depth = 0 must work too (then it is only final LN), test_gradients uses it
    - must work for any T

    Args:
        cfg: ViTConfig

    Shapes:
        input:  (N, T, D)
        output: (N, T, D), normalized
    """

    def __init__(self, cfg: ViTConfig):
        super().__init__()
        # TODO: nn.ModuleList of TransformerBlock, final LayerNorm

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: tokens, (N, T, D)
        Returns:
            LN(z_L), (N, T, D)
        """
        # TODO
        raise NotImplementedError
