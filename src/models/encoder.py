"""
Transformer block and full encoder (L blocks + final LayerNorm).

Paper reference: ViT Eq. 2-3 (pre-norm blocks) and Eq. 4 (final LN), Section 3.1
Author: <name>
Adapted from: none
"""

from __future__ import annotations

from torch import Tensor, nn

from configs.config import ViTConfig

from src.models.attention import MultiHeadSelfAttention as mha
from src.models.mlp import MLP as mlp
import src.models.stochastic_depth as stochastic_depth

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

        self.drop_path = stochastic_depth.DropPath(drop_path)

        # Attention
        self.layer_norm1 = nn.LayerNorm(dim)
        self.mha = mha(dim, num_heads, attn_dropout, dropout)

        # MLP
        self.layer_norm2 = nn.LayerNorm(dim)
        self.mlp = mlp(dim, mlp_dim, dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: tokens, (N, T, D)
        Returns:
            (N, T, D)
        """
        # Attention (N,T,D)
        x = x + self.drop_path(self.mha(self.layer_norm1(x)))

        # MLP (N,T,D)
        x = x + self.drop_path(self.mlp(self.layer_norm2(x)))

        return x


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

        # stochastic depth probability lists
        drop_probs = stochastic_depth.drop_path_schedule(cfg.drop_path_rate, cfg.depth)

        # Build transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=cfg.embed_dim,
                             num_heads=cfg.num_heads,
                             mlp_dim=int(cfg.embed_dim*cfg.mlp_ratio),
                             dropout=cfg.dropout,
                             attn_dropout=cfg.attn_dropout,
                             drop_path=drop_probs[i])
            for i in range(cfg.depth)
        ])

        # Final layer norm
        self.layer_norm = nn.LayerNorm(cfg.embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: tokens, (N, T, D)
        Returns:
            LN(z_L), (N, T, D)
        """
        for block in self.blocks:
            x = block(x)
        x = self.layer_norm(x)
        return x