"""
Multi-head self-attention, written by hand (no nn.MultiheadAttention here).

Paper reference: ViT Eq. 2, Appendix A (Eq. 5-8); Vaswani et al., 2017
Author: <name>
Adapted from: none
"""

from __future__ import annotations

from torch import Tensor, nn


class MultiHeadSelfAttention(nn.Module):
    """
    Steps (shapes in comments when implementing):
        1. qkv = Linear(D, 3D)(x)                -> (N, T, 3D), rows ordered [q; k; v]
        2. reshape and split                     -> q, k, v each (N, h, T, d_h)
        3. scores = q @ k^T / sqrt(d_h)          -> (N, h, T, T)
        4. softmax over last axis, attn dropout  -> (N, h, T, T)
        5. out = attn @ v                        -> (N, h, T, d_h)
        6. merge heads                           -> (N, T, D)
        7. proj = Linear(D, D), proj dropout     -> (N, T, D)

    Must work for any T (patch dropout changes T).

    Args:
        dim:          D
        num_heads:    h (D must divide by h)
        attn_dropout: dropout on attention weights
        proj_dropout: dropout after output projection

    Shapes:
        input:  (N, T, D)
        output: (N, T, D), and attention weights (N, h, T, T) if return_attn=True
    """

    def __init__(self, dim: int, num_heads: int, attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        # TODO: qkv linear, proj linear, dropouts, scale = d_h ** -0.5

    def forward(self, x: Tensor, return_attn: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """
        Args:
            x:           tokens, (N, T, D)
            return_attn: also return attention weights (for tests and visualization)
        Returns:
            out (N, T, D), or (out, attn) where attn is (N, h, T, T)
        """
        # TODO
        raise NotImplementedError
