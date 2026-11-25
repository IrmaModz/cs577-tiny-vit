"""
Feed-forward network (MLP) inside each transformer block.

Paper reference: ViT Eq. 3, Section 3.1 (two layers with GELU)
Author: <name>
Adapted from: none
"""

from __future__ import annotations

from torch import Tensor, nn


class MLP(nn.Module):
    """
    Linear(D, M) -> GELU -> Dropout -> Linear(M, D) -> Dropout
    Works on each token separately.

    Args:
        dim:        D
        hidden_dim: M (= mlp_ratio * D)
        dropout:    dropout rate

    Shapes:
        input:  (N, T, D)
        output: (N, T, D)
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        # TODO: two linear layers, GELU, dropout

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: tokens, (N, T, D)
        Returns:
            (N, T, D)
        """
        # TODO
        raise NotImplementedError
