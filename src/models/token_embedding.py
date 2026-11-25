"""
Add CLS token in front of patch sequence and add learned 1D positional embedding.

Paper reference: ViT Eq. 1 (x_class, E_pos), Section 3.1
Author: <name>
Adapted from: none
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TokenEmbedding(nn.Module):
    """
    z0 = [cls ; patches] + pos_embed, then dropout.
    CLS token is always at index 0.

    Note: we create parameters with zeros here. Real init is done only in vit.py (_init_weights).

    Args:
        num_patches: P
        embed_dim:   D
        dropout:     dropout after adding positions

    Attributes:
        cls_token: (1, 1, D)
        pos_embed: (1, P + 1, D)

    Shapes:
        input:  (N, P, D)
        output: (N, P + 1, D)
    """

    def __init__(self, num_patches: int, embed_dim: int, dropout: float = 0.0):
        super().__init__()
        # one learned CLS vector, same for all images
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # (1, 1, D)
        # one learned position vector for each token: CLS + P patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))  # (1, P+1, D)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: patch vectors, (N, P, D)
        Returns:
            tokens with CLS and positions, (N, P + 1, D)
        """
        n = x.shape[0]
        # expand makes N "copies" without copying memory
        cls = self.cls_token.expand(n, -1, -1)  # (N, 1, D)
        x = torch.cat([cls, x], dim=1)  # (N, P+1, D), CLS is at index 0
        x = x + self.pos_embed  # (N, P+1, D) + (1, P+1, D) -> (N, P+1, D)
        return self.dropout(x)  # (N, P+1, D)
