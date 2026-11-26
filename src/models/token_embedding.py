"""
Add CLS token in front of patch sequence and add learned 1D positional embedding.

Paper reference: ViT Eq. 1 (x_class, E_pos), Section 3.1
Author: <name>
Adapted from: none
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch
import numpy as np

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
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # static positional encoding / might change to learnable as paper
        pos_embed = sinusoidal_positions(num_patches+1, embed_dim)
        pos_embed = torch.tensor(pos_embed, dtype=torch.float32).unsqueeze(0)

        self.register_buffer("pos_embedding", pos_embed)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: patch vectors, (N, P, D)
        Returns:
            tokens with CLS and positions, (N, P + 1, D)
        """
        N = x.shape[0]

        # copy cls_token as number of batch
        cls_token = self.cls_token.expand(N, -1, -1)

        # concat
        x = torch.cat((cls_token, x), dim=1)

        #pos encoding
        x = x + self.pos_embedding

        # dropout
        x = self.dropout(x)
        return x

def sinusoidal_positions(length: int, model_dim: int) -> np.ndarray:
    if model_dim%2 != 0:
        raise ValueError("model_dim should be even")

    pe = np.zeros((length, model_dim))

    m = np.arange(model_dim/2)
    p = np.arange(length).reshape(-1, 1)

    wm = np.power(10000., -(2*m)/model_dim)

    pe[:, 0::2] = np.sin(p*wm)
    pe[:, 1::2] = np.cos(p*wm)

    return pe
