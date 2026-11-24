"""
Add CLS token in front of patch sequence and add learned 1D positional embedding.

Paper reference: ViT Eq. 1 (x_class, E_pos), Section 3.1
Author: <name>
Adapted from: none
"""

from __future__ import annotations

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
        # TODO: cls_token, pos_embed (nn.Parameter with zeros), dropout layer

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: patch vectors, (N, P, D)
        Returns:
            tokens with CLS and positions, (N, P + 1, D)
        """
        # TODO
        raise NotImplementedError
