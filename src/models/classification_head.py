"""
Classification head: take CLS token and give 10 class scores (logits).

Paper reference: ViT Eq. 4 (y = LN(z_L^0)), Section 3.1
Author: <name>
Adapted from: none
"""

from __future__ import annotations

from torch import Tensor, nn


class ClassificationHead(nn.Module):
    """
    Takes CLS token x[:, 0] and puts it through one Linear(D, num_classes).

    NO LayerNorm here, it is already in encoder. This is still same like paper Eq. 4,
    because LN works on each token separately: LN(z_L)[:, 0] == LN(z_L[:, 0]).

    Args:
        embed_dim:   D
        num_classes: 10

    Shapes:
        input:  (N, T, D)
        output: (N, num_classes)
    """

    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        # TODO: one linear layer

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: encoder output, (N, T, D)
        Returns:
            logits, (N, num_classes)
        """
        # TODO
        raise NotImplementedError
