"""
Patch dropout: in training we randomly keep only some patch tokens (CLS always stays).
This is our regularization fallback from proposal, against overfitting.

Paper reference: not in ViT paper. Idea from PatchDropout (Liu et al., 2023)
Author: <name>
Adapted from: none
"""

from __future__ import annotations

from torch import Tensor, nn


class PatchDropout(nn.Module):
    """
    Keep random K patch tokens for each sample, K = max(1, round((1 - rate) * P)).

    IMPORTANT: use it AFTER TokenEmbedding (positions already added), otherwise kept
    tokens lose info where they come from.

    - CLS token (index 0) is always kept
    - each sample in batch gets its own random patches
    - in eval mode or rate == 0 it does nothing (returns input)

    Note: it makes sequence shorter, so encoder must work with any T.

    Args:
        rate: fraction of patch tokens to drop, in [0, 1)

    Shapes:
        input:  (N, 1 + P, D)
        output: (N, 1 + K, D)
    """

    def __init__(self, rate: float = 0.0):
        super().__init__()
        # TODO: save rate

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: tokens, (N, 1 + P, D)
        Returns:
            kept tokens, (N, 1 + K, D)
        """
        # TODO
        raise NotImplementedError
