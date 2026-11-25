"""
Patch dropout: in training we randomly keep only some patch tokens (CLS always stays).
This is our regularization fallback from proposal, against overfitting.

Paper reference: not in ViT paper. Idea from PatchDropout (Liu et al., 2023)
Author: <name>
Adapted from: none
"""

from __future__ import annotations

import torch
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
        if not 0.0 <= rate < 1.0:
            raise ValueError(f"PatchDropout rate must be in [0, 1), but it is {rate}")
        self.rate = rate

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: tokens, (N, 1 + P, D)
        Returns:
            kept tokens, (N, 1 + K, D)
        """
        # in eval mode or when it is off, do nothing
        if not self.training or self.rate == 0.0:
            return x

        n, t, d = x.shape  # t = 1 + P
        cls = x[:, :1]  # (N, 1, D)
        patches = x[:, 1:]  # (N, P, D)
        p = t - 1
        k = max(1, round((1 - self.rate) * p))  # how many patches we keep

        # own random order for each sample, take first K -> random K patches
        # random tensor must be on same device like x
        keep_idx = torch.rand(n, p, device=x.device).argsort(dim=1)[:, :k]  # (N, K)

        # gather needs index with same number of dims, so repeat index for all D
        keep_idx = keep_idx.unsqueeze(-1).expand(-1, -1, d)  # (N, K, D)
        kept = torch.gather(patches, dim=1, index=keep_idx)  # (N, K, D)

        return torch.cat([cls, kept], dim=1)
