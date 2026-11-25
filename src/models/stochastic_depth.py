"""
Stochastic depth (DropPath): in training randomly skip whole residual branch for some samples.

Paper reference: not in ViT paper. Huang et al., 2016 (Deep Networks with Stochastic Depth)
Author: <name>
Adapted from: none   (if you look at timm DropPath, write it here!)
"""

from __future__ import annotations

from torch import Tensor, nn
import torch

class DropPath(nn.Module):
    """
    For each sample: with probability p output is zero, otherwise output is x / (1 - p).
    So expected value stays same as x.
    Mask shape is (N, 1, 1), so one decision for whole sample.
    In eval mode or p == 0 it does nothing.

    Args:
        p: drop probability, in [0, 1)

    Shapes:
        input:  (N, T, D)
        output: (N, T, D)
    """

    def __init__(self, p: float = 0.0):
        super().__init__()

        self.p = p

        assert p >= 0.0 and p < 1.0

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: output of residual branch, (N, T, D)
        Returns:
            (N, T, D)
        """
        if not self.training or self.p == 0.0:
            return x

        N, T, D = x.shape

        mask = torch.rand(N, 1, 1, device=x.device) < (1.0-self.p)
        x = x * mask / (1.0-self.p)

        return x


def drop_path_schedule(max_rate: float, depth: int) -> list[float]:
    """
    Drop rate for each block, growing linear from 0 to max_rate.
    Example: drop_path_schedule(0.1, 6) -> [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]
    """
    if depth == 0:
        return []
    if depth == 1:
        return [0.0]
    
    gap = max_rate / (depth-1)
    
    return [gap * i for i in range(depth)]
