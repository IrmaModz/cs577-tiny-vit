"""
Cut image into square patches and project each patch into vector of size D.

Paper reference: ViT Eq. 1 (x_p E), Section 3.1
Author: <name>
Adapted from: none
"""

from __future__ import annotations

from torch import Tensor, nn


class PatchEmbedding(nn.Module):
    """
    Image -> sequence of patch vectors.

    We do it with Conv2d(C, D, kernel_size=p, stride=p), then flatten and transpose.
    This is same as cutting image into p x p patches, flatten each one to p*p*C numbers,
    and multiply with matrix E of shape (p*p*C, D). (explain this also in report)

    Patch order is row by row: left to right, top to bottom.

    Args:
        image_size:  H = W (32)
        patch_size:  p (4 or 8)
        in_channels: C (3)
        embed_dim:   D

    Attributes:
        num_patches: P = (H / p) ** 2

    Shapes:
        input:  (N, C, H, W)
        output: (N, P, D)
    """

    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        # TODO: check image_size % patch_size == 0, save num_patches, create conv layer

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: images, (N, C, H, W)
        Returns:
            patch vectors, (N, P, D)
        """
        # TODO: give ValueError if H, W is not image_size
        raise NotImplementedError
