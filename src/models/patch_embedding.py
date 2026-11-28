"""
Patch embedding: Split an input image into non-overlapping patches, 
flatten each patch, and project it into a D-dimensional embedding space.

Reshape + permute + Linear is equivalent to Conv2d with
kernel_size = stride = patch_size.

Paper reference: ViT Fig. 1 (Patch Embedding), Eq. 1 and Section 3.1
Author: <name>
Adapted from: none
"""

from torch import Tensor, nn

class PatchEmbedding(nn.Module):
    """
    Image -> sequence of patch vectors.

    We cut image into p x p patches with reshape + permute, flatten each patch to
    p*p*C numbers, and project with Linear(p*p*C, D). This is matrix E from paper Eq. 1.
    Same result like Conv2d(C, D, kernel_size=p, stride=p) (test_embeddings.py checks this).

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

        if image_size % patch_size != 0:
            raise ValueError("image_size should be divided by patch_size")

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels

        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = self.in_channels * self.patch_size**2

        self.linear1 = nn.Linear(self.patch_dim, embed_dim)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: images, (N, C, H, W)
        Returns:
            patch vectors, (N, P, D)
        """
        N, C, H, W = x.shape

        # 1. give ValueError if H, W is not image_size
        if H != self.image_size or W != self.image_size:
            raise ValueError("H, W does not match image_size")

        # 2. reshape   (N, C, H, W) -> (N, C, H/p, p, W/p, p)
        x = x.reshape(N, C, H//self.patch_size, self.patch_size, W//self.patch_size, self.patch_size)

        # 3. permute   -> (N, H/p, W/p, C, p, p)      order (0, 2, 4, 1, 3, 5)
        x = x.permute(0, 2, 4, 1, 3, 5)

        # 4. reshape   -> (N, P, p*p*C)
        x = x.reshape(N, self.num_patches, self.patch_dim)

        # 5. Linear    -> (N, P, D)
        x = self.linear1(x)
        return x
