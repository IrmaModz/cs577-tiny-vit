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
        # TODO: check image_size % patch_size == 0, save num_patches, create Linear(p*p*C, D)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: images, (N, C, H, W)
        Returns:
            patch vectors, (N, P, D)
        """
        # TODO:
        # 1. give ValueError if H, W is not image_size
        # 2. reshape   (N, C, H, W) -> (N, C, H/p, p, W/p, p)
        # 3. permute   -> (N, H/p, W/p, C, p, p)      order (0, 2, 4, 1, 3, 5)
        # 4. reshape   -> (N, P, p*p*C)
        # 5. Linear    -> (N, P, D)
        raise NotImplementedError
