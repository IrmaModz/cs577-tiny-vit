"""
Multi-head self-attention, written by hand (no nn.MultiheadAttention here).

Paper reference: ViT Eq. 2, Appendix A (Eq. 5-8); Vaswani et al., 2017
Author: <name>
Adapted from: none
"""

from __future__ import annotations

from torch import Tensor, nn
import torch

class MultiHeadSelfAttention(nn.Module):
    """
    Steps (shapes in comments when implementing):
        1. qkv = Linear(D, 3D)(x)                -> (N, T, 3D), rows ordered [q; k; v]
        2. reshape and split                     -> q, k, v each (N, h, T, d_h)
        3. scores = q @ k^T / sqrt(d_h)          -> (N, h, T, T)
        4. softmax over last axis, attn dropout  -> (N, h, T, T)
        5. out = attn @ v                        -> (N, h, T, d_h)
        6. merge heads                           -> (N, T, D)
        7. proj = Linear(D, D), proj dropout     -> (N, T, D)

    Must work for any T (patch dropout changes T).

    Args:
        dim:          D
        num_heads:    h (D must divide by h)
        attn_dropout: dropout on attention weights
        proj_dropout: dropout after output projection

    Shapes:
        input:  (N, T, D)
        output: (N, T, D), and attention weights (N, h, T, T) if return_attn=True
    """

    def __init__(self, dim: int, num_heads: int, attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

        # initialize qkv and fc weights
        self.qkv_weights = nn.Linear(dim, 3*dim)
        self.linear1 = nn.Linear(self.dim, self.dim)

        assert dim % num_heads == 0

    def forward(self, x: Tensor, return_attn: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """
        Args:
            x:           tokens, (N, T, D)
            return_attn: also return attention weights (for tests and visualization)
        Returns:
            out (N, T, D), or (out, attn) where attn is (N, h, T, T)
        """
        # B: batch size, T: token count, D: embedding dim
        N, T, _ = x.shape

        qkv = self.qkv_weights(x) # (N,T,3D)
        q, k, v = qkv.chunk(3, dim=-1) # (N,T,D)

        # To (N,num_heads,T,d_h)
        d_h = self.dim//self.num_heads
        q = q.reshape(N, T, self.num_heads, d_h).transpose(1, 2)
        k = k.reshape(N, T, self.num_heads, d_h).transpose(1, 2)
        v = v.reshape(N, T, self.num_heads, d_h).transpose(1, 2)

        # (N,num_heads,T,T)
        scores = q @ k.transpose(-2, -1) * d_h**(-0.5)

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # (N,num_heads,T,d_h)
        out = attn @ v

        # merge heads (N,T,D)
        out = out.transpose(1, 2).reshape(N, T, self.dim)

        # proj and dropout
        proj = self.linear1(out)
        proj = self.proj_dropout(proj)

        if return_attn:
            return proj, attn
        
        return proj
