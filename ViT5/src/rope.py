"""
2-D Rotary Position Embedding for ViT-5.

Adapted from the official ViT-5 implementation:
    https://github.com/wangf3014/ViT-5/blob/main/rope.py

Key change from upstream: replaced hard-coded ``.cuda()`` calls with
device-agnostic ``device=x.device`` so the module works on CPU and MPS too.
"""

from math import pi
import math

import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange, repeat
import numpy as np


def broadcat(freqss, dim=-1):
    num_freqss = len(freqss)
    shape_lens = set(list(map(lambda t: len(t.shape), freqss)))
    assert len(shape_lens) == 1, "freqss must all have the same number of dimensions"
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), freqss)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(
        [*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]
    ), "invalid dimensions for broadcastable concatenation"
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_freqss), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    freqss = list(map(lambda t: t[0].expand(*t[1]), zip(freqss, expandable_shapes)))
    return torch.cat(freqss, dim=dim)


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class VisionRotaryEmbedding(nn.Module):
    """
    2-D rotary embedding for vision patch sequences.

    ``pt_seq_len`` is the training-time grid side-length (e.g. 14 for a
    224-image with patch_size=16).  At inference the frequencies are
    interpolated to fit the actual sequence length.

    Args:
        dim:        Half of the head dimension (``head_dim // 2``).
        pt_seq_len: Training spatial grid size (one side of the square grid).
        theta:      Base frequency (default 10000, set to 100 for registers
                    per the ViT-5 paper).
    """

    def __init__(
        self,
        dim,
        pt_seq_len=14,
        custom_freqs=None,
        freqs_for="lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        elif freqs_for == "pixel":
            freqs = torch.linspace(1.0, max_freq / 2, dim // 2) * pi
        elif freqs_for == "constant":
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f"unknown modality {freqs_for}")

        self.pt_seq_len = pt_seq_len
        self.register_buffer("freqs", freqs)

    def forward(self, x):
        """
        Apply 2-D RoPE to a patch sequence.

        Args:
            x: Tensor of shape ``(B, N, num_heads, head_dim)`` where
               ``N`` must be a perfect square (the spatial patch grid).

        Returns:
            Tensor of the same shape with rotary encodings applied.
        """
        ft_seq_len = int(np.sqrt(x.shape[1]))
        # Device-agnostic range (upstream used hard-coded .cuda())
        t = torch.arange(ft_seq_len, device=x.device).float() / ft_seq_len * self.pt_seq_len

        freqs = torch.einsum("..., f -> ... f", t, self.freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)          # (ft_seq_len, dim)
        freqs = broadcat((freqs[:, None, :], freqs[None, :, :]), dim=-1)  # (ft_seq_len, ft_seq_len, 2*dim)

        freqs_cos = freqs.cos().view(-1, 1, freqs.shape[-1])   # (N, 1, head_dim)
        freqs_sin = freqs.sin().view(-1, 1, freqs.shape[-1])
        return x * freqs_cos + rotate_half(x) * freqs_sin


def rotate_freqs(freqs, angle_deg):
    """Rotate a 2-D frequency map by ``angle_deg`` degrees (bilinear interpolation)."""
    assert freqs.ndim == 4 and freqs.shape[0] == freqs.shape[1], (
        "Input must have shape (n, n, d1, d2)"
    )
    n, _, d1, d2 = freqs.shape
    freq_type = freqs.dtype
    angle_rad = math.radians(angle_deg)

    freqs = freqs.reshape(n, n, -1)
    freqs = freqs.permute(2, 0, 1).unsqueeze(0)

    theta = torch.tensor(
        [
            [math.cos(angle_rad), -math.sin(angle_rad), 0.0],
            [math.sin(angle_rad), math.cos(angle_rad), 0.0],
        ],
        dtype=torch.float32,
        device=freqs.device,
    ).unsqueeze(0)

    freqs = freqs.to(torch.float32)
    grid = F.affine_grid(theta, freqs.size(), align_corners=True)
    rotated = F.grid_sample(
        freqs, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    rotated = rotated.squeeze(0).permute(1, 2, 0).to(freq_type)
    return rotated.reshape(n, n, d1, d2)
