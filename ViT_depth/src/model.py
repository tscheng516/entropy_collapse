"""
HookedViTDepth — ViT-B/16 backbone with a lightweight depth-prediction head.

Architecture
------------
* Backbone: ``timm`` ViT-B/16 (or any supported ViT), with ``num_classes=0``
  and ``global_pool=''`` so that ``forward_features`` returns all token
  embeddings ``(B, 1 + N, D)`` where ``N = (img_size / patch_size)^2``.
* Depth head: LayerNorm + 2-layer convolutional projection
  (embed_dim → 256 → 1) applied to the 2-D patch-feature grid, followed
  by bilinear upsampling to the full input resolution and Softplus to
  guarantee positive depth values.

All hooks from ``ViT/src/model.py`` are preserved verbatim:
  * Flash / fused attention disabled on every block.
  * Per-layer attention matrix caching (``block.attn.last_att``) for entropy
    computation.
  * Optional QK normalisation.
  * Runtime attention temperature support.

The model exposes a ``blocks`` property that delegates to
``backbone.blocks`` so that all helpers in ``src/helpers.py`` (curvature
masks, attention entropy) work without modification.

Usage::

    from src.model import build_hooked_vit_depth

    model = build_hooked_vit_depth(
        model_name="vit_base_patch16_224",
        img_size=448,
        patch_size=16,
        init_std=0.02,
    )
    # model(x) → (B, 1, img_size, img_size)  depth in metres (positive)
"""

from __future__ import annotations

import math
import types
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Minimum depth offset added after Softplus to prevent numerically-zero
# predictions, which would produce -inf when computing log(pred) in SILog.
_MIN_DEPTH_OFFSET: float = 1e-3


# ---------------------------------------------------------------------------
# Patched attention forward — identical to ViT/src/model.py
# ---------------------------------------------------------------------------

def _patched_attn_forward(
    self,
    x: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Drop-in replacement for timm's ``Attention.forward`` that:
      1. Computes explicit (non-fused) softmax attention.
      2. Caches the attention weight tensor as ``self.last_att``.
      3. Applies an optional ``self.temperature`` scale to the logits.
    """
    B, N, C = x.shape
    head_dim: int = getattr(self, "head_dim", C // self.num_heads)
    attn_dim: int = getattr(self, "attn_dim", C)

    qkv = (
        self.qkv(x)
        .reshape(B, N, 3, self.num_heads, head_dim)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, v = qkv.unbind(0)

    q = self.q_norm(q)
    k = self.k_norm(k)

    temperature = getattr(self, "temperature", 1.0)
    attn = (q * (self.scale / temperature)) @ k.transpose(-2, -1)

    if is_causal:
        causal_bias = attn.new_full((N, N), float("-inf")).triu_(1)
        attn = attn + causal_bias
    elif attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            bias = torch.zeros_like(attn_mask, dtype=attn.dtype)
            bias.masked_fill_(~attn_mask, float("-inf"))
            attn = attn + bias
        else:
            attn = attn + attn_mask

    attn = attn.softmax(dim=-1)

    if getattr(self, "_cache_attn", False):
        self.last_att = attn.detach()

    attn = self.attn_drop(attn)
    x = (attn @ v).transpose(1, 2).reshape(B, N, attn_dim)
    _norm = getattr(self, "norm", None)
    if _norm is not None:
        x = _norm(x)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


# ---------------------------------------------------------------------------
# Depth head
# ---------------------------------------------------------------------------

class _DepthHead(nn.Module):
    """
    Lightweight convolutional depth-prediction head.

    Takes a 2-D patch feature grid of shape ``(B, D, H', W')`` and produces
    a coarse depth map ``(B, 1, H', W')``.  The calling code upsamples this
    back to the full ``(B, 1, H, W)`` input resolution.

    Architecture: LayerNorm → Conv1×1(D→256) → GELU → Conv1×1(256→1).

    Args:
        embed_dim: Transformer embedding dimension (input channel count).
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.norm  = nn.LayerNorm(embed_dim)
        self.conv1 = nn.Conv2d(embed_dim, 256, kernel_size=1)
        self.act   = nn.GELU()
        self.conv2 = nn.Conv2d(256, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, D, H', W')`` patch feature grid.

        Returns:
            ``(B, 1, H', W')`` raw (pre-softplus) depth predictions.
        """
        B, D, H, W = x.shape
        # Apply LayerNorm in feature-last convention, then restore.
        x = x.permute(0, 2, 3, 1)   # (B, H', W', D)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)   # (B, D, H', W')
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)            # (B, 1, H', W')
        return x


# ---------------------------------------------------------------------------
# Wrapper model
# ---------------------------------------------------------------------------

class HookedViTDepth(nn.Module):
    """
    ViT-B/16 (or any timm ViT) with hooked attention and a depth head.

    The model exposes a ``blocks`` property that mirrors the backbone's
    transformer blocks so that all helpers in ``src/helpers.py`` work
    without modification.

    Args:
        backbone:     timm ViT model built with ``num_classes=0`` and
                      ``global_pool=''``.
        depth_head:   ``_DepthHead`` instance.
        n_patches_h:  Number of patches along the height dimension
                      (``img_size // patch_size``).
        n_patches_w:  Number of patches along the width dimension.
    """

    def __init__(
        self,
        backbone: nn.Module,
        depth_head: _DepthHead,
        n_patches_h: int,
        n_patches_w: int,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.depth_head = depth_head
        self.n_patches_h = n_patches_h
        self.n_patches_w = n_patches_w

    @property
    def blocks(self):
        """Delegate to backbone transformer blocks."""
        return self.backbone.blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``(B, 3, H, W)`` RGB image tensor.

        Returns:
            ``(B, 1, H, W)`` predicted depth map in metres (positive values).
        """
        B, C, H, W = x.shape

        # Backbone → (B, 1+N, D) where 1 is the CLS token.
        tokens = self.backbone.forward_features(x)

        # Drop CLS, reshape patch tokens to 2-D grid.
        patch_tokens = tokens[:, 1:, :]                # (B, N, D)
        D = patch_tokens.size(-1)
        patch_tokens = patch_tokens.reshape(
            B, self.n_patches_h, self.n_patches_w, D
        ).permute(0, 3, 1, 2)                           # (B, D, H', W')

        # Depth head → coarse depth (B, 1, H', W').
        depth_coarse = self.depth_head(patch_tokens)

        # Upsample to input resolution.
        depth = F.interpolate(
            depth_coarse, size=(H, W), mode="bilinear", align_corners=True
        )

        # Softplus ensures strictly positive depth; offset avoids near-zero.
        depth = F.softplus(depth) + _MIN_DEPTH_OFFSET      # (B, 1, H, W)
        return depth


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_hooked_vit_depth(
    model_name: str = "vit_base_patch16_224",
    img_size: int = 448,
    patch_size: int = 16,
    init_std: float = 0.02,
    use_scaled_init: bool = True,
    qk_norm: bool = False,
    depth: Optional[int] = None,
    num_heads: Optional[int] = None,
    embed_dim: Optional[int] = None,
    device: str = "cuda",
) -> HookedViTDepth:
    """
    Build and return a ``HookedViTDepth`` model ready for depth-estimation
    entropy-collapse experiments.

    The returned model:
      * Has flash / fused attention disabled on every block.
      * Caches the attention weight matrix as ``block.attn.last_att`` after
        each forward pass (used by ``get_attention_entropy``).
      * Re-initialises weights from scratch with ``init_std`` (no pretrained
        weights — entropy dynamics are most vivid when trained from scratch).
      * Applies optional QK normalisation.

    Args:
        model_name:      timm model string (default: ``'vit_base_patch16_224'``).
        img_size:        Input spatial resolution (square).  448 gives 28×28=784
                         patches for patch_size=16.
        patch_size:      Patch size passed to timm.  Must evenly divide img_size.
        init_std:        Std for weight initialisation (default 0.02, standard ViT).
        use_scaled_init: Scale each attention output-projection by
                         ``init_std / sqrt(2 * depth)`` (depth-scaled residual init).
        qk_norm:         Install per-head LayerNorm on Q/K projections.
        depth:           Override number of transformer layers (``None`` = timm default).
        num_heads:       Override number of attention heads (``None`` = timm default).
        embed_dim:       Override embedding dimension (``None`` = timm default).
        device:          Target device string.

    Returns:
        A ``HookedViTDepth`` model on ``device``.
    """
    import timm

    arch_kwargs: dict = {"patch_size": patch_size}
    if depth is not None:
        arch_kwargs["depth"] = depth
    if num_heads is not None:
        arch_kwargs["num_heads"] = num_heads
    if embed_dim is not None:
        arch_kwargs["embed_dim"] = embed_dim

    backbone = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=0,          # no classification head
        global_pool="",         # return all tokens (B, 1+N, D)
        img_size=img_size,
        qk_norm=qk_norm,
        **arch_kwargs,
    )

    # Infer embedding dimension from the backbone.
    _embed_dim: int = backbone.embed_dim

    # Compute number of patches per spatial dimension.
    n_patches_h = img_size // patch_size
    n_patches_w = img_size // patch_size

    depth_head = _DepthHead(embed_dim=_embed_dim)

    # ------------------------------------------------------------------ #
    # Re-initialise all weights from scratch
    # ------------------------------------------------------------------ #
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.normal_(module.weight, mean=0.0, std=init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=init_std)

    backbone.apply(_init_weights)
    depth_head.apply(_init_weights)

    if use_scaled_init:
        n_layers = len(backbone.blocks)
        scale = init_std / math.sqrt(2 * n_layers)
        for name, param in backbone.named_parameters():
            if name.endswith("attn.proj.weight"):
                nn.init.normal_(param, mean=0.0, std=scale)

    # ------------------------------------------------------------------ #
    # Patch every attention block to cache attention matrices
    # ------------------------------------------------------------------ #
    for block in backbone.blocks:
        attn = block.attn
        if hasattr(attn, "fused_attn"):
            attn.fused_attn = False
        attn.forward = types.MethodType(_patched_attn_forward, attn)

    model = HookedViTDepth(
        backbone=backbone,
        depth_head=depth_head,
        n_patches_h=n_patches_h,
        n_patches_w=n_patches_w,
    )
    model.to(device)
    return model


def set_attention_temperature(model: nn.Module, temperature: float) -> None:
    """
    Set the attention temperature on every transformer block.

    Identical to the ViT/ version; works with ``HookedViTDepth`` and its
    ``DistributedDataParallel`` wrapper.

    Args:
        model:       A ``HookedViTDepth`` (or DDP wrapper).
        temperature: Positive float.  1.0 restores default behaviour.
    """
    raw = model.module if hasattr(model, "module") else model
    for block in raw.blocks:
        block.attn.temperature = temperature
