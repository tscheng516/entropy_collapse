"""
HookedViT5 — ViT-5-Base model for entropy-collapse experiments.

ViT-5 (Wang et al., 2026, arXiv:2602.08071) modernises the canonical ViT
architecture with the following components, all active by default in the
Base variant used here:

  * 2-D Rotary Position Embedding (RoPE) on patch tokens, plus a separate
    low-theta RoPE for register tokens.
  * Register tokens (4 by default) appended after the patch sequence.
  * RMSNorm as the normalisation layer (instead of LayerNorm).
  * QK-normalisation on queries and keys (per-head RMSNorm).
  * Layer-scale initialisation (γ₁ / γ₂ learnable scalars per block).
  * Standard Transformer Mlp with GELU activation (mlp_ratio=4).

Additions for entropy-collapse experiments (mirrors ViT/src/model.py):
  * Per-layer attention-matrix caching (``block.attn.last_att``) so that
    entropy can be computed after every forward pass without re-running
    inference.  Enabled per-block via ``block.attn._cache_attn = True``.
  * Flash / SDPA attention is **always disabled**; the explicit softmax
    path is required for (a) attention caching and (b) second-order
    gradient computation via ``torch.autograd``.
  * Runtime attention temperature support: set ``attn.temperature`` on
    any block (or call ``set_attention_temperature``) to scale logits for
    entropy-collapse intervention experiments.

Usage::

    from src.model import build_hooked_vit5

    # CIFAR-100 (32×32, patch_size=4, 64 patches)
    model = build_hooked_vit5(
        num_classes=100,
        img_size=32,
        patch_size=4,
        drop_path_rate=0.2,
    )

    # ImageNet-1k (192×192, patch_size=16, 144 patches)
    model = build_hooked_vit5(
        num_classes=1000,
        img_size=192,
        patch_size=16,
        drop_path_rate=0.2,
    )
"""

from __future__ import annotations

import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import Mlp, PatchEmbed
from timm.models.layers import DropPath, trunc_normal_

from .rope import VisionRotaryEmbedding


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root-Mean-Square layer normalisation (no bias, single weight scale)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block (defined for completeness; not used in Base)."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer=nn.SiLU,
        drop: float = 0.0,
        norm_layer=nn.LayerNorm,
        subln: bool = False,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features, bias=False)
        self.w2 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """
    Multi-head self-attention with:
      * Optional QK-normalisation (per-head RMSNorm on Q and K).
      * 2-D RoPE for patch tokens and register tokens.
      * Attention caching: set ``self._cache_attn = True`` before the
        forward pass to store the attention weights in ``self.last_att``
        (shape: B × num_heads × N × N).
      * Runtime temperature scaling via ``self.temperature`` (default 1.0).
        Values > 1 soften the distribution; < 1 sharpen it.

    Flash / SDPA is always disabled here to keep the explicit softmax path
    required for Hessian computation and attention entropy measurement.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope_size: int = 0,
        rope_reg_size: int = 0,
        num_registers: int = 0,
        reg_theta: float = 10000,
        qk_norm: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.num_registers = num_registers

        # 2-D RoPE for patch tokens.
        self.rope = (
            VisionRotaryEmbedding(head_dim // 2, rope_size)
            if rope_size > 0
            else None
        )
        # Separate low-theta RoPE for register tokens.
        self.rope_reg = (
            VisionRotaryEmbedding(head_dim // 2, rope_reg_size, theta=reg_theta)
            if rope_reg_size > 0
            else None
        )

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # Register tokens are appended at the end of the sequence:
        #   [cls | patches | registers]
        reg_idx = N - self.num_registers

        # Fused QKV projection; split into (B, N, num_heads, head_dim) per tensor.
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(dim=2)  # each: (B, N, num_heads, head_dim)

        # Optional per-head QK normalisation (default: on in ViT-5-Base).
        if self.qk_norm:
            qk_dtype = q.dtype
            q = self.q_norm(q).to(qk_dtype)
            k = self.k_norm(k).to(qk_dtype)

        # Apply patch RoPE (tokens 1 … reg_idx-1; token 0 is the CLS token).
        if self.rope is not None:
            q = torch.cat(
                (q[:, :1], self.rope(q[:, 1:reg_idx]), q[:, reg_idx:]), dim=1
            )
            k = torch.cat(
                (k[:, :1], self.rope(k[:, 1:reg_idx]), k[:, reg_idx:]), dim=1
            )

        # Apply register RoPE (tokens reg_idx … N-1).
        if self.rope_reg is not None:
            q = torch.cat(
                (q[:, :1], q[:, 1:reg_idx], self.rope_reg(q[:, reg_idx:])), dim=1
            )
            k = torch.cat(
                (k[:, :1], k[:, 1:reg_idx], self.rope_reg(k[:, reg_idx:])), dim=1
            )

        # Transpose to (B, num_heads, N, head_dim) for matmul.
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        # Explicit softmax attention — always used (no flash / SDPA).
        temperature = getattr(self, "temperature", 1.0)
        attn = (q_t * (self.scale / temperature)) @ k_t.transpose(-2, -1)  # (B, nh, N, N)
        attn = attn.softmax(dim=-1)

        # Cache for entropy computation — only when enabled.
        if getattr(self, "_cache_attn", False):
            self.last_att = attn.detach()

        attn = self.attn_drop(attn)
        x = (attn @ v_t).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """
    Transformer block with layer-scale and optional stochastic depth.

    Layer scale (Touvron et al., 2021) initialises the residual branch
    weights at a small value (``init_values``) and learns them.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        Attention_block=Attention,
        Mlp_block=Mlp,
        init_values: float = 1e-4,
        rope_size: int = 0,
        rope_reg_size: int = 0,
        reg_theta: float = 10000,
        num_registers: int = 0,
        qk_norm: bool = False,
        layer_scale: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            rope_size=rope_size,
            rope_reg_size=rope_reg_size,
            num_registers=num_registers,
            qk_norm=qk_norm,
            reg_theta=reg_theta,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.layer_scale = layer_scale
        if layer_scale:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones(dim), requires_grad=True
            )
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones(dim), requires_grad=True
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layer_scale:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class vit_models(nn.Module):
    """
    ViT-5 Vision Transformer backbone.

    Structural differences from canonical timm ViT:
      * RMSNorm instead of LayerNorm.
      * QK-normalisation (per-head RMSNorm on Q and K).
      * 2-D RoPE for patch and register token positions.
      * Register tokens (``num_registers``) appended after patches.
      * Layer-scale (learnable γ₁ / γ₂ per block, init = ``init_scale``).
      * No QKV bias (``qkv_bias=False``).

    The absolute position embedding (APE) is kept alongside RoPE as in the
    official ViT-5 implementation.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer=nn.LayerNorm,
        ape: bool = True,
        block_layers=Block,
        Patch_layer=PatchEmbed,
        act_layer=nn.GELU,
        Attention_block=Attention,
        Mlp_block=Mlp,
        init_scale: float = 1e-4,
        rope: bool = False,
        num_registers: int = 0,
        qk_norm: bool = False,
        reg_theta: float = 10000,
        layer_scale: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.dropout_rate = drop_rate
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_registers = num_registers

        self.patch_embed = Patch_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.reg_token = (
            nn.Parameter(torch.zeros(1, num_registers, embed_dim))
            if num_registers > 0
            else None
        )

        # Register RoPE grid size: sqrt(num_registers) must be an integer.
        rope_reg_size = int(num_registers ** 0.5)
        assert rope_reg_size ** 2 == num_registers, (
            f"num_registers must be a perfect square, got {num_registers}"
        )

        # Absolute position embedding (APE) for the patch tokens.
        self.pos_embed = (
            nn.Parameter(torch.zeros(1, num_patches, embed_dim)) if ape else None
        )

        # Stochastic depth: linearly spaced from 0 to drop_path_rate.
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()

        self.blocks = nn.ModuleList(
            [
                block_layers(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=0.0,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    Attention_block=Attention_block,
                    Mlp_block=Mlp_block,
                    init_values=init_scale,
                    rope_size=img_size // patch_size if rope else 0,
                    rope_reg_size=rope_reg_size,
                    num_registers=num_registers,
                    qk_norm=qk_norm,
                    reg_theta=reg_theta,
                    layer_scale=layer_scale,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)

        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        # Weight initialisation.
        trunc_normal_(self.cls_token, std=0.02)
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        if num_registers > 0:
            trunc_normal_(self.reg_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "reg_token"}

    def get_classifier(self):
        return self.head

    def get_num_layers(self):
        return len(self.blocks)

    def reset_classifier(self, num_classes: int, global_pool: str = "") -> None:
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        registers = (
            self.reg_token.expand(B, -1, -1) if self.reg_token is not None else None
        )

        # Add absolute position embedding to patch tokens.
        if self.pos_embed is not None:
            x = x + self.pos_embed

        # Concatenate: [cls | patches | registers]
        x = torch.cat((cls_tokens, x), dim=1)
        if registers is not None:
            x = torch.cat((x, registers), dim=1)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]  # CLS token output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        if self.dropout_rate:
            x = F.dropout(x, p=float(self.dropout_rate), training=self.training)
        x = self.head(x)
        return x


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_hooked_vit5(
    num_classes: int = 1000,
    img_size: int = 192,
    patch_size: int = 16,
    drop_path_rate: float = 0.2,
    num_registers: int = 4,
    init_std: float = 0.02,
    use_scaled_init: bool = False,
    device: str = "cuda",
) -> torch.nn.Module:
    """
    Build and return a ViT-5-Base model ready for entropy-collapse experiments.

    Architecture (fixed, matches ``vit5_base`` in the official repo):
      * embed_dim = 768, depth = 12, num_heads = 12, mlp_ratio = 4
      * RMSNorm, QK-norm, 2-D RoPE, register tokens, layer scale
      * Absolute position embedding + RoPE (as in the official code)

    Only ``num_classes``, ``img_size``, ``patch_size``, and
    ``drop_path_rate`` differ between the CIFAR-100 and ImageNet-1k
    presets.

    Args:
        num_classes:      Output vocabulary size (100 for CIFAR-100,
                          1000 for ImageNet-1k).
        img_size:         Spatial resolution (32 for CIFAR-100, 192 for
                          ImageNet-1k as per the ViT-5-Base paper recipe).
        patch_size:       Patch stride (4 for CIFAR-100 → 64 patches;
                          16 for ImageNet-1k → 144 patches at 192×192).
        drop_path_rate:   Stochastic depth rate (ViT-5-Base default: 0.2).
        num_registers:    Number of register tokens (default 4, must be a
                          perfect square).
        init_std:         Std for weight re-initialisation.  The in-class
                          default of 0.02 matches the ViT-5 paper.
        use_scaled_init:  If True, scales ``attn.proj.weight`` by
                          ``init_std / sqrt(2 * depth)`` — the same
                          NanoGPT-style depth scaling used in ViT/.
                          Defaults to False since ViT-5 uses layer-scale.
        device:           Target device string.

    Returns:
        A ``vit_models`` instance with flash attention disabled and
        hooks installed for entropy and Hessian experiments.
    """
    model = vit_models(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        num_registers=num_registers,
        norm_layer=partial(RMSNorm, eps=1e-6),
        block_layers=Block,
        rope=True,
        reg_theta=100,
        qk_norm=True,
        drop_path_rate=drop_path_rate,
        num_classes=num_classes,
        layer_scale=True,
        init_scale=1e-4,
    )

    # Re-apply custom weight initialisation (overrides in-class default).
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=init_std)

    model.apply(_init_weights)

    if use_scaled_init:
        n_layers = len(model.blocks)
        scale = init_std / math.sqrt(2 * n_layers)
        for name, param in model.named_parameters():
            if name.endswith("attn.proj.weight"):
                nn.init.normal_(param, mean=0.0, std=scale)

    model.to(device)
    return model


def set_attention_temperature(
    model: torch.nn.Module, temperature: float
) -> None:
    """
    Set the attention temperature on every transformer block.

    A temperature > 1 softens the attention distribution (higher entropy);
    < 1 sharpens it.  The change is in-place and persistent.

    Args:
        model:       A ``vit_models`` instance or its DDP wrapper.
        temperature: Positive float.  1.0 restores default behaviour.
    """
    raw = model.module if hasattr(model, "module") else model
    for block in raw.blocks:
        block.attn.temperature = temperature
