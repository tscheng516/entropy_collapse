"""
HookedViT — a thin wrapper around a timm ViT model.

Additions over the base timm ViT:
  * Custom small-std weight initialisation for stability experiments.
  * Per-layer attention-matrix caching (``block.attn.last_att``) so that
    entropy can be computed after every forward pass without re-running
    inference.
  * Flash / fused attention is **disabled** on every block; the explicit
    softmax path is required for (a) caching attention matrices and (b)
    second-order gradient computations via ``torch.autograd``.
  * Optional QK normalisation (``qk_norm``).  The default is ``False``,
    which matches the NanoGPT / LLM experiment setup where no QK-norm is
    applied.  Set ``qk_norm=True`` to install per-head LayerNorm on the
    query and key projections (timm's native ``qk_norm`` feature).
  * Runtime attention temperature support: set ``attn.temperature`` on
    any attention block (or call ``set_attention_temperature``) to scale
    the attention logits, producing sharper (T < 1) or softer (T > 1)
    attention distributions for entropy-collapse interventions.

Usage::

    from src.model import build_hooked_vit

    # NanoGPT-parity — no QK norm (default)
    model = build_hooked_vit(
        model_name="vit_small_patch16_224",
        num_classes=10,
        init_std=0.002,
        use_scaled_init=True,
        qk_norm=False,
    )

    # With QK norm
    model = build_hooked_vit(
        model_name="vit_small_patch16_224",
        num_classes=10,
        qk_norm=True,
    )
"""

from __future__ import annotations

import math
import types
from typing import Optional

import torch
import torch.nn.functional as F


def _patched_attn_forward(
    self,
    x: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Drop-in replacement for timm's ``Attention.forward`` that:
      1. Computes the explicit (non-fused) softmax-attention.
      2. Caches the attention weight tensor as ``self.last_att`` for
         downstream entropy measurements.
      3. Applies an optional ``self.temperature`` scale factor to the
         attention logits (default 1.0 = no change).  Values > 1 soften
         the distribution; values < 1 sharpen it.

    The signature matches timm ≥ 0.9 (which passes ``attn_mask`` and
    ``is_causal`` from the enclosing Block) and is backward-compatible
    with older timm releases.

    For standard ViT image classification both kwargs are always their
    defaults (``None`` / ``False``), so the mask path is a no-op.
    """
    B, N, C = x.shape

    # head_dim is an explicit attribute in timm ≥ 0.9; fall back for older.
    head_dim: int = getattr(self, "head_dim", C // self.num_heads)
    # attn_dim may differ from C when a wider projection is used.
    attn_dim: int = getattr(self, "attn_dim", C)

    qkv = (
        self.qkv(x)
        .reshape(B, N, 3, self.num_heads, head_dim)
        .permute(2, 0, 3, 1, 4)
    )
    q, k, v = qkv.unbind(0)  # each: (B, num_heads, N, head_dim)

    # Per-head query / key normalisation (identity by default in standard ViT).
    q = self.q_norm(q)
    k = self.k_norm(k)

    # Explicit softmax attention — bypass fused_attn / SDPA entirely.
    temperature = getattr(self, "temperature", 1.0)
    attn = (q * (self.scale / temperature)) @ k.transpose(-2, -1)  # (B, heads, N, N)

    # Handle optional attention mask (mirrors timm's non-fused path).
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

    # Cache for entropy computation — detach so it doesn't pollute grads.
    self.last_att = attn.detach()

    attn = self.attn_drop(attn)
    x = (attn @ v).transpose(1, 2).reshape(B, N, attn_dim)
    _norm = getattr(self, "norm", None)                                                                                                                                          
    if _norm is not None:                                                                                                                                                        
        x = _norm(x)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def build_hooked_vit(
    model_name: str = "vit_small_patch16_224",
    num_classes: int = 10,
    pretrained: bool = False,
    img_size: int = 224,
    init_std: float = 0.002,
    use_scaled_init: bool = True,
    qk_norm: bool = False,
    depth: Optional[int] = None,
    num_heads: Optional[int] = None,
    embed_dim: Optional[int] = None,
    patch_size: Optional[int] = None,
    device: str = "cuda",
) -> torch.nn.Module:
    """
    Build and return a timm ViT model patched for entropy-collapse experiments.

    The returned model:
      * Has flash / fused attention disabled on every block.
      * Caches the attention weight matrix as ``block.attn.last_att`` after
        each forward pass (used by ``get_attention_entropy``).
      * Optionally re-initialises weights with a small std (``init_std``).
      * Applies QK normalisation only when ``qk_norm=True``.

    Args:
        model_name:       timm model string (default: ``'vit_small_patch16_224'``).
        num_classes:      Number of output classes.
        pretrained:       Load pretrained timm weights.  Set ``False`` for
                          scratch training (entropy dynamics are most vivid).
        img_size:         Input spatial resolution passed to timm.
        init_std:         Std for all ``nn.Linear`` / ``nn.Embedding`` weights
                          when ``pretrained=False``.
        use_scaled_init:  Scale each attention output-projection (``attn.proj``)
                          by ``init_std / sqrt(2 * depth)`` — ViT analogue of
                          NanoGPT residual-depth scaling.
        qk_norm:          If ``False`` (default), ``q_norm`` and ``k_norm`` are
                          ``Identity`` — matching the NanoGPT / LLM experiment
                          setup.  If ``True``, timm installs per-head LayerNorm
                          on the query and key projections.
        depth:            Override the number of transformer layers.  ``None``
                          uses the timm model default.  Set to 6 to match
                          NanoGPT-small (``n_layer=6``).
        num_heads:        Override the number of attention heads.  ``None``
                          uses the timm model default.  Set to 6 to match
                          NanoGPT-small (``n_head=6``).
        embed_dim:        Override the embedding dimension.  ``None`` uses the
                          timm model default.  Set to 384 to match NanoGPT-small
                          (``n_embd=384``).
        patch_size:       Override the patch size.  ``None`` uses the timm model
                          default.  Set to 4 with ``img_size=32`` for CIFAR
                          (64 patches = NanoGPT-small ``block_size=64``).
        device:           Target device string.

    Returns:
        A timm ViT model ready for entropy-collapse experiments.
    """
    import timm  # imported here so the module is importable without timm installed

    # Collect architecture overrides; only pass non-None values so the timm
    # model's built-in defaults are preserved when not explicitly overridden.
    arch_kwargs: dict = {}
    if depth is not None:
        arch_kwargs["depth"] = depth
    if num_heads is not None:
        arch_kwargs["num_heads"] = num_heads
    if embed_dim is not None:
        arch_kwargs["embed_dim"] = embed_dim
    if patch_size is not None:
        arch_kwargs["patch_size"] = patch_size

    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        img_size=img_size,
        qk_norm=qk_norm,
        **arch_kwargs,
    )

    # ------------------------------------------------------------------ #
    # Re-initialise weights with small std (scratch experiments only)
    # ------------------------------------------------------------------ #
    if not pretrained:
        def _init_weights(module: torch.nn.Module) -> None:
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=init_std)

        model.apply(_init_weights)

        if use_scaled_init:
            n_layers = len(model.blocks)
            scale = init_std / math.sqrt(2 * n_layers)
            for name, param in model.named_parameters():
                # attn.proj is the attention output projection (analogous to c_proj)
                if name.endswith("attn.proj.weight"):
                    torch.nn.init.normal_(param, mean=0.0, std=scale)

    # ------------------------------------------------------------------ #
    # Patch every attention block to cache attention matrices
    # ------------------------------------------------------------------ #
    for block in model.blocks:
        attn = block.attn
        # Disable fused / SDPA attention (timm ≥ 0.9 flag)
        if hasattr(attn, "fused_attn"):
            attn.fused_attn = False
        attn.forward = types.MethodType(_patched_attn_forward, attn)

    model.to(device)
    return model


def set_attention_temperature(model: torch.nn.Module, temperature: float) -> None:
    """
    Set the attention temperature on every transformer block of a
    (possibly DDP-wrapped) timm ViT model.

    A temperature > 1 softens the attention distribution (higher entropy);
    a temperature < 1 sharpens it (lower entropy).  The change is applied
    in-place and persists for all subsequent forward passes.

    Args:
        model:       A timm ViT instance, or a ``DistributedDataParallel``
                     wrapper around one.
        temperature: Positive float.  1.0 restores the default behaviour.
    """
    raw = model.module if hasattr(model, "module") else model
    for block in raw.blocks:
        block.attn.temperature = temperature
