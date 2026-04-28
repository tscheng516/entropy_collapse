"""
HookedGPT — a thin wrapper around nanochat's GPT model.

Additions over the base nanochat GPT:
  * Per-layer attention-matrix caching (``block.attn.last_att``) so that
    per-layer Shannon entropy can be computed after every forward pass.
  * Flash / FA3 attention is replaced with an explicit SDPA softmax
    path, which is required for (a) attention caching and (b) second-order
    gradient computations via ``torch.autograd``.
  * Runtime attention temperature support via ``attn.temperature`` — scale
    the post-QK-norm dot products to sharpen (T < 1) or soften (T > 1) the
    attention distributions.

Differences from nanochat's production CausalSelfAttention.forward:
  * Sliding window (``window_size``) is **ignored** — the patched forward
    always uses full causal context.  Sliding windows require FA3's native
    API and are incompatible with autograd's create_graph=True path.
  * The FA3 / SDPA call is replaced by an explicit (q @ k.T).softmax
    computation, keeping the graph fully differentiable.
  * GQA (``n_kv_head < n_head``) is handled by ``repeat_interleave`` to
    expand key/value heads before the matmul.

Usage::

    from src.model import build_hooked_gpt

    model = build_hooked_gpt(gpt_cfg, device="cuda")
    # model.transformer.h[i].attn.last_att  ← (B, n_head, T, T) after fwd
"""

from __future__ import annotations

import math
import types
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Patched CausalSelfAttention forward
# ---------------------------------------------------------------------------

def _patched_attn_forward(
    self,
    x: torch.Tensor,
    ve: torch.Tensor | None,
    cos_sin: tuple[torch.Tensor, torch.Tensor],
    window_size: tuple[int, int],   # accepted for signature compat, ignored
    kv_cache,                       # None during training
) -> torch.Tensor:
    """
    Drop-in replacement for nanochat's CausalSelfAttention.forward that:

    1. Computes explicit softmax attention instead of calling FA3 /
       ``flash_attn``.  This enables second-order autograd (HVPs) and
       attention caching.
    2. Handles GQA (``n_kv_head < n_head``) via ``repeat_interleave``.
    3. Caches the attention weight tensor as ``self.last_att`` whenever
       ``self._cache_attn is True``.
    4. Applies an optional ``self.temperature`` scale factor (default 1.0).

    The ``window_size`` argument is accepted for API compatibility but
    ignored — the patched forward always uses full causal context.

    Only the training path (``kv_cache is None``) is supported.  Inference
    with kv-cache should use the original nanochat forward.
    """
    assert kv_cache is None, (
        "HookedGPT patched forward does not support kv_cache inference. "
        "Use the original nanochat GPT for generation."
    )

    B, T, C = x.size()

    # --- Q, K, V projections ---
    q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
    k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
    v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

    # --- Value residual (ResFormer) — keep identical to nanochat ---
    if ve is not None:
        ve = ve.view(B, T, self.n_kv_head, self.head_dim)
        gate = 3 * torch.sigmoid(
            self.ve_gate(x[..., : self.ve_gate_channels])
        )  # (B, T, n_kv_head)
        v = v + gate.unsqueeze(-1) * ve

    # --- Rotary positional embeddings (RoPE) ---
    cos, sin = cos_sin  # each: (1, T, 1, head_dim//2)
    d = self.head_dim // 2
    q1, q2 = q[..., :d], q[..., d:]
    q = torch.cat([q1 * cos + q2 * sin, q1 * (-sin) + q2 * cos], dim=-1)
    k1, k2 = k[..., :d], k[..., d:]
    k = torch.cat([k1 * cos + k2 * sin, k1 * (-sin) + k2 * cos], dim=-1)

    # --- QK norm (functional RMSNorm, matching nanochat's norm()) ---
    q = F.rms_norm(q, (q.size(-1),))
    k = F.rms_norm(k, (k.size(-1),))

    # --- Nanochat's sharpening scale (q *= 1.2, k *= 1.2) ---
    q = q * 1.2
    k = k * 1.2

    # --- Transpose to (B, heads, T, head_dim) for matmul ---
    q = q.transpose(1, 2)  # (B, n_head, T, head_dim)
    k = k.transpose(1, 2)  # (B, n_kv_head, T, head_dim)
    v = v.transpose(1, 2)  # (B, n_kv_head, T, head_dim)

    # --- GQA expansion: tile k/v to match n_head ---
    if self.n_kv_head != self.n_head:
        repeat = self.n_head // self.n_kv_head
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

    # --- Explicit scaled dot-product attention ---
    temperature = getattr(self, "temperature", 1.0)
    # scale matches flash_attn's default (1/sqrt(head_dim)), divided by T.
    scale = self.head_dim ** -0.5 / temperature
    attn = (q * scale) @ k.transpose(-2, -1)  # (B, n_head, T, T)

    # Causal mask (lower-triangular)
    causal_bias = attn.new_full((T, T), float("-inf")).triu_(1)
    attn = attn + causal_bias
    attn = attn.softmax(dim=-1)

    # Cache for entropy measurement — only when explicitly requested.
    if getattr(self, "_cache_attn", False):
        self.last_att = attn.detach()

    y = (attn @ v)  # (B, n_head, T, head_dim)
    y = y.transpose(1, 2).contiguous().view(B, T, -1)
    y = self.c_proj(y)
    return y


# ---------------------------------------------------------------------------
# Build & patch
# ---------------------------------------------------------------------------

def build_hooked_gpt(
    gpt_cfg,          # nanochat.gpt.GPTConfig instance
    device: str = "cuda",
) -> torch.nn.Module:
    """
    Build a nanochat GPT and patch every CausalSelfAttention block with the
    explicit-softmax, attention-caching forward.

    The model is returned in ``train()`` mode on ``device``.

    Args:
        gpt_cfg:  A nanochat ``GPTConfig`` instance.
        device:   Target device string.

    Returns:
        A nanochat GPT model ready for entropy-collapse experiments.
    """
    # Import here so the module is importable without nanochat installed.
    from nanochat.gpt import GPT  # noqa: PLC0415

    model = GPT(gpt_cfg)
    model.to(device)
    model.init_weights()  # fills parameters with nanochat's intended init

    # Patch every attention block to use the explicit-softmax forward.
    for block in model.transformer.h:
        block.attn.forward = types.MethodType(_patched_attn_forward, block.attn)

    model.train()
    return model


def set_attention_temperature(model: torch.nn.Module, temperature: float) -> None:
    """
    Set the attention temperature on every transformer block of a
    (possibly DDP-wrapped) nanochat GPT model.

    A temperature > 1 softens the attention distribution (higher entropy);
    a temperature < 1 sharpens it (lower entropy).

    Args:
        model:       A nanochat GPT or a ``DistributedDataParallel`` wrapper.
        temperature: Positive float.  1.0 restores the default behaviour.
    """
    raw = model.module if hasattr(model, "module") else model
    for block in raw.transformer.h:
        block.attn.temperature = temperature
