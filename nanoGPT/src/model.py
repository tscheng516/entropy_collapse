"""
HookedGPT — a thin wrapper around Karpathy's NanoGPT ``GPT`` class.

Additions over the base ``GPT``:
  * Custom small-std weight initialisation for stability experiments.
  * Per-layer attention-matrix caching (``block.attn.last_att``) so that
    entropy can be computed after every forward pass without re-running
    inference.
  * Flash-attention is **disabled** by default; the explicit softmax path
    is required for (a) caching attention matrices and (b) second-order
    gradient computations via ``torch.autograd``.
  * Runtime attention temperature support: set ``attn.temperature`` on
    any attention block (or call ``set_attention_temperature``) to scale
    the attention logits, producing sharper (T < 1) or softer (T > 1)
    attention distributions for entropy-collapse interventions.

The NanoGPT repository must be on ``sys.path`` before importing this
module (see ``base_train.py`` for how the path is appended at runtime).
"""

import math
import types

import torch
import torch.nn.functional as F


def _patched_attn_forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Drop-in replacement for ``CausalSelfAttention.forward`` that:
      1. Computes the explicit (non-flash) causal softmax-attention.
      2. Caches the attention weight tensor as ``self.last_att`` for
         downstream entropy measurements.
      3. Applies an optional ``self.temperature`` scale factor to the
         attention logits (default 1.0 = no change).  Values > 1 soften
         the distribution; values < 1 sharpen it.

    The signature and output are identical to the original, so the rest
    of the model is unaffected.
    """
    B, T, C = x.size()
    q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
    head_size = C // self.n_head
    k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
    q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
    v = v.view(B, T, self.n_head, head_size).transpose(1, 2)

    temperature = getattr(self, "temperature", 1.0)
    att = (q @ k.transpose(-2, -1)) * (1.0 / (math.sqrt(head_size) * temperature))
    att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
    att = F.softmax(att, dim=-1)

    # Cache for entropy computation — detach so it doesn't pollute grads
    self.last_att = att.detach()

    att = self.attn_dropout(att)
    y = att @ v
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    y = self.resid_dropout(self.c_proj(y))
    return y


class HookedGPT:
    """
    Factory function that returns a ``GPT`` instance patched for the
    entropy-collapse experiments.

    We deliberately *not* subclass ``GPT`` here — instead we build the
    model normally and then:
      * Override the attention forward method on every block.
      * Re-initialise weights with the requested ``init_std``.

    This keeps the NanoGPT code completely unmodified and allows the
    caller to simply do::

        from src.model import build_hooked_gpt
        model = build_hooked_gpt(config, init_std=0.002, scaled_init=True)
    """


def build_hooked_gpt(
    config,
    init_std: float = 0.002,
    use_scaled_init: bool = True,
    device: str = "cuda",
) -> "GPT":  # noqa: F821 — GPT imported at call-time via sys.path
    """
    Build and return a NanoGPT ``GPT`` model with:
      * Flash-attention disabled.
      * Explicit causal-softmax forward patched onto every attention block.
      * Custom weight initialisation.

    Args:
        config:           ``GPTConfig`` instance.
        init_std:         Std for all nn.Linear / nn.Embedding weights.
        use_scaled_init:  Apply NanoGPT residual-depth scaling to c_proj.
        device:           Target device string.

    Returns:
        A ``GPT`` model ready for entropy-collapse experiments.
    """
    # Import at call-time so callers can add nanoGPT to sys.path first
    from model import GPT  # type: ignore[import]

    config.flash = False  # disable flash attention globally
    model = GPT(config)

    # ------------------------------------------------------------------ #
    # Re-initialise weights with small std
    # ------------------------------------------------------------------ #
    def _init_weights(module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=init_std)

    model.apply(_init_weights)

    if use_scaled_init:
        scale = init_std / math.sqrt(2 * config.n_layer)
        for name, param in model.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(param, mean=0.0, std=scale)

    # ------------------------------------------------------------------ #
    # Patch every attention block to cache attention matrices
    # ------------------------------------------------------------------ #
    for block in model.transformer.h:
        block.attn.flash = False
        causal_mask = torch.tril(
            torch.ones(config.block_size, config.block_size)
        ).view(1, 1, config.block_size, config.block_size)
        block.attn.register_buffer("bias", causal_mask)
        block.attn.forward = types.MethodType(_patched_attn_forward, block.attn)

    model.to(device)
    return model


def set_attention_temperature(model: "GPT", temperature: float) -> None:  # noqa: F821
    """
    Set the attention temperature on every transformer block of a
    (possibly DDP-wrapped) ``GPT`` model.

    A temperature > 1 softens the attention distribution (higher entropy);
    a temperature < 1 sharpens it (lower entropy).  The change is applied
    in-place and persists for all subsequent forward passes.

    Args:
        model:       A ``GPT`` instance, or a ``DistributedDataParallel``
                     wrapper around one.
        temperature: Positive float.  1.0 restores the default behaviour.
    """
    raw = model.module if hasattr(model, "module") else model
    for block in raw.transformer.h:
        block.attn.temperature = temperature
