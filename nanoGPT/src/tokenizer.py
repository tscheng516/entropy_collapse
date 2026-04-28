"""
Tokenizer utilities for the NanoGPT entropy-collapse experiments.

Provides a unified ``get_tokenizer(dataset)`` interface that returns an
encode / decode function pair and the vocabulary size for each supported
dataset.

Supported datasets
------------------
* ``shakespeare_char``  — Character-level (65 unique chars, matches the
                          canonical NanoGPT ``data/shakespeare_char/``
                          vocabulary).
* ``fineweb_edu``       — GPT-2 BPE via tiktoken (vocab_size = 50 257).
* ``climbmix``          — GPT-2 BPE via tiktoken (vocab_size = 50 257).

Usage::

    from src.tokenizer import get_tokenizer

    tok = get_tokenizer("shakespeare_char")
    ids  = tok["encode"]("Hello world!")  # list[int]
    text = tok["decode"](ids)             # str
    vs   = tok["vocab_size"]              # 65

Notes
-----
* The Shakespeare character vocabulary is built at call time from the
  canonical 65-character set; no file I/O is required.
* For GPT-2 BPE datasets the ``tiktoken`` package must be installed::

      pip install tiktoken
"""

from __future__ import annotations

from typing import Callable


# ======================================================================
# Shakespeare — character-level
# ======================================================================

# The 65-character set produced by NanoGPT's prepare.py for the
# Tiny-Shakespeare corpus (sorted unique characters in the text).
_SHAKESPEARE_CHARS: str = (
    "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
)


def _make_shakespeare_tokenizer() -> dict:
    chars = sorted(set(_SHAKESPEARE_CHARS))
    vocab_size = len(chars)
    stoi: dict[str, int] = {ch: i for i, ch in enumerate(chars)}
    itos: dict[int, str] = {i: ch for i, ch in enumerate(chars)}

    def encode(s: str) -> list[int]:
        return [stoi[c] for c in s if c in stoi]

    def decode(ids) -> str:
        return "".join(itos.get(int(i), "") for i in ids)

    return {
        "encode": encode,
        "decode": decode,
        "vocab_size": vocab_size,
        "name": "shakespeare_char",
    }


# ======================================================================
# GPT-2 BPE — used for fineweb_edu and climbmix
# ======================================================================


def _make_gpt2_tokenizer(name: str) -> dict:
    try:
        import tiktoken
    except ImportError as exc:
        raise ImportError(
            "tiktoken is required for GPT-2 BPE tokenization.\n"
            "Install with:  pip install tiktoken"
        ) from exc

    enc = tiktoken.get_encoding("gpt2")

    def encode(s: str) -> list[int]:
        return enc.encode_ordinary(s)

    def decode(ids) -> str:
        return enc.decode(list(ids))

    return {
        "encode": encode,
        "decode": decode,
        "vocab_size": 50_257,
        "name": name,
        # Expose the raw encoder for scripts that need eot_token etc.
        "_enc": enc,
    }


# ======================================================================
# Public API
# ======================================================================


def get_tokenizer(dataset: str) -> dict:
    """
    Return a tokenizer dict for the given dataset.

    Args:
        dataset: One of ``"shakespeare_char"`` (also ``"shakespeare"``),
                 ``"fineweb_edu"`` / ``"fineweb-edu"`` / ``"fineweb"``,
                 or ``"climbmix"`` / ``"nemotron_climbmix"``.

    Returns:
        Dict with keys:
          * ``encode``      — ``Callable[[str], list[int]]``
          * ``decode``      — ``Callable[[list[int]], str]``
          * ``vocab_size``  — ``int``
          * ``name``        — ``str``  (canonical dataset identifier)

    Raises:
        ValueError: for unknown dataset names.
        ImportError: if tiktoken is not installed and a BPE tokenizer is
                     requested.
    """
    ds = dataset.lower().strip()
    if ds in ("shakespeare_char", "shakespeare"):
        return _make_shakespeare_tokenizer()
    if ds in ("fineweb_edu", "fineweb-edu", "fineweb"):
        return _make_gpt2_tokenizer("fineweb_edu")
    if ds in ("climbmix", "nemotron_climbmix", "nemotron-climbmix"):
        return _make_gpt2_tokenizer("climbmix")
    raise ValueError(
        f"Unknown dataset '{dataset}'. "
        "Supported: 'shakespeare_char', 'fineweb_edu', 'climbmix'."
    )
