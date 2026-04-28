"""
Data utilities for the NanoGPT entropy-collapse experiments.

Reuses the NanoGPT binary data format (uint16 token ids).  A lightweight
wrapper around ``numpy.memmap`` is provided so the data file never has to
fit in RAM.

Dataset support
---------------
* ``shakespeare_char``  — Char-level Shakespeare (~1 MB).  Prepared by
                          ``prepare_shakespeare``; no external dependencies.
* ``fineweb_edu``       — HuggingFace ``HuggingFaceFW/fineweb-edu``
                          (GPT-2 BPE via tiktoken, ``sample-10BT`` split).
* ``climbmix``          — HuggingFace ``nvidia/ClimbMix``
                          (GPT-2 BPE via tiktoken).

All datasets are prepared once and stored as ``train.bin`` / ``val.bin``
(uint16, same format as NanoGPT).  Subsequent runs skip preparation if
the files already exist.

Quick-start
-----------
Call ``ensure_data(dataset, data_dir)`` once (at the top of your training
script) and it will automatically download and tokenise the dataset if
needed::

    from src.data_utils import ensure_data, load_data, get_batch

    ensure_data("shakespeare_char", "data/shakespeare_char")
    train_data, val_data = load_data("data/shakespeare_char")
"""

from __future__ import annotations

import os
import pickle
import numpy as np
import torch


# ===========================================================================
# Low-level I/O helpers
# ===========================================================================


def load_data(data_dir: str):
    """
    Load the train and (optionally) val token arrays from *data_dir*.

    The files are memory-mapped so that only the requested slices are
    actually read from disk.

    Args:
        data_dir: Path to the directory containing ``train.bin``.

    Returns:
        train_data (np.memmap): uint16 token array for training.
        val_data   (np.memmap | None): uint16 token array for validation,
                   or ``None`` if ``val.bin`` is not present.
    """
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"train.bin not found in '{data_dir}'. "
            "Call ensure_data() first or run base_train.py with auto-prepare."
        )

    train_data = np.memmap(train_path, dtype=np.uint16, mode="r")
    val_data = (
        np.memmap(val_path, dtype=np.uint16, mode="r")
        if os.path.exists(val_path)
        else None
    )
    return train_data, val_data


def get_batch(
    data: np.ndarray,
    batch_size: int,
    block_size: int,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a random batch of (input, target) token sequences.

    Args:
        data:       uint16 numpy array of token ids (from ``load_data``).
        batch_size: Number of sequences per batch.
        block_size: Context window length (tokens per sequence).
        device:     Target torch device.

    Returns:
        x: (batch_size, block_size) int64 tensor — input tokens.
        y: (batch_size, block_size) int64 tensor — next-token targets.
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix]
    )
    return x.to(device), y.to(device)


# ===========================================================================
# Dataset preparation helpers
# ===========================================================================


def prepare_shakespeare(data_dir: str) -> None:
    """
    Download and tokenise the Tiny Shakespeare dataset (character-level).

    The character vocabulary is the sorted unique characters that appear in
    the corpus (matches NanoGPT's ``data/shakespeare_char/prepare.py``).
    Writes ``train.bin``, ``val.bin``, and ``meta.pkl`` to *data_dir*.

    Preparation is skipped silently if both bin files already exist.

    Args:
        data_dir: Destination directory for the prepared files.
    """
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")
    if os.path.exists(train_path) and os.path.exists(val_path):
        return

    os.makedirs(data_dir, exist_ok=True)

    # Download Tiny Shakespeare from the char-rnn repository.
    import urllib.request

    url = (
        "https://raw.githubusercontent.com/karpathy/char-rnn/"
        "master/data/tinyshakespeare/input.txt"
    )
    txt_path = os.path.join(data_dir, "input.txt")
    if not os.path.exists(txt_path):
        print(f"[data] downloading Shakespeare from {url} …")
        urllib.request.urlretrieve(url, txt_path)

    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"[data] Shakespeare: {len(text):,} characters")

    chars = sorted(set(text))
    vocab_size = len(chars)
    print(f"[data] vocab_size = {vocab_size}")

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    ids = np.array([stoi[c] for c in text], dtype=np.uint16)

    n_train = int(len(ids) * 0.9)
    ids[:n_train].tofile(train_path)
    ids[n_train:].tofile(val_path)
    print(
        f"[data] train: {n_train:,} tokens, "
        f"val: {len(ids) - n_train:,} tokens → {data_dir}"
    )

    meta = {"vocab_size": vocab_size, "itos": itos, "stoi": stoi}
    with open(os.path.join(data_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)


def prepare_fineweb_edu(
    data_dir: str,
    num_proc: int = 4,
    max_tokens: int = int(1e9),
    val_fraction: float = 0.005,
) -> None:
    """
    Download and tokenise a slice of the FineWeb-Edu dataset (GPT-2 BPE).

    Streams from ``HuggingFaceFW/fineweb-edu`` (``sample-10BT`` split)
    until *max_tokens* tokens have been collected, then writes ``train.bin``
    and ``val.bin`` to *data_dir*.

    Preparation is skipped silently if both bin files already exist.

    Args:
        data_dir:      Destination directory.
        num_proc:      Unused (streaming mode); kept for API symmetry.
        max_tokens:    Maximum token count to collect (default 1 B).
        val_fraction:  Fraction of documents used for validation (0.5%).
    """
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")
    if os.path.exists(train_path) and os.path.exists(val_path):
        return

    os.makedirs(data_dir, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets is required for FineWeb-Edu preparation.\n"
            "Install with:  pip install datasets"
        ) from exc

    try:
        import tiktoken
    except ImportError as exc:
        raise ImportError(
            "tiktoken is required for GPT-2 BPE tokenization.\n"
            "Install with:  pip install tiktoken"
        ) from exc

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    print("[data] streaming HuggingFaceFW/fineweb-edu (sample-10BT) …")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
        trust_remote_code=False,
    )

    train_ids: list[np.ndarray] = []
    val_ids: list[np.ndarray] = []
    total = 0
    val_every = max(1, int(1.0 / val_fraction))

    for doc_count, doc in enumerate(ds):
        text = doc.get("text", "")
        if not text:
            continue
        toks = enc.encode_ordinary(text)
        toks.append(eot)
        arr = np.array(toks, dtype=np.uint16)
        if doc_count % val_every == 0:
            val_ids.append(arr)
        else:
            train_ids.append(arr)
        total += len(arr)
        if total >= max_tokens:
            break

    train_arr = np.concatenate(train_ids) if train_ids else np.array([], dtype=np.uint16)
    val_arr = np.concatenate(val_ids) if val_ids else np.array([], dtype=np.uint16)
    train_arr.tofile(train_path)
    val_arr.tofile(val_path)
    print(
        f"[data] fineweb-edu: train {len(train_arr):,} tokens, "
        f"val {len(val_arr):,} tokens → {data_dir}"
    )


def prepare_climbmix(
    data_dir: str,
    num_proc: int = 4,
    max_tokens: int = int(1e9),
    val_fraction: float = 0.005,
) -> None:
    """
    Download and tokenise a slice of the Nemotron ClimbMix dataset (GPT-2 BPE).

    Streams from ``nvidia/ClimbMix`` until *max_tokens* tokens have been
    collected, then writes ``train.bin`` and ``val.bin`` to *data_dir*.

    Preparation is skipped silently if both bin files already exist.

    Args:
        data_dir:      Destination directory.
        num_proc:      Unused (streaming mode); kept for API symmetry.
        max_tokens:    Maximum token count to collect (default 1 B).
        val_fraction:  Fraction of documents used for validation (0.5%).
    """
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")
    if os.path.exists(train_path) and os.path.exists(val_path):
        return

    os.makedirs(data_dir, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets is required for ClimbMix preparation.\n"
            "Install with:  pip install datasets"
        ) from exc

    try:
        import tiktoken
    except ImportError as exc:
        raise ImportError(
            "tiktoken is required for GPT-2 BPE tokenization.\n"
            "Install with:  pip install tiktoken"
        ) from exc

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    print("[data] streaming nvidia/ClimbMix …")
    ds = load_dataset(
        "nvidia/ClimbMix",
        split="train",
        streaming=True,
        trust_remote_code=False,
    )

    train_ids: list[np.ndarray] = []
    val_ids: list[np.ndarray] = []
    total = 0
    val_every = max(1, int(1.0 / val_fraction))

    for doc_count, doc in enumerate(ds):
        # ClimbMix uses either 'text' or 'content' field
        text = doc.get("text", doc.get("content", ""))
        if not text:
            continue
        toks = enc.encode_ordinary(text)
        toks.append(eot)
        arr = np.array(toks, dtype=np.uint16)
        if doc_count % val_every == 0:
            val_ids.append(arr)
        else:
            train_ids.append(arr)
        total += len(arr)
        if total >= max_tokens:
            break

    train_arr = np.concatenate(train_ids) if train_ids else np.array([], dtype=np.uint16)
    val_arr = np.concatenate(val_ids) if val_ids else np.array([], dtype=np.uint16)
    train_arr.tofile(train_path)
    val_arr.tofile(val_path)
    print(
        f"[data] climbmix: train {len(train_arr):,} tokens, "
        f"val {len(val_arr):,} tokens → {data_dir}"
    )


# ===========================================================================
# Convenience: auto-prepare
# ===========================================================================


def ensure_data(dataset: str, data_dir: str, num_proc: int = 4) -> None:
    """
    Ensure ``train.bin`` and ``val.bin`` exist in *data_dir* for *dataset*.

    Selects and calls the appropriate ``prepare_*`` function.  If both
    files already exist, returns immediately without downloading anything.

    Args:
        dataset:  ``"shakespeare_char"``, ``"fineweb_edu"``, or ``"climbmix"``.
        data_dir: Directory where the binary token files should reside.
        num_proc: Parallel-worker hint passed to the prepare function.

    Raises:
        ValueError: for unknown dataset names.
    """
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")
    if os.path.exists(train_path) and os.path.exists(val_path):
        return  # already prepared

    ds = dataset.lower().strip()
    if ds in ("shakespeare_char", "shakespeare"):
        prepare_shakespeare(data_dir)
    elif ds in ("fineweb_edu", "fineweb-edu", "fineweb"):
        prepare_fineweb_edu(data_dir, num_proc=num_proc)
    elif ds in ("climbmix", "nemotron_climbmix", "nemotron-climbmix"):
        prepare_climbmix(data_dir, num_proc=num_proc)
    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            "Supported: 'shakespeare_char', 'fineweb_edu', 'climbmix'."
        )
