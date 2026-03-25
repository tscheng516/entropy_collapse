"""
Data utilities for the NanoGPT entropy-collapse experiments.

Reuses the NanoGPT binary data format (uint16 token ids written by
``data/<dataset>/prepare.py``).  A lightweight wrapper around
``numpy.memmap`` is provided so the data file never has to fit in RAM.
"""

import os
import numpy as np
import torch


def load_data(data_dir: str):
    """
    Load the train and (optionally) val token arrays from *data_dir*.

    The files are memory-mapped so that only the requested slices are
    actually read from disk.

    Args:
        data_dir: Path to the directory produced by a NanoGPT
                  ``prepare.py`` script (must contain ``train.bin``).

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
            "Run the corresponding NanoGPT prepare.py first."
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
        data:       uint16 numpy array of token ids (e.g. from load_data).
        batch_size: Number of sequences in the batch.
        block_size: Length of each sequence (context window).
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
