"""
Data utilities for the depth/ entropy-collapse experiments.

Dataset backend: NYU Depth V2 (Silberman et al., 2012).

  * Auto-downloaded from Hugging Face ``Intel/nyu_depth_v2`` on first run;
    subsequent runs reuse the local cache.
  * Every sample is a paired (RGB image, depth map) where depth values are
    in metres (approx. 0 – 10 m).  Invalid / sky pixels are marked 0.

Returned loaders yield ``(images, depths)`` pairs:
  * ``images``: ``(B, 3, img_size, img_size)`` float32 tensors, normalised
    with ImageNet statistics.
  * ``depths``: ``(B, 1, img_size, img_size)`` float32 tensors in metres.
    Zero pixels remain zero (invalid mask).

Paired spatial augmentations (train split only):
  * Random horizontal flip (same for image and depth).
  * Resize to ``img_size × img_size``.
  * Image-only colour jitter.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ImageNet statistics — same as ViT/ for consistency.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# NYU Depth V2 Hugging Face repository identifier.
_HF_REPO = "Intel/nyu_depth_v2"


class _NYUDepthDataset(Dataset):
    """
    Wrapper around a Hugging Face NYU Depth V2 split.

    Each item is a ``(image_tensor, depth_tensor)`` pair where:
      * ``image_tensor``: ``(3, H, W)`` normalised RGB float32.
      * ``depth_tensor``: ``(1, H, W)`` depth in metres (float32).
        Zero pixels are invalid and should be excluded from loss computation.

    Args:
        hf_ds:     A Hugging Face ``Dataset`` object (train or test split).
        img_size:  Target spatial resolution (square).
        augment:   If ``True``, apply random horizontal flip + colour jitter
                   (for the training split).
    """

    def __init__(self, hf_ds, img_size: int, augment: bool = False) -> None:
        self.hf_ds = hf_ds
        self.img_size = img_size
        self.augment = augment

        self._to_tensor = T.ToTensor()
        self._normalise = T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)
        self._color_jitter = T.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        )

    def __len__(self) -> int:
        return len(self.hf_ds)

    def __getitem__(self, idx: int):
        ex = self.hf_ds[idx]

        # --- Image ---
        img: Image.Image = ex["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        img = img.convert("RGB")

        # --- Depth map ---
        depth_raw = ex["depth_map"]
        if isinstance(depth_raw, Image.Image):
            depth_np = np.array(depth_raw, dtype=np.float32)
        else:
            depth_np = np.array(depth_raw, dtype=np.float32)
        # Depth may be 2-D (H, W) or 3-D (H, W, 1) — squeeze to 2-D.
        if depth_np.ndim == 3:
            depth_np = depth_np.squeeze(-1)
        depth_pil = Image.fromarray(depth_np, mode="F")  # PIL float32 image

        # --- Paired spatial transforms ---
        if self.augment:
            # Random horizontal flip (same seed for both).
            if random.random() > 0.5:
                img = TF.hflip(img)
                depth_pil = TF.hflip(depth_pil)

        # Resize to square target resolution.
        img = TF.resize(img, [self.img_size, self.img_size], antialias=True)
        depth_pil = TF.resize(
            depth_pil,
            [self.img_size, self.img_size],
            interpolation=TF.InterpolationMode.NEAREST,
        )

        # Image-only colour jitter (train only).
        if self.augment:
            img = self._color_jitter(img)

        # Convert image to tensor and normalise.
        img_t = self._normalise(self._to_tensor(img))  # (3, H, W)

        # Convert depth PIL image to tensor.
        depth_np2 = np.array(depth_pil, dtype=np.float32)
        depth_t = torch.from_numpy(depth_np2).unsqueeze(0)  # (1, H, W)

        return img_t, depth_t


def load_data(
    data_dir: str,
    img_size: int,
    batch_size: int,
    num_workers: int = 8,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and validation ``DataLoader`` objects for NYU Depth V2.

    On the first call the dataset is downloaded from Hugging Face
    (``Intel/nyu_depth_v2``) and cached under ``data_dir``.  Subsequent
    calls reuse the cached files.

    Args:
        data_dir:    Local cache directory for the HF dataset.
        img_size:    Square spatial size fed to the model.
        batch_size:  Images per batch.
        num_workers: DataLoader worker processes.
        pin_memory:  Pin DataLoader memory for faster GPU transfer.

    Returns:
        train_loader: Iterable DataLoader yielding ``(images, depths)``.
        val_loader:   Idem for validation (no augmentation).

    Raises:
        ImportError: When the ``datasets`` package is not installed.
    """
    os.makedirs(data_dir, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "NYU Depth V2 loading requires the 'datasets' package. "
            "Install with: pip install datasets"
        ) from exc

    _rank = int(os.environ.get("RANK", "0"))
    _world_size = int(os.environ.get("WORLD_SIZE", "1"))
    _is_ddp = _world_size > 1

    hf_cache = os.path.abspath(data_dir)

    if _rank == 0:
        print(f"[data] loading NYU Depth V2 from Hugging Face (cache: '{hf_cache}')")

    # DDP-safe: rank-0 downloads/verifies; others wait at barrier.
    if _is_ddp and _rank == 0:
        load_dataset(_HF_REPO, split="train", cache_dir=hf_cache)
        load_dataset(_HF_REPO, split="test",  cache_dir=hf_cache)
    if _is_ddp:
        dist.barrier()

    train_hf = load_dataset(
        _HF_REPO, split="train", cache_dir=hf_cache
    )
    val_hf = load_dataset(
        _HF_REPO, split="test", cache_dir=hf_cache
    )

    train_ds = _NYUDepthDataset(train_hf, img_size=img_size, augment=True)
    val_ds = _NYUDepthDataset(val_hf,   img_size=img_size, augment=False)

    if _rank == 0:
        print(
            f"[data] NYU Depth V2 — train: {len(train_ds)}  val: {len(val_ds)}"
        )

    _loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **_loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **_loader_kwargs,
    )
    return train_loader, val_loader


def infinite_loader(loader: DataLoader):
    """
    Wrap a DataLoader to yield batches indefinitely (epoch-cycling).

    Args:
        loader: A finite DataLoader.

    Yields:
        ``(images, depths)`` tensors.
    """
    while True:
        yield from loader
