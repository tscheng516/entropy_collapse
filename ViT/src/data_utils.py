"""
Data utilities for the ViT entropy-collapse experiments.

Supports the following dataset backends:
  * CIFAR-10   — auto-downloaded via torchvision (default for pilot tests).
  * CIFAR-100  — auto-downloaded via torchvision.
  * ImageNet-1k — loaded from a local ImageFolder directory tree if present,
                  otherwise downloaded and cached via Hugging Face ``datasets``.

All datasets are returned as ``(train_loader, val_loader)`` pairs.  Images
are resized to ``img_size × img_size`` and normalised with ImageNet statistics
so that pretrained timm weights can be used directly.
"""

from __future__ import annotations

import os
import warnings

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torchvision
import torchvision.transforms as T
from PIL import ImageFile

# Allow PIL to load truncated / partially corrupt images instead of stalling.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Suppress noisy PIL warnings about known ImageNet-1k EXIF issues.
warnings.filterwarnings("ignore", message="Corrupt EXIF data")
warnings.filterwarnings("ignore", message="Truncated File Read")
warnings.filterwarnings("ignore", message="Metadata Warning, tag 274")


# ImageNet channel statistics (mean / std per channel).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class _HFDatasetWrapper(Dataset):
    """Minimal wrapper to apply torchvision transforms on HF datasets."""

    def __init__(self, hf_ds, transform):
        self.hf_ds = hf_ds
        self.transform = transform

    def __len__(self) -> int:
        return len(self.hf_ds)

    def __getitem__(self, idx: int):
        ex = self.hf_ds[idx]
        img = ex["image"]
        if hasattr(img, "convert"):
            img = img.convert("RGB")
        x = self.transform(img)
        y = int(ex["label"])
        return x, y


def _train_transform(img_size: int) -> T.Compose:
    """Standard training augmentation pipeline."""
    return T.Compose(
        [
            T.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )


def _val_transform(img_size: int) -> T.Compose:
    """Deterministic validation pipeline (centre-crop after resize)."""
    return T.Compose(
        [
            T.Resize(int(img_size * 256 / 224)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )


def load_data(
    dataset: str,
    data_dir: str,
    img_size: int,
    batch_size: int,
    num_workers: int = 8,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DistributedSampler | None]:
    """
    Build train and validation ``DataLoader`` objects for the chosen dataset.

    In DDP mode (``WORLD_SIZE > 1``) the train loader uses a
    ``DistributedSampler`` so each rank receives a disjoint shard of the
    dataset (effective per-GPU batch = ``batch_size / world_size``).
    The sampler is returned as the third element so callers can call
    ``sampler.set_epoch(epoch)`` to re-shuffle across epochs.

    Args:
        dataset:     One of ``'cifar10'``, ``'cifar100'``, ``'imagenet'``,
             ``'imagenet1k'``, ``'imagenet_hf'``.
        data_dir:    Root directory for data storage / ImageFolder tree.
        img_size:    Spatial size fed to the model (images are resized).
        batch_size:  Per-GPU batch size (passed directly to DataLoader).
        num_workers: DataLoader worker processes.
        pin_memory:  Pin DataLoader memory for faster GPU transfer.

    Returns:
        train_loader:   Iterable DataLoader yielding ``(images, labels)``.
        val_loader:     Idem for validation (full dataset on every rank).
        train_sampler:  ``DistributedSampler`` when running DDP, else ``None``.

    Raises:
        ValueError: When ``dataset`` is not a recognised string.
        FileNotFoundError: When ImageNet ``data_dir`` is missing the expected
                   ``train/`` or ``val/`` sub-directories.
    """
    os.makedirs(data_dir, exist_ok=True)
    train_tf = _train_transform(img_size)
    val_tf = _val_transform(img_size)

    dataset_key = dataset.lower()

    # DDP-safe download: only rank 0 downloads, others wait at a barrier.
    _rank = int(os.environ.get("RANK", "0"))
    _world_size = int(os.environ.get("WORLD_SIZE", "1"))
    _is_ddp = _world_size > 1

    if dataset_key == "cifar10":
        if _is_ddp:
            if _rank == 0:
                torchvision.datasets.CIFAR10(
                    root=data_dir, train=True, download=True,
                )
                torchvision.datasets.CIFAR10(
                    root=data_dir, train=False, download=True,
                )
            dist.barrier()
        train_ds = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=not _is_ddp, transform=train_tf
        )
        val_ds = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=not _is_ddp, transform=val_tf
        )

    elif dataset_key == "cifar100":
        if _is_ddp:
            if _rank == 0:
                torchvision.datasets.CIFAR100(
                    root=data_dir, train=True, download=True,
                )
                torchvision.datasets.CIFAR100(
                    root=data_dir, train=False, download=True,
                )
            dist.barrier()
        train_ds = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=not _is_ddp, transform=train_tf
        )
        val_ds = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=not _is_ddp, transform=val_tf
        )

    elif dataset_key in (
        "imagenet", "imagenet1k",
        "imagenet_hf", "imagenet1k_hf", "hf_imagenet",
    ):
        train_path = os.path.join(data_dir, "train")
        val_path = os.path.join(data_dir, "val")
        if os.path.isdir(train_path) and os.path.isdir(val_path):
            if int(os.environ.get("RANK", "0")) == 0:
                print(f"[data] loading local ImageNet from '{data_dir}'")
            train_ds = torchvision.datasets.ImageFolder(train_path, transform=train_tf)
            val_ds = torchvision.datasets.ImageFolder(val_path, transform=val_tf)
        else:
            # Load via HF; subsequent runs reuse the cache.
            if int(os.environ.get("RANK", "0")) == 0:
                print(f"[data] loading ImageNet via Hugging Face (cache: '{data_dir}')")
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise ImportError(
                    "Hugging Face fallback requires the 'datasets' package. "
                    "Install with: pip install datasets"
                ) from exc

            hf_cache = os.path.abspath(data_dir)
            train_hf = load_dataset(
                "imagenet-1k", split="train",
                cache_dir=hf_cache, token=True,
            )
            val_hf = load_dataset(
                "imagenet-1k", split="validation",
                cache_dir=hf_cache, token=True,
            )
            train_ds = _HFDatasetWrapper(train_hf, transform=train_tf)
            val_ds = _HFDatasetWrapper(val_hf, transform=val_tf)

    else:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            "Supported values: 'cifar10', 'cifar100', 'imagenet', 'imagenet1k', 'imagenet_hf'."
        )

    _loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        prefetch_factor=8 if num_workers > 0 else None,
    )

    # In DDP mode each rank receives a disjoint shard via DistributedSampler;
    # shuffle is delegated to the sampler (set per-epoch via set_epoch).
    train_sampler: DistributedSampler | None = None
    if _is_ddp:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=_world_size,
            rank=_rank,
            shuffle=True,
            drop_last=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),  # sampler and shuffle are mutually exclusive
        sampler=train_sampler,
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
    return train_loader, val_loader, train_sampler


def infinite_loader(loader: DataLoader, sampler: DistributedSampler | None = None):
    """
    Wrap a DataLoader to yield batches indefinitely (epoch-cycling).

    When a ``DistributedSampler`` is supplied, calls ``sampler.set_epoch``
    at the start of each epoch so each rank sees a different shuffle order.

    Args:
        loader:  A finite DataLoader.
        sampler: Optional ``DistributedSampler`` used by *loader*.

    Yields:
        ``(images, labels)`` tensors.
    """
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        yield from loader
        epoch += 1
