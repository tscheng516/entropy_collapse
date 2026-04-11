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

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T


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
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and validation ``DataLoader`` objects for the chosen dataset.

    Args:
        dataset:     One of ``'cifar10'``, ``'cifar100'``, ``'imagenet'``,
             ``'imagenet1k'``, ``'imagenet_hf'``.
        data_dir:    Root directory for data storage / ImageFolder tree.
        img_size:    Spatial size fed to the model (images are resized).
        batch_size:  Images per batch.
        num_workers: DataLoader worker processes.
        pin_memory:  Pin DataLoader memory for faster GPU transfer.

    Returns:
        train_loader: Iterable DataLoader yielding ``(images, labels)``.
        val_loader:   Idem for validation.

    Raises:
        ValueError: When ``dataset`` is not a recognised string.
        FileNotFoundError: When ImageNet ``data_dir`` is missing the expected
                   ``train/`` or ``val/`` sub-directories.
    """
    os.makedirs(data_dir, exist_ok=True)
    train_tf = _train_transform(img_size)
    val_tf = _val_transform(img_size)

    dataset_key = dataset.lower()

    if dataset_key == "cifar10":
        train_ds = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=train_tf
        )
        val_ds = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=val_tf
        )

    elif dataset_key == "cifar100":
        train_ds = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=train_tf
        )
        val_ds = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=val_tf
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
            # Download via HF and cache locally; subsequent runs reuse the cache.
            if int(os.environ.get("RANK", "0")) == 0:
                print(
                    f"[data] local ImageNet not found at '{data_dir}', "
                    "downloading from Hugging Face (cached for future runs) …"
                )
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

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader


def infinite_loader(loader: DataLoader):
    """
    Wrap a DataLoader to yield batches indefinitely (epoch-cycling).

    Args:
        loader: A finite DataLoader.

    Yields:
        ``(images, labels)`` tensors.
    """
    while True:
        yield from loader
