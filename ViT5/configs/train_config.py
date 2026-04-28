"""
Training configuration for the ViT-5 entropy-collapse experiments.

All configs use ViT-5-Base (87 M parameters):
  embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
  RMSNorm, QK-norm, 2-D RoPE, 4 register tokens, layer-scale.

Named presets (``CONFIGS`` registry)
-------------------------------------
| Key                  | Dataset     | img_size | patch | Notes              |
|----------------------|-------------|----------|-------|--------------------|
| ``cifar100_base``    | CIFAR-100   |   32     |   4   | 64 patches         |
| ``imagenet1k_base``  | ImageNet-1k |  192     |  16   | 144 patches        |

Select a preset via the CLI::

    python base_train.py config=imagenet1k_base
    python base_train.py config=cifar100_base --lr 3e-3

Hyperparameter references
--------------------------
ViT-5-Base pre-training defaults (LAMB, 800 epochs, 8-GPU):
  lr=3e-3, weight_decay=0.05, drop_path=0.2, warmup_epochs=5,
  img_size=192, batch=256, smoothing=0.0 (BCE loss).

Our AdamW adaptation keeps the same lr, weight_decay, and drop_path but
uses cosine-decay schedule with 50 k iterations (matching the ViT/
counterpart), label_smoothing=0.1, and a batch size of 256.

Paper: https://arxiv.org/abs/2602.08071
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time


_DATASET_DEFAULTS: dict[str, dict[str, int]] = {
    "cifar10":       {"num_classes": 10,   "img_size": 32},
    "cifar100":      {"num_classes": 100,  "img_size": 32},
    "imagenet":      {"num_classes": 1000, "img_size": 192},
    "imagenet1k":    {"num_classes": 1000, "img_size": 192},
    "imagenet_hf":   {"num_classes": 1000, "img_size": 192},
    "imagenet1k_hf": {"num_classes": 1000, "img_size": 192},
    "hf_imagenet":   {"num_classes": 1000, "img_size": 192},
}


@dataclass
class TrainConfig:
    # ------------------------------------------------------------------ #
    # I/O
    # ------------------------------------------------------------------ #
    out_dir: str = "out"
    eval_interval: int = 500
    log_interval: int = 1
    checkpoint_interval: int = -1
    save_checkpoint: bool = False
    init_from: str = "scratch"
    # 'scratch' | 'resume' | '<path-to-checkpoint>'

    # ------------------------------------------------------------------ #
    # Weights & Biases
    # ------------------------------------------------------------------ #
    wandb_log: bool = True
    wandb_project: str = "entropy-collapse-vit5"
    wandb_run_name: str = "run"

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    dataset: str = "cifar100"
    # 'cifar100' | 'imagenet1k' | 'imagenet_hf'

    data_dir: str = "./data"
    # CIFAR: torchvision download root (shared with ViT/ when run from
    #   the repo root with the same data_dir default).
    # ImageNet: directory with train/ and val/ in ImageFolder layout.

    batch_size: int = 256
    num_workers: int = 8

    # ------------------------------------------------------------------ #
    # Model — ViT-5-Base (fixed architecture, params below are read-only)
    # ------------------------------------------------------------------ #
    # embed_dim=768, depth=12, num_heads=12, mlp_ratio=4 — not exposed as
    # CLI fields since they are fixed for ViT-5-Base.

    num_classes: Optional[int] = None   # inferred from dataset when None
    img_size: Optional[int] = None      # inferred from dataset when None
    patch_size: Optional[int] = 4       # 4 for CIFAR-100; 16 for ImageNet

    # ViT-5-Base specific knobs.
    drop_path_rate: float = 0.2         # stochastic depth (ViT-5-Base default)
    num_registers: int = 4              # register tokens (must be a perfect square)
    qk_norm: bool = True                # QK-normalisation — always True for ViT-5

    # ------------------------------------------------------------------ #
    # Weight initialisation
    # ------------------------------------------------------------------ #
    init_std: float = 0.02
    # Matches ViT-5 paper (trunc_normal std=0.02).
    use_scaled_init: bool = False
    # False = keep ViT-5 defaults (layer-scale already provides depth scaling).
    label_smoothing: float = 0.1
    # Soft-label cross-entropy (user-specified; ViT-5 paper uses BCE with smoothing=0).

    # ------------------------------------------------------------------ #
    # Optimiser — AdamW (ViT-5 paper used LAMB; we use AdamW for comparability
    # with the ViT/ experiments)
    # ------------------------------------------------------------------ #
    optimizer: str = "adamw"            # 'adamw' | 'sgd'
    learning_rate: float = 3e-3         # ViT-5-Base default (LAMB lr transplanted)
    max_iters: int = 1000
    weight_decay: float = 0.05          # ViT-5-Base default
    beta1: float = 0.9
    beta2: float = 0.999
    grad_clip: float = 1.0
    eps: float = 1e-8

    # ------------------------------------------------------------------ #
    # LR schedule — cosine decay with linear warm-up
    # ------------------------------------------------------------------ #
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 50000
    min_lr: float = 3e-5                # ~1% of peak lr (3e-3 * 0.01)

    # ------------------------------------------------------------------ #
    # Hessian metrics
    # ------------------------------------------------------------------ #
    hessian_freq: int = 100
    hessian_max_iter: int = 10
    hessian_batch_size: int = 128
    compute_fd: bool = False

    # ------------------------------------------------------------------ #
    # Attention entropy
    # ------------------------------------------------------------------ #
    entropy_freq: int = 100

    # ------------------------------------------------------------------ #
    # Temperature-shift intervention
    # ------------------------------------------------------------------ #
    temp_shift_step: int = -1
    temp_shift_factor: float = 2.0

    # ------------------------------------------------------------------ #
    # Compute / device
    # ------------------------------------------------------------------ #
    device: str = "cuda"
    compile: bool = False
    dtype: str = "bfloat16"
    use_grad_ckpt: bool = False
    seed: int = 1337

    def __post_init__(self) -> None:
        _VALID_DTYPES = {"float32", "bfloat16", "float16", "float8"}
        if self.dtype not in _VALID_DTYPES:
            raise ValueError(
                f"dtype must be one of {sorted(_VALID_DTYPES)}, got '{self.dtype}'."
            )
        ds = self.dataset.lower()
        defaults = _DATASET_DEFAULTS.get(ds)
        if defaults is None:
            return
        if self.num_classes is None:
            self.num_classes = defaults["num_classes"]
        if self.img_size is None:
            self.img_size = defaults["img_size"]


# ---------------------------------------------------------------------------
# Named preset configs
# ---------------------------------------------------------------------------


@dataclass
class ViT5BaseCIFAR100Config(TrainConfig):
    """ViT-5-Base on CIFAR-100.

    32×32 images with patch_size=4 → 64 patches.  Register tokens (4)
    bring the total sequence length to 64 + 1 (CLS) + 4 = 69 tokens.

    Usage::

        python base_train.py config=cifar100_base
    """

    # ----- Data -----
    dataset: str = "cifar100"
    num_classes: int = 100
    batch_size: int = 256

    # ----- Image / patch -----
    img_size: int = 32
    patch_size: int = 4              # 32/4 = 8 → 64 patches

    # ----- ViT-5-Base knobs -----
    drop_path_rate: float = 0.2      # ViT-5-Base default
    num_registers: int = 4

    # ----- Init -----
    init_std: float = 0.02
    use_scaled_init: bool = False
    label_smoothing: float = 0.1

    # ----- Optimiser (AdamW, ViT-5-Base lr reference) -----
    learning_rate: float = 3e-3
    weight_decay: float = 0.05
    beta2: float = 0.999
    eps: float = 1e-8

    # ----- Schedule -----
    max_iters: int = 50000
    warmup_iters: int = 2000
    lr_decay_iters: int = 50000
    min_lr: float = 3e-5

    # ----- Output -----
    out_dir: str = "out/cifar100/vit5b16"
    wandb_project: str = "entropy-collapse-cifar100"
    wandb_run_name: str = time.strftime("%Y%m%d-%H%M%S")


@dataclass
class ViT5BaseImageNet1kConfig(TrainConfig):
    """ViT-5-Base on ImageNet-1k.

    192×192 images (ViT-5-Base paper default) with patch_size=16 →
    144 patches.  Register tokens bring the total to 149 tokens.

    The official pre-training used LAMB with lr=3e-3 for 800 epochs.
    Here we use AdamW with the same lr and 50 k iterations to match the
    ViT/ imagenet1k_base run length.

    Usage::

        python base_train.py config=imagenet1k_base
    """

    # ----- Data -----
    dataset: str = "imagenet1k"
    num_classes: int = 1000
    batch_size: int = 256
    num_workers: int = 8

    # ----- Image / patch -----
    img_size: int = 192              # ViT-5-Base pre-training resolution
    patch_size: int = 16             # 192/16 = 12 → 144 patches

    # ----- ViT-5-Base knobs -----
    drop_path_rate: float = 0.2
    num_registers: int = 4

    # ----- Init -----
    init_std: float = 0.02
    use_scaled_init: bool = False
    label_smoothing: float = 0.1

    # ----- Optimiser -----
    learning_rate: float = 3e-3
    weight_decay: float = 0.05
    beta2: float = 0.999
    eps: float = 1e-8

    # ----- Schedule -----
    max_iters: int = 50000
    warmup_iters: int = 5000         # ~1 warm-up epoch at batch=256
    lr_decay_iters: int = 50000
    min_lr: float = 3e-5

    # ----- Compute -----
    use_grad_ckpt: bool = False

    # ----- Output -----
    out_dir: str = "out/imagenet1k/vit5b16"
    wandb_project: str = "entropy-collapse-imagenet1k"
    wandb_run_name: str = time.strftime("%Y%m%d-%H%M%S")


# Registry — add entries here to expose new presets to the CLI.
CONFIGS: dict[str, type[TrainConfig]] = {
    "default":          TrainConfig,
    "cifar100_base":    ViT5BaseCIFAR100Config,
    "imagenet1k_base":  ViT5BaseImageNet1kConfig,
}
