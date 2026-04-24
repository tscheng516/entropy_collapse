"""
Training configuration for the ViT entropy-collapse experiments.

Default config (``TrainConfig``) is ViT-B/16 on CIFAR-100 with a DeiT recipe.
Run ``python base_train.py`` to use it directly.

Named presets (``CONFIGS`` registry)
-------------------------------------
| Key                  | Model    | Dataset     | Notes                   |
|----------------------|----------|-------------|-------------------------|
| ``cifar100_base``    | ViT-B/16 | CIFAR-100   | same as default + paths |
| ``imagenet1k_base``  | ViT-B/16 | ImageNet-1k | DeiT recipe             |
| ``cifar100_large``   | ViT-L/16 | CIFAR-100   | patch_size=4            |
| ``imagenet1k_large`` | ViT-L/16 | ImageNet-1k |                         |
| ``cifar100_huge``    | ViT-H/14 | CIFAR-100   | patch_size=4            |
| ``imagenet1k_huge``  | ViT-H/14 | ImageNet-1k |                         |

Select a preset via the CLI::

    python base_train.py config=imagenet1k_base
    python base_train.py config=cifar100_large --lr 5e-4
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time


_DATASET_DEFAULTS: dict[str, dict[str, int]] = {
    "cifar10": {"num_classes": 10, "img_size": 32},
    "cifar100": {"num_classes": 100, "img_size": 32},
    "imagenet": {"num_classes": 1000, "img_size": 224},
    "imagenet1k": {"num_classes": 1000, "img_size": 224},
    "imagenet_hf": {"num_classes": 1000, "img_size": 224},
    "imagenet1k_hf": {"num_classes": 1000, "img_size": 224},
    "hf_imagenet": {"num_classes": 1000, "img_size": 224},
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
    wandb_project: str = "entropy-collapse-vit"
    wandb_run_name: str = "run"

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    dataset: str = "cifar100"
    # 'cifar10' | 'cifar100' | 'imagenet1k' | 'imagenet_hf'

    data_dir: str = "./data"
    # CIFAR: torchvision download root.
    # ImageNet: directory with train/ and val/ in ImageFolder layout.
    # ImageNet HF: Hugging Face cache directory.

    batch_size: int = 128
    num_workers: int = 8

    # ------------------------------------------------------------------ #
    # Model — default: ViT-B/16 on CIFAR-100 (patch_size=4 → 64 patches)
    # ------------------------------------------------------------------ #
    model_name: str = "vit_base_patch16_224"
    # Any timm ViT with a fused .blocks[i].attn.qkv projection is supported.

    pretrained: bool = False
    # False = scratch training (best for entropy-collapse dynamics).

    num_classes: Optional[int] = None   # inferred from dataset when None
    img_size: Optional[int] = None      # inferred from dataset when None

    # Architecture overrides passed to timm.create_model (None = timm default).
    depth: Optional[int] = 12
    num_heads: Optional[int] = 12
    embed_dim: Optional[int] = 768
    patch_size: Optional[int] = 4       # 4 for CIFAR-100 32×32 → 64 patches

    # ------------------------------------------------------------------ #
    # Weight initialisation
    # ------------------------------------------------------------------ #
    init_std: float = 0.02
    # Standard ViT truncated-normal init; 0.002 causes flat-loss dead zones.
    use_scaled_init: bool = True
    qk_norm: bool = False
    label_smoothing: float = 0.1

    # ------------------------------------------------------------------ #
    # Optimiser
    # ------------------------------------------------------------------ #
    optimizer: str = "adamw"            # 'adamw' | 'sgd'
    learning_rate: float = 1e-3
    max_iters: int = 50000
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.999
    # beta2=0.999 is the DeiT/ViT standard; 0.95 suits GPT-style LMs.
    grad_clip: float = 1.0
    eps: float = 1e-8

    # ------------------------------------------------------------------ #
    # LR schedule — cosine decay with linear warm-up
    # ------------------------------------------------------------------ #
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 50000
    min_lr: float = 1e-5

    # ------------------------------------------------------------------ #
    # Hessian metrics
    # ------------------------------------------------------------------ #
    hessian_freq: int = 500
    # Compute all nine curvature proxies every N iterations.
    hessian_max_iter: int = 10
    # Power-iteration steps for λ_max estimation.
    compute_fd: bool = False
    # Enable finite-difference proxies (BFGS, FD); costs an extra fwd/bwd pass.

    # ------------------------------------------------------------------ #
    # Attention entropy
    # ------------------------------------------------------------------ #
    entropy_freq: int = 500
    # Compute per-layer attention entropy every N iterations.

    # ------------------------------------------------------------------ #
    # Temperature-shift intervention
    # ------------------------------------------------------------------ #
    temp_shift_step: int = -1
    temp_shift_factor: float = 2.0

    # ------------------------------------------------------------------ #
    # Compute / device
    # ------------------------------------------------------------------ #
    device: str = "cuda"                # 'cuda' | 'cpu' | 'mps'
    compile: bool = False               # disable when computing 2nd-order grads
    dtype: str = "float32"              # 'float32' | 'bfloat16'
    seed: int = 1337

    def __post_init__(self) -> None:
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
class ViTBaseCIFAR100Config(TrainConfig):
    """ViT-B/16 on CIFAR-100 — same recipe as TrainConfig with labelled outputs.

    Usage::

        python base_train.py config=cifar100_base
    """

    # ----- Data -----
    dataset: str = "cifar100"
    num_classes: int = 100
    batch_size: int = 128

    # ----- Image / patch -----
    img_size: int = 32
    patch_size: int = 4           # 32/4 = 8 → 64 patches

    # ----- Architecture -----
    model_name: str = "vit_base_patch16_224"
    depth: int = 12
    num_heads: int = 12
    embed_dim: int = 768

    # ----- Init -----
    init_std: float = 0.02
    use_scaled_init: bool = True
    label_smoothing: float = 0.1

    # ----- Optimiser -----
    learning_rate: float = 1e-3
    weight_decay: float = 0.05
    beta2: float = 0.999
    eps: float = 1e-8

    # ----- Schedule -----
    max_iters: int = 50000
    warmup_iters: int = 2000
    lr_decay_iters: int = 50000
    min_lr: float = 1e-5

    # ----- Output -----
    out_dir: str = "out/cifar100_vitb16"
    wandb_project: str = "entropy-collapse-cifar100_vitb16"
    wandb_run_name: str = time.strftime("%Y%m%d-%H%M%S")


@dataclass
class ViTBaseImageNet1kConfig(TrainConfig):
    """ViT-B/16 on ImageNet-1k — DeiT recipe (Touvron et al., 2021).

    patch_size=16 on 224×224 → 196 patches.  50 000 iters × batch 256 ≈ 64 epochs.

    Usage::


        python base_train.py config=imagenet1k_base

    Override individual fields::

        python base_train.py config=imagenet1k_base --lr 5e-4 max_iters=100000
    """

    # ----- Data -----
    dataset: str = "imagenet1k"
    num_classes: int = 1000
    batch_size: int = 256
    num_workers: int = 8

    # ----- Image / patch -----
    img_size: int = 224
    patch_size: int = 16          # 224/16 = 14 → 196 patches

    # ----- Architecture -----
    model_name: str = "vit_base_patch16_224"
    depth: int = 12
    num_heads: int = 12
    embed_dim: int = 768

    # ----- Init -----
    init_std: float = 0.02
    use_scaled_init: bool = True
    label_smoothing: float = 0.1

    # ----- Optimiser -----
    learning_rate: float = 1e-3
    weight_decay: float = 0.05
    beta2: float = 0.999
    eps: float = 1e-8

    # ----- Schedule -----
    max_iters: int = 50000
    warmup_iters: int = 5000
    lr_decay_iters: int = 50000
    min_lr: float = 1e-5

    # ----- Output -----
    out_dir: str = "out/imagenet1k_vitb16"
    wandb_project: str = "entropy-collapse-imagenet1k_vitb16"
    wandb_run_name: str = time.strftime("%Y%m%d-%H%M%S")


@dataclass
class ViTLargeCIFAR100Config(TrainConfig):
    """ViT-L/16 on CIFAR-100. patch_size=4 on 32×32 → 64 patches. ~307 M params.

    Usage::

        python base_train.py config=cifar100_large
    """

    # ----- Data -----
    dataset: str = "cifar100"
    num_classes: int = 100
    batch_size: int = 64          # smaller batch; ViT-L is memory-heavy

    # ----- Image / patch -----
    img_size: int = 32
    patch_size: int = 4

    # ----- Architecture -----
    model_name: str = "vit_large_patch16_224"
    depth: int = 24
    num_heads: int = 16
    embed_dim: int = 1024

    # ----- Init -----
    init_std: float = 0.02
    use_scaled_init: bool = True
    label_smoothing: float = 0.1

    # ----- Optimiser -----
    learning_rate: float = 5e-4   # lower LR for larger model
    weight_decay: float = 0.05
    beta2: float = 0.999
    eps: float = 1e-8

    # ----- Schedule -----
    max_iters: int = 50000
    warmup_iters: int = 2000
    lr_decay_iters: int = 50000
    min_lr: float = 5e-6

    # ----- Output -----
    out_dir: str = "out/cifar100_vitl16"
    wandb_project: str = "entropy-collapse-cifar100_vitl16"
    wandb_run_name: str = time.strftime("%Y%m%d-%H%M%S")


@dataclass
class ViTLargeImageNet1kConfig(TrainConfig):
    """ViT-L/16 on ImageNet-1k. patch_size=16 on 224×224 → 196 patches. ~307 M params.

    Usage::

        python base_train.py config=imagenet1k_large
    """

    # ----- Data -----
    dataset: str = "imagenet1k"
    num_classes: int = 1000
    batch_size: int = 128         # reduced vs ViT-B to fit memory
    num_workers: int = 8

    # ----- Image / patch -----
    img_size: int = 224
    patch_size: int = 16

    # ----- Architecture -----
    model_name: str = "vit_large_patch16_224"
    depth: int = 24
    num_heads: int = 16
    embed_dim: int = 1024

    # ----- Init -----
    init_std: float = 0.02
    use_scaled_init: bool = True
    label_smoothing: float = 0.1

    # ----- Optimiser -----
    learning_rate: float = 5e-4
    weight_decay: float = 0.05
    beta2: float = 0.999
    eps: float = 1e-8

    # ----- Schedule -----
    max_iters: int = 50000
    warmup_iters: int = 5000
    lr_decay_iters: int = 50000
    min_lr: float = 5e-6

    # ----- Output -----
    out_dir: str = "out/imagenet1k_vitl16"
    wandb_project: str = "entropy-collapse-imagenet1k_vitl16"
    wandb_run_name: str = time.strftime("%Y%m%d-%H%M%S")


@dataclass
class ViTHugeCIFAR100Config(TrainConfig):
    """ViT-H/14 on CIFAR-100. patch_size=4 on 32×32 → 64 patches. ~632 M params.

    Usage::

        python base_train.py config=cifar100_huge
    """

    # ----- Data -----
    dataset: str = "cifar100"
    num_classes: int = 100
    batch_size: int = 32          # very large model; reduce further if OOM

    # ----- Image / patch -----
    img_size: int = 32
    patch_size: int = 4

    # ----- Architecture -----
    model_name: str = "vit_huge_patch14_224"
    depth: int = 32
    num_heads: int = 16
    embed_dim: int = 1280

    # ----- Init -----
    init_std: float = 0.02
    use_scaled_init: bool = True
    label_smoothing: float = 0.1

    # ----- Optimiser -----
    learning_rate: float = 3e-4   # conservative LR for very large model
    weight_decay: float = 0.05
    beta2: float = 0.999
    eps: float = 1e-8

    # ----- Schedule -----
    max_iters: int = 50000
    warmup_iters: int = 2000
    lr_decay_iters: int = 50000
    min_lr: float = 3e-6

    # ----- Output -----
    out_dir: str = "out/cifar100_vith14"
    wandb_project: str = "entropy-collapse-cifar100_vith14"
    wandb_run_name: str = time.strftime("%Y%m%d-%H%M%S")


@dataclass
class ViTHugeImageNet1kConfig(TrainConfig):
    """ViT-H/14 on ImageNet-1k. patch_size=14 on 224×224 → 256 patches. ~632 M params.

    Usage::

        python base_train.py config=imagenet1k_huge
    """

    # ----- Data -----
    dataset: str = "imagenet1k"
    num_classes: int = 1000
    batch_size: int = 64
    num_workers: int = 8

    # ----- Image / patch -----
    img_size: int = 224
    patch_size: int = 14          # 224/14 = 16 → 256 patches

    # ----- Architecture -----
    model_name: str = "vit_huge_patch14_224"
    depth: int = 32
    num_heads: int = 16
    embed_dim: int = 1280

    # ----- Init -----
    init_std: float = 0.02
    use_scaled_init: bool = True
    label_smoothing: float = 0.1

    # ----- Optimiser -----
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    beta2: float = 0.999
    eps: float = 1e-8

    # ----- Schedule -----
    max_iters: int = 50000
    warmup_iters: int = 5000
    lr_decay_iters: int = 50000
    min_lr: float = 3e-6

    # ----- Output -----
    out_dir: str = "out/imagenet1k_vith14"
    wandb_project: str = "entropy-collapse-imagenet1k_vith14"
    wandb_run_name: str = time.strftime("%Y%m%d-%H%M%S")


# Registry — add entries here to expose new presets to the CLI.
CONFIGS: dict[str, type[TrainConfig]] = {
    "default":           TrainConfig,
    "cifar100_base":     ViTBaseCIFAR100Config,
    "imagenet1k_base":   ViTBaseImageNet1kConfig,
    "cifar100_large":    ViTLargeCIFAR100Config,
    "imagenet1k_large":  ViTLargeImageNet1kConfig,
    "cifar100_huge":     ViTHugeCIFAR100Config,
    "imagenet1k_huge":   ViTHugeImageNet1kConfig,
}
