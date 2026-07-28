"""
Training configuration for the ViT5 (ViT-5) entropy-collapse experiments.

ViT-5 (Wang et al., 2026, arXiv:2602.08071) architecture, always active:
  RMSNorm, QK-norm, 2-D RoPE (patch tokens theta=10000, register tokens
  theta=100), 4 register tokens, layer-scale, no QKV bias.

Named presets (``CONFIGS`` registry)
-------------------------------------
| Key                    | Model      | Dataset     | Notes            |
|------------------------|------------|-------------|-------------------|
| ``cifar100_small``     | ViT-5-S    | CIFAR-100   | 64 patches        |
| ``cifar100_base``      | ViT-5-B    | CIFAR-100   | 64 patches        |
| ``cifar100_large``     | ViT-5-L    | CIFAR-100   | 64 patches        |
| ``imagenet1k_small``   | ViT-5-S    | ImageNet-1k | 144 patches       |
| ``imagenet1k_base``    | ViT-5-B    | ImageNet-1k | 144 patches       |
| ``imagenet1k_large``   | ViT-5-L    | ImageNet-1k | 144 patches       |

Select a preset via the CLI::

    python base_train.py config=cifar100_base
    python base_train.py config=imagenet1k_large learning_rate=1e-3

Paper: https://arxiv.org/abs/2602.08071
Official implementation: https://github.com/wangf3014/ViT-5
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time


_DATASET_DEFAULTS: dict[str, dict[str, int]] = {
    "cifar100": {"num_classes": 100, "img_size": 32},
    "imagenet1k": {"num_classes": 1000, "img_size": 192},
}


@dataclass
class TrainConfig:
    # ------------------------------------------------------------------ #
    # I/O
    # ------------------------------------------------------------------ #
    out_dir: str = "out/pilot"
    eval_interval: int = 500
    log_interval: int = 10
    checkpoint_interval: int = -1
    save_checkpoint: bool = False
    init_from: str = "scratch"
    # 'scratch' | 'resume' | '<path-to-checkpoint>'

    # ------------------------------------------------------------------ #
    # Weights & Biases
    # ------------------------------------------------------------------ #
    wandb_log: bool = False  # disabled in pilot run with default config
    wandb_project: str = "entropy-collapse-vit5"
    wandb_run_name: str = "run"

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    dataset: str = "cifar100"
    # 'cifar100' | 'imagenet1k'

    data_dir: str = "./data"
    # CIFAR: torchvision download root (shared with ViT/ when run from the
    #   repo root with the same data_dir default).
    # ImageNet: directory with train/ and val/ in ImageFolder layout.

    batch_size: int = 32  # small batch for pilot run; increase to 256 for main experiments
    num_workers: int = 8

    # ------------------------------------------------------------------ #
    # Model — ViT-5 (fixed architecture family; size selected via model_name)
    # ------------------------------------------------------------------ #
    model_name: str = "vit5_base"
    # 'vit5_small' | 'vit5_base' | 'vit5_large' (see src/model.py MODEL_SIZES)

    num_classes: Optional[int] = None   # inferred from dataset when None
    img_size: Optional[int] = None      # inferred from dataset when None
    patch_size: Optional[int] = 4       # 4 for CIFAR-100; 16 for ImageNet-1k

    # Architecture overrides (None = use model_name preset default).
    depth: Optional[int] = None
    num_heads: Optional[int] = None
    embed_dim: Optional[int] = None

    # ViT-5-specific knobs.
    num_registers: int = 4              # register tokens; must be a perfect square
    qk_norm: bool = True                # QK-normalisation — on by default for ViT-5
    reg_theta: float = 100              # RoPE base frequency for register tokens
    drop_path_rate: float = 0.1         # stochastic depth

    # ------------------------------------------------------------------ #
    # Weight initialisation
    # ------------------------------------------------------------------ #
    init_std: float = 0.02
    # Matches the ViT-5 paper (trunc_normal std=0.02).
    use_scaled_init: bool = False
    # False = keep ViT-5 defaults (layer-scale already provides depth scaling).
    label_smoothing: float = 0.1

    # ------------------------------------------------------------------ #
    # Optimiser
    # ------------------------------------------------------------------ #
    optimizer: str = "adamw"            # 'adamw' | 'sgd'
    learning_rate: float = 3e-4
    max_iters: int = 100  # small number for pilot run; increase for main experiments
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.999
    grad_clip: float = 1.0
    eps: float = 1e-8

    # ------------------------------------------------------------------ #
    # LR schedule — cosine decay with linear warm-up
    # ------------------------------------------------------------------ #
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 20000
    min_lr: float = 3e-6

    # ------------------------------------------------------------------ #
    # Hessian metrics
    # ------------------------------------------------------------------ #
    hessian_intv: int = 50
    # Compute all nine curvature proxies every N iterations.
    hessian_max_iter: int = 10
    # Power-iteration steps for lambda_max estimation.
    hessian_batch_size: int = 128
    # Samples sliced from the training batch for curvature estimation.
    compute_fd: bool = False
    # Enable finite-difference proxies (BFGS, FD) and K-FAC; costs extra passes.

    # ------------------------------------------------------------------ #
    # Attention entropy
    # ------------------------------------------------------------------ #
    entropy_intv: int = 50
    # Compute per-layer attention entropy every N iterations.
    att_sim: bool = False
    # When True, also snapshot attention heatmaps / Gram matrices for plotting.

    # ------------------------------------------------------------------ #
    # Temperature-shift intervention
    # ------------------------------------------------------------------ #
    temp_shift_step: int = -1
    temp_shift_factor: float = 0.25

    # ------------------------------------------------------------------ #
    # Compute / device
    # ------------------------------------------------------------------ #
    device: str = "cuda"                # 'cuda' | 'cpu' | 'mps'
    compile: bool = False               # disable when computing 2nd-order grads
    dtype: str = "bfloat16"             # 'float32' | 'bfloat16' | 'float16' | 'float8'
    seed: int = 1337

    def __post_init__(self) -> None:
        _VALID_DTYPES = {"float32", "bfloat16", "float16", "float8", "float8_e4m3fn", "float8_e5m2"}
        if self.dtype not in _VALID_DTYPES:
            raise ValueError(
                f"dtype must be one of {sorted(_VALID_DTYPES)}, got '{self.dtype}'."
            )

        ds = self.dataset.lower()
        defaults = _DATASET_DEFAULTS.get(ds)
        if defaults is not None:
            if self.num_classes is None:
                self.num_classes = defaults["num_classes"]
            if self.img_size is None:
                self.img_size = defaults["img_size"]

        # Validate num_registers at config-resolve time rather than at model
        # init (fixes a bug present in the ViT5_old/ prototype, where an
        # invalid value only surfaced as a late assertion error).
        root = int(self.num_registers ** 0.5)
        if root * root != self.num_registers:
            raise ValueError(
                f"num_registers must be a perfect square, got {self.num_registers}."
            )


# ---------------------------------------------------------------------------
# Named preset configs
# ---------------------------------------------------------------------------

@dataclass
class ViT5SmallCIFAR100Config(TrainConfig):
    """ViT-5-Small on CIFAR-100.

    Usage::

        python base_train.py config=cifar100_small
    """

    dataset: str = "cifar100"
    num_classes: int = 100
    batch_size: int = 256

    img_size: int = 32
    patch_size: int = 4            # 32/4 = 8 -> 64 patches

    model_name: str = "vit5_small"

    init_std: float = 0.02
    use_scaled_init: bool = False
    label_smoothing: float = 0.1
    drop_path_rate: float = 0.05

    learning_rate: float = 3e-3
    weight_decay: float = 0.05
    beta2: float = 0.999
    eps: float = 1e-8

    max_iters: int = 20000
    warmup_iters: int = 2000
    lr_decay_iters: int = 20000
    min_lr: float = 3e-5

    out_dir: str = "out/cifar100/vit5s"
    wandb_log: bool = True
    # wandb_project: str = "entropy-collapse-vit5-cifar100"
    wandb_run_name: str = time.strftime("%Y%m%d-%H%M%S")


@dataclass
class ViT5BaseCIFAR100Config(TrainConfig):
    """ViT-5-Base on CIFAR-100.

    Usage::

        python base_train.py config=cifar100_base
    """

    dataset: str = "cifar100"
    num_classes: int = 100
    batch_size: int = 256

    img_size: int = 32
    patch_size: int = 4            # 32/4 = 8 -> 64 patches

    model_name: str = "vit5_base"

    init_std: float = 0.02
    use_scaled_init: bool = False
    label_smoothing: float = 0.1
    drop_path_rate: float = 0.0

    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    beta2: float = 0.999
    eps: float = 1e-8

    max_iters: int = 20000
    warmup_iters: int = 2000
    lr_decay_iters: int = 20000
    min_lr: float = 3e-5

    out_dir: str = "out/cifar100/vit5b"
    wandb_log: bool = True
    # wandb_project: str = "entropy-collapse-vit5-cifar100"
    wandb_run_name: str = time.strftime("%Y%m%d-%H%M%S")


@dataclass
class ViT5LargeCIFAR100Config(TrainConfig):
    """ViT-5-Large on CIFAR-100.

    Usage::

        python base_train.py config=cifar100_large
    """

    dataset: str = "cifar100"
    num_classes: int = 100
    batch_size: int = 256

    img_size: int = 32
    patch_size: int = 4            # 32/4 = 8 -> 64 patches

    model_name: str = "vit5_large"

    init_std: float = 0.02
    use_scaled_init: bool = False
    label_smoothing: float = 0.1
    drop_path_rate: float = 0.0

    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    beta2: float = 0.999
    eps: float = 1e-8

    max_iters: int = 20000
    warmup_iters: int = 2000
    lr_decay_iters: int = 20000
    min_lr: float = 3e-6

    out_dir: str = "out/cifar100/vit5l"
    wandb_log: bool = True
    # wandb_project: str = "entropy-collapse-vit5-cifar100"
    wandb_run_name: str = time.strftime("%Y%m%d-%H%M%S")


@dataclass
class ViT5SmallImageNet1kConfig(TrainConfig):
    """ViT-5-Small on ImageNet-1k.

    patch_size=16 on 192x192 -> 144 patches.

    Usage::

        python base_train.py config=imagenet1k_small
    """

    dataset: str = "imagenet1k"
    num_classes: int = 1000
    batch_size: int = 256
    num_workers: int = 8

    img_size: int = 192
    patch_size: int = 16           # 192/16 = 12 -> 144 patches

    model_name: str = "vit5_small"

    init_std: float = 0.02
    use_scaled_init: bool = False
    label_smoothing: float = 0.1
    drop_path_rate: float = 0.0

    learning_rate: float = 4e-3
    weight_decay: float = 0.05
    beta2: float = 0.999
    eps: float = 1e-8

    max_iters: int = 50000
    warmup_iters: int = 5000
    lr_decay_iters: int = 50000
    min_lr: float = 1e-5

    out_dir: str = "out/imagenet1k/vit5s"
    wandb_log: bool = True
    # wandb_project: str = "entropy-collapse-vit5-imagenet1k
    wandb_run_name: str = time.strftime("%Y%m%d-%H%M%S")


@dataclass
class ViT5BaseImageNet1kConfig(TrainConfig):
    """ViT-5-Base on ImageNet-1k.

    patch_size=16 on 192x192 -> 144 patches.

    Usage::

        python base_train.py config=imagenet1k_base
    """

    dataset: str = "imagenet1k"
    num_classes: int = 1000
    batch_size: int = 256
    num_workers: int = 8

    img_size: int = 192
    patch_size: int = 16           # 192/16 = 12 -> 144 patches

    model_name: str = "vit5_base"

    init_std: float = 0.02
    use_scaled_init: bool = False
    label_smoothing: float = 0.1
    drop_path_rate: float = 0.0

    learning_rate: float = 3e-3
    weight_decay: float = 0.05
    beta2: float = 0.999
    eps: float = 1e-8

    max_iters: int = 50000
    warmup_iters: int = 5000
    lr_decay_iters: int = 50000
    min_lr: float = 1e-5

    out_dir: str = "out/imagenet1k/vit5b"
    wandb_log: bool = True
    # wandb_project: str = "entropy-collapse-vit5-imagenet1k"
    wandb_run_name: str = time.strftime("%Y%m%d-%H%M%S")


@dataclass
class ViT5LargeImageNet1kConfig(TrainConfig):
    """ViT-5-Large on ImageNet-1k.

    patch_size=16 on 192x192 -> 144 patches.

    Usage::

        python base_train.py config=imagenet1k_large
    """

    dataset: str = "imagenet1k"
    num_classes: int = 1000
    batch_size: int = 256
    num_workers: int = 8

    img_size: int = 192
    patch_size: int = 16           # 192/16 = 12 -> 144 patches

    model_name: str = "vit5_large"

    init_std: float = 0.02
    use_scaled_init: bool = False
    label_smoothing: float = 0.1
    drop_path_rate: float = 0.0

    learning_rate: float = 3e-3
    weight_decay: float = 0.05
    beta2: float = 0.999
    eps: float = 1e-8

    max_iters: int = 50000
    warmup_iters: int = 5000
    lr_decay_iters: int = 50000
    min_lr: float = 1e-5

    out_dir: str = "out/imagenet1k/vit5l"
    wandb_log: bool = True
    # wandb_project: str = "entropy-collapse-vit5-imagenet1k"
    wandb_run_name: str = time.strftime("%Y%m%d-%H%M%S")


# Registry — add entries here to expose new presets to the CLI.
CONFIGS: dict[str, type[TrainConfig]] = {
    "default":             TrainConfig,
    "cifar100_small":      ViT5SmallCIFAR100Config,
    "cifar100_base":       ViT5BaseCIFAR100Config,
    "cifar100_large":      ViT5LargeCIFAR100Config,
    "imagenet1k_small":    ViT5SmallImageNet1kConfig,
    "imagenet1k_base":     ViT5BaseImageNet1kConfig,
    "imagenet1k_large":    ViT5LargeImageNet1kConfig,
}
