"""
Training configuration for the depth/ entropy-collapse experiments.

Default config (``DepthTrainConfig``) is ViT-B/16 on NYU Depth V2 with a
DeiT-style recipe adapted for depth estimation from scratch.

Named presets (``CONFIGS`` registry)
--------------------------------------
| Key                    | Model      | Dataset      | Notes                  |
|------------------------|------------|--------------|------------------------|
| ``nyudepth_base``      | ViT-B/16   | NYU Depth V2 | same as default        |
| ``nyudepth_large``     | ViT-L/16   | NYU Depth V2 | larger backbone        |

Select a preset via the CLI::

    python base_train.py config=nyudepth_base
    python base_train.py config=nyudepth_large --lr 5e-4
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time


@dataclass
class DepthTrainConfig:
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
    wandb_project: str = "entropy-collapse-depth"
    wandb_run_name: str = "run"

    # ------------------------------------------------------------------ #
    # Data — NYU Depth V2 via Hugging Face
    # ------------------------------------------------------------------ #
    data_dir: str = "./data/nyu_depth_v2"
    # Local Hugging Face cache directory for Intel/nyu_depth_v2.

    batch_size: int = 16
    num_workers: int = 8

    # ------------------------------------------------------------------ #
    # Model — default: ViT-B/16 on NYU Depth V2
    # ------------------------------------------------------------------ #
    model_name: str = "vit_base_patch16_224"
    # Any timm ViT with a fused .blocks[i].attn.qkv projection is supported.

    img_size: int = 448
    # 448 / 16 = 28 → 784 patches (square).

    patch_size: int = 16

    # Architecture overrides passed to timm.create_model (None = timm default).
    depth: Optional[int] = 12
    num_heads: Optional[int] = 12
    embed_dim: Optional[int] = 768

    # ------------------------------------------------------------------ #
    # Weight initialisation
    # ------------------------------------------------------------------ #
    init_std: float = 0.02
    # Standard ViT init; 0.002 causes flat-loss issues (see ViT/ memories).
    use_scaled_init: bool = True
    qk_norm: bool = False

    # ------------------------------------------------------------------ #
    # Loss
    # ------------------------------------------------------------------ #
    silog_lambda: float = 0.5
    # Scale-invariant log loss variance-balancing coefficient.

    # ------------------------------------------------------------------ #
    # Optimiser
    # ------------------------------------------------------------------ #
    optimizer: str = "adamw"            # 'adamw' | 'sgd'
    learning_rate: float = 5e-4
    max_iters: int = 50000
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
    lr_decay_iters: int = 50000
    min_lr: float = 1e-5

    # ------------------------------------------------------------------ #
    # Hessian metrics
    # ------------------------------------------------------------------ #
    hessian_freq: int = 500
    hessian_max_iter: int = 10
    compute_fd: bool = False

    # ------------------------------------------------------------------ #
    # Attention entropy
    # ------------------------------------------------------------------ #
    entropy_freq: int = 500

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
    dtype: str = "float32"
    seed: int = 1337


# ---------------------------------------------------------------------------
# Named preset configs
# ---------------------------------------------------------------------------

@dataclass
class ViTBaseNYUDepthConfig(DepthTrainConfig):
    """ViT-B/16 on NYU Depth V2 — DeiT-style recipe trained from scratch.

    img_size=448, patch_size=16 → 28×28=784 patches.
    50 000 iters × batch 16 ≈ ~17 epochs over the 47 584-sample train split.

    Usage::

        python base_train.py config=nyudepth_base
    """

    # ----- Data -----
    batch_size: int = 16
    num_workers: int = 8

    # ----- Image / patch -----
    img_size:   int = 448
    patch_size: int = 16

    # ----- Architecture -----
    model_name: str = "vit_base_patch16_224"
    depth:      int = 12
    num_heads:  int = 12
    embed_dim:  int = 768

    # ----- Init -----
    init_std:        float = 0.02
    use_scaled_init: bool  = True

    # ----- Optimiser -----
    learning_rate: float = 5e-4
    weight_decay:  float = 0.05
    beta2:         float = 0.999
    eps:           float = 1e-8

    # ----- Schedule -----
    max_iters:       int   = 50000
    warmup_iters:    int   = 2000
    lr_decay_iters:  int   = 50000
    min_lr:          float = 1e-5

    # ----- Output -----
    out_dir:        str = "out/nyudepth_vitb16"
    wandb_project:  str = "entropy-collapse-depth"
    wandb_run_name: str = time.strftime("%Y%m%d-%H%M%S")


@dataclass
class ViTLargeNYUDepthConfig(DepthTrainConfig):
    """ViT-L/16 on NYU Depth V2.  img_size=448, patch_size=16 → 784 patches.
    ~307 M parameters.

    Usage::

        python base_train.py config=nyudepth_large
    """

    # ----- Data -----
    batch_size: int = 8          # ViT-L is memory-heavy; reduce if OOM
    num_workers: int = 8

    # ----- Image / patch -----
    img_size:   int = 448
    patch_size: int = 16

    # ----- Architecture -----
    model_name: str = "vit_large_patch16_224"
    depth:      int = 24
    num_heads:  int = 16
    embed_dim:  int = 1024

    # ----- Init -----
    init_std:        float = 0.02
    use_scaled_init: bool  = True

    # ----- Optimiser -----
    learning_rate: float = 2e-4   # lower LR for larger model
    weight_decay:  float = 0.05
    beta2:         float = 0.999
    eps:           float = 1e-8

    # ----- Schedule -----
    max_iters:       int   = 50000
    warmup_iters:    int   = 2000
    lr_decay_iters:  int   = 50000
    min_lr:          float = 2e-6

    # ----- Output -----
    out_dir:        str = "out/nyudepth_vitl16"
    wandb_project:  str = "entropy-collapse-depth"
    wandb_run_name: str = time.strftime("%Y%m%d-%H%M%S")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CONFIGS: dict[str, type] = {
    "nyudepth_base":  ViTBaseNYUDepthConfig,
    "nyudepth_large": ViTLargeNYUDepthConfig,
}
