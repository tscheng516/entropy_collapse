"""
Training configuration for the ViT entropy-collapse experiments.

Preset configs
--------------
A registry of named presets is exported as ``CONFIGS``:

    from configs.train_config import CONFIGS
    cfg = CONFIGS["cifar100_small"]()

Or load a preset directly from the CLI::

    python base_train.py config=cifar100_small
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


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
    checkpoint_interval: int = 5000
    save_checkpoint: bool = True
    init_from: str = "scratch"
    """
    'scratch'  — initialise a fresh model with custom init_std weights.
    'resume'   — load the latest checkpoint from out_dir and continue.
    '<path>'   — load a checkpoint file and fine-tune.
    """

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
    """
    Dataset to use.  Supported values:
      'cifar10'   — CIFAR-10 (auto-downloaded, 10 classes, 32×32 → resized).
      'cifar100'  — CIFAR-100 (auto-downloaded, 100 classes).
      'imagenet'  — ImageNet-1k (requires data_dir with train/ and val/ folders).
      'imagenet_hf' — ImageNet-1k loaded directly from Hugging Face datasets.
    """

    data_dir: str = "./data"
    """
    Root directory for dataset storage.
    For CIFAR-10/100: the torchvision download root.
    For ImageNet: should contain ``train/`` and ``val/`` sub-folders
    in ImageFolder format.
    For ImageNet HF: used as the Hugging Face cache directory.
    """

    batch_size: int = 64
    num_workers: int = 4

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    model_name: str = "vit_small_patch16_224"
    """
    timm model identifier.  Defaults to ViT-Small/16 (22 M params,
    384-dim, 12 layers, 6 heads) — a lightweight pilot model.
    Any timm ViT variant with a standard ``.blocks[i].attn.qkv`` fused
    projection is supported.
    """

    pretrained: bool = False
    """
    Load ImageNet-pretrained weights from timm.
    Use ``False`` for scratch training (entropy-collapse dynamics are
    most visible from random initialisation).
    """

    num_classes: Optional[int] = None
    """
    Number of output classes.
    If ``None``, this is inferred from ``dataset``.
    """

    img_size: Optional[int] = None
    """
    Spatial resolution fed to the model.
    If ``None``, this is inferred from ``dataset``.
    """

    # Architecture overrides (passed as kwargs to timm.create_model).
    # ``None`` means "use the timm model's built-in default".
    depth: Optional[int] = None
    num_heads: Optional[int] = None
    embed_dim: Optional[int] = None
    patch_size: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Weight initialisation
    # ------------------------------------------------------------------ #
    init_std: float = 0.002
    use_scaled_init: bool = True
    qk_norm: bool = False
    label_smoothing: float = 0.0

    # ------------------------------------------------------------------ #
    # Optimiser
    # ------------------------------------------------------------------ #
    optimizer: str = "adamw"
    """'adamw' or 'sgd'."""

    learning_rate: float = 1e-5
    """Peak learning rate."""

    max_iters: int = 1000
    """Total number of training iterations."""

    weight_decay: float = 1e-2
    """AdamW weight decay."""

    beta1: float = 0.9
    beta2: float = 0.95
    """AdamW beta coefficients."""

    grad_clip: float = 1.0
    """Gradient norm clipping (0.0 to disable)."""

    eps: float = 1e-5
    """
    AdamW epsilon.  Slightly larger than PyTorch's default (1e-8) for
    numerical stability at the small weight magnitudes used here.
    """

    # ------------------------------------------------------------------ #
    # Learning-rate schedule
    # ------------------------------------------------------------------ #
    decay_lr: bool = True
    """Enable cosine LR decay with linear warm-up."""

    warmup_iters: int = 100
    """Number of linear warm-up iterations."""

    lr_decay_iters: int = 1000
    """Decay LR over this many iterations (should equal max_iters)."""

    min_lr: float = 1e-5
    """Minimum LR at the end of the cosine schedule (≈ learning_rate / 10)."""

    # ------------------------------------------------------------------ #
    # Hessian metrics
    # ------------------------------------------------------------------ #
    hessian_freq: int = 500
    """
    Compute all Hessian proxy metrics (λ_max, H_tilde, H_VV, H_GN, FD,
    Diag_H, Fisher, BFGS, KFAC) every N iterations.  Use a larger value
    to reduce overhead.
    """

    hessian_max_iter: int = 10
    """Number of power-iteration steps for λ_max estimation."""

    compute_fd: bool = False
    """
    Whether to compute the Finite-Difference sharpness proxy.
    This requires an extra forward/backward pass and can be disabled
    when running on tight compute budgets.
    """

    # ------------------------------------------------------------------ #
    # Attention entropy
    # ------------------------------------------------------------------ #
    entropy_freq: int = 500
    """
    Compute per-layer attention entropy every N iterations.
    Logged both to stdout and to wandb as entropy/layer_<k>.
    """

    # ------------------------------------------------------------------ #
    # Temperature-shift intervention
    # ------------------------------------------------------------------ #
    temp_shift_step: int = -1
    temp_shift_factor: float = 2.0

    # ------------------------------------------------------------------ #
    # Compute / device
    # ------------------------------------------------------------------ #
    device: str = "cuda"
    """Target device: 'cuda', 'cpu', or 'mps'."""

    compile: bool = False
    """
    If True, wrap the model with torch.compile() (PyTorch ≥ 2.0).
    Disable when computing second-order gradients (autograd through
    the compiled graph is not supported in all versions).
    """

    dtype: str = "float32"
    """'float32' or 'bfloat16'."""

    seed: int = 1337
    """Random seed for reproducibility."""

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
class ViTSmallCIFAR100Config(TrainConfig):
    """
    ViT configuration comparable to the NanoGPT-small (d=6) LLM experiment,
    adapted for CIFAR-100.

    Architecture parity with NanoGPT-small (LLM/configs/train_config.py):

    +--------------+-----------------------+------------------+
    | Dimension    | NanoGPT-small         | This ViT config  |
    +--------------+-----------------------+------------------+
    | Depth        | n_layer = 6           | depth = 6        |
    | Heads        | n_head  = 6           | num_heads = 6    |
    | Hidden dim   | n_embd  = 384         | embed_dim = 384  |
    | Seq length   | block_size = 64 tok   | 64 patches *     |
    +--------------+-----------------------+------------------+

    * patch_size=4 on 32×32 CIFAR images → (32/4)² = 64 patches, exactly
      matching NanoGPT-small's context length of 64 tokens.  Both models
      therefore process sequences of length 64 per sample.

    Both architectures use:
      - 6 transformer layers
      - 6 attention heads
      - 384-dimensional embeddings
      - MLP expansion ratio of 4× (≈ 1536 hidden units)
      - No QK normalisation (``qk_norm=False``)
      - Small-std weight initialisation with residual-depth scaling

    This yields roughly comparable total parameter counts (~10.7 M each)
    and makes cross-modal comparisons of entropy / Hessian dynamics
    straightforward.

    Usage::

        python base_train.py config=cifar100_small

    Individual fields can still be overridden via the usual CLI syntax::

        python base_train.py config=cifar100_small --lr 5e-4 max_iters=2000
    """

    # ----- Data -----
    dataset: str = "cifar100"
    num_classes: int = 100
    # ----- Image / patch geometry -----
    img_size: int = 32
    patch_size: int = 4
    # ----- Architecture (NanoGPT-small parity) -----
    model_name: str = "vit_small_patch16_224"
    depth: int = 6
    num_heads: int = 6
    embed_dim: int = 384

    # ----- Output -----
    out_dir: str = "out/cifar100_d6"
    wandb_run_name: str = "cifar100_d6"


@dataclass
class ViTBaseImageNet1kConfig(TrainConfig):
    """
    ViT-B/16 trained from scratch on ImageNet-1k.

    Architecture (standard ViT-Base):
      - 12 transformer layers, 12 heads, 768-dim embeddings
      - patch_size = 16 on 224×224 images → (224/16)² = 196 patches

    Training recipe follows DeiT-style defaults (Touvron et al., 2021)
    adapted for the entropy-collapse experimental loop.

    Usage::

        python base_train.py config=imagenet1k_base

    Individual fields can still be overridden::

        python base_train.py config=imagenet1k_base --lr 5e-4 max_iters=50000
    """

    # ----- Data -----
    dataset: str = "imagenet1k"
    num_classes: int = 1000
    batch_size: int = 128

    # ----- Image / patch geometry -----
    img_size: int = 224
    patch_size: int = 16

    # ----- Architecture (ViT-Base) -----
    model_name: str = "vit_base_patch16_224"
    depth: int = 12
    num_heads: int = 12
    embed_dim: int = 768

    # ----- Optimiser -----
    learning_rate: float = 1e-3
    weight_decay: float = 0.05
    beta2: float = 0.999
    eps: float = 1e-8

    # ----- LR schedule -----
    max_iters: int = 10000
    warmup_iters: int = 500
    lr_decay_iters: int = 10000
    min_lr: float = 1e-6

    # ----- Output -----
    out_dir: str = "out/imagenet1k_vitb16"
    wandb_run_name: str = "imagenet1k_vitb16"


@dataclass
class ViTCurrentImageNet1kConfig(TrainConfig):
    """
    ViT-Base/16 trained from scratch on ImageNet-1k with a modern recipe.

    This config fixes the flat-loss issue of ``ViTBaseImageNet1kConfig`` by
    adopting DeiT-style (Touvron et al., 2021) training defaults:

      * **init_std = 0.02** — the standard ViT initialisation scale.  The
        previous ``0.002`` trapped the model in a near-zero dead zone where
        all logits (and therefore softmax outputs) are near-uniform, causing
        the loss to stay flat at ≈ ln(1000) ≈ 6.908.
      * **label_smoothing = 0.1** — softens hard one-hot targets, improving
        generalisation and stabilising early training.
      * **Longer training schedule** — ``max_iters = 50000`` with a 2000-step
        warm-up gives the optimiser enough runway to converge.

    Architecture (standard ViT-Base):
      - 12 transformer layers, 12 heads, 768-dim embeddings
      - patch_size = 16 on 224×224 images → (224/16)² = 196 patches

    Usage::

        python base_train.py config=imagenet1k_current

    Individual fields can still be overridden::

        python base_train.py config=imagenet1k_current --lr 5e-4 max_iters=100000
    """

    # ----- Data -----
    dataset: str = "imagenet1k"
    num_classes: int = 1000
    batch_size: int = 256

    # ----- Architecture -----
    model_name: str = "vit_base_patch16_224"

    # ----- Initialisation -----
    init_std: float = 0.02
    use_scaled_init: bool = True

    # ----- Optimiser -----
    learning_rate: float = 1e-3
    weight_decay: float = 0.05
    beta2: float = 0.999          # base default 0.95 is for GPT; ViT needs 0.999
    eps: float = 1e-8
    label_smoothing: float = 0.1

    # ----- LR schedule -----
    max_iters: int = 50000
    warmup_iters: int = 5000
    lr_decay_iters: int = 50000
    min_lr: float = 1e-5

    # ----- Output -----
    out_dir: str = "out/imagenet1k_current"
    wandb_run_name: str = "imagenet1k_current"


# Registry mapping preset names to config classes.  Add entries here to
# expose additional presets to the ``config=<name>`` CLI argument.
CONFIGS: dict[str, type[TrainConfig]] = {
    "default": TrainConfig,
    "cifar100_small": ViTSmallCIFAR100Config,
    "imagenet1k_base": ViTBaseImageNet1kConfig,
    "imagenet1k_current": ViTCurrentImageNet1kConfig,
}
