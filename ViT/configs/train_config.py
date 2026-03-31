"""
Training configuration for the ViT entropy-collapse experiments.

All flags that control learning rate, model initialisation, training
dynamics, Hessian metric computation frequency, and attention entropy
logging are gathered here in a single dataclass so that experiments are
fully reproducible from a printed config snapshot.
"""

from dataclasses import dataclass


@dataclass
class TrainConfig:
    # ------------------------------------------------------------------ #
    # I/O
    # ------------------------------------------------------------------ #
    out_dir: str = "vit_out"
    """Directory for checkpoints and logs."""

    eval_interval: int = 200
    """How often (in iterations) to evaluate on the validation set."""

    log_interval: int = 1
    """How often (in iterations) to log train loss to stdout / wandb."""

    checkpoint_interval: int = 500
    """Save a checkpoint every N iterations (in addition to best-loss saves)."""

    always_save_checkpoint: bool = True
    """If True, save a checkpoint after every eval, not just when val loss improves."""

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
    """Enable Weights & Biases logging."""

    wandb_project: str = "entropy-collapse-vit"
    """W&B project name."""

    wandb_run_name: str = "run"
    """W&B run name (auto-appended with timestamp when empty)."""

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    dataset: str = "cifar10"
    """
    Dataset to use.  Supported values:
      'cifar10'   — CIFAR-10 (auto-downloaded, 10 classes, 32×32 → resized).
      'cifar100'  — CIFAR-100 (auto-downloaded, 100 classes).
      'imagenet'  — ImageNet-1k (requires data_dir with train/ and val/ folders).
    """

    data_dir: str = "./data"
    """
    Root directory for dataset storage.
    For CIFAR-10/100: the torchvision download root.
    For ImageNet: should contain ``train/`` and ``val/`` sub-folders
    in ImageFolder format.
    """

    batch_size: int = 64
    """Number of images per batch."""

    num_workers: int = 4
    """Number of DataLoader worker processes."""

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

    num_classes: int = 10
    """Number of output classes (10 for CIFAR-10, 100 for CIFAR-100, 1000 for ImageNet)."""

    img_size: int = 224
    """
    Spatial resolution fed to the model.
    CIFAR images (32×32) are upsampled to this size by the data pipeline.
    """

    # ------------------------------------------------------------------ #
    # Weight initialisation
    # ------------------------------------------------------------------ #
    init_std: float = 0.002
    """
    Standard deviation for all weight matrices when ``init_from='scratch'``.
    A small value places the model near the origin, making early-training
    Hessian dynamics easier to observe.
    """

    use_scaled_init: bool = True
    """
    If True, scale each attention output-projection (``attn.proj``)
    weight by ``init_std / sqrt(2 * n_layers)`` — the ViT analogue of
    NanoGPT's residual-depth scaling.
    """

    qk_norm: bool = False
    """
    If False (default), query and key projections use no normalisation —
    matching the NanoGPT / LLM experiment setup where no QK-norm is applied.
    If True, per-head LayerNorm is installed on the query and key projections
    via timm's native ``qk_norm`` option.
    """

    # ------------------------------------------------------------------ #
    # Optimiser
    # ------------------------------------------------------------------ #
    optimizer: str = "adamw"
    """'adamw' or 'sgd'."""

    learning_rate: float = 1e-3
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
    hessian_freq: int = 10
    """
    Compute all Hessian proxy metrics (λ_max, H_tilde, H_VV, H_GN, FD)
    every N iterations.  Use a larger value to reduce overhead.
    """

    hessian_max_iter: int = 10
    """Number of power-iteration steps for λ_max estimation."""

    compute_fd: bool = True
    """
    Whether to compute the Finite-Difference sharpness proxy.
    This requires an extra forward/backward pass and can be disabled
    when running on tight compute budgets.
    """

    # ------------------------------------------------------------------ #
    # Attention entropy
    # ------------------------------------------------------------------ #
    entropy_freq: int = 10
    """
    Compute per-layer attention entropy every N iterations.
    Logged both to stdout and to wandb as entropy/layer_<k>.
    """

    # ------------------------------------------------------------------ #
    # Spike detection / MAD analysis
    # ------------------------------------------------------------------ #
    z_score: float = 3.0
    """MAD z-score multiplier for spike detection (used in spike plots)."""

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
