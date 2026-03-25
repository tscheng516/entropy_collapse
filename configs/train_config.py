"""
Training configuration for the NanoGPT entropy-collapse experiments.

All flags that control learning rate, model init, training dynamics,
Hessian metric computation frequency, and attention entropy logging
are gathered here in a single dataclass so that experiments are fully
reproducible from a printed config snapshot.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainConfig:
    # ------------------------------------------------------------------ #
    # I/O
    # ------------------------------------------------------------------ #
    out_dir: str = "out"
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
    '<path>'   — load a NanoGPT-style checkpoint file and fine-tune.
    """

    # ------------------------------------------------------------------ #
    # Weights & Biases
    # ------------------------------------------------------------------ #
    wandb_log: bool = True
    """Enable Weights & Biases logging."""

    wandb_project: str = "entropy-collapse"
    """W&B project name."""

    wandb_run_name: str = "run"
    """W&B run name (auto-appended with timestamp when empty)."""

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    dataset: str = "shakespeare_char"
    """Dataset name (subdirectory inside data_dir, or a custom label)."""

    data_dir: str = "nanoGPT/data/shakespeare_char"
    """Path to the folder containing train.bin (and optionally val.bin)."""

    batch_size: int = 32
    """Number of sequences per batch."""

    block_size: int = 64
    """Context length (tokens per sequence)."""

    # ------------------------------------------------------------------ #
    # Model architecture
    # ------------------------------------------------------------------ #
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    bias: bool = False
    vocab_size: int = 65
    """
    Vocab size for character-level Shakespeare (65 unique chars).
    Set to 50257 for GPT-2 tokeniser.
    """

    # ------------------------------------------------------------------ #
    # Weight initialisation
    # ------------------------------------------------------------------ #
    init_std: float = 0.002
    """
    Standard deviation for all weight matrices.
    A small value (0.002) places the model near the origin, making
    early-training Hessian dynamics easier to observe.
    """

    use_scaled_init: bool = True
    """
    If True, scale c_proj weights by 1/sqrt(2 * n_layer) (NanoGPT-style
    residual depth scaling).
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
    AdamW epsilon.  A slightly larger value than PyTorch's default (1e-8)
    improves numerical stability at the small weight magnitudes used in
    these stability experiments.
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
