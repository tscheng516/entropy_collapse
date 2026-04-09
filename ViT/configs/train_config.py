"""
Training configuration for the ViT entropy-collapse experiments.

All flags that control learning rate, model initialisation, training
dynamics, Hessian metric computation frequency, and attention entropy
logging are gathered here in a single dataclass so that experiments are
fully reproducible from a printed config snapshot.

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

    # Architecture overrides (passed as kwargs to timm.create_model).
    # ``None`` means "use the timm model's built-in default".
    depth: Optional[int] = None
    """
    Number of transformer layers.  ``None`` uses the timm model default
    (e.g. 12 for ``vit_small_patch16_224``).
    Set to 6 to match the NanoGPT-small depth (``n_layer=6``).
    """

    num_heads: Optional[int] = None
    """
    Number of attention heads per layer.  ``None`` uses the timm model
    default (e.g. 6 for ``vit_small_patch16_224``).
    Set to 6 to match NanoGPT-small (``n_head=6``).
    """

    embed_dim: Optional[int] = None
    """
    Embedding / hidden dimension.  ``None`` uses the timm model default
    (e.g. 384 for ``vit_small_patch16_224``).
    Set to 384 to match NanoGPT-small (``n_embd=384``).
    """

    patch_size: Optional[int] = None
    """
    Patch size for the patch-embedding convolution.  ``None`` uses the
    timm model default (e.g. 16 for ``vit_small_patch16_224``).
    Set to 4 with ``img_size=32`` for native CIFAR resolution: (32/4)² = 64
    patches, which matches NanoGPT-small's context length (``block_size=64``).
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
    # Temperature-shift intervention
    # ------------------------------------------------------------------ #
    temp_shift_step: int = -1
    """
    Training step at which a one-time temperature-shift intervention is
    applied to all attention heads.  A value of -1 (default) disables
    the intervention entirely.  When set to a non-negative integer, at
    that iteration the attention logit scale is divided by
    ``temp_shift_factor``, making the softmax distribution softer
    (higher entropy) for factor > 1 and sharper (lower entropy) for
    factor < 1.  This provides an alternative to large-LR perturbations
    for producing spikes in the Hessian / entropy metrics.
    """

    temp_shift_factor: float = 2.0
    """
    Multiplicative factor applied to the attention temperature at
    ``temp_shift_step``.  Values > 1 soften the attention distribution
    (entropy increases); values < 1 sharpen it (entropy decreases).
    A value of 2.0 matches the typical intervention magnitude used in
    the entropy-collapse literature.
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
    """Native CIFAR resolution — no upsampling needed with patch_size=4."""
    patch_size: int = 4
    """4×4 patches on 32×32 images → 64 patches (= NanoGPT block_size=64)."""

    # ----- Architecture (NanoGPT-small parity) -----
    model_name: str = "vit_small_patch16_224"
    """timm base template; depth/embed_dim/num_heads are overridden below."""
    depth: int = 6
    """6 transformer layers — matches NanoGPT-small n_layer=6."""
    num_heads: int = 6
    """6 attention heads — matches NanoGPT-small n_head=6."""
    embed_dim: int = 384
    """384-dim embeddings — matches NanoGPT-small n_embd=384."""

    # ----- Output -----
    out_dir: str = "out/cifar100_d6"
    wandb_run_name: str = "cifar100_d6"


# Registry mapping preset names to config classes.  Add entries here to
# expose additional presets to the ``config=<name>`` CLI argument.
CONFIGS: dict[str, type[TrainConfig]] = {
    "default": TrainConfig,
    "cifar100_small": ViTSmallCIFAR100Config,
}
