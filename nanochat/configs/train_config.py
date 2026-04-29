"""
Training configuration for nanochat entropy-collapse experiments.

Depth is the single complexity dial, just like nanochat itself.
All other architectural hyperparameters follow it automatically via the
named preset dataclasses.  ``TrainConfig`` is a minimal d12 research run
(~85 M parameters, ~5 min on 8×H100).  Larger presets mirror the nanochat
speedrun configurations.

Named presets  (n_embd = depth×64, n_head = n_embd/128 — nanochat formula)
-------------
| Key   | n_layer | n_embd | n_head | n_kv_head | ~params | Use case              |
|-------|---------|--------|--------|-----------|---------|----------------------|
| d8    |  8      |  512   |  4     |  4        |  ~42 M  | Quick iteration       |
| d12   | 12      |  768   |  6     |  6        | ~110 M  | Default research run  |
| d24   | 24      | 1536   | 12     | 12        | ~730 M  | GPT-2 scale speedrun  |

max_iters calibrated for param:data ratio ≈ 10 on 4 GPUs (train.sh default),
batch_size=8 per GPU.  Scale proportionally for other GPU counts.

Select a preset on the CLI::

    python nanochat/base_train.py config=d12
    python nanochat/base_train.py config=d24 --wandb true
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class TrainConfig:
    # ------------------------------------------------------------------ #
    # I/O
    # ------------------------------------------------------------------ #
    out_dir: str = "out/nanochat/pilot"
    eval_interval: int = 500
    log_interval: int = 10
    checkpoint_interval: int = -1          # -1 = disabled
    save_checkpoint: bool = False
    init_from: str = "scratch"
    # 'scratch' | 'resume' | '<path-to-checkpoint>'

    # ------------------------------------------------------------------ #
    # Weights & Biases
    # ------------------------------------------------------------------ #
    wandb_log: bool = False
    wandb_project: str = "entropy-collapse-nanochat"
    wandb_run_name: str = "run"

    # ------------------------------------------------------------------ #
    # Data — ClimbMix shards loaded via nanochat's tokenizing data loader.
    # Run ``python -m nanochat.dataset`` inside the nanochat clone to
    # download the parquet shards before launching training.
    # ------------------------------------------------------------------ #
    nanochat_dir: str = "nanochat/nanochat_repo"
    # Path to the cloned nanochat repo relative to the repo root.
    # Override with the absolute path if you cloned elsewhere, e.g.
    #   nanochat_dir=/path/to/nanochat_repo

    # ------------------------------------------------------------------ #
    # Model — nanochat GPTConfig fields
    # ------------------------------------------------------------------ #
    n_layer: int = 12
    n_head: int = 6               # query heads
    n_kv_head: int = 6            # key/value heads (GQA; set < n_head for MQA)
    n_embd: int = 768
    sequence_len: int = 512
    # Keep sequence_len short (512) for research runs; set to 2048 for
    # full-context pre-training.  Hessian memory scales O(B·T²) per layer.
    vocab_size: int = 32768       # nanochat BPE tokenizer
    window_pattern: str = "SSSL"
    # Sliding window pattern: 'L'=full context, 'S'=quarter context.
    # The patched attention ignores this and always uses full causal context
    # (sliding windows are incompatible with explicit softmax + second-order
    # autograd).  window_pattern is kept so the GPTConfig round-trips.

    # ------------------------------------------------------------------ #
    # Weight initialisation
    # ------------------------------------------------------------------ #
    init_std: float = 0.02
    # nanochat uses uniform init in init_weights(); this field allows
    # re-initialising with a smaller std for entropy-collapse pilot tests.
    # Set to None to use nanochat's native init_weights() unchanged.
    use_scaled_init: bool = False
    # Whether to apply residual-depth scaling (NanoGPT-style) on top of
    # nanochat's init.  Disabled by default to stay close to nanochat's
    # training setup.

    # ------------------------------------------------------------------ #
    # Optimiser
    # ------------------------------------------------------------------ #
    optimizer: str = "muon_adamw"
    # 'muon_adamw'  — nanochat's Muon+AdamW (nanochat default; fastest).
    #                 prec_H applies only to AdamW param groups; Muon
    #                 groups fall back to raw H.
    #                 dtype compat: fp32 only — nanochat's compiled
    #                 adamw_step_fused lerp_ crashes with bf16 embeddings.
    # 'adamw'       — plain AdamW on all params.  Slower but prec_H works
    #                 everywhere.  Compatible with both fp32 AND bf16:
    #                 PyTorch's native AdamW handles mixed-dtype grads
    #                 (wte + value_embeds stay bf16, rest fp32).

    # AdamW shared fields (used by both optimizer paths)
    learning_rate: float = 3e-4     # AdamW: single LR for all param groups
    max_iters: int = 2000
    weight_decay: float = 0.28      # nanochat default; cosine-decayed to 0
                                    # over training for muon_adamw
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eps: float = 1e-8

    # MuonAdamW hyperparameters — passed directly to model.setup_optimizer().
    # Per-group LRs from nanochat scripts/base_train.py defaults.
    # The trapezoidal schedule scales all groups via their ``initial_lr``.
    muon_matrix_lr: float = 0.02        # Muon: transformer.h matrices
    muon_embedding_lr: float = 0.3      # AdamW: wte + value_embeds
    muon_unembedding_lr: float = 0.008  # AdamW: lm_head
    muon_scalar_lr: float = 0.5         # AdamW: resid/x0/smear scalars
    muon_ns_steps: int = 5              # Newton-Schulz / Polar Express iters
    # Muon momentum is scheduled dynamically in base_train.py (0.85→0.97→0.90);
    # it is not stored here.  WD is also cosine-scheduled from weight_decay→0.

    # ------------------------------------------------------------------ #
    # LR schedule — trapezoidal (nanochat default)
    # linear warm-up → constant plateau → linear warm-down
    # ------------------------------------------------------------------ #
    warmup_iters: int = 500
    warmdown_ratio: float = 0.65    # fraction of max_iters spent in warmdown
    min_lr_frac: float = 0.05       # final LR as fraction of initial LR

    # ------------------------------------------------------------------ #
    # Hessian metrics
    # ------------------------------------------------------------------ #
    hessian_intv: int = 500
    hessian_max_iter: int = 10
    hessian_batch_size: int = 2
    # Keep small for long sequences: attention memory ∝ B·T².
    # 2 sequences at T=512 uses ~50 MB for attention during HVP; safe
    # on any GPU.  Increase to 4–8 if your GPU has ≥40 GB.
    compute_fd: bool = False
    label_smoothing: float = 0.0

    # ------------------------------------------------------------------ #
    # Attention entropy
    # ------------------------------------------------------------------ #
    entropy_intv: int = 500

    # ------------------------------------------------------------------ #
    # Temperature-shift intervention
    # ------------------------------------------------------------------ #
    temp_shift_step: int = -1       # -1 = disabled
    temp_shift_factor: float = 0.25

    # ------------------------------------------------------------------ #
    # Compute / device
    # ------------------------------------------------------------------ #
    device: str = "cuda"
    compile: bool = False           # disable when computing 2nd-order grads
    seed: int = 1337
    num_workers: int = 4            # tokenizer threads for nanochat dataloader
    batch_size: int = 8             # sequences per GPU per step
    compute_dtype: str = "fp32"
    # 'fp32' — float32 master weights throughout (default, recommended).
    #           Avoids a dtype mismatch in nanochat's @torch.compile'd
    #           ``adamw_step_fused``: when COMPUTE_DTYPE=bf16, wte and
    #           value_embeds are cast to bf16, their gradients are bf16, but
    #           the optimizer's hyperparameter 0-D tensors stay float32.
    #           torch.compile traces ``lerp_(bf16, bf16, float32)`` → crash.
    # 'bf16' — bfloat16 embeddings/activations (nanochat speedrun setting).
    #           Only safe with optimizer='adamw' (PyTorch AdamW handles mixed
    #           dtypes); muon_adamw + bf16 will hit the compiled lerp_ bug.

    # skip_intv: carry-forward mode for plot_history (True = interval-skip)
    skip_intv: bool = False


# ---------------------------------------------------------------------------
# Named preset configs
# ---------------------------------------------------------------------------

@dataclass
class D8Config(TrainConfig):
    """Fast pilot: 8-layer GPT, ~30 M params.  Trains in ~3 min on 8×H100.

    Usage::

        python nanochat/base_train.py config=d8
    """
    n_layer: int = 8
    n_head: int = 4
    n_kv_head: int = 4
    n_embd: int = 512
    sequence_len: int = 512

    out_dir: str = "out/nanochat/d8"
    wandb_project: str = "entropy-collapse-nanochat-d8"
    wandb_run_name: str = time.strftime("%Y%m%d-%H%M%S")

    # max_iters calibrated for ratio=10 at 4 GPUs, B=8, T=512.
    # Scale: ×2 for 2 GPUs, ×8 for 1 GPU.
    max_iters: int = 25_000
    warmup_iters: int = 250           # ~1% of max_iters
    learning_rate: float = 5e-4

    hessian_intv: int = 250
    entropy_intv: int = 50


@dataclass
class D12Config(TrainConfig):
    """Standard research run: 12-layer GPT, ~85 M params.

    Usage::

        python nanochat/base_train.py config=d12
    """
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    sequence_len: int = 512

    out_dir: str = "out/nanochat/d12"
    wandb_log: bool = True
    wandb_project: str = "entropy-collapse-nanochat-d12"
    wandb_run_name: str = time.strftime("%Y%m%d-%H%M%S")

    batch_size: int = 8
    # max_iters calibrated for ratio=10 at 4 GPUs, B=8, T=512.
    # Scale: ×2 for 2 GPUs, ×8 for 1 GPU.
    max_iters: int = 65_000
    warmup_iters: int = 650           # ~1% of max_iters
    learning_rate: float = 3e-4

    hessian_intv: int = 500
    entropy_intv: int = 100


@dataclass
class D24Config(TrainConfig):
    """GPT-2 scale: 24-layer GPT, ~350 M params.  Mirrors nanochat speedrun d24.

    Usage::

        python nanochat/base_train.py config=d24 --wandb true

    Note: Reduce hessian_batch_size to 1 on GPUs with < 80 GB if you OOM
    during curvature computation.
    """
    # nanochat formula: n_embd=24×64=1536, n_head=1536/128=12, n_kv_head=12
    n_layer: int = 24
    n_head: int = 12
    n_kv_head: int = 12
    n_embd: int = 1536
    sequence_len: int = 2048

    out_dir: str = "out/nanochat/d24"
    wandb_log: bool = True
    wandb_project: str = "entropy-collapse-nanochat-d24"
    wandb_run_name: str = time.strftime("%Y%m%d-%H%M%S")

    batch_size: int = 8
    # max_iters calibrated for ratio=10 at 4 GPUs, B=8, T=2048.
    # Scale: ×2 for 2 GPUs, ×8 for 1 GPU.
    max_iters: int = 110_000
    warmup_iters: int = 1000          # ~1% of max_iters
    learning_rate: float = 2e-4

    hessian_intv: int = 1000
    hessian_batch_size: int = 1   # large model + long seq → keep at 1
    entropy_intv: int = 200


# Registry used by CLI ``config=<key>``
CONFIGS: dict[str, type] = {
    "d8":  D8Config,
    "d12": D12Config,
    "d24": D24Config,
}
