"""
Training configuration for nanochat entropy-collapse experiments.

Depth is the single complexity dial, just like nanochat itself.
All other architectural hyperparameters follow it automatically via the
named preset dataclasses.  ``TrainConfig`` is a minimal d12 research run
(~85 M parameters, ~5 min on 8×H100).  Larger presets mirror the nanochat
speedrun configurations.

Named presets
-------------
| Key   | n_layer | n_embd | n_head | n_kv_head | ~params | Use case              |
|-------|---------|--------|--------|-----------|---------|----------------------|
| d8    |  8      |  512   |  4     |  4        |  ~30 M  | Quick iteration       |
| d12   | 12      |  768   |  6     |  6        |  ~85 M  | Default research run  |
| d24   | 24      | 1024   |  8     |  4        | ~350 M  | GPT-2 scale speedrun  |

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
    # 'muon_adamw'  — nanochat's Muon+AdamW (nanochat default; prec_H
    #                 applies only to AdamW param groups, Muon groups fall
    #                 back to raw H)
    # 'adamw'       — plain AdamW on all params (slower but prec_H works
    #                 everywhere; useful for ablations)

    # AdamW-only case: single LR applied to all param groups
    learning_rate: float = 3e-4
    max_iters: int = 2000
    weight_decay: float = 0.0      # nanochat default; set >0 for regularisation
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eps: float = 1e-8

    # MuonAdamW hyperparameters — passed directly to model.setup_optimizer().
    # Each param group gets its own tuned LR; the cosine schedule scales
    # all groups proportionally via their stored ``initial_lr``.
    muon_matrix_lr: float = 0.02        # Muon groups: transformer.h matrices
    muon_embedding_lr: float = 0.2      # AdamW: token embedding + value embeds
    muon_unembedding_lr: float = 0.004  # AdamW: lm_head
    muon_scalar_lr: float = 0.5         # AdamW: resid_lambdas, x0_lambdas, smear
    muon_momentum: float = 0.95         # Muon momentum coefficient
    muon_ns_steps: int = 5              # Newton-Schulz / Polar Express iterations
    muon_beta2: float = 0.9             # Muon variance-reduction beta

    # ------------------------------------------------------------------ #
    # LR schedule — cosine decay with linear warm-up
    # ------------------------------------------------------------------ #
    decay_lr: bool = True
    warmup_iters: int = 200
    lr_decay_iters: int = 2000
    min_lr: float = 3e-5

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

    max_iters: int = 2000
    warmup_iters: int = 200
    lr_decay_iters: int = 2000
    learning_rate: float = 5e-4
    min_lr: float = 5e-5

    hessian_intv: int = 200
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
    max_iters: int = 10000
    warmup_iters: int = 500
    lr_decay_iters: int = 10000
    learning_rate: float = 3e-4
    min_lr: float = 3e-5

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
    n_layer: int = 24
    n_head: int = 8
    n_kv_head: int = 4            # GQA: 2× more query heads than KV heads
    n_embd: int = 1024
    sequence_len: int = 2048

    out_dir: str = "out/nanochat/d24"
    wandb_log: bool = True
    wandb_project: str = "entropy-collapse-nanochat-d24"
    wandb_run_name: str = time.strftime("%Y%m%d-%H%M%S")

    batch_size: int = 8
    max_iters: int = 50000
    warmup_iters: int = 2000
    lr_decay_iters: int = 50000
    learning_rate: float = 2e-4
    min_lr: float = 2e-5

    hessian_intv: int = 1000
    hessian_batch_size: int = 1   # large model + long seq → keep at 1
    entropy_intv: int = 200


# Registry used by CLI ``config=<key>``
CONFIGS: dict[str, type] = {
    "d8":  D8Config,
    "d12": D12Config,
    "d24": D24Config,
}
