"""
Training configuration for nanochat entropy-collapse experiments.

Depth is the single complexity dial, just like nanochat itself.
All other architectural hyperparameters follow it automatically via the
named preset dataclasses.  ``TrainConfig`` is a minimal d6 research run
(~100 M parameters).  Larger presets mirror the nanochat
speedrun configurations.

Named presets  (n_embd = depth×64, n_head = n_embd/128 — nanochat formula)
-------------
| Key   | n_layer | n_embd | n_head  | Use case             |
|-------|---------|--------|---------|----------------------|
| d8    |  8      |  512   |  4      |  Quick iteration     |
| d12   | 12      |  768   |  6      | Default research run |
| d24   | 24      | 1536   | 12      | GPT-2 scale speedrun |

max_iters calibrated for param:data ratio ≈ 10 on 4 GPUs (train.sh default),
batch_size=8 per GPU.  Scale proportionally for other GPU counts.

Select a preset on the CLI::

    python nanochat/base_train.py config=d12
    python nanochat/base_train.py config=d24 --seq_len 2048 
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
    n_layer: int = 6
    n_head: int = 3               # query heads
    n_kv_head: int = 3            # key/value heads (GQA; set < n_head for MQA)
    n_embd: int = 384
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
    # Optimiser
    # ------------------------------------------------------------------ #
    optimizer: str = "muon_adamw"
    # 'muon_adamw'  — nanochat's Muon+AdamW (nanochat default; fastest).
    #                 Matrix params use Muon; scalar params use AdamW with
    #                 fixed per-group LRs/betas/wd from gpt.py.
    #                 dtype compat: fp32 only — nanochat's compiled
    #                 adamw_step_fused lerp_ crashes with bf16 embeddings.
    # 'adamw'       — pure AdamW.  Replaces Muon on matrix params while
    #                 keeping scalar groups (embeddings, lm_head, scalars)
    #                 identical to muon_adamw (same fixed LRs/betas/wd from
    #                 gpt.py).  Compatible with both fp32 and bf16.

    # Shared optimiser fields — control the matrix param group in both paths.
    # learning_rate = matrix_lr: Muon LR (muon_adamw) or AdamW LR (adamw)
    #   for transformer weight matrices.  nanochat default: 0.02.
    learning_rate: float = 0.02
    max_iters: int = 2000
    weight_decay: float = 0.28      # weight decay for matrix params;
                                    # cosine-decayed to 0 for muon_adamw.
                                    # scalar params use fixed wd from gpt.py.
    beta1: float = 0.9              # AdamW beta1 (adamw) / Muon momentum (muon_adamw)
    beta2: float = 0.95             # AdamW beta2 (adamw) / Muon beta2 (muon_adamw)

    # Muon momentum is scheduled dynamically in base_train.py (0.85→0.97→0.90).
    # WD is cosine-scheduled from weight_decay→0 for muon_adamw.
    # Scalar param groups (embeddings, lm_head, scalars) use the defaults
    # defined in nanochat gpt.py setup_optimizer() and are not overridden.

    # ------------------------------------------------------------------ #
    # Hessian metrics
    # ------------------------------------------------------------------ #
    hessian_intv: int = 50
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
    entropy_intv: int = 50

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


# ---------------------------------------------------------------------------
# Named preset configs
# ---------------------------------------------------------------------------

@dataclass
class D8Config(TrainConfig):
    n_layer: int = 8
    n_head: int = 4
    n_kv_head: int = 4
    n_embd: int = 512
    sequence_len: int = 512

    out_dir: str = "out/nanochat/d8"
    wandb_log: bool = True
    wandb_project: str = "entropy-collapse-nanochat"
    wandb_run_name: str = time.strftime("%Y%m%d-%H%M%S")

    batch_size: int = 8
    max_iters: int = 20_000

    hessian_batch_size: int = 4  


@dataclass
class D12Config(TrainConfig):
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    sequence_len: int = 1024

    out_dir: str = "out/nanochat/d12"
    wandb_log: bool = True
    wandb_project: str = "entropy-collapse-nanochat"
    wandb_run_name: str = time.strftime("%Y%m%d-%H%M%S")

    batch_size: int = 8
    max_iters: int = 20_000
    
    hessian_batch_size: int = 2  



@dataclass
class D24Config(TrainConfig):
    n_layer: int = 24
    n_head: int = 12
    n_kv_head: int = 12
    n_embd: int = 1536
    sequence_len: int = 2048

    out_dir: str = "out/nanochat/d24"
    wandb_log: bool = True
    wandb_project: str = "entropy-collapse-nanochat"
    wandb_run_name: str = time.strftime("%Y%m%d-%H%M%S")

    batch_size: int = 8
    max_iters: int = 20_000

    hessian_batch_size: int = 1   # large model + long seq → keep at 1



# Registry used by CLI ``config=<key>``
CONFIGS: dict[str, type] = {
    "d8":  D8Config,
    "d12": D12Config,
    "d24": D24Config,
}
