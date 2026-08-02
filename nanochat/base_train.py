"""
base_train.py — nanochat entropy-collapse training script.

Trains a nanochat GPT with HookedGPT attention caching, logging per-layer
attention entropy and nine curvature proxy metrics (λ_max of H, Prec_H,
H_VV, GN, Diag_H, Fisher, + KFAC / BFGS / FD when compute_fd=True).


Usage
-----
Pilot (d6):

    python base_train.py

Named preset::

    python base_train.py config=d12

Override individual fields::

    python base_train.py config=d12 learning_rate=1e-4 max_iters=5000

Multi-GPU via torchrun::

    torchrun --nproc_per_node=4 base_train.py config=d12

All config fields can be overridden as ``key=value`` arguments.

"""

from __future__ import annotations

import os
import sys
import math
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

# True on rank-0 (or single-GPU); works before dist.init_process_group.
_is_master = int(os.environ.get("RANK", "0")) == 0

# ---------------------------------------------------------------------------
# 0.  Path setup — add nanochat/ so ``configs`` and ``src`` sub-packages
#     resolve.  The nanochat repo itself is added after cfg is parsed.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# ---------------------------------------------------------------------------
# 1.  Configuration
# ---------------------------------------------------------------------------
from configs.train_config import TrainConfig, CONFIGS  # noqa: E402
from common.train_utils import (  # noqa: E402
    resolve_config,
    setup_ddp_and_run_dir,
    wrap_model_ddp,
)
from common.helpers import strip_compile_prefix  # noqa: E402
from common.pretrain import run_training  # noqa: E402

cfg = resolve_config(TrainConfig, CONFIGS, _is_master)

# ---------------------------------------------------------------------------
# 2.  Add nanochat repo to sys.path so ``nanochat.*`` is importable
# ---------------------------------------------------------------------------
_nanochat_dir = os.path.join(REPO_ROOT, cfg.nanochat_dir)
if not os.path.isdir(_nanochat_dir):
    raise FileNotFoundError(
        f"nanochat_dir='{cfg.nanochat_dir}' resolves to '{_nanochat_dir}' "
        "which does not exist.  Clone the nanochat repo there or override "
        "nanochat_dir=<absolute-path> on the CLI."
    )
if _nanochat_dir not in sys.path:
    sys.path.insert(0, _nanochat_dir)

# ---------------------------------------------------------------------------
# 2b. Patch nanochat.common.COMPUTE_DTYPE BEFORE any nanochat module is
#     imported.  nanochat's gpt.py and optim.py do
#     ``from nanochat.common import COMPUTE_DTYPE`` at module load time, so
#     the patch must land in sys.modules before their first import.
#
#     Root cause of the bf16 crash:
#       init_weights() casts wte + value_embeds to COMPUTE_DTYPE (bf16).
#       Their gradients are also bf16.  adamw_step_fused is
#       @torch.compile(fullgraph=True) and calls
#           exp_avg.lerp_(grad, 1 - beta1_t)
#       where beta1_t is a float32 0-D CPU tensor.  torch.compile traces
#       lerp_(bf16, bf16, float32) → TorchRuntimeError dtype mismatch.
#       The Muon path avoids this with an explicit .to(stacked_grads.dtype)
#       cast; the AdamW path does not.
# ---------------------------------------------------------------------------
import nanochat.common as _nanochat_common  # noqa: E402

_nc_dtype = (
    torch.bfloat16
    if cfg.compute_dtype.lower() in ("bf16", "bfloat16")
    else torch.float32
)
_nanochat_common.COMPUTE_DTYPE = _nc_dtype

if _is_master:
    print(f"[dtype] nanochat COMPUTE_DTYPE → {_nc_dtype}")

if _nc_dtype == torch.bfloat16 and cfg.optimizer.lower() == "muon_adamw":
    if _is_master:
        print(
            "[warn] compute_dtype=bf16 + optimizer=muon_adamw will crash: "
            "nanochat's compiled adamw_step_fused does not cast its float32 "
            "hyperparameter scalars to bf16 embedding gradients "
            "(lerp_ dtype mismatch inside @torch.compile).  "
            "Use compute_dtype=fp32 for muon_adamw, or switch to "
            "optimizer=adamw which is safe with both fp32 and bf16."
        )

# ---------------------------------------------------------------------------
# 3.  Reproducibility & device
# ---------------------------------------------------------------------------
torch.set_float32_matmul_precision("high")  # enable TF32 on Ampere+
torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)

device = cfg.device
# nanochat manages dtype internally — we never wrap with autocast.
ctx = nullcontext()

os.makedirs(cfg.out_dir, exist_ok=True)

use_ddp, rank, world_size, local_rank, device, run_out_dir = setup_ddp_and_run_dir(cfg, _is_master)

# ---------------------------------------------------------------------------
# 4.  Data — ClimbMix via nanochat's tokenizing dataloader
# ---------------------------------------------------------------------------
from nanochat.tokenizer import get_tokenizer                              # noqa: E402
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit  # noqa: E402

# Loads RustBPETokenizer trained by ``python -m scripts.tok_train``
tokenizer = get_tokenizer()

# Each call returns a fresh generator yielding (inputs, targets) as (B, T)
# long tensors.  Use separate iterators for train and val.
def _make_loader(split: str):
    return tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer,
        cfg.batch_size,
        cfg.sequence_len,
        split,
        device=device,
    )

train_iter = _make_loader("train")
val_iter = _make_loader("val")

if _is_master:
    print(
        f"[data] nanochat ClimbMix  "
        f"batch_size={cfg.batch_size}  seq_len={cfg.sequence_len}  "
        f"device={device}"
    )

# ---------------------------------------------------------------------------
# 5.  Model
# ---------------------------------------------------------------------------
from nanochat.gpt import GPTConfig                  # noqa: E402
from src.model import build_hooked_gpt, set_attention_temperature  # noqa: E402

iter_num = 0
best_val_loss = float("inf")


def _make_gpt_cfg() -> GPTConfig:
    return GPTConfig(
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_kv_head=cfg.n_kv_head,
        n_embd=cfg.n_embd,
        sequence_len=cfg.sequence_len,
        vocab_size=cfg.vocab_size,
        window_pattern=cfg.window_pattern,
    )


if cfg.init_from == "scratch":
    if _is_master:
        print(
            f"[model] building nanochat GPT from scratch  "
            f"n_layer={cfg.n_layer}  n_embd={cfg.n_embd}  "
            f"n_head={cfg.n_head}  n_kv_head={cfg.n_kv_head} …"
        )
    gpt_cfg = _make_gpt_cfg()
    model = build_hooked_gpt(gpt_cfg, device=device)

elif cfg.init_from == "resume":
    ckpt_path = os.path.join(cfg.out_dir, "ckpt.pt")
    if _is_master:
        print(f"[model] resuming from {ckpt_path} …")
    checkpoint = torch.load(ckpt_path, map_location=device)
    # Restore GPTConfig fields from checkpoint so architecture matches.
    for _k in ("n_layer", "n_head", "n_kv_head", "n_embd", "sequence_len",
               "vocab_size", "window_pattern"):
        if _k in checkpoint:
            setattr(cfg, _k, checkpoint[_k])
    gpt_cfg = _make_gpt_cfg()
    model = build_hooked_gpt(gpt_cfg, device=device)
    model.load_state_dict(strip_compile_prefix(checkpoint["model"]))
    iter_num = checkpoint.get("iter_num", 0)
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))

else:
    if _is_master:
        print(f"[model] fine-tuning from checkpoint {cfg.init_from} …")
    checkpoint = torch.load(cfg.init_from, map_location=device)
    for _k in ("n_layer", "n_head", "n_kv_head", "n_embd", "sequence_len",
               "vocab_size", "window_pattern"):
        if _k in checkpoint:
            setattr(cfg, _k, checkpoint[_k])
    gpt_cfg = _make_gpt_cfg()
    model = build_hooked_gpt(gpt_cfg, device=device)
    model.load_state_dict(strip_compile_prefix(checkpoint["model"]))

n_params = sum(p.numel() for p in model.parameters()) / 1e6
if _is_master:
    print(f"[model] nanochat GPT  {n_params:.2f}M parameters")

if cfg.compile:
    if _is_master:
        print("[model] compiling with torch.compile (disable for Hessian metrics)")
    model = torch.compile(model)

model = wrap_model_ddp(model, use_ddp, device, local_rank)

# ---------------------------------------------------------------------------
# 6.  Optimiser
# ---------------------------------------------------------------------------
def _build_optimizer(model_: torch.nn.Module) -> torch.optim.Optimizer:
    opt_name = cfg.optimizer.lower()
    _raw = model_.module if use_ddp else model_
    if opt_name == "adamw":
        # Build param groups via nanochat's setup_optimizer so that scalar
        # groups (embeddings, lm_head, resid/x0/smear/backout) keep their
        # exact LRs, betas, and wd from gpt.py unchanged.  Only the Muon
        # matrix groups are replaced with AdamW using cfg settings.
        _ref_opt = _raw.setup_optimizer(
            matrix_lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        adamw_groups = []
        for pg in _ref_opt.param_groups:
            if pg.get("kind") == "muon":
                # Replace Muon group with AdamW using cfg-controlled settings.
                adamw_groups.append({
                    "params": pg["params"],
                    "lr": cfg.learning_rate,
                    "betas": (cfg.beta1, cfg.beta2),
                    "weight_decay": cfg.weight_decay,
                    "eps": 1e-8,
                })
            else:
                # Scalar AdamW group — preserve exact values from gpt.py.
                adamw_groups.append({
                    "params": pg["params"],
                    "lr": pg["lr"],
                    "betas": pg.get("betas", (0.8, 0.95)),
                    "weight_decay": pg.get("weight_decay", 0.0),
                    "eps": pg.get("eps", 1e-10),
                })
        opt = torch.optim.AdamW(adamw_groups)
        for pg in opt.param_groups:
            pg["initial_lr"] = pg["lr"]
        return opt
    elif opt_name == "muon_adamw":
        # Delegates to nanochat's model.setup_optimizer() which builds
        # MuonAdamW (or DistMuonAdamW for DDP) with per-group LRs.
        # Each group stores group["initial_lr"] so the trapezoidal schedule
        # can scale all groups proportionally.
        # Note: prec_H (Adam-preconditioned Hessian) only works for the AdamW
        # param groups; Muon groups fall back to the raw Hessian value.
        return _raw.setup_optimizer(
            matrix_lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
    else:
        raise ValueError(
            f"Unknown optimizer '{cfg.optimizer}'. "
            "Choose 'muon_adamw' (nanochat default) or 'adamw'."
        )


optimizer = _build_optimizer(model)

# ---------------------------------------------------------------------------
# 7.  LR / momentum / weight-decay schedules  (nanochat trapezoidal)
# ---------------------------------------------------------------------------
def get_lr_multiplier(it: int) -> float:
    """Trapezoidal LR multiplier in [min_lr_frac, 1.0].

    nanochat defaults: warmup_steps=40, warmdown_ratio=0.65, final_lr_frac=0.05.
    Multiply each param group's ``initial_lr`` by this value to get the
    actual LR for that step.
    """
    _warmup = 40
    _warmdown_ratio = 0.65
    _min_lr_frac = 0.05
    warmdown_iters = round(_warmdown_ratio * cfg.max_iters)
    warmdown_start = cfg.max_iters - warmdown_iters
    if it < _warmup:
        # Linear warmup from _min_lr_frac to 1.0
        return _min_lr_frac + (1.0 - _min_lr_frac) * (it + 1) / max(1, _warmup)
    elif it < warmdown_start:
        return 1.0
    else:
        # Linear warmdown from 1.0 to _min_lr_frac
        progress = (cfg.max_iters - it) / max(1, warmdown_iters)
        return _min_lr_frac + (1.0 - _min_lr_frac) * progress


def get_muon_momentum(it: int) -> float:
    """Muon momentum schedule (nanochat default).

    Ramps from 0.85 → 0.97 over the first 400 steps, stays at 0.97 through
    the plateau, then decays back to 0.90 during LR warmdown.
    """
    warmdown_iters = round(0.65 * cfg.max_iters)
    warmdown_start = cfg.max_iters - warmdown_iters
    if it < 400:
        frac = it / 400
        return (1 - frac) * 0.85 + frac * 0.97
    elif it >= warmdown_start:
        progress = (it - warmdown_start) / max(1, warmdown_iters)
        return 0.97 * (1 - progress) + 0.90 * progress
    else:
        return 0.97


def get_muon_wd(it: int) -> float:
    """Cosine weight-decay schedule for Muon: decays from cfg.weight_decay → 0."""
    return cfg.weight_decay * 0.5 * (1.0 + math.cos(math.pi * it / max(1, cfg.max_iters)))


# ---------------------------------------------------------------------------
# 8.  Loss/metrics step, LR/momentum/weight-decay schedule, and validation
# ---------------------------------------------------------------------------
def update_schedule(optimizer_: torch.optim.Optimizer, it: int) -> float:
    lrm = get_lr_multiplier(it)
    if cfg.optimizer.lower() == "muon_adamw":
        muon_mom = get_muon_momentum(it)
        muon_wd = get_muon_wd(it)
        for pg in optimizer_.param_groups:
            pg["lr"] = pg.get("initial_lr", pg["lr"]) * lrm
            if pg.get("kind") == "muon":
                pg["momentum"] = muon_mom
                pg["weight_decay"] = muon_wd
    else:
        for pg in optimizer_.param_groups:
            pg["lr"] = pg.get("initial_lr", pg["lr"]) * lrm
    return lrm * cfg.learning_rate  # representative value for logging


_raw_model = model.module if use_ddp else model


def step_fn(raw_model: torch.nn.Module, X: torch.Tensor, Y: torch.Tensor):
    logits = raw_model(X)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        Y.view(-1),
        label_smoothing=cfg.label_smoothing,
    )
    return loss, {}


@torch.no_grad()
def estimate_val() -> dict:
    """Return {"loss": mean_val_loss} over n_batches token sequences."""
    n_batches = 20
    _raw_model.eval()
    losses = []
    _val_iter = _make_loader("val")
    for _ in range(n_batches):
        try:
            xv, yv = next(_val_iter)
        except StopIteration:
            break
        logits = _raw_model(xv)
        lv = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            yv.view(-1),
        )
        losses.append(lv.item())
    _raw_model.train()
    return {"loss": float(np.mean(losses)) if losses else float("inf")}


# ---------------------------------------------------------------------------
# 9.  Shared training loop (common/pretrain.py)
# ---------------------------------------------------------------------------
ckpt_extra_fields = {
    "n_layer": cfg.n_layer,
    "n_head": cfg.n_head,
    "n_kv_head": cfg.n_kv_head,
    "n_embd": cfg.n_embd,
    "sequence_len": cfg.sequence_len,
    "vocab_size": cfg.vocab_size,
    "window_pattern": cfg.window_pattern,
}

run_training(
    cfg,
    model,
    optimizer,
    train_iter,
    update_schedule,
    step_fn,
    estimate_val,
    set_attention_temperature,
    use_ddp=use_ddp,
    rank=rank,
    device=device,
    run_out_dir=run_out_dir,
    ctx=ctx,
    ckpt_extra_fields=ckpt_extra_fields,
    has_accuracy=False,
    qq_kk_masks=None,
    save_periodic_ckpt=False,
    initial_iter_num=iter_num,
    initial_best_val_loss=best_val_loss,
)


