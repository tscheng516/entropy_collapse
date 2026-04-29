"""
base_train.py — nanochat entropy-collapse training script.

Trains a nanochat GPT with HookedGPT attention caching, logging per-layer
attention entropy and nine curvature proxy metrics (λ_max of H, Prec_H,
H_VV, GN, Diag_H, Fisher, KFAC; + BFGS / FD when compute_fd=True).

Preset configs (depth is the single complexity dial):
  d8  | 8-layer  ~30 M  — quick iteration (~3 min on 8×H100)
  d12 | 12-layer ~85 M  — standard research run (default)
  d24 | 24-layer ~350 M — GPT-2 scale speedrun

Usage
-----
Default (d12, ~85 M params)::

    python nanochat/base_train.py

Named preset::

    python nanochat/base_train.py config=d8

Override individual fields::

    python nanochat/base_train.py config=d12 learning_rate=1e-4 max_iters=5000

Multi-GPU via torchrun::

    torchrun --nproc_per_node=4 nanochat/base_train.py config=d12

Key=value arguments are ast.literal_eval'd (NanoGPT-style).
Argparse shortcuts: --lr, --optim, --max_it, --wandb, --cp, --temp_shift.

Prerequisites
-------------
1. Clone nanochat at the pinned SHA and install its uv environment::

       git clone https://github.com/karpathy/nanochat
       cd nanochat && git checkout 0aaca56
       uv sync --extra gpu
       uv pip install -r ../entropy_collapse/nanochat/requirements.txt

2. Download ClimbMix data shards (inside the nanochat clone)::

       python -m nanochat.dataset

3. Set ``nanochat_dir=<path>`` (default: ``nanochat/nanochat_repo``).

Precision note
--------------
nanochat manages compute dtype globally via ``COMPUTE_DTYPE`` and casts
Linear weights to ``x.dtype`` in each forward.  We therefore do **not** use
``torch.amp.autocast``; instead a ``nullcontext()`` is used throughout.  The
HVP graph still flows through bf16 weight casts, which is sufficient for
λ_max trend monitoring.
"""

from __future__ import annotations

import ast
import os
import sys
import math
import time
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist

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

_config_cls = TrainConfig
for _arg in sys.argv[1:]:
    if _arg.startswith("config="):
        _preset = _arg.split("=", 1)[1].strip()
        if _preset in CONFIGS:
            _config_cls = CONFIGS[_preset]
            if _is_master:
                print(f"[config] using preset '{_preset}' ({_config_cls.__name__})")
        else:
            raise ValueError(
                f"Unknown config preset '{_preset}'. "
                f"Available presets: {list(CONFIGS.keys())}."
            )
        break

cfg = _config_cls()

# NanoGPT-style key=value overrides
for arg in sys.argv[1:]:
    if "=" in arg:
        key, val = arg.split("=", 1)
        if key == "config":
            continue
        if hasattr(cfg, key):
            try:
                setattr(cfg, key, ast.literal_eval(val))
            except (ValueError, SyntaxError):
                setattr(cfg, key, val)
        else:
            if _is_master:
                print(f"[warn] unknown config key '{key}', ignoring.")

import argparse  # noqa: E402

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--cp", type=str)
parser.add_argument("--optim", type=str)
parser.add_argument("--lr", type=float)
parser.add_argument("--bs", type=int)
parser.add_argument("--max_it", type=int)
parser.add_argument("--num_workers", type=int)
parser.add_argument("--hessian_intv", type=int)
parser.add_argument("--entropy_intv", type=int)
parser.add_argument("--wandb", type=str)
parser.add_argument(
    "--temp_shift",
    type=int,
    default=None,
    metavar="STEP",
    help=(
        "Training step at which a one-time temperature-shift intervention is "
        "applied to all attention heads (-1 disables)."
    ),
)
known_args, _ = parser.parse_known_args()


def _maybe_set(attr, val, conv=lambda x: x):
    if val is None:
        return
    try:
        setattr(cfg, attr, conv(val))
    except Exception:
        if _is_master:
            print(f"[warn] failed to set cfg.{attr} from CLI value {val}")


_maybe_set("init_from", known_args.cp)
_maybe_set("optimizer", known_args.optim)
_maybe_set("learning_rate", known_args.lr)
_maybe_set("batch_size", known_args.bs)
_maybe_set("max_iters", known_args.max_it)
_maybe_set("num_workers", known_args.num_workers)
_maybe_set("hessian_intv", known_args.hessian_intv)
_maybe_set("entropy_intv", known_args.entropy_intv)
if known_args.wandb is not None:
    sval = str(known_args.wandb).lower()
    _maybe_set("wandb_log", sval in ("1", "true", "yes", "y"))
_maybe_set("temp_shift_step", known_args.temp_shift)

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
# 3.  Reproducibility & device
# ---------------------------------------------------------------------------
torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)

device = cfg.device
# nanochat manages dtype internally — we never wrap with autocast.
ctx = nullcontext()

os.makedirs(cfg.out_dir, exist_ok=True)

# --- Distributed setup (torchrun) -----------------------------------------
use_ddp = False
rank = 0
world_size = 1
local_rank = 0

if int(os.environ.get("WORLD_SIZE", "1")) > 1:
    use_ddp = True
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() and cfg.device.startswith("cuda") else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if torch.cuda.is_available() and cfg.device.startswith("cuda"):
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"

# Per-run timestamped sub-directory ----------------------------------------
if cfg.init_from == "resume":
    run_out_dir = cfg.out_dir
else:
    if use_ddp:
        if rank == 0:
            run_id = time.strftime("%Y%m%d-%H%M%S")
        else:
            run_id = None
        run_id_list = [run_id]
        dist.broadcast_object_list(run_id_list, src=0)
        run_id = run_id_list[0]
        run_out_dir = os.path.join(cfg.out_dir, run_id)
        if rank == 0:
            os.makedirs(run_out_dir, exist_ok=True)
        dist.barrier()
    else:
        run_id = time.strftime("%Y%m%d-%H%M%S")
        run_out_dir = os.path.join(cfg.out_dir, run_id)
        os.makedirs(run_out_dir, exist_ok=True)

if rank == 0:
    print(f"[io] outputs → {run_out_dir}")

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


def _strip_compile_prefix(state_dict: dict) -> dict:
    """Remove the ``_orig_mod.`` key prefix added by ``torch.compile``."""
    return {
        (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
        for k, v in state_dict.items()
    }


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
    model.load_state_dict(_strip_compile_prefix(checkpoint["model"]))
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
    model.load_state_dict(_strip_compile_prefix(checkpoint["model"]))

n_params = sum(p.numel() for p in model.parameters()) / 1e6
if _is_master:
    print(f"[model] nanochat GPT  {n_params:.2f}M parameters")

if cfg.compile:
    if _is_master:
        print("[model] compiling with torch.compile (disable for Hessian metrics)")
    model = torch.compile(model)

if use_ddp:
    if torch.cuda.is_available() and cfg.device.startswith("cuda"):
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )
    else:
        model = torch.nn.parallel.DistributedDataParallel(model)

# ---------------------------------------------------------------------------
# 6.  Optimiser
# ---------------------------------------------------------------------------
def _build_optimizer(model_: torch.nn.Module) -> torch.optim.Optimizer:
    opt_name = cfg.optimizer.lower()
    if opt_name == "adamw":
        decay_params = [
            p for n, p in model_.named_parameters()
            if p.requires_grad and p.ndim >= 2
        ]
        no_decay_params = [
            p for n, p in model_.named_parameters()
            if p.requires_grad and p.ndim < 2
        ]
        param_groups = [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        opt = torch.optim.AdamW(
            param_groups,
            lr=cfg.learning_rate,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps,
        )
        for pg in opt.param_groups:
            pg["initial_lr"] = pg["lr"]
        return opt
    elif opt_name == "muon_adamw":
        # Delegates to nanochat's model.setup_optimizer() which builds
        # MuonAdamW (or DistMuonAdamW for DDP) with per-group LRs.
        # Each group stores group["initial_lr"] so the cosine schedule can
        # scale all groups proportionally rather than overwriting them.
        # Note: prec_H (Adam-preconditioned Hessian) only works for the AdamW
        # param groups; Muon groups fall back to the raw Hessian value.
        _raw = model_.module if use_ddp else model_
        return _raw.setup_optimizer(
            matrix_lr=cfg.muon_matrix_lr,
            embedding_lr=cfg.muon_embedding_lr,
            unembedding_lr=cfg.muon_unembedding_lr,
            scalar_lr=cfg.muon_scalar_lr,
            weight_decay=cfg.weight_decay,
        )
    else:
        raise ValueError(
            f"Unknown optimizer '{cfg.optimizer}'. "
            "Choose 'muon_adamw' (nanochat default) or 'adamw'."
        )


optimizer = _build_optimizer(model)

# ---------------------------------------------------------------------------
# 7.  LR schedule  (cosine with linear warm-up)
# ---------------------------------------------------------------------------
def get_lr(it: int) -> float:
    """Absolute LR for the AdamW-only optimizer path."""
    if not cfg.decay_lr:
        return cfg.learning_rate
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / cfg.warmup_iters
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    ratio = (it - cfg.warmup_iters) / max(1, cfg.lr_decay_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def get_lr_scale(it: int) -> float:
    """Fractional LR scale in [min_lr/learning_rate, 1] for MuonAdamW.

    Each param group has its own tuned ``initial_lr``; multiplying by this
    scale applies the same cosine decay to all groups proportionally.
    """
    if not cfg.decay_lr:
        return 1.0
    floor = cfg.min_lr / cfg.learning_rate if cfg.learning_rate > 0 else 0.0
    if it < cfg.warmup_iters:
        return floor + (1.0 - floor) * (it + 1) / cfg.warmup_iters
    if it > cfg.lr_decay_iters:
        return floor
    ratio = (it - cfg.warmup_iters) / max(1, cfg.lr_decay_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return floor + coeff * (1.0 - floor)


# ---------------------------------------------------------------------------
# 8.  W&B
# ---------------------------------------------------------------------------
if cfg.wandb_log:
    import wandb  # noqa: E402
    if not use_ddp or rank == 0:
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name or None,
            config=vars(cfg),
        )

# ---------------------------------------------------------------------------
# 9.  Curvature and entropy helpers
# ---------------------------------------------------------------------------
from src.helpers import (  # noqa: E402
    get_VV_subspace_mask,
    get_curvature_metrics,
    get_attention_entropy,
)

_raw_model = model.module if use_ddp else model
vv_mask = get_VV_subspace_mask(_raw_model).to(device)


@torch.no_grad()
def estimate_val_loss(n_batches: int = 20) -> float:
    """Return mean val loss over n_batches token sequences."""
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
    return float(np.mean(losses)) if losses else float("inf")


def _save_checkpoint(suffix: str = "ckpt") -> None:
    if not use_ddp or rank == 0:
        checkpoint = {
            "model": _raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter_num": iter_num,
            "best_val_loss": best_val_loss,
            # Architecture fields for resume
            "n_layer": cfg.n_layer,
            "n_head": cfg.n_head,
            "n_kv_head": cfg.n_kv_head,
            "n_embd": cfg.n_embd,
            "sequence_len": cfg.sequence_len,
            "vocab_size": cfg.vocab_size,
            "window_pattern": cfg.window_pattern,
            "config": vars(cfg),
        }
        path = os.path.join(run_out_dir, f"{suffix}.pt")
        torch.save(checkpoint, path)
        print(f"[ckpt] saved → {path}")


# ---------------------------------------------------------------------------
# 10.  Training loop
# ---------------------------------------------------------------------------
n_layers = len(_raw_model.transformer.h)

history: dict[str, list] = {
    "loss": [],
    "val_loss": [],
    "hessian": [],
    "prec_h": [],
    "hessian_vv": [],
    "gn": [],
    "fd": [],
    "diag_h": [],
    "fisher": [],
    "bfgs": [],
    "kfac": [],
    "entropy": [],
    "lr": [],
}

if _is_master:
    print(f"\n[train] starting — max_iters={cfg.max_iters}  device={device}\n")
t0 = time.time()
model.train()

X, Y = next(train_iter)

for iter_num in range(iter_num, cfg.max_iters):

    # ---- LR update ----
    if cfg.optimizer.lower() == "muon_adamw":
        # Scale each group's initial_lr proportionally — avoids clobbering
        # the per-group LR ratios set by setup_optimizer().
        lr_scale = get_lr_scale(iter_num)
        for pg in optimizer.param_groups:
            pg["lr"] = pg.get("initial_lr", pg["lr"]) * lr_scale
        lr = lr_scale * cfg.muon_matrix_lr  # representative value for logging
    else:
        lr = get_lr(iter_num)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

    # ---- Temperature-shift intervention ----
    if cfg.temp_shift_step >= 0 and iter_num == cfg.temp_shift_step:
        set_attention_temperature(model, cfg.temp_shift_factor)
        if _is_master:
            print(
                f"[temp_shift] iter {iter_num}: applied temperature={cfg.temp_shift_factor:.4g} "
                f"to all attention heads"
            )
        if cfg.wandb_log and (not use_ddp or rank == 0):
            wandb.log({"intervention/temp_shift_factor": cfg.temp_shift_factor}, step=iter_num)

    # ---- Periodic evaluation ----
    if iter_num % cfg.eval_interval == 0 or iter_num == cfg.max_iters - 1:
        val_loss = estimate_val_loss()
        history["val_loss"].append((iter_num, val_loss))
        if _is_master:
            print(f"[eval] iter {iter_num:5d} | val_loss {val_loss:.4f}")
        if cfg.wandb_log and (not use_ddp or rank == 0):
            wandb.log({"val/loss": val_loss}, step=iter_num)

        if cfg.save_checkpoint and val_loss < best_val_loss and iter_num > 0:
            best_val_loss = val_loss

    if cfg.checkpoint_interval > 0 and iter_num % cfg.checkpoint_interval == 0 and iter_num > 0:
        _save_checkpoint(f"ckpt_iter{iter_num:06d}")

    # ---- Curvature metrics (spectral norm) ----
    curvature: dict[str, float] = {
        "hessian": 0.0, "prec_h": 0.0, "hessian_vv": 0.0,
        "gn": 0.0, "fd": 0.0, "diag_h": 0.0,
        "fisher": 0.0, "bfgs": 0.0, "kfac": 0.0,
    }
    if iter_num % cfg.hessian_intv == 0:
        _raw_model.train()
        optimizer.zero_grad()
        try:
            curvature = get_curvature_metrics(
                _raw_model,
                optimizer,
                X,
                Y,
                vv_mask,
                max_iter=cfg.hessian_max_iter,
                compute_fd=cfg.compute_fd,
                hessian_batch_size=cfg.hessian_batch_size,
                label_smoothing=cfg.label_smoothing,
            )
        except Exception as exc:
            if _is_master:
                print(f"[warn] curvature metrics failed at iter {iter_num}: {exc}")
        finally:
            optimizer.zero_grad()

    for k in ("hessian", "prec_h", "hessian_vv", "gn", "fd", "diag_h", "fisher", "bfgs", "kfac"):
        history[k].append(curvature[k])

    # ---- Standard training step ----
    layer_entropies: list[float] = [0.0] * n_layers
    _need_entropy = iter_num % cfg.entropy_intv == 0

    if _need_entropy:
        for blk in _raw_model.transformer.h:
            blk.attn._cache_attn = True

    optimizer.zero_grad(set_to_none=True)
    with ctx:
        logits = _raw_model(X)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            Y.view(-1),
            label_smoothing=cfg.label_smoothing,
        )

    if _need_entropy:
        with torch.no_grad():
            layer_entropies = get_attention_entropy(_raw_model)
        for blk in _raw_model.transformer.h:
            blk.attn._cache_attn = False
            blk.attn.last_att = None  # free attention cache immediately

    loss.backward()
    if cfg.grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()

    loss_val = loss.item()
    history["loss"].append(loss_val)
    history["lr"].append(lr)
    history["entropy"].append(layer_entropies)

    # pre-fetch next batch
    X, Y = next(train_iter)

    # ---- Logging ----
    if iter_num % cfg.log_interval == 0:
        dt = time.time() - t0
        t0 = time.time()
        if _is_master:
            print(
                f"iter {iter_num:5d} | loss {loss_val:.4f} "
                f"| lr {lr:.2e} | dt {dt * 1000:.1f}ms"
            )
            if iter_num % cfg.hessian_intv == 0:
                _cmsg = (
                    f"  H {curvature['hessian']:.3f} | H~(prec) {curvature['prec_h']:.3f} "
                    f"| H_VV {curvature['hessian_vv']:.3f} | GN {curvature['gn']:.3f} "
                    f"| DiagH {curvature['diag_h']:.3f} | Fisher {curvature['fisher']:.3f}"
                )
                if cfg.compute_fd:
                    _cmsg += (
                        f" | BFGS {curvature['bfgs']:.3f}"
                        f" | FD {curvature['fd']:.3f}"
                        f" | KFAC {curvature['kfac']:.3f}"
                    )
                print(_cmsg)
        if cfg.wandb_log and (not use_ddp or rank == 0):
            log_dict: dict = {
                "train/loss": loss_val,
                "train/lr": lr,
            }
            if iter_num % cfg.hessian_intv == 0:
                log_dict.update({
                    "hessian/lambda_max": curvature["hessian"],
                    "hessian/prec_H": curvature["prec_h"],
                    "hessian/H_VV": curvature["hessian_vv"],
                    "hessian/GN": curvature["gn"],
                    "hessian/diag_H": curvature["diag_h"],
                    "hessian/fisher": curvature["fisher"],
                })
                if cfg.compute_fd:
                    log_dict.update({
                        "hessian/FD": curvature["fd"],
                        "hessian/BFGS": curvature["bfgs"],
                        "hessian/KFAC": curvature["kfac"],
                    })
            if iter_num % cfg.entropy_intv == 0:
                log_dict.update({
                    f"entropy/layer_{i}": v
                    for i, v in enumerate(layer_entropies)
                })
            wandb.log(log_dict, step=iter_num)

# ---------------------------------------------------------------------------
# 11.  Final checkpoint & save history
# ---------------------------------------------------------------------------
_save_checkpoint("final_ckpt")

if not use_ddp or rank == 0:
    history_path = os.path.join(run_out_dir, "history.pkl")
    with open(history_path, "wb") as f:
        pickle.dump(history, f)
    print(f"[done] history saved → {history_path}")

    if cfg.wandb_log:
        wandb.finish()

# ---------------------------------------------------------------------------
# 12.  Post-training plots
# ---------------------------------------------------------------------------
if not use_ddp or rank == 0:
    from plot_history import plot_history  # noqa: E402

    plot_history(
        pkl_path=history_path,
        out_dir=run_out_dir,
        hessian_intv=cfg.hessian_intv,
        entropy_intv=cfg.entropy_intv,
        skip_intv=True,
        lam=10.0,
        compute_fd=cfg.compute_fd,
    )

# ---------------------------------------------------------------------------
# 13.  DDP teardown
# ---------------------------------------------------------------------------
if use_ddp:
    dist.destroy_process_group()
