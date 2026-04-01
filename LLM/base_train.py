"""
base_train.py — NanoGPT entropy-collapse training script.

This script trains a HookedGPT model from scratch (or from a checkpoint)
while logging:
  * Train loss     — every iteration
  * Val loss       — every ``eval_interval`` iterations
  * Hessian proxies (H, H_tilde, H_VV, H_GN, FD)
                   — every ``hessian_freq`` iterations
  * Per-layer attention entropy
                   — every ``entropy_freq`` iterations

All metrics are emitted to stdout and (optionally) to Weights & Biases.
Checkpoints are saved to ``out_dir`` at regular intervals and whenever
the validation loss improves.

Usage
-----
Default config (edit configs/train_config.py or pass overrides on CLI)::

    python base_train.py

Override individual flags::

    python base_train.py --lr 5e-4 --optim sgd --max_it 2000

The override syntax reuses the NanoGPT ``configurator.py`` convention:
any ``key=value`` argument is eval'd and injected into the config
dataclass.  For argparse-style flags, use short names such as
``--cp``, ``--optim``, ``--lr``, ``--max_it``, ``--wandb``, and ``--z``.

Setup
-----
1. Clone NanoGPT inside the  directory and prepare data::

    cd LLM
    git clone https://github.com/karpathy/nanoGPT.git
    cd nanoGPT && python data/shakespeare_char/prepare.py && cd ../..

2. Install dependencies::

    pip install -r requirements.txt

3. Run (from the repo root)::

    python base_train.py

    or

    torchrun --nproc_per_node=4 base_train.py --cp scratch --optim adamw --lr 1e-3 --max_it 1000 --wandb true data_dir=nanoGPT/data/shakespeare_char

Note: ``data_dir`` in TrainConfig defaults to ``"nanoGPT/data/shakespeare_char"``
(relative to the working directory).  If you run from the repo root, set::

    python base_train.py data_dir=nanoGPT/data/shakespeare_char
"""

from __future__ import annotations

import os
import sys
import math
import time
import pickle
from contextlib import nullcontext

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless servers
import numpy as np
import torch
import torch.distributed as dist


def _sdp_math_only_context(device_name: str):
    """Use the non-deprecated SDPA context when available."""
    if device_name == "cpu":
        return nullcontext()

    attn_mod = getattr(torch.nn, "attention", None)
    if (
        attn_mod is not None
        and hasattr(attn_mod, "sdpa_kernel")
        and hasattr(attn_mod, "SDPBackend")
    ):
        return attn_mod.sdpa_kernel(backends=[attn_mod.SDPBackend.MATH])

    # Backward compatibility for older torch versions.
    return torch.backends.cuda.sdp_kernel(
        enable_flash=False,
        enable_mem_efficient=False,
        enable_math=True,
    )

# ---------------------------------------------------------------------------
# 0.  Path setup
#     a) Add repo root to sys.path so that the ``common`` package is importable.
#     b) Locate NanoGPT (expected at LLM/nanoGPT/) and add to sys.path.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Add the LLM/ directory so that ``configs`` and ``src`` sub-packages resolve.
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

NANOGPT_DIR = os.path.join(_SCRIPT_DIR, "nanoGPT")
if NANOGPT_DIR not in sys.path:
    sys.path.insert(0, NANOGPT_DIR)

# ---------------------------------------------------------------------------
# 1.  Default configuration (loaded from dataclass, then CLI overrides)
# ---------------------------------------------------------------------------
from configs.train_config import TrainConfig  # noqa: E402

cfg = TrainConfig()

# NanoGPT-style CLI overrides: python base_train.py learning_rate=1e-4 ...
# Short argparse flags are also supported, e.g. ``--lr 1e-4 --optim adamw``.
import ast

for arg in sys.argv[1:]:
    if "=" in arg:
        key, val = arg.split("=", 1)
        if hasattr(cfg, key):
            try:
                # ast.literal_eval handles ints, floats, bools, strings safely
                setattr(cfg, key, ast.literal_eval(val))
            except (ValueError, SyntaxError):
                # Fall back to raw string for values like device='mps' or run names
                setattr(cfg, key, val)
        else:
            print(f"[warn] unknown config key '{key}', ignoring.")

# --- argparse for common overrides (works with torchrun) ------------
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--cp", type=str, help="checkpoint path or 'scratch'/'resume'")
parser.add_argument("--optim", type=str, help="optimizer name (adamw or sgd)")
parser.add_argument("--lr", type=float, help="peak learning rate")
parser.add_argument("--max_it", type=int, help="number of training iterations")
parser.add_argument("--wandb", type=str, help="enable wandb logging (true/false)")
parser.add_argument("--z", type=float, help="MAD z-score for spike detection/plots")
known_args, _ = parser.parse_known_args()

def _maybe_set(attr, val, conv=lambda x: x):
    if val is None:
        return
    try:
        setattr(cfg, attr, conv(val))
    except Exception:
        print(f"[warn] failed to set cfg.{attr} from CLI value {val}")

_maybe_set("init_from", known_args.cp)
_maybe_set("optimizer", known_args.optim)
_maybe_set("learning_rate", known_args.lr)
_maybe_set("max_iters", known_args.max_it)
if known_args.wandb is not None:
    sval = str(known_args.wandb).lower()
    _maybe_set("wandb_log", sval in ("1", "true", "yes", "y"))
_maybe_set("z_score", known_args.z)

# ---------------------------------------------------------------------------
# 2.  Reproducibility & device
# ---------------------------------------------------------------------------
torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)

device = cfg.device
dtype_map = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}
ptdtype = dtype_map.get(cfg.dtype, torch.float32)
ctx = (
    nullcontext()
    if device == "cpu"
    else torch.amp.autocast(device_type=device.split(":")[0], dtype=ptdtype)
)

os.makedirs(cfg.out_dir, exist_ok=True)

# --- Distributed setup (torchrun) -------------------------------------
# Detect multi-process launch via WORLD_SIZE; init process group and set
# device / local rank accordingly. Non-distributed runs fall back to
# single-process behaviour.
use_ddp = False
rank = 0
world_size = 1
local_rank = 0

if int(os.environ.get("WORLD_SIZE", "1")) > 1:
    use_ddp = True
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # set CUDA device for this process
    if torch.cuda.is_available() and cfg.device.startswith("cuda"):
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"

# Create a per-run timestamped subdirectory to avoid overwriting previous
# runs. When distributed, the rank 0 process chooses the run id and the
# remainder receive it via broadcast so all ranks share the same folder.
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
# 3.  Data
# ---------------------------------------------------------------------------
from src.data_utils import load_data, get_batch  # noqa: E402

train_data, val_data = load_data(cfg.data_dir)
print(
    f"[data] train tokens: {len(train_data):,}  "
    f"val tokens: {len(val_data) if val_data is not None else 0:,}"
)


def _get_train_batch():
    return get_batch(train_data, cfg.batch_size, cfg.block_size, device)


def _get_val_batch():
    if val_data is None:
        return _get_train_batch()
    return get_batch(val_data, cfg.batch_size, cfg.block_size, device)


# ---------------------------------------------------------------------------
# 4.  Model
# ---------------------------------------------------------------------------
from model import GPTConfig  # noqa: E402  (via NANOGPT_DIR)
from src.model import build_hooked_gpt  # noqa: E402

iter_num = 0
best_val_loss = float("inf")

if cfg.init_from == "scratch":
    print("[model] initialising fresh HookedGPT …")
    gpt_cfg = GPTConfig(
        block_size=cfg.block_size,
        vocab_size=cfg.vocab_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
        bias=cfg.bias,
    )
    model = build_hooked_gpt(
        gpt_cfg,
        init_std=cfg.init_std,
        use_scaled_init=cfg.use_scaled_init,
        device=device,
    )

elif cfg.init_from == "resume":
    ckpt_path = os.path.join(cfg.out_dir, "ckpt.pt")
    print(f"[model] resuming from {ckpt_path} …")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gpt_cfg = GPTConfig(**checkpoint["model_args"])
    model = build_hooked_gpt(
        gpt_cfg,
        init_std=cfg.init_std,
        use_scaled_init=False,  # don't re-init when resuming
        device=device,
    )
    # Strip compile prefix if present
    state_dict = checkpoint["model"]
    for k in list(state_dict.keys()):
        if k.startswith("_orig_mod."):
            state_dict[k[len("_orig_mod."):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint.get("iter_num", 0)
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))

else:
    print(f"[model] fine-tuning from checkpoint {cfg.init_from} …")
    checkpoint = torch.load(cfg.init_from, map_location=device)
    gpt_cfg = GPTConfig(**checkpoint["model_args"])
    model = build_hooked_gpt(
        gpt_cfg,
        init_std=cfg.init_std,
        use_scaled_init=False,
        device=device,
    )
    state_dict = checkpoint["model"]
    for k in list(state_dict.keys()):
        if k.startswith("_orig_mod."):
            state_dict[k[len("_orig_mod."):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

n_params = model.get_num_params() / 1e6
print(f"[model] {n_params:.2f}M parameters")

if cfg.compile:
    print("[model] compiling with torch.compile (disable for Hessian metrics)")
    model = torch.compile(model)

# If running under torchrun, wrap the model in DistributedDataParallel so
# gradients are synced and parameters are consistent across processes.
if use_ddp:
    if torch.cuda.is_available() and cfg.device.startswith("cuda"):
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )
    else:
        model = torch.nn.parallel.DistributedDataParallel(model)
else:
    model.to(device)

# ---------------------------------------------------------------------------
# 5.  Optimiser
# ---------------------------------------------------------------------------
def _build_optimizer(model_: torch.nn.Module) -> torch.optim.Optimizer:
    if cfg.optimizer.lower() == "adamw":
        # Reuse NanoGPT's configure_optimizers for correct weight-decay split
        try:
            return model_.configure_optimizers(
                cfg.weight_decay,
                cfg.learning_rate,
                (cfg.beta1, cfg.beta2),
                device,
            )
        except AttributeError:
            return torch.optim.AdamW(
                model_.parameters(),
                lr=cfg.learning_rate,
                betas=(cfg.beta1, cfg.beta2),
                weight_decay=cfg.weight_decay,
                eps=cfg.eps,
            )
    elif cfg.optimizer.lower() == "sgd":
        return torch.optim.SGD(model_.parameters(), lr=cfg.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer '{cfg.optimizer}'")


optimizer = _build_optimizer(model)

# ---------------------------------------------------------------------------
# 6.  LR schedule  (cosine with linear warm-up, mirrors NanoGPT)
# ---------------------------------------------------------------------------
def get_lr(it: int) -> float:
    if not cfg.decay_lr:
        return cfg.learning_rate
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / cfg.warmup_iters
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    ratio = (it - cfg.warmup_iters) / max(1, cfg.lr_decay_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


# ---------------------------------------------------------------------------
# 7.  W&B
# ---------------------------------------------------------------------------
if cfg.wandb_log:
    import wandb
    if not use_ddp or rank == 0:
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name or None,
            config=vars(cfg),
        )

# ---------------------------------------------------------------------------
# 8.  Helper: Hessian metrics & entropy
# ---------------------------------------------------------------------------
from src.helpers import (  # noqa: E402
    get_VV_subspace_mask,
    get_curvature_metrics,
    get_attention_entropy,
)

vv_mask = get_VV_subspace_mask(model).to(device)


@torch.no_grad()
def estimate_val_loss(n_batches: int = 10) -> float:
    model.eval()
    losses = []
    for _ in range(n_batches):
        xv, yv = _get_val_batch()
        with ctx:
            _, lv = model(xv, yv)
        losses.append(lv.item())
    model.train()
    return float(np.mean(losses))


def _save_checkpoint(suffix: str = "ckpt") -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter_num": iter_num,
        "best_val_loss": best_val_loss,
        "model_args": {
            "block_size": cfg.block_size,
            "vocab_size": cfg.vocab_size,
            "n_layer": cfg.n_layer,
            "n_head": cfg.n_head,
            "n_embd": cfg.n_embd,
            "dropout": cfg.dropout,
            "bias": cfg.bias,
        },
        "config": vars(cfg),
    }
    # Only the rank-0 process writes checkpoints to avoid concurrent
    # file writes when running under torchrun / DDP.
    if not use_ddp or rank == 0:
        # If wrapped in DDP, the underlying module holds the real params.
        sd = checkpoint["model"]
        if use_ddp and hasattr(model, "module"):
            sd = model.module.state_dict()
            checkpoint["model"] = sd
        path = os.path.join(run_out_dir, f"{suffix}.pt")
        torch.save(checkpoint, path)
        print(f"[ckpt] saved → {path}")


# ---------------------------------------------------------------------------
# 9.  Training loop
# ---------------------------------------------------------------------------
history: dict[str, list] = {
    "loss": [],
    "val_loss": [],
    "hessian": [],
    "prec_h": [],
    "hessian_vv": [],
    "gn": [],
    "fd": [],
    "entropy": [],
    "lr": [],
}

print(f"\n[train] starting — max_iters={cfg.max_iters}  device={device}\n")
t0 = time.time()
model.train()

X, Y = _get_train_batch()  # pre-fetch first batch

for iter_num in range(iter_num, cfg.max_iters):

    # ---- LR update ----
    lr = get_lr(iter_num)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    # ---- Periodic evaluation ----
    if iter_num % cfg.eval_interval == 0 or iter_num == cfg.max_iters - 1:
        val_loss = estimate_val_loss()
        history["val_loss"].append((iter_num, val_loss))
        print(
            f"[eval] iter {iter_num:5d} | val_loss {val_loss:.4f} "
            f"| best {best_val_loss:.4f}"
        )
        if cfg.wandb_log and (not use_ddp or rank == 0):
            wandb.log({"val/loss": val_loss}, step=iter_num)

        if val_loss < best_val_loss or cfg.always_save_checkpoint:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                _save_checkpoint("best_ckpt")
            _save_checkpoint("ckpt")

    if iter_num % cfg.checkpoint_interval == 0 and iter_num > 0:
        _save_checkpoint(f"ckpt_iter{iter_num:06d}")

    # ---- Hessian metrics ----
    curvature: dict[str, float] = {
        "hessian": 0.0,
        "prec_h": 0.0,
        "hessian_vv": 0.0,
        "gn": 0.0,
        "fd": 0.0,
    }
    if iter_num % cfg.hessian_freq == 0:
        model.train()
        optimizer.zero_grad()
        with _sdp_math_only_context(device):
            _, loss_for_hess = model(X, Y)

        try:
            curvature = get_curvature_metrics(
                model,
                optimizer,
                X,
                Y,
                loss_for_hess,
                vv_mask,
                max_iter=cfg.hessian_max_iter,
                compute_fd=cfg.compute_fd,
            )
        except Exception as exc:
            print(f"[warn] curvature metrics failed at iter {iter_num}: {exc}")
        finally:
            optimizer.zero_grad()

    for k in ("hessian", "prec_h", "hessian_vv", "gn", "fd"):
        history[k].append(curvature[k])

    # ---- Attention entropy ----
    # Capture attention entropy from the most recent forward pass.  We
    # compute it immediately after the model forward inside the normal
    # training context so the patched attention forward has executed
    # and populated `block.attn.last_att` reliably for both optimisers
    # (avoids subtle differences between eval/train contexts and
    # compiled vs. eager execution).
    layer_entropies: list[float] = [0.0] * cfg.n_layer

    # ---- Standard training step ----
    optimizer.zero_grad(set_to_none=True)
    with ctx:
        _, loss = model(X, Y)

        # Record attention entropy immediately after the forward so the
        # patched attention modules have populated their cached
        # `last_att` tensors.  Wrap in `no_grad` to avoid touching the
        # autograd graph.
        if iter_num % cfg.entropy_freq == 0:
            with torch.no_grad():
                layer_entropies = get_attention_entropy(model)

    loss.backward()
    if cfg.grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()

    loss_val = loss.item()
    history["loss"].append(loss_val)
    history["lr"].append(lr)

    # Record entropy history (may be the zero-default when not sampled)
    history["entropy"].append(layer_entropies)

    # pre-fetch next batch
    X, Y = _get_train_batch()

    # ---- Logging ----
    if iter_num % cfg.log_interval == 0:
        dt = time.time() - t0
        t0 = time.time()
        print(f"iter {iter_num:5d} | loss {loss_val:.4f} | lr {lr:.2e}  | dt {dt*1000:.1f}ms")
        if iter_num % cfg.hessian_freq == 0:
            print(f"| H {curvature['hessian']:.1f} | H_VV {curvature['hessian_vv']:.1f} | GN {curvature['gn']:.1f}")
        if cfg.wandb_log and (not use_ddp or rank == 0):
            log_dict: dict[str, float | int] = {
                "train/loss": loss_val,
                "train/lr": lr,
            }
            if iter_num % cfg.hessian_freq == 0:
                log_dict.update(
                    {
                        "hessian/lambda_max": curvature["hessian"],
                        "hessian/prec_H": curvature["prec_h"],
                        "hessian/H_VV": curvature["hessian_vv"],
                        "hessian/GN": curvature["gn"],
                        "hessian/FD": curvature["fd"],
                    }
                )
            if iter_num % cfg.entropy_freq == 0:
                log_dict.update(
                    {
                        f"entropy/layer_{i}": v
                        for i, v in enumerate(layer_entropies)
                    }
                )
            wandb.log(log_dict, step=iter_num)

# ---------------------------------------------------------------------------
# 10.  Final checkpoint & save history
# ---------------------------------------------------------------------------
_save_checkpoint("final_ckpt")

# Only the main process writes the history file and final plots when
# running distributed to avoid races and duplicate outputs.
if not use_ddp or rank == 0:
    history_path = os.path.join(run_out_dir, "history.pkl")
    with open(history_path, "wb") as f:
        pickle.dump(history, f)
    print(f"[done] history saved → {history_path}")

    if cfg.wandb_log:
        wandb.finish()

# ---------------------------------------------------------------------------
# 11.  Post-training plots (saved to out_dir, not shown interactively)
# ---------------------------------------------------------------------------

if not use_ddp or rank == 0:
    from src.plotting import plot_training_dynamics, plot_spike_cooccurrence  # noqa: E402

    fig = plot_training_dynamics(
        histories={"Run": history},
        lrs={"Run": cfg.learning_rate},
        save_path=os.path.join(run_out_dir, "training_dynamics.png"),
    )
    print(f"[plot] training dynamics → {os.path.join(run_out_dir, 'training_dynamics.png')}")

    spike_targets = [
        ("hessian_vv", "H_VV", "hessian_vv"),
        ("prec_h", "Prec_H", "hessian_prec"),
        ("gn", "GN", "hessian_gn"),
    ]
    for z in (3, 10):
        for key, label, suffix in spike_targets:
            _, res = plot_spike_cooccurrence(
                history["hessian"],
                history[key],
                x_name="Exact H",
                y_name=label,
                window=15,
                z_score=z,
                save_path=os.path.join(
                    run_out_dir,
                    f"spike_cooccurrence_H_vs_{suffix}_z{z}.png",
                ),
            )
            print(
                f"[plot] z={z} spike co-occurrence: "
                f"P({label} spike | H spike) = {res['P(Y_spike | X_spike)']:.3f}"
            )
