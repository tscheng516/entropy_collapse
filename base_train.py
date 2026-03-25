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

    python base_train.py learning_rate=5e-4 optimizer=sgd max_iters=2000

The override syntax reuses the NanoGPT ``configurator.py`` convention:
any ``key=value`` argument is eval'd and injected into the config
dataclass.  Pass ``wandb_log=True`` to enable W&B tracking.

Setup
-----
1. Clone NanoGPT and prepare data::

    git clone https://github.com/karpathy/nanoGPT.git
    cd nanoGPT && python data/shakespeare_char/prepare.py && cd ..

2. Install dependencies::

    pip install -r requirements.txt

3. Run::

    python base_train.py
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

# ---------------------------------------------------------------------------
# 0.  Locate NanoGPT and add to sys.path so we can import model.py
# ---------------------------------------------------------------------------
NANOGPT_DIR = os.path.join(os.path.dirname(__file__), "nanoGPT")
if NANOGPT_DIR not in sys.path:
    sys.path.insert(0, NANOGPT_DIR)

# ---------------------------------------------------------------------------
# 1.  Default configuration (loaded from dataclass, then CLI overrides)
# ---------------------------------------------------------------------------
from configs.train_config import TrainConfig  # noqa: E402

cfg = TrainConfig()

# NanoGPT-style CLI overrides: python base_train.py learning_rate=1e-4 ...
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

# Create a per-run timestamped subdirectory to avoid overwriting previous
# runs. Keep the original `cfg.out_dir` path when resuming so callers can
# continue an existing run.
if cfg.init_from == "resume":
    run_out_dir = cfg.out_dir
else:
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_out_dir = os.path.join(cfg.out_dir, run_id)
    os.makedirs(run_out_dir, exist_ok=True)

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
        if cfg.wandb_log:
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
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ) if device != "cpu" else nullcontext():
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
        if cfg.wandb_log:
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

history_path = os.path.join(run_out_dir, "history.pkl")
with open(history_path, "wb") as f:
    pickle.dump(history, f)
print(f"[done] history saved → {history_path}")

if cfg.wandb_log:
    wandb.finish()

# ---------------------------------------------------------------------------
# 11.  Post-training plots (saved to out_dir, not shown interactively)
# ---------------------------------------------------------------------------
from src.plotting import plot_training_dynamics, plot_spike_cooccurrence  # noqa: E402

fig = plot_training_dynamics(
    histories={"Run": history},
    lrs={"Run": cfg.learning_rate},
    save_path=os.path.join(run_out_dir, "training_dynamics.png"),
)
print(f"[plot] training dynamics → {os.path.join(run_out_dir, 'training_dynamics.png')}")

fig2, res = plot_spike_cooccurrence(
    history["hessian"],
    history["hessian_vv"],
    x_name="Exact H",
    y_name="H_VV",
    window=15,
    z_score=10.0,
    save_path=os.path.join(run_out_dir, "spike_cooccurrence_H_vs_HVV.png"),
)
print(f"[plot] spike co-occurrence: P(H_VV spike | H spike) = {res['P(Y_spike | X_spike)']:.3f}")
