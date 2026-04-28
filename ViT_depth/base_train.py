"""
base_train.py — depth/ entropy-collapse training script.

Trains ViT-B/16 from scratch on NYU Depth V2 with the scale-invariant
log loss (SILog).  The training dynamics, per-layer attention entropy,
and all nine Hessian-proxy curvature metrics are tracked and logged in
the same format as ``ViT/base_train.py``.

Select a named preset via ``config=<name>``; available presets:
  nyudepth_base | nyudepth_large

Logged every iteration:
  * Train SILog loss
  * Learning rate

Logged every ``eval_interval``:
  * Val SILog loss, RMSE (log-scale), δ<1.25 accuracy

Logged every ``hessian_freq``:
  * Curvature proxies — λ_max of H, Prec_H, H_VV, GN, Diag_H, Fisher, KFAC

Logged every ``entropy_freq``:
  * Per-layer attention entropy

Usage
-----
Default (ViT-B/16, NYU Depth V2)::

    python base_train.py

Named preset::

    python base_train.py config=nyudepth_base

Override individual fields::

    python base_train.py --lr 5e-4 --max_it 10000 hessian_freq=100

Multi-GPU via torchrun::

    torchrun --nproc_per_node=4 base_train.py config=nyudepth_base \\
        data_dir=./data/nyu_depth_v2 --wandb true

Key=value arguments are ast.literal_eval'd (NanoGPT-style).
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

_is_master = int(os.environ.get("RANK", "0")) == 0

# ---------------------------------------------------------------------------
# 0.  Path setup
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
from configs.train_config import DepthTrainConfig, CONFIGS  # noqa: E402

_config_cls = DepthTrainConfig
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

# NanoGPT-style CLI overrides
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

import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--cp", type=str)
parser.add_argument("--optim", type=str)
parser.add_argument("--lr", type=float)
parser.add_argument("--max_it", type=int)
parser.add_argument("--num_workers", type=int)
parser.add_argument("--hessian_freq", type=int)
parser.add_argument("--entropy_freq", type=int)
parser.add_argument("--wandb", type=str)
parser.add_argument(
    "--temp_shift",
    type=int,
    default=None,
    metavar="STEP",
    help="Training step for one-time temperature-shift intervention (-1 disables).",
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


_maybe_set("init_from",    known_args.cp)
_maybe_set("optimizer",    known_args.optim)
_maybe_set("learning_rate", known_args.lr)
_maybe_set("max_iters",    known_args.max_it)
_maybe_set("num_workers",  known_args.num_workers)
_maybe_set("hessian_freq", known_args.hessian_freq)
_maybe_set("entropy_freq", known_args.entropy_freq)
if known_args.wandb is not None:
    sval = str(known_args.wandb).lower()
    _maybe_set("wandb_log", sval in ("1", "true", "yes", "y"))
_maybe_set("temp_shift_step", known_args.temp_shift)

# ---------------------------------------------------------------------------
# 2.  Reproducibility & device
# ---------------------------------------------------------------------------
torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)

device = cfg.device
dtype_map = {
    "float32":  torch.float32,
    "bfloat16": torch.bfloat16,
    "float16":  torch.float16,
}
ptdtype = dtype_map.get(cfg.dtype, torch.float32)
ctx = (
    nullcontext()
    if device == "cpu"
    else torch.amp.autocast(device_type=device.split(":")[0], dtype=ptdtype)
)

os.makedirs(cfg.out_dir, exist_ok=True)

# --- Distributed setup (torchrun) -----------------------------------------
use_ddp = False
rank = 0
world_size = 1
local_rank = 0

if int(os.environ.get("WORLD_SIZE", "1")) > 1:
    use_ddp = True
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = (
        "nccl"
        if torch.cuda.is_available() and cfg.device.startswith("cuda")
        else "gloo"
    )
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if torch.cuda.is_available() and cfg.device.startswith("cuda"):
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"

# Per-run timestamped sub-directory
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
from src.data_utils import load_data, infinite_loader  # noqa: E402

train_loader, val_loader = load_data(
    data_dir=cfg.data_dir,
    img_size=cfg.img_size,
    batch_size=cfg.batch_size,
    num_workers=cfg.num_workers,
)

train_iter = infinite_loader(train_loader)
if _is_master:
    print(
        f"[data] NYU Depth V2  "
        f"train_batches={len(train_loader)}  "
        f"val_batches={len(val_loader)}"
    )

# ---------------------------------------------------------------------------
# 4.  Model
# ---------------------------------------------------------------------------
from src.model import build_hooked_vit_depth, set_attention_temperature  # noqa: E402

iter_num = 0
best_val_loss = float("inf")


def _strip_compile_prefix(state_dict: dict) -> dict:
    return {
        (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
        for k, v in state_dict.items()
    }


if cfg.init_from == "scratch":
    if _is_master:
        print(f"[model] building {cfg.model_name} from scratch …")
    model = build_hooked_vit_depth(
        model_name=cfg.model_name,
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        init_std=cfg.init_std,
        use_scaled_init=cfg.use_scaled_init,
        qk_norm=cfg.qk_norm,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        embed_dim=cfg.embed_dim,
        device=device,
    )

elif cfg.init_from == "resume":
    ckpt_path = os.path.join(cfg.out_dir, "ckpt.pt")
    if _is_master:
        print(f"[model] resuming from {ckpt_path} …")
    checkpoint = torch.load(ckpt_path, map_location=device)
    ckpt_qk_norm = checkpoint.get("qk_norm", False)
    model = build_hooked_vit_depth(
        model_name=checkpoint["model_name"],
        img_size=cfg.img_size,
        patch_size=checkpoint.get("patch_size", cfg.patch_size),
        init_std=cfg.init_std,
        use_scaled_init=False,
        qk_norm=ckpt_qk_norm,
        depth=checkpoint.get("depth"),
        num_heads=checkpoint.get("num_heads"),
        embed_dim=checkpoint.get("embed_dim"),
        device=device,
    )
    state_dict = _strip_compile_prefix(checkpoint["model"])
    model.load_state_dict(state_dict)
    iter_num = checkpoint.get("iter_num", 0)
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))

else:
    if _is_master:
        print(f"[model] loading checkpoint {cfg.init_from} …")
    checkpoint = torch.load(cfg.init_from, map_location=device)
    ckpt_qk_norm = checkpoint.get("qk_norm", False)
    model = build_hooked_vit_depth(
        model_name=checkpoint.get("model_name", cfg.model_name),
        img_size=cfg.img_size,
        patch_size=checkpoint.get("patch_size", cfg.patch_size),
        init_std=cfg.init_std,
        use_scaled_init=False,
        qk_norm=ckpt_qk_norm,
        depth=checkpoint.get("depth"),
        num_heads=checkpoint.get("num_heads"),
        embed_dim=checkpoint.get("embed_dim"),
        device=device,
    )
    state_dict = _strip_compile_prefix(checkpoint["model"])
    model.load_state_dict(state_dict)

n_params = sum(p.numel() for p in model.parameters()) / 1e6
if _is_master:
    print(f"[model] {cfg.model_name}  {n_params:.2f}M parameters")

if cfg.compile:
    if _is_master:
        print("[model] compiling with torch.compile (disable for Hessian metrics)")
    model = torch.compile(model)

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
        decay_params = [
            p for n, p in model_.named_parameters()
            if p.requires_grad and p.ndim >= 2
        ]
        no_decay_params = [
            p for n, p in model_.named_parameters()
            if p.requires_grad and p.ndim < 2
        ]
        param_groups = [
            {"params": decay_params,    "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(
            param_groups,
            lr=cfg.learning_rate,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.eps,
        )
    elif cfg.optimizer.lower() == "sgd":
        return torch.optim.SGD(model_.parameters(), lr=cfg.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer '{cfg.optimizer}'")


optimizer = _build_optimizer(model)

# ---------------------------------------------------------------------------
# 6.  LR schedule  (cosine with linear warm-up)
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
# 8.  Helpers
# ---------------------------------------------------------------------------
from src.helpers import (  # noqa: E402
    get_VV_subspace_mask,
    get_curvature_metrics,
    get_attention_entropy,
    scale_invariant_log_loss,
)

_raw_model = model.module if use_ddp else model
vv_mask = get_VV_subspace_mask(_raw_model).to(device)


def _depth_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[float, float]:
    """
    Compute RMSE (log-scale) and δ<1.25 accuracy for a batch.

    Args:
        pred:   ``(B, 1, H, W)`` predicted depth in metres.
        target: ``(B, 1, H, W)`` ground-truth depth in metres.

    Returns:
        (rmse_log, delta1) where
          * ``rmse_log`` = sqrt(mean( (log pred - log gt)^2 )) over valid pixels.
          * ``delta1``   = fraction of valid pixels where
                           max(pred/gt, gt/pred) < 1.25  (expressed in %).
    """
    pred   = pred.squeeze(1)
    target = target.squeeze(1)
    valid  = (target > 0.0) & torch.isfinite(target)

    p_v = pred[valid].clamp(min=eps)
    g_v = target[valid].clamp(min=eps)

    if g_v.numel() == 0:
        return 0.0, 0.0

    log_diff = torch.log(p_v) - torch.log(g_v)
    rmse_log = torch.sqrt((log_diff ** 2).mean()).item()

    ratio = torch.max(p_v / g_v, g_v / p_v)
    delta1 = (ratio < 1.25).float().mean().item() * 100.0

    return rmse_log, delta1


@torch.no_grad()
def estimate_val_metrics(n_batches: int = 10) -> tuple[float, float, float]:
    """Return (mean_val_silog, mean_rmse_log, mean_delta1) over n_batches."""
    _raw_model.eval()
    silog_losses, rmses, delta1s = [], [], []
    val_iter_local = iter(val_loader)
    for _ in range(n_batches):
        try:
            xv, yv = next(val_iter_local)
        except StopIteration:
            break
        xv, yv = xv.to(device), yv.to(device)
        with ctx:
            pred = _raw_model(xv)
        lv = scale_invariant_log_loss(pred, yv, lam=cfg.silog_lambda)
        silog_losses.append(lv.item())
        rmse, d1 = _depth_metrics(pred, yv)
        rmses.append(rmse)
        delta1s.append(d1)
    _raw_model.train()
    return (
        float(np.mean(silog_losses)) if silog_losses else float("inf"),
        float(np.mean(rmses))        if rmses       else 0.0,
        float(np.mean(delta1s))      if delta1s     else 0.0,
    )


def _save_checkpoint(suffix: str = "ckpt") -> None:
    if not use_ddp or rank == 0:
        checkpoint = {
            "model":          _raw_model.state_dict(),
            "optimizer":      optimizer.state_dict(),
            "iter_num":       iter_num,
            "best_val_loss":  best_val_loss,
            "model_name":     cfg.model_name,
            "qk_norm":        cfg.qk_norm,
            "depth":          cfg.depth,
            "num_heads":      cfg.num_heads,
            "embed_dim":      cfg.embed_dim,
            "patch_size":     cfg.patch_size,
            "img_size":       cfg.img_size,
            "config":         vars(cfg),
        }
        path = os.path.join(run_out_dir, f"{suffix}.pt")
        torch.save(checkpoint, path)
        print(f"[ckpt] saved → {path}")


# ---------------------------------------------------------------------------
# 9.  Training loop
# ---------------------------------------------------------------------------
n_layers = len(_raw_model.blocks)

history: dict[str, list] = {
    "loss":       [],
    "val_loss":   [],
    "val_rmse":   [],
    "val_delta1": [],
    "hessian":    [],
    "prec_h":     [],
    "hessian_vv": [],
    "gn":         [],
    "fd":         [],
    "diag_h":     [],
    "fisher":     [],
    "bfgs":       [],
    "kfac":       [],
    "entropy":    [],
    "lr":         [],
}

if _is_master:
    print(f"\n[train] starting — max_iters={cfg.max_iters}  device={device}\n")
t0 = time.time()
model.train()

X, Y = next(train_iter)
X, Y = X.to(device), Y.to(device)

for iter_num in range(iter_num, cfg.max_iters):

    # ---- LR update ----
    lr = get_lr(iter_num)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    # ---- Temperature-shift intervention ----
    if cfg.temp_shift_step >= 0 and iter_num == cfg.temp_shift_step:
        set_attention_temperature(model, cfg.temp_shift_factor)
        if _is_master:
            print(
                f"[temp_shift] iter {iter_num}: temperature={cfg.temp_shift_factor:.4g}"
            )
        if cfg.wandb_log and (not use_ddp or rank == 0):
            wandb.log(
                {"intervention/temp_shift_factor": cfg.temp_shift_factor},
                step=iter_num,
            )

    # ---- Periodic evaluation ----
    if iter_num % cfg.eval_interval == 0 or iter_num == cfg.max_iters - 1:
        val_silog, val_rmse, val_delta1 = estimate_val_metrics()
        history["val_loss"].append((iter_num, val_silog))
        history["val_rmse"].append((iter_num, val_rmse))
        history["val_delta1"].append((iter_num, val_delta1))
        if _is_master:
            print(
                f"[eval] iter {iter_num:5d} | val_silog {val_silog:.4f} "
                f"| rmse_log {val_rmse:.4f} | δ<1.25 {val_delta1:.2f}% "
                f"| best {best_val_loss:.4f}"
            )
        if cfg.wandb_log and (not use_ddp or rank == 0):
            wandb.log(
                {
                    "val/silog":   val_silog,
                    "val/rmse":    val_rmse,
                    "val/delta1":  val_delta1,
                },
                step=iter_num,
            )
        if (cfg.save_checkpoint or val_silog < best_val_loss) and iter_num > 0:
            if val_silog < best_val_loss:
                best_val_loss = val_silog

    if (
        cfg.checkpoint_interval > 0
        and iter_num % cfg.checkpoint_interval == 0
        and iter_num > 0
    ):
        _save_checkpoint(f"ckpt_iter{iter_num:06d}")

    # ---- Curvature metrics ----
    curvature: dict[str, float] = {
        k: 0.0
        for k in ("hessian", "prec_h", "hessian_vv", "gn", "fd",
                  "diag_h", "fisher", "bfgs", "kfac")
    }
    if iter_num % cfg.hessian_freq == 0:
        _raw_model.train()
        optimizer.zero_grad()
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ) if device != "cpu" else nullcontext():
            pred_h = _raw_model(X)
            loss_for_hess = scale_invariant_log_loss(
                pred_h, Y, lam=cfg.silog_lambda
            )

        try:
            curvature = get_curvature_metrics(
                _raw_model,
                optimizer,
                X,
                Y,
                loss_for_hess,
                vv_mask,
                max_iter=cfg.hessian_max_iter,
                compute_fd=cfg.compute_fd,
            )
        except Exception as exc:
            if _is_master:
                print(f"[warn] curvature metrics failed at iter {iter_num}: {exc}")
        finally:
            optimizer.zero_grad()

    for k in ("hessian", "prec_h", "hessian_vv", "gn", "fd", "diag_h",
              "fisher", "bfgs", "kfac"):
        history[k].append(curvature[k])

    # ---- Standard training step ----
    layer_entropies: list[float] = [0.0] * n_layers
    _need_entropy = iter_num % cfg.entropy_freq == 0

    if _need_entropy:
        for blk in _raw_model.blocks:
            blk.attn._cache_attn = True

    optimizer.zero_grad(set_to_none=True)
    with ctx:
        pred = _raw_model(X)
        loss = scale_invariant_log_loss(pred, Y, lam=cfg.silog_lambda)

        if _need_entropy:
            with torch.no_grad():
                layer_entropies = get_attention_entropy(_raw_model)
            for blk in _raw_model.blocks:
                blk.attn._cache_attn = False
                blk.attn.last_att = None

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
    X, Y = X.to(device), Y.to(device)

    # ---- Logging ----
    if iter_num % cfg.log_interval == 0:
        dt = time.time() - t0
        t0 = time.time()
        if _is_master:
            print(
                f"iter {iter_num:5d} | silog {loss_val:.4f} "
                f"| lr {lr:.2e} | dt {dt * 1000:.1f}ms"
            )
            if iter_num % cfg.hessian_freq == 0:
                print(
                    f"  H {curvature['hessian']:.3f} | H~(prec) {curvature['prec_h']:.3f} "
                    f"| H_VV {curvature['hessian_vv']:.3f} | GN {curvature['gn']:.3f} "
                    f"| BFGS {curvature['bfgs']:.3f} | FD {curvature['fd']:.3f} "
                    f"| DiagH {curvature['diag_h']:.3f} | Fisher {curvature['fisher']:.3f} "
                    f"| KFAC {curvature['kfac']:.3f}"
                )
        if cfg.wandb_log and (not use_ddp or rank == 0):
            log_dict: dict = {
                "train/silog": loss_val,
                "train/lr":    lr,
            }
            if iter_num % cfg.hessian_freq == 0:
                log_dict.update(
                    {
                        "hessian/lambda_max": curvature["hessian"],
                        "hessian/prec_H":     curvature["prec_h"],
                        "hessian/H_VV":       curvature["hessian_vv"],
                        "hessian/GN":         curvature["gn"],
                        "hessian/FD":         curvature["fd"],
                        "hessian/diag_H":     curvature["diag_h"],
                        "hessian/fisher":     curvature["fisher"],
                        "hessian/BFGS":       curvature["bfgs"],
                        "hessian/KFAC":       curvature["kfac"],
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

if not use_ddp or rank == 0:
    history_path = os.path.join(run_out_dir, "history.pkl")
    with open(history_path, "wb") as f:
        pickle.dump(history, f)
    print(f"[done] history saved → {history_path}")

    if cfg.wandb_log:
        wandb.finish()

# ---------------------------------------------------------------------------
# 11.  Post-training plots
# ---------------------------------------------------------------------------
if not use_ddp or rank == 0:
    from plot_history import plot_history  # noqa: E402

    plot_history(
        pkl_path=history_path,
        out_dir=run_out_dir,
        hessian_freq=cfg.hessian_freq,
        entropy_freq=cfg.entropy_freq,
        skip_intv=True,
        lam=100.0,
        compute_fd=cfg.compute_fd,
    )
