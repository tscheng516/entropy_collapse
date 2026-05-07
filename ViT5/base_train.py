"""
base_train.py — ViT-5 entropy-collapse training script.

Default config: ViT-5-Base on CIFAR-100 (see TrainConfig in configs/train_config.py).
Select a different preset via ``config=<name>``; available presets:
  cifar100_base | imagenet1k_base

Logged every iteration:
  * Train loss / accuracy
  * Learning rate

Logged every ``eval_interval``:
  * Val loss / accuracy

Logged every ``hessian_intv``:
  * Curvature proxies — λ_max of H, Prec_H, H_VV, GN, Diag_H, Fisher, KFAC
    (+ BFGS / FD when compute_fd=True)

Logged every ``entropy_intv``:
  * Per-layer attention entropy

Architecture: ViT-5-Base (fixed)
  embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
  RMSNorm, QK-norm, 2-D RoPE, 4 register tokens, layer-scale.

Usage
-----
Default (ViT-5-Base, CIFAR-100)::

    python base_train.py

Named preset::

    python base_train.py config=imagenet1k_base data_dir=/data/imagenet

Override individual fields::

    python base_train.py --lr 1e-3 --max_it 10000 hessian_intv=100

Multi-GPU via torchrun::

    torchrun --nproc_per_node=4 base_train.py config=imagenet1k_base \\
        data_dir=/data/imagenet --wandb true

Key=value arguments are ast.literal_eval'd (NanoGPT-style).
Argparse shortcuts: --lr, --optim, --max_it, --wandb, --cp, --temp_shift.
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
# 0.  Path setup — add ViT5/ so ``configs`` and ``src`` sub-packages resolve.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# ---------------------------------------------------------------------------
# 1.  Default configuration (loaded from dataclass, then CLI overrides)
# ---------------------------------------------------------------------------
from configs.train_config import TrainConfig, CONFIGS

# Allow ``config=<preset>`` as a CLI argument to select a named config class
# before the dataclass is instantiated.  This must be parsed first so that
# preset defaults are in place before individual key=value overrides are
# applied further below.
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

# NanoGPT-style CLI overrides: python base_train.py learning_rate=1e-4 ...
# Short argparse flags are also supported, e.g. ``--lr 1e-4 --optim adamw``.
for arg in sys.argv[1:]:
    if "=" in arg:
        key, val = arg.split("=", 1)
        if key == "config":
            continue  # already handled above
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
        "applied to all attention heads (-1 disables, default: cfg.temp_shift_step)."
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


def _dataset_num_classes(dataset_name: str) -> int | None:
    """Return the canonical num_classes for known datasets, else None."""
    ds = dataset_name.lower()
    if ds == "cifar10":
        return 10
    if ds == "cifar100":
        return 100
    if ds in ("imagenet", "imagenet1k", "imagenet_hf", "imagenet1k_hf", "hf_imagenet"):
        return 1000
    return None


expected_classes: int | None = _dataset_num_classes(cfg.dataset)
if expected_classes is not None:
    if cfg.num_classes != expected_classes:
        if _is_master:
            print(
                f"[config] dataset='{cfg.dataset}' => overriding num_classes "
                f"{cfg.num_classes} -> {expected_classes}"
            )
        cfg.num_classes = expected_classes
elif cfg.num_classes is None:
    raise ValueError(
        f"Unknown dataset '{cfg.dataset}'. Please set num_classes explicitly."
    )

if cfg.img_size is None:
    raise ValueError(
        f"img_size is None for dataset='{cfg.dataset}'. "
        "Please set img_size explicitly or use a named preset config."
    )

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
_FP8_ALIASES = {"float8", "float8_e4m3fn", "float8_e5m2"}
if cfg.dtype in _FP8_ALIASES:
    if _is_master:
        print(
            f"[warn] dtype='{cfg.dtype}' — FP8 autocast is not natively supported "
            "by torch.amp. Falling back to bfloat16."
        )
    ptdtype = torch.bfloat16
else:
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
    if not cfg.wandb_run_name or cfg.wandb_run_name == "run":
        cfg.wandb_run_name = run_id

if rank == 0:
    print(f"[io] outputs → {run_out_dir}")

# ---------------------------------------------------------------------------
# 3.  Data
# ---------------------------------------------------------------------------
from src.data_utils import load_data, infinite_loader

train_loader, val_loader, train_sampler = load_data(
    dataset=cfg.dataset,
    data_dir=cfg.data_dir,
    img_size=cfg.img_size,
    batch_size=cfg.batch_size,
    num_workers=cfg.num_workers,
)

train_iter = infinite_loader(train_loader, train_sampler)
if _is_master:
    print(
        f"[data] dataset={cfg.dataset}  "
        f"train_batches={len(train_loader)}  "
        f"val_batches={len(val_loader)}"
    )

# ---------------------------------------------------------------------------
# 4.  Model — ViT-5-Base (fixed architecture)
# ---------------------------------------------------------------------------
from src.model import build_hooked_vit5, set_attention_temperature

iter_num = 0
best_val_loss = float("inf")


def _strip_compile_prefix(state_dict: dict) -> dict:
    """Remove the ``_orig_mod.`` key prefix added by ``torch.compile``."""
    return {
        (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
        for k, v in state_dict.items()
    }


if cfg.init_from == "scratch":
    if _is_master:
        print("[model] building vit5_base from scratch …")
    model = build_hooked_vit5(
        num_classes=cfg.num_classes,
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        drop_path_rate=cfg.drop_path_rate,
        num_registers=cfg.num_registers,
        init_std=cfg.init_std,
        use_scaled_init=cfg.use_scaled_init,
        device=device,
    )

elif cfg.init_from == "resume":
    ckpt_path = os.path.join(cfg.out_dir, "ckpt.pt")
    if _is_master:
        print(f"[model] resuming from {ckpt_path} …")
    checkpoint = torch.load(ckpt_path, map_location=device)
    if expected_classes is not None and checkpoint["num_classes"] != expected_classes:
        raise ValueError(
            f"dataset='{cfg.dataset}' expects num_classes={expected_classes}, "
            f"but checkpoint has num_classes={checkpoint['num_classes']}. "
            "Use a matching dataset/checkpoint pair or train from scratch."
        )
    model = build_hooked_vit5(
        num_classes=checkpoint["num_classes"],
        img_size=cfg.img_size,
        patch_size=checkpoint.get("patch_size", cfg.patch_size),
        drop_path_rate=checkpoint.get("drop_path_rate", cfg.drop_path_rate),
        num_registers=checkpoint.get("num_registers", cfg.num_registers),
        init_std=cfg.init_std,
        use_scaled_init=False,
        device=device,
    )
    state_dict = _strip_compile_prefix(checkpoint["model"])
    model.load_state_dict(state_dict)
    iter_num = checkpoint.get("iter_num", 0)
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))

else:
    if _is_master:
        print(f"[model] fine-tuning from checkpoint {cfg.init_from} …")
    checkpoint = torch.load(cfg.init_from, map_location=device)
    ckpt_num_classes = checkpoint.get("num_classes", cfg.num_classes)
    if expected_classes is not None and ckpt_num_classes != expected_classes:
        raise ValueError(
            f"dataset='{cfg.dataset}' expects num_classes={expected_classes}, "
            f"but checkpoint has num_classes={ckpt_num_classes}. "
            "Use a matching dataset/checkpoint pair or start from scratch."
        )
    model = build_hooked_vit5(
        num_classes=ckpt_num_classes,
        img_size=cfg.img_size,
        patch_size=checkpoint.get("patch_size", cfg.patch_size),
        drop_path_rate=checkpoint.get("drop_path_rate", cfg.drop_path_rate),
        num_registers=checkpoint.get("num_registers", cfg.num_registers),
        init_std=cfg.init_std,
        use_scaled_init=False,
        device=device,
    )
    state_dict = _strip_compile_prefix(checkpoint["model"])
    model.load_state_dict(state_dict)

n_params = sum(p.numel() for p in model.parameters()) / 1e6
if _is_master:
    print(f"[model] vit5_base  {n_params:.2f}M parameters")

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
        # Weight-decay split: apply only to weight tensors (not biases / norms)
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
# 8.  Helpers: curvature (spectral-norm) metrics & attention entropy
#
#   get_curvature_metrics  returns λ_max estimates for nine curvature proxies:
#     hessian   — λ_max(H)               exact Hessian, power iteration
#     prec_h    — λ_max(D^{-½} H D^{-½}) Adam-preconditioned Hessian
#     hessian_vv— λ_max(H_VV)            H restricted to value-proj subspace
#     gn        — λ_max(H_GN)            Gauss-Newton (J^T H_L J)
#     bfgs      — λ_max(H)               central-difference FD (O(ε²))
#     fd        — λ_max(H)               forward-difference FD (O(ε))
#     diag_h    — max(diag(H))           Bekas–Kokiopoulou–Saad estimator
#     fisher    — λ_max(F)               empirical Fisher
#     kfac      — max λ_max(A)·λ_max(G)  K-FAC Kronecker proxy
# ---------------------------------------------------------------------------
from common.helpers import (
    get_VV_subspace_mask,
    get_curvature_metrics,
    get_attention_entropy,
)

# Unwrap DDP to get the underlying module for mask / entropy helpers
_raw_model = model.module if use_ddp else model
vv_mask = get_VV_subspace_mask(_raw_model).to(device)


@torch.no_grad()
def estimate_val_metrics(n_batches: int = 10) -> tuple[float, float]:
    """Return (mean_val_loss, top1_accuracy_percent) over n_batches."""
    _raw_model.eval()
    losses, correct, total = [], 0, 0
    val_iter_local = iter(val_loader)
    for _ in range(n_batches):
        try:
            xv, yv = next(val_iter_local)
        except StopIteration:
            break
        xv, yv = xv.to(device), yv.to(device)
        with ctx:
            logits = _raw_model(xv)
        lv = F.cross_entropy(logits, yv)
        losses.append(lv.item())
        correct += (logits.argmax(dim=-1) == yv).sum().item()
        total += yv.size(0)
    _raw_model.train()
    val_loss = float(np.mean(losses)) if losses else float("inf")
    val_acc = 100.0 * correct / total if total > 0 else 0.0
    return val_loss, val_acc


def _save_checkpoint(suffix: str = "ckpt") -> None:
    if not use_ddp or rank == 0:
        checkpoint = {
            "model": _raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter_num": iter_num,
            "best_val_loss": best_val_loss,
            # ViT-5-Base architecture identifiers
            "model_arch": "vit5_base",
            "num_classes": cfg.num_classes,
            "img_size": cfg.img_size,
            "patch_size": cfg.patch_size,
            "drop_path_rate": cfg.drop_path_rate,
            "num_registers": cfg.num_registers,
            "qk_norm": cfg.qk_norm,
            "config": vars(cfg),
        }
        path = os.path.join(run_out_dir, f"{suffix}.pt")
        torch.save(checkpoint, path)
        print(f"[ckpt] saved → {path}")


# ---------------------------------------------------------------------------
# 9.  Training loop
# ---------------------------------------------------------------------------
n_layers = len(_raw_model.blocks)

history: dict[str, list] = {
    "loss": [],
    "acc": [],
    "val_loss": [],
    "val_acc": [],
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
                f"[temp_shift] iter {iter_num}: applied temperature={cfg.temp_shift_factor:.4g} "
                f"to all attention heads"
            )
        if cfg.wandb_log and (not use_ddp or rank == 0):
            wandb.log({"intervention/temp_shift_factor": cfg.temp_shift_factor}, step=iter_num)

    # ---- Periodic evaluation ----
    if iter_num % cfg.eval_interval == 0 or iter_num == cfg.max_iters - 1:
        val_loss, val_acc = estimate_val_metrics()
        history["val_loss"].append((iter_num, val_loss))
        history["val_acc"].append((iter_num, val_acc))
        if _is_master:
            print(
                f"[eval] iter {iter_num:5d} | val_loss {val_loss:.4f} "
                f"| val_acc {val_acc:.2f}%  | best {best_val_loss:.4f}"
            )
        if cfg.wandb_log and (not use_ddp or rank == 0):
            wandb.log(
                {"val/loss": val_loss, "val/acc": val_acc},
                step=iter_num,
            )

        if (cfg.save_checkpoint or val_loss < best_val_loss) and iter_num > 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # _save_checkpoint("best_ckpt")
            # _save_checkpoint("ckpt")

    if cfg.checkpoint_interval > 0 and iter_num % cfg.checkpoint_interval == 0 and iter_num > 0:
        _save_checkpoint(f"ckpt_iter{iter_num:06d}")

    # ---- Curvature metrics (spectral norm) ----
    # All nine proxies are λ_max (or max-diagonal) estimates; reset to 0 on
    # non-measurement iterations so history entries have consistent length.
    curvature: dict[str, float] = {
        "hessian": 0.0,
        "prec_h": 0.0,
        "hessian_vv": 0.0,
        "gn": 0.0,
        "fd": 0.0,
        "diag_h": 0.0,
        "fisher": 0.0,
        "bfgs": 0.0,
        "kfac": 0.0,
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
                use_grad_ckpt=cfg.use_grad_ckpt,
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

    # Enable attention caching only when entropy will be read.
    if _need_entropy:
        for blk in _raw_model.blocks:
            blk.attn._cache_attn = True

    optimizer.zero_grad(set_to_none=True)
    with ctx:
        logits = _raw_model(X)
        loss = F.cross_entropy(logits, Y, label_smoothing=cfg.label_smoothing)

        if _need_entropy:
            with torch.no_grad():
                layer_entropies = get_attention_entropy(_raw_model)
            for blk in _raw_model.blocks:
                blk.attn._cache_attn = False
                blk.attn.last_att = None  # free cached attention immediately

    loss.backward()
    if cfg.grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()

    loss_val = loss.item()
    with torch.no_grad():
        train_acc = 100.0 * (logits.argmax(dim=-1) == Y).float().mean().item()

    history["loss"].append(loss_val)
    history["acc"].append(train_acc)
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
                f"iter {iter_num:5d} | loss {loss_val:.4f} | acc {train_acc:.1f}% "
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
                "train/acc": train_acc,
                "train/lr": lr,
            }
            if iter_num % cfg.hessian_intv == 0:
                log_dict.update(
                    {
                        "hessian/lambda_max": curvature["hessian"],
                        "hessian/prec_H": curvature["prec_h"],
                        "hessian/H_VV": curvature["hessian_vv"],
                        "hessian/GN": curvature["gn"],
                        "hessian/diag_H": curvature["diag_h"],
                        "hessian/fisher": curvature["fisher"],
                    }
                )
                if cfg.compute_fd:
                    log_dict.update(
                        {
                            "hessian/FD": curvature["fd"],
                            "hessian/BFGS": curvature["bfgs"],
                            "hessian/KFAC": curvature["kfac"],
                        }
                    )
            if iter_num % cfg.entropy_intv == 0:
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
    import dataclasses
    history["config"] = dataclasses.asdict(cfg)
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
    from common.plot_results import plot_results  

    plot_results(
        pkl_path=history_path,
        save_path=os.path.join(run_out_dir, f"plot_results_22.png"),
        hessian_intv=cfg.hessian_intv,
        entropy_intv=cfg.entropy_intv,
        skip_intv=True,
        vs_H_prec=True, 
        compute_fd=cfg.compute_fd,
        fmt="png",
    )
    
# ---------------------------------------------------------------------------
# 12.  DDP teardown
# ---------------------------------------------------------------------------
if use_ddp:
    dist.destroy_process_group()
