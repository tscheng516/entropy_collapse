"""
base_train.py — ViT entropy-collapse training script.

This script trains a HookedViT model (timm ViT-Small by default) while
logging:
  * Train loss / accuracy — every iteration
  * Val loss / accuracy   — every ``eval_interval`` iterations
  * Hessian proxies (H, H_tilde, H_VV, H_GN, FD)
                         — every ``hessian_freq`` iterations
  * Per-layer attention entropy
                         — every ``entropy_freq`` iterations

All metrics are emitted to stdout and (optionally) to Weights & Biases.
Checkpoints are saved to ``out_dir`` at regular intervals and whenever
the validation loss improves.

Usage
-----
Default config (CIFAR-10, ViT-Small/16, AdamW)::

    python base_train.py

Override individual flags::

    python base_train.py --lr 5e-4 --optim sgd --max_it 2000

The override syntax is identical to NanoGPT's ``configurator.py`` convention:
any ``key=value`` argument is ``ast.literal_eval``'d and injected into the
config dataclass.  For argparse-style flags, use short names such as
``--cp``, ``--optim``, ``--lr``, ``--max_it``, ``--wandb``, and ``--z``.

Setup
-----
1. Install dependencies::

    pip install -r requirements.txt

2. Run from the repo root::

    python base_train.py

   or with torchrun::

    torchrun --nproc_per_node=4 base_train.py \\
        dataset=imagenet data_dir=/data/imagenet num_classes=1000 \\
        --wandb true --lr 1e-3 --max_it 5000

Note: CIFAR-10 images are 32×32 but are up-sampled to 224×224 by the
data pipeline so the same ViT-Small/16 architecture can be used for a
lightweight pilot without any model changes.
"""

from __future__ import annotations

import ast
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
import torch.nn.functional as F
import torch.distributed as dist

# ---------------------------------------------------------------------------
# 0.  Path setup — add ViT/ so ``configs`` and ``src`` sub-packages resolve.
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
from configs.train_config import TrainConfig, CONFIGS  # noqa: E402

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
        if hasattr(cfg, key):
            try:
                setattr(cfg, key, ast.literal_eval(val))
            except (ValueError, SyntaxError):
                setattr(cfg, key, val)
        else:
            print(f"[warn] unknown config key '{key}', ignoring.")

import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--cp", type=str)
parser.add_argument("--optim", type=str)
parser.add_argument("--lr", type=float)
parser.add_argument("--max_it", type=int)
parser.add_argument("--wandb", type=str)
parser.add_argument("--z", type=float)
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


def _expected_num_classes(dataset_name: str) -> int | None:
    ds = dataset_name.lower()
    if ds == "cifar10":
        return 10
    if ds == "cifar100":
        return 100
    if ds in ("imagenet", "imagenet_hf", "imagenet1k_hf", "hf_imagenet"):
        return 1000
    return None


expected_classes = _expected_num_classes(cfg.dataset)
if cfg.init_from == "scratch" and expected_classes is not None and cfg.num_classes != expected_classes:
    print(
        f"[warn] dataset='{cfg.dataset}' typically uses num_classes={expected_classes}, "
        f"but got {cfg.num_classes}; overriding to {expected_classes} for compatibility."
    )
    cfg.num_classes = expected_classes

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
# 3.  Data
# ---------------------------------------------------------------------------
from src.data_utils import load_data, infinite_loader  # noqa: E402

train_loader, val_loader = load_data(
    dataset=cfg.dataset,
    data_dir=cfg.data_dir,
    img_size=cfg.img_size,
    batch_size=cfg.batch_size,
    num_workers=cfg.num_workers,
)

train_iter = infinite_loader(train_loader)
print(
    f"[data] dataset={cfg.dataset}  "
    f"train_batches={len(train_loader)}  "
    f"val_batches={len(val_loader)}"
)

# ---------------------------------------------------------------------------
# 4.  Model
# ---------------------------------------------------------------------------
from src.model import build_hooked_vit  # noqa: E402

iter_num = 0
best_val_loss = float("inf")


def _strip_compile_prefix(state_dict: dict) -> dict:
    """Remove the ``_orig_mod.`` key prefix added by ``torch.compile``."""
    return {
        (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
        for k, v in state_dict.items()
    }


if cfg.init_from == "scratch":
    print(f"[model] building {cfg.model_name} from scratch …")
    model = build_hooked_vit(
        model_name=cfg.model_name,
        num_classes=cfg.num_classes,
        pretrained=False,
        img_size=cfg.img_size,
        init_std=cfg.init_std,
        use_scaled_init=cfg.use_scaled_init,
        qk_norm=cfg.qk_norm,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        embed_dim=cfg.embed_dim,
        patch_size=cfg.patch_size,
        device=device,
    )

elif cfg.init_from == "resume":
    ckpt_path = os.path.join(cfg.out_dir, "ckpt.pt")
    print(f"[model] resuming from {ckpt_path} …")
    checkpoint = torch.load(ckpt_path, map_location=device)
    if expected_classes is not None and checkpoint["num_classes"] != expected_classes:
        raise ValueError(
            f"dataset='{cfg.dataset}' expects num_classes={expected_classes}, "
            f"but checkpoint has num_classes={checkpoint['num_classes']}. "
            "Use a matching dataset/checkpoint pair or train from scratch."
        )
    # Older checkpoints (before qk_norm was saved) default to False.
    ckpt_qk_norm = checkpoint.get("qk_norm", False)
    if "qk_norm" not in checkpoint and cfg.qk_norm != ckpt_qk_norm:
        print(
            f"[warn] checkpoint has no 'qk_norm' field (assumed False); "
            f"cfg.qk_norm={cfg.qk_norm} will be ignored — using False to "
            "match the checkpoint architecture."
        )
    model = build_hooked_vit(
        model_name=checkpoint["model_name"],
        num_classes=checkpoint["num_classes"],
        pretrained=False,
        img_size=cfg.img_size,
        init_std=cfg.init_std,
        use_scaled_init=False,
        qk_norm=ckpt_qk_norm,
        depth=checkpoint.get("depth"),
        num_heads=checkpoint.get("num_heads"),
        embed_dim=checkpoint.get("embed_dim"),
        patch_size=checkpoint.get("patch_size"),
        device=device,
    )
    state_dict = _strip_compile_prefix(checkpoint["model"])
    model.load_state_dict(state_dict)
    iter_num = checkpoint.get("iter_num", 0)
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))

else:
    print(f"[model] fine-tuning from checkpoint {cfg.init_from} …")
    checkpoint = torch.load(cfg.init_from, map_location=device)
    ckpt_num_classes = checkpoint.get("num_classes", cfg.num_classes)
    if expected_classes is not None and ckpt_num_classes != expected_classes:
        raise ValueError(
            f"dataset='{cfg.dataset}' expects num_classes={expected_classes}, "
            f"but checkpoint has num_classes={ckpt_num_classes}. "
            "Use a matching dataset/checkpoint pair or start from scratch."
        )
    # Older checkpoints (before qk_norm was saved) default to False.
    ckpt_qk_norm = checkpoint.get("qk_norm", False)
    if "qk_norm" not in checkpoint and cfg.qk_norm != ckpt_qk_norm:
        print(
            f"[warn] checkpoint has no 'qk_norm' field (assumed False); "
            f"cfg.qk_norm={cfg.qk_norm} will be ignored — using False to "
            "match the checkpoint architecture."
        )
    model = build_hooked_vit(
        model_name=checkpoint.get("model_name", cfg.model_name),
        num_classes=checkpoint.get("num_classes", cfg.num_classes),
        pretrained=False,
        img_size=cfg.img_size,
        init_std=cfg.init_std,
        use_scaled_init=False,
        qk_norm=ckpt_qk_norm,
        depth=checkpoint.get("depth"),
        num_heads=checkpoint.get("num_heads"),
        embed_dim=checkpoint.get("embed_dim"),
        patch_size=checkpoint.get("patch_size"),
        device=device,
    )
    state_dict = _strip_compile_prefix(checkpoint["model"])
    model.load_state_dict(state_dict)

n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"[model] {cfg.model_name}  {n_params:.2f}M parameters")

if cfg.compile:
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
# 8.  Helper: Hessian metrics & entropy
# ---------------------------------------------------------------------------
from src.helpers import (  # noqa: E402
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
            "model_name": cfg.model_name,
            "num_classes": cfg.num_classes,
            "qk_norm": cfg.qk_norm,
            "depth": cfg.depth,
            "num_heads": cfg.num_heads,
            "embed_dim": cfg.embed_dim,
            "patch_size": cfg.patch_size,
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
    "entropy": [],
    "lr": [],
}

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

    # ---- Periodic evaluation ----
    if iter_num % cfg.eval_interval == 0 or iter_num == cfg.max_iters - 1:
        val_loss, val_acc = estimate_val_metrics()
        history["val_loss"].append((iter_num, val_loss))
        history["val_acc"].append((iter_num, val_acc))
        print(
            f"[eval] iter {iter_num:5d} | val_loss {val_loss:.4f} "
            f"| val_acc {val_acc:.2f}%  | best {best_val_loss:.4f}"
        )
        if cfg.wandb_log and (not use_ddp or rank == 0):
            wandb.log(
                {"val/loss": val_loss, "val/acc": val_acc},
                step=iter_num,
            )

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
        _raw_model.train()
        optimizer.zero_grad()
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ) if device != "cpu" else nullcontext():
            logits_h = _raw_model(X)
            loss_for_hess = F.cross_entropy(logits_h, Y)

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
            print(f"[warn] curvature metrics failed at iter {iter_num}: {exc}")
        finally:
            optimizer.zero_grad()

    for k in ("hessian", "prec_h", "hessian_vv", "gn", "fd"):
        history[k].append(curvature[k])

    # ---- Standard training step ----
    layer_entropies: list[float] = [0.0] * n_layers

    optimizer.zero_grad(set_to_none=True)
    with ctx:
        logits = _raw_model(X)
        loss = F.cross_entropy(logits, Y)

        if iter_num % cfg.entropy_freq == 0:
            with torch.no_grad():
                layer_entropies = get_attention_entropy(_raw_model)

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
        print(
            f"iter {iter_num:5d} | loss {loss_val:.4f} | acc {train_acc:.1f}% "
            f"| lr {lr:.2e} | dt {dt * 1000:.1f}ms"
        )
        if iter_num % cfg.hessian_freq == 0:
            print(
                f"  H {curvature['hessian']:.3f} | H_VV {curvature['hessian_vv']:.3f} "
                f"| GN {curvature['gn']:.3f}"
            )
        if cfg.wandb_log and (not use_ddp or rank == 0):
            log_dict: dict = {
                "train/loss": loss_val,
                "train/acc": train_acc,
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
