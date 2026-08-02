"""
base_train.py — ViT entropy-collapse training script.

Default config: ViT-B/16 on CIFAR-100 with a DeiT recipe (see TrainConfig).

Logged every ``log_interval``:
  * Train loss / accuracy
  * Learning rate

Logged every ``eval_interval``:
  * Val loss / accuracy

Logged every ``hessian_intv``:
  * Curvature proxies — λ_max of H, Prec_H, H_VV, GN, Diag_H, Fisher
    (+  KFAC, BFGS and FD when compute_fd=True)

Logged every ``entropy_intv``:
  * Per-layer attention entropy

Usage
-----
Default - pilot run (ViT-B/16, CIFAR-100)::

    python base_train.py

Named preset::

    python base_train.py config=imagenet1k_base data_dir=/data/imagenet

Override individual fields::

    python base_train.py config=imagenet1k_base learning_rate=3e-4 max_iters=15000

Multi-GPU via torchrun::

    torchrun --nproc_per_node=4 base_train.py config=imagenet1k_base

All config fields can be overridden as ``key=value`` arguments.
"""

from __future__ import annotations

import os
import sys
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

# True on rank-0 (or single-GPU); works before dist.init_process_group.
_is_master = int(os.environ.get("RANK", "0")) == 0

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
# 1.  Configuration
# ---------------------------------------------------------------------------
from configs.train_config import TrainConfig, CONFIGS
from common.train_utils import (
    resolve_config,
    setup_ddp_and_run_dir,
    build_adamw_with_decay_split,
    cosine_warmup_lr,
    wrap_model_ddp,
)
from common.helpers import strip_compile_prefix
from common.pretrain import run_training, estimate_classification_val_metrics

cfg = resolve_config(TrainConfig, CONFIGS, _is_master)


def _dataset_defaults(dataset_name: str) -> tuple[int, int] | None:
    ds = dataset_name.lower()
    if ds == "cifar100":
        return 100, 32
    if ds == "imagenet1k":
        return 1000, 224
    return None


dataset_defaults = _dataset_defaults(cfg.dataset)
expected_classes: int | None = None
if dataset_defaults is not None:
    expected_classes, expected_img_size = dataset_defaults
    if cfg.num_classes != expected_classes:
        if _is_master:
            print(
                f"[config] dataset='{cfg.dataset}' => overriding num_classes "
                f"{cfg.num_classes} -> {expected_classes}"
            )
        cfg.num_classes = expected_classes
    if cfg.img_size != expected_img_size:
        if _is_master:
            print(
                f"[config] dataset='{cfg.dataset}' => overriding img_size "
                f"{cfg.img_size} -> {expected_img_size}"
            )
        cfg.img_size = expected_img_size
elif cfg.num_classes is None or cfg.img_size is None:
    raise ValueError(
        f"Unknown dataset '{cfg.dataset}'. Please set num_classes and img_size explicitly."
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
    # FP8 via torch.amp.autocast is not yet natively supported.
    # Use NVIDIA TransformerEngine for true FP8 training on H100/H200.
    # Falling back to bfloat16 for AMP.
    if _is_master:
        print(
            f"[warn] dtype='{cfg.dtype}' — FP8 autocast is not natively supported "
            "by torch.amp. Install TransformerEngine for true FP8 training. "
            "Falling back to bfloat16."
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

use_ddp, rank, world_size, local_rank, device, run_out_dir = setup_ddp_and_run_dir(cfg, _is_master)

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
# 4.  Model
# ---------------------------------------------------------------------------
from src.model import build_hooked_vit, set_attention_temperature  

iter_num = 0
best_val_loss = float("inf")


if cfg.init_from == "scratch":
    if _is_master:
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
    if _is_master:
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
        if _is_master:
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
    state_dict = strip_compile_prefix(checkpoint["model"])
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
    # Older checkpoints (before qk_norm was saved) default to False.
    ckpt_qk_norm = checkpoint.get("qk_norm", False)
    if "qk_norm" not in checkpoint and cfg.qk_norm != ckpt_qk_norm:
        if _is_master:
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
    state_dict = strip_compile_prefix(checkpoint["model"])
    model.load_state_dict(state_dict)

n_params = sum(p.numel() for p in model.parameters()) / 1e6
if _is_master:
    print(f"[model] {cfg.model_name}  {n_params:.2f}M parameters")

if cfg.compile:
    if _is_master:
        print("[model] compiling with torch.compile (disable for Hessian metrics)")
    model = torch.compile(model)

model = wrap_model_ddp(model, use_ddp, device, local_rank)

# ---------------------------------------------------------------------------
# 5.  Optimiser
# ---------------------------------------------------------------------------
optimizer = build_adamw_with_decay_split(model, cfg)


# ---------------------------------------------------------------------------
# 6.  LR schedule  (cosine with linear warm-up)
# ---------------------------------------------------------------------------
def update_schedule(optimizer_: torch.optim.Optimizer, it: int) -> float:
    lr = cosine_warmup_lr(it, cfg)
    for pg in optimizer_.param_groups:
        pg["lr"] = lr
    return lr


# ---------------------------------------------------------------------------
# 7.  Loss / metrics step & validation estimation
# ---------------------------------------------------------------------------
_raw_model = model.module if use_ddp else model


def step_fn(raw_model: torch.nn.Module, X: torch.Tensor, Y: torch.Tensor):
    logits = raw_model(X)
    loss = F.cross_entropy(logits, Y, label_smoothing=cfg.label_smoothing)
    acc = 100.0 * (logits.argmax(dim=-1) == Y).float().mean().item()
    return loss, {"acc": acc}


def estimate_val() -> dict:
    return estimate_classification_val_metrics(_raw_model, val_loader, ctx, device)


# ---------------------------------------------------------------------------
# 8.  Shared training loop (common/pretrain.py)
# ---------------------------------------------------------------------------
ckpt_extra_fields = {
    "model_name": cfg.model_name,
    "num_classes": cfg.num_classes,
    "qk_norm": cfg.qk_norm,
    "depth": cfg.depth,
    "num_heads": cfg.num_heads,
    "embed_dim": cfg.embed_dim,
    "patch_size": cfg.patch_size,
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
    has_accuracy=True,
    qq_kk_masks=None,
    save_periodic_ckpt=False,
    initial_iter_num=iter_num,
    initial_best_val_loss=best_val_loss,
)

