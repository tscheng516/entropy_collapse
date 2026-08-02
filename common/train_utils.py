"""Common training utilities shared across ViT/, nanochat/, and ViT5/.

Functions
---------
resolve_config(default_cls, configs, is_master) -> cfg
    Parse ``config=<preset>`` and ``key=value`` CLI overrides.
    No argparse — every dataclass field is addressable directly as key=value.

setup_ddp_and_run_dir(cfg, is_master)
    -> (use_ddp, rank, world_size, local_rank, device, run_out_dir)
    Initialise distributed training (if torchrun) and create a
    timestamped output sub-directory, broadcast across ranks.

init_wandb(cfg, use_ddp, rank)
    Initialise W&B on rank-0 (or single-GPU) if cfg.wandb_log is True.

save_history_and_plot(history, cfg, run_out_dir, use_ddp, rank)
    Serialize history.pkl and call plot_results (rank-0 only).
"""

from __future__ import annotations

import ast
import math
import os
import sys
import time

import torch
import torch.distributed as dist


def resolve_config(default_cls, configs: dict, is_master: bool):
    """Select a config preset and apply key=value CLI overrides.

    Scans ``sys.argv`` for:

    * ``config=<preset>`` — select a named preset class from *configs*.
    * ``key=value``       — override any field on the resulting dataclass.

    Every dataclass field is directly addressable as a key=value argument,
    so no argparse shortcuts are needed.

    Args:
        default_cls:  The default config dataclass class.
        configs:      The CONFIGS registry dict ``{preset_name: cls}``.
        is_master:    Whether this process is rank-0.

    Returns:
        cfg: Instantiated (and overridden) config dataclass.
    """
    config_cls = default_cls
    for arg in sys.argv[1:]:
        if arg.startswith("config="):
            preset = arg.split("=", 1)[1].strip()
            if preset in configs:
                config_cls = configs[preset]
                if is_master:
                    print(f"[config] using preset '{preset}' ({config_cls.__name__})")
            else:
                raise ValueError(
                    f"Unknown config preset '{preset}'. "
                    f"Available: {list(configs.keys())}."
                )
            break

    cfg = config_cls()

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
                if is_master:
                    print(f"[warn] unknown config key '{key}', ignoring.")

    return cfg


def setup_ddp_and_run_dir(cfg, is_master: bool):
    """Initialise DDP (if running under torchrun) and create the per-run
    timestamped output sub-directory.

    The run-id timestamp is generated on rank-0 and broadcast to all
    workers so every rank writes to the same directory.

    Args:
        cfg:       Config dataclass; must have ``out_dir``, ``init_from``,
                   ``wandb_run_name``, and ``device``.
        is_master: Whether this process is rank-0.

    Returns:
        use_ddp (bool), rank (int), world_size (int),
        local_rank (int), device (str), run_out_dir (str)
    """
    use_ddp = False
    rank = 0
    world_size = 1
    local_rank = 0
    device = cfg.device

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

    os.makedirs(cfg.out_dir, exist_ok=True)

    if cfg.init_from == "resume":
        run_out_dir = cfg.out_dir
    else:
        if use_ddp:
            run_id = time.strftime("%Y%m%d-%H%M%S") if rank == 0 else None
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
        print(f"[io] outputs \u2192 {run_out_dir}")

    return use_ddp, rank, world_size, local_rank, device, run_out_dir


def wrap_model_ddp(model, use_ddp: bool, device: str, local_rank: int):
    """Place *model* on *device* and, if running under DDP, wrap it.

    ``model.to(device)`` is idempotent, so this is safe to call even when
    the model (e.g. nanochat's GPT) was already constructed directly on
    *device*.

    Args:
        model:      The (unwrapped) model to place/wrap.
        use_ddp:    Whether distributed training is active.
        device:     Target device string (e.g. ``"cuda:0"``).
        local_rank: Local rank, used for ``device_ids`` under DDP+CUDA.

    Returns:
        The (possibly ``DistributedDataParallel``-wrapped) model.
    """
    if use_ddp:
        if torch.cuda.is_available() and device.startswith("cuda"):
            model.to(device)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank
            )
        else:
            model.to(device)
            model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        model.to(device)
    return model


def build_adamw_with_decay_split(model, cfg) -> torch.optim.Optimizer:
    """Build an AdamW/SGD optimizer with a weight-decay split.

    Weight tensors (``ndim >= 2``) get ``cfg.weight_decay``; biases and
    norm parameters (``ndim < 2``) get ``0.0``. Lifted from the identical
    ``_build_optimizer`` in ViT/base_train.py and ViT5/base_train.py.

    Args:
        model: The (possibly DDP-wrapped) model whose parameters to optimize.
        cfg:   Config dataclass; must have ``optimizer``, ``learning_rate``,
               ``weight_decay``, ``beta1``, ``beta2``, ``eps``.

    Returns:
        A ``torch.optim.AdamW`` or ``torch.optim.SGD`` instance.
    """
    if cfg.optimizer.lower() == "adamw":
        decay_params = [
            p for n, p in model.named_parameters()
            if p.requires_grad and p.ndim >= 2
        ]
        no_decay_params = [
            p for n, p in model.named_parameters()
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
        return torch.optim.SGD(model.parameters(), lr=cfg.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer '{cfg.optimizer}'")


def cosine_warmup_lr(it: int, cfg) -> float:
    """Cosine LR schedule with linear warm-up.

    Lifted from the identical ``get_lr(it)`` in ViT/base_train.py and
    ViT5/base_train.py.

    Args:
        it:  Current iteration.
        cfg: Config dataclass; must have ``decay_lr``, ``warmup_iters``,
             ``learning_rate``, ``lr_decay_iters``, ``min_lr``.

    Returns:
        Learning rate for iteration *it*.
    """
    if not cfg.decay_lr:
        return cfg.learning_rate
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / cfg.warmup_iters
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    ratio = (it - cfg.warmup_iters) / max(1, cfg.lr_decay_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def init_wandb(cfg, use_ddp: bool, rank: int) -> None:
    """Initialise W&B on rank-0 (or single-GPU) if cfg.wandb_log is set."""
    if cfg.wandb_log and (not use_ddp or rank == 0):
        import wandb  # noqa: PLC0415

        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name or None,
            config=vars(cfg),
        )


def save_history_and_plot(
    history: dict,
    cfg,
    run_out_dir: str,
    use_ddp: bool,
    rank: int,    att_sim: bool = False,) -> None:
    """Serialize ``history`` to ``history.pkl`` and run ``plot_results``
    (rank-0 / single-GPU only).

    Also stores a snapshot of the config dataclass under
    ``history[\"config\"]``.
    """
    if use_ddp and rank != 0:
        return

    import dataclasses
    import pickle

    import matplotlib.pyplot as plt

    from common.plot_result import plot_results  # noqa: PLC0415

    history["config"] = dataclasses.asdict(cfg)

    history_path = os.path.join(run_out_dir, "history.pkl")
    with open(history_path, "wb") as f:
        pickle.dump(history, f)

    print(f"[done] history saved \u2192 {history_path}")

    if cfg.wandb_log:
        import wandb  # noqa: PLC0415

        wandb.finish()

    plot_results(
        pkl_path=history_path,
        save_path=os.path.join(run_out_dir, "results.png"),
        hessian_intv=cfg.hessian_intv,
        entropy_intv=cfg.entropy_intv,
        skip_intv=True,
        vs_H_prec=True,
        compute_fd=cfg.compute_fd,
        compute_qqkk=getattr(cfg, "compute_qqkk", False),
        att_sim=att_sim,
        fmt="png",
    )
    plt.close("all")
