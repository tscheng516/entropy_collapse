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
    rank: int,
) -> None:
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
        fmt="png",
    )
    plt.close("all")
