"""common/pretrain.py — shared pretraining loop for entropy-collapse experiments.

``run_training()`` contains the model-agnostic training loop shared by
ViT/base_train.py, ViT5/base_train.py, and nanochat/base_train.py: periodic
validation, curvature-proxy / attention-entropy / attention-similarity /
feature-covariance-stable-rank tracking, attention-heatmap & Gram-matrix
snapshots, checkpointing, console/W&B logging, and final history/plot saving.

Each project's ``base_train.py`` keeps only what is genuinely model-specific
(data loading, model construction, optimizer, LR/momentum schedule, the
loss/metrics step, and validation estimation) and supplies these as small
callables/kwargs to ``run_training(...)``.

Design notes
------------
* ``.to(device)`` is applied unconditionally to every batch — it is a no-op
  when the tensor is already on *device*, so this works whether the caller's
  ``train_iter``/``estimate_val`` places tensors on-device itself
  (nanochat's streaming loader) or not (ViT/ViT5's ``DataLoader``).
* Optional metric groups are each gated by a single boolean config field,
  read via ``getattr(cfg, "<flag>", False)`` so configs that don't define a
  given flag simply keep it disabled instead of erroring:
    - ``compute_fd``          — bfgs, fd, kfac curvature proxies.
    - ``compute_qqkk``        — hessian_qq, hessian_kk curvature proxies
                                 (query/key-projection subspaces; masks are
                                 built internally, no caller wiring needed).
    - ``compute_spectrum``    — similarity, cov_stable_rank_post_attn/ffn,
                                 att_heatmaps(+iters), gram_hessian(+iters).
    - ``compute_grad_norm``   — grad_norm_full, grad_norm_attn.
  When a flag is off, the corresponding history keys are omitted entirely
  (not just left empty) and the underlying computation is skipped.
* ``getattr(cfg, "grad_clip", 0.0)`` lets configs without that field
  (nanochat) fall back to the historical no-op (no clipping) behaviour.
* ``save_periodic_ckpt`` preserves each project's historical
  best/periodic-checkpoint-on-improvement behaviour: ViT and nanochat only
  update the ``best_val_loss`` bookkeeping without writing ``best_ckpt``/
  ``ckpt`` to disk on every improved eval (pass ``False``); ViT5 also writes
  them (pass ``True``).
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from common.helpers import (
    get_blocks,
    get_VV_subspace_mask,
    get_curvature_metrics,
    get_attention_entropy,
    get_attention_similarity,
    get_attention_heatmap_all,
    get_attention_gram_head0,
    get_feature_covariance_stable_rank,
    get_grad_norm,
)
from common.train_utils import init_wandb, save_history_and_plot


@torch.no_grad()
def estimate_classification_val_metrics(
    model: torch.nn.Module,
    val_loader,
    ctx,
    device: str,
    n_batches: int = 10,
) -> dict:
    """Return ``{"loss": mean_val_loss, "acc": top1_accuracy_percent}`` over
    ``n_batches`` of *val_loader*.

    Shared by ViT/ and ViT5/ (identical ``estimate_val_metrics`` logic
    previously duplicated in both ``base_train.py`` files).
    """
    model.eval()
    losses, correct, total = [], 0, 0
    val_iter_local = iter(val_loader)
    for _ in range(n_batches):
        try:
            xv, yv = next(val_iter_local)
        except StopIteration:
            break
        xv, yv = xv.to(device), yv.to(device)
        with ctx:
            logits = model(xv)
        lv = F.cross_entropy(logits, yv)
        losses.append(lv.item())
        correct += (logits.argmax(dim=-1) == yv).sum().item()
        total += yv.size(0)
    model.train()
    val_loss = float(np.mean(losses)) if losses else float("inf")
    val_acc = 100.0 * correct / total if total > 0 else 0.0
    return {"loss": val_loss, "acc": val_acc}


def run_training(
    cfg,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_iter,
    update_schedule,
    step_fn,
    estimate_val,
    set_attention_temperature,
    *,
    use_ddp: bool,
    rank: int,
    device: str,
    run_out_dir: str,
    ctx,
    ckpt_extra_fields: dict,
    has_accuracy: bool = False,
    save_periodic_ckpt: bool = True,
    initial_iter_num: int = 0,
    initial_best_val_loss: float = float("inf"),
) -> dict:
    """Run the shared entropy-collapse pretraining loop.

    Args:
        cfg:            Resolved ``TrainConfig`` dataclass.
        model:          Model, already device-placed and (if applicable)
                        DDP-wrapped via ``common.train_utils.wrap_model_ddp``.
        optimizer:      Already-constructed optimizer.
        train_iter:     Iterator yielding ``(X, Y)`` batches (device
                        placement not required — ``.to(device)`` is applied
                        unconditionally and is a no-op if already placed).
        update_schedule: ``(optimizer, it) -> float``. Mutates optimizer
                        hyperparameters (lr, and anything else, e.g. Muon
                        momentum/weight-decay) in place for iteration *it*
                        and returns a representative lr value for logging.
        step_fn:        ``(raw_model, X, Y) -> (loss, metrics)``. Runs the
                        forward pass + loss computation (called inside the
                        ``ctx`` autocast context). ``metrics`` may contain
                        ``"acc"``.
        estimate_val:   ``() -> dict`` with key ``"loss"`` and, when
                        ``has_accuracy``, ``"acc"``.
        set_attention_temperature: ``(model, temperature) -> None``,
                        imported from the project's ``src/model.py``.
        use_ddp, rank, device, run_out_dir:
                        From ``common.train_utils.setup_ddp_and_run_dir``.
        ctx:            Autocast context manager (or ``nullcontext()``).
        ckpt_extra_fields: Static dict of architecture fields merged into
                        every saved checkpoint (e.g. ``model_name``,
                        ``depth``, ...).
        has_accuracy:   Whether ``step_fn``/``estimate_val`` populate
                        ``"acc"`` — controls whether ``acc``/``val_acc``
                        history keys and log lines are used.
        save_periodic_ckpt: Whether to write ``best_ckpt``/``ckpt`` to disk
                        on validation improvement (in addition to
                        ``checkpoint_interval`` and the final checkpoint).
        initial_iter_num: Starting iteration (non-zero when resuming from a
                        checkpoint).
        initial_best_val_loss: Starting ``best_val_loss`` (from checkpoint
                        when resuming).

    Returns:
        The final ``history`` dict (already serialized to ``history.pkl``
        and plotted via ``save_history_and_plot``).
    """
    _is_master = rank == 0

    init_wandb(cfg, use_ddp, rank)
    if cfg.wandb_log:
        import wandb  # noqa: PLC0415

    _raw_model = model.module if use_ddp else model
    blocks = get_blocks(_raw_model)
    n_layers = len(blocks)

    vv_mask = get_VV_subspace_mask(_raw_model).to(device)
    compute_fd = cfg.compute_fd
    compute_qqkk = getattr(cfg, "compute_qqkk", False)
    compute_spectrum = getattr(cfg, "compute_spectrum", False)
    compute_grad_norm = getattr(cfg, "compute_grad_norm", False)
    qq_mask = get_VV_subspace_mask(_raw_model, "q").to(device) if compute_qqkk else None
    kk_mask = get_VV_subspace_mask(_raw_model, "k").to(device) if compute_qqkk else None

    iter_num = initial_iter_num
    best_val_loss = initial_best_val_loss

    def _save_checkpoint(suffix: str = "ckpt") -> None:
        if not use_ddp or rank == 0:
            checkpoint = {
                "model": _raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
                "config": vars(cfg),
                **ckpt_extra_fields,
            }
            path = os.path.join(run_out_dir, f"{suffix}.pt")
            torch.save(checkpoint, path)
            print(f"[ckpt] saved → {path}")

    history: dict[str, list] = {
        "loss": [],
        "val_loss": [],
        "hessian": [],
        "prec_h": [],
        "hessian_vv": [],
        "gn": [],
        "diag_h": [],
        "fisher": [],
        "entropy": [],
        "lr": [],
    }
    if has_accuracy:
        history["acc"] = []
        history["val_acc"] = []
    if compute_fd:
        history["fd"] = []
        history["bfgs"] = []
        history["kfac"] = []
    if compute_qqkk:
        history["hessian_qq"] = []
        history["hessian_kk"] = []
    if compute_spectrum:
        history["similarity"] = []
        history["cov_stable_rank_post_attn"] = []
        history["cov_stable_rank_post_ffn"] = []
        history["att_heatmaps"] = []
        history["att_heatmap_iters"] = []
        history["gram_hessian"] = []
        history["gram_hessian_iters"] = []
    if compute_grad_norm:
        history["grad_norm_full"] = []
        history["grad_norm_attn"] = []

    if _is_master:
        print(f"\n[train] starting — max_iters={cfg.max_iters}  device={device}\n")
    t0 = time.time()
    model.train()

    X, Y = next(train_iter)
    X, Y = X.to(device), Y.to(device)

    for iter_num in range(iter_num, cfg.max_iters):

        # ---- LR / schedule update ----
        lr = update_schedule(optimizer, iter_num)

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
            val_metrics = estimate_val()
            val_loss = val_metrics["loss"]
            val_acc = val_metrics.get("acc")
            history["val_loss"].append((iter_num, val_loss))
            if has_accuracy:
                history["val_acc"].append((iter_num, val_acc))
            if _is_master:
                _msg = f"[eval] iter {iter_num:5d} | val_loss {val_loss:.4f}"
                if has_accuracy:
                    _msg += f" | val_acc {val_acc:.2f}%"
                print(_msg)
            if cfg.wandb_log and (not use_ddp or rank == 0):
                _log = {"val/loss": val_loss}
                if has_accuracy:
                    _log["val/acc"] = val_acc
                wandb.log(_log, step=iter_num)

            if (cfg.save_checkpoint or val_loss < best_val_loss) and iter_num > 0:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if save_periodic_ckpt:
                        _save_checkpoint("best_ckpt")
                if save_periodic_ckpt:
                    _save_checkpoint("ckpt")

            # ---- Attention heatmap snapshot (all layers & heads) ----
            if compute_spectrum and (not use_ddp or rank == 0):
                _raw_model.eval()
                for blk in blocks:
                    blk.attn._cache_attn = True
                with torch.no_grad():
                    with ctx:
                        _ = _raw_model(X)
                _snapshot = get_attention_heatmap_all(_raw_model)
                if _snapshot is not None:
                    history["att_heatmaps"].append(_snapshot)
                    history["att_heatmap_iters"].append(iter_num)
                for blk in blocks:
                    blk.attn._cache_attn = False
                    blk.attn.last_att = None
                _raw_model.train()

        if cfg.checkpoint_interval > 0 and iter_num % cfg.checkpoint_interval == 0 and iter_num > 0:
            _save_checkpoint(f"ckpt_iter{iter_num:06d}")

        # ---- Curvature metrics (spectral norm) ----
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
            "hessian_qq": 0.0,
            "hessian_kk": 0.0,
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
                    compute_fd=compute_fd,
                    hessian_batch_size=cfg.hessian_batch_size,
                    label_smoothing=cfg.label_smoothing,
                    compute_qqkk=compute_qqkk,
                    qq_mask=qq_mask,
                    kk_mask=kk_mask,
                )
            except Exception as exc:
                if _is_master:
                    print(f"[warn] curvature metrics failed at iter {iter_num}: {exc}")
            finally:
                optimizer.zero_grad()

        for k in ("hessian", "prec_h", "hessian_vv", "gn", "diag_h", "fisher"):
            history[k].append(curvature[k])
        if compute_fd:
            history["fd"].append(curvature["fd"])
            history["bfgs"].append(curvature["bfgs"])
            history["kfac"].append(curvature["kfac"])
        if compute_qqkk:
            history["hessian_qq"].append(curvature["hessian_qq"])
            history["hessian_kk"].append(curvature["hessian_kk"])

        # ---- Attention Gram matrix snapshot (head=0, all layers, hessian batch) ----
        if iter_num % cfg.hessian_intv == 0 and compute_spectrum and (not use_ddp or rank == 0):
            _raw_model.eval()
            _Xc = X[:cfg.hessian_batch_size]
            for blk in blocks:
                blk.attn._cache_attn = True
            with torch.no_grad():
                with ctx:
                    _ = _raw_model(_Xc)
            _grams = get_attention_gram_head0(_raw_model, head=0)
            if _grams is not None:
                history["gram_hessian"].append(_grams)
                history["gram_hessian_iters"].append(iter_num)
            for blk in blocks:
                blk.attn._cache_attn = False
                blk.attn.last_att = None
            _raw_model.train()

        # ---- Standard training step ----
        layer_entropies: list[float] = [0.0] * n_layers
        layer_sims: list[list[float]] = [[] for _ in range(n_layers)]
        cov_stable_rank_post_attn: list[float] = [0.0] * n_layers
        cov_stable_rank_post_ffn: list[float] = [0.0] * n_layers
        _need_entropy = iter_num % cfg.entropy_intv == 0

        # Enable attention caching only when entropy/similarity will be read.
        if _need_entropy:
            for blk in blocks:
                blk.attn._cache_attn = True

        optimizer.zero_grad(set_to_none=True)
        with ctx:
            loss, metrics = step_fn(_raw_model, X, Y)

            if _need_entropy:
                with torch.no_grad():
                    layer_entropies = get_attention_entropy(_raw_model)
                    if compute_spectrum:
                        layer_sims = get_attention_similarity(_raw_model)
                for blk in blocks:
                    blk.attn._cache_attn = False
                    blk.attn.last_att = None  # free memory immediately

        loss.backward()
        if compute_grad_norm:
            grad_norm_full = get_grad_norm(_raw_model, attn_only=False)
            grad_norm_attn = get_grad_norm(_raw_model, attn_only=True)
        if getattr(cfg, "grad_clip", 0.0) > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        # ---- Feature covariance stable rank (compute_spectrum only) ----
        if _need_entropy and compute_spectrum:
            _cov = get_feature_covariance_stable_rank(
                _raw_model,
                X,
                hessian_batch_size=cfg.hessian_batch_size,
                max_iter=cfg.hessian_max_iter,
            )
            cov_stable_rank_post_attn = _cov["post_attn"]
            cov_stable_rank_post_ffn = _cov["post_ffn"]

        loss_val = loss.item()
        train_acc = metrics.get("acc", 0.0) if has_accuracy else None

        history["loss"].append(loss_val)
        if has_accuracy:
            history["acc"].append(train_acc)
        history["lr"].append(lr)
        history["entropy"].append(layer_entropies)
        if compute_spectrum:
            history["similarity"].append(layer_sims)
            history["cov_stable_rank_post_attn"].append(cov_stable_rank_post_attn)
            history["cov_stable_rank_post_ffn"].append(cov_stable_rank_post_ffn)
        if compute_grad_norm:
            history["grad_norm_full"].append(grad_norm_full)
            history["grad_norm_attn"].append(grad_norm_attn)

        # pre-fetch next batch
        X, Y = next(train_iter)
        X, Y = X.to(device), Y.to(device)

        # ---- Logging ----
        if iter_num % cfg.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            if _is_master:
                _line = f"iter {iter_num:5d} | loss {loss_val:.4f} "
                if has_accuracy:
                    _line += f"| acc {train_acc:.1f}% "
                _line += f"| lr {lr:.2e} | dt {dt * 1000:.1f}ms"
                if compute_grad_norm:
                    _line += f" | gnorm {grad_norm_full:.3f} (attn {grad_norm_attn:.3f})"
                print(_line)
                if iter_num % cfg.entropy_intv == 0 and compute_spectrum:
                    _sim_str = "  ".join(
                        f"L{i}:[" + ",".join(f"{v:.3f}" for v in hs) + "]"
                        for i, hs in enumerate(layer_sims)
                    )
                    print(f"  attn_sim(all_h): {_sim_str}")
                    _cov_str_attn = "  ".join(
                        f"L{i}:{v:.3f}" for i, v in enumerate(cov_stable_rank_post_attn)
                    )
                    _cov_str_ffn = "  ".join(
                        f"L{i}:{v:.3f}" for i, v in enumerate(cov_stable_rank_post_ffn)
                    )
                    print(f"  cov_sr(post_attn): {_cov_str_attn}")
                    print(f"  cov_sr(post_ffn):  {_cov_str_ffn}")
                if iter_num % cfg.hessian_intv == 0:
                    _cmsg = (
                        f"  H {curvature['hessian']:.3f} | H~(prec) {curvature['prec_h']:.3f} "
                        f"| H_VV {curvature['hessian_vv']:.3f} | GN {curvature['gn']:.3f} "
                        f"| DiagH {curvature['diag_h']:.3f} | Fisher {curvature['fisher']:.3f}"
                    )
                    if compute_fd:
                        _cmsg += (
                            f" | BFGS {curvature['bfgs']:.3f}"
                            f" | FD {curvature['fd']:.3f}"
                            f" | KFAC {curvature['kfac']:.3f}"
                        )
                    if compute_qqkk:
                        _cmsg += (
                            f" | H_QQ {curvature['hessian_qq']:.3f}"
                            f" | H_KK {curvature['hessian_kk']:.3f}"
                        )
                    print(_cmsg)
            if cfg.wandb_log and (not use_ddp or rank == 0):
                log_dict: dict = {
                    "train/loss": loss_val,
                    "train/lr": lr,
                }
                if has_accuracy:
                    log_dict["train/acc"] = train_acc
                if compute_grad_norm:
                    log_dict["grad_norm/full"] = grad_norm_full
                    log_dict["grad_norm/attn"] = grad_norm_attn
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
                    if compute_fd:
                        log_dict.update(
                            {
                                "hessian/FD": curvature["fd"],
                                "hessian/BFGS": curvature["bfgs"],
                                "hessian/KFAC": curvature["kfac"],
                            }
                        )
                    if compute_qqkk:
                        log_dict.update(
                            {
                                "hessian/H_QQ": curvature["hessian_qq"],
                                "hessian/H_KK": curvature["hessian_kk"],
                            }
                        )
                if iter_num % cfg.entropy_intv == 0:
                    log_dict.update(
                        {f"entropy/layer_{i}": v for i, v in enumerate(layer_entropies)}
                    )
                    if compute_spectrum:
                        log_dict.update(
                            {
                                f"attn_sim/layer_{i}_head_{j}": v
                                for i, hs in enumerate(layer_sims)
                                for j, v in enumerate(hs)
                            }
                        )
                        log_dict.update(
                            {
                                f"cov_sr/post_attn_layer_{i}": v
                                for i, v in enumerate(cov_stable_rank_post_attn)
                            }
                        )
                        log_dict.update(
                            {
                                f"cov_sr/post_ffn_layer_{i}": v
                                for i, v in enumerate(cov_stable_rank_post_ffn)
                            }
                        )
                wandb.log(log_dict, step=iter_num)

    # ---------------------------------------------------------------------
    # Final checkpoint & history
    # ---------------------------------------------------------------------
    _save_checkpoint("final_ckpt")
    save_history_and_plot(history, cfg, run_out_dir, use_ddp, rank, compute_spectrum)

    # ---------------------------------------------------------------------
    # DDP teardown
    # ---------------------------------------------------------------------
    if use_ddp:
        import torch.distributed as dist  # noqa: PLC0415

        dist.destroy_process_group()

    return history
