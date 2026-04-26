"""
Plotting utilities for the depth/ entropy-collapse experiments.

This module is self-contained and mirrors ``ViT/src/plotting.py``.

The key difference: the training-dynamics plot shows SILog loss and RMSE
instead of cross-entropy loss and top-1 accuracy.  All curvature and spike
co-occurrence plots are identical to the ViT/ version.

Families of utilities provided:

1. ``plot_curvature_smoothed_comparison`` — raw vs smoothed for all nine
                                 spectral-norm curvature metrics.
2. ``plot_spike_cooccurrence`` — MAD-based joint/disjoint spike timeline.
3. ``plot_all_spike_cooccurrences`` — λ_max(H) vs each other proxy.
4. ``print_correlations`` — raw + smoothed correlation report.
5. ``plot_training_dynamics`` — compact training figure (loss, RMSE, entropy).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy import stats as sp_stats

from src.helpers import smooth_log_trend

_MAD_TO_STD = 1.4826


def _carry_forward_positive(series: np.ndarray | list) -> np.ndarray:
    arr = np.asarray(series, dtype=float).copy().ravel()
    if arr.size == 0:
        return arr
    valid = np.isfinite(arr) & (arr > 0)
    if not valid.any():
        return arr
    first_valid = int(np.argmax(valid))
    arr[:first_valid] = arr[first_valid]
    for i in range(first_valid + 1, arr.size):
        if not (np.isfinite(arr[i]) and arr[i] > 0):
            arr[i] = arr[i - 1]
    return arr


def _carry_forward_positive_2d(matrix: np.ndarray | list) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return arr
    out = arr.copy()
    for j in range(out.shape[1]):
        out[:, j] = _carry_forward_positive(out[:, j])
    return out


def _has_positive_finite(arr: np.ndarray | list) -> bool:
    a = np.asarray(arr, dtype=float).ravel()
    if a.size == 0:
        return False
    return bool(np.any(np.isfinite(a) & (a > 0)))


def _extract_positive(series: np.ndarray | list) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(series, dtype=float).ravel()
    if arr.size == 0:
        return np.array([]), np.array([], dtype=int)
    valid = np.isfinite(arr) & (arr > 0)
    indices = np.where(valid)[0]
    return arr[valid], indices


def _extract_positive_2d(matrix: np.ndarray | list) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return arr, np.arange(max(len(arr), 0), dtype=int)
    valid_rows = np.any(np.isfinite(arr) & (arr > 0), axis=1)
    indices = np.where(valid_rows)[0]
    return arr[valid_rows], indices


# ======================================================================
# Curvature metrics — raw vs smoothed comparison
# ======================================================================


def plot_curvature_smoothed_comparison(
    history: dict,
    lam: float = 100.0,
    save_path: str | None = None,
    skip_intv: bool = True,
    hessian_freq: int = 1,
    compute_fd: bool = False,
) -> plt.Figure:
    """
    Plot all nine spectral-norm curvature metrics (raw + smoothed).

    Identical to ``ViT/src/plotting.py:plot_curvature_smoothed_comparison``.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    def _as1d(key):
        val = history.get(key)
        if val is None:
            return np.asarray([])
        arr = np.asarray(val)
        if arr.ndim == 0:
            return arr.reshape(-1)
        return arr.ravel()

    def _prep(key):
        raw = _as1d(key)
        if skip_intv:
            return _extract_positive(raw)
        else:
            vals = _carry_forward_positive(raw)
            return vals, np.arange(len(vals))

    h_arr, h_idx = _prep("hessian")
    if _has_positive_finite(h_arr):
        ax.plot(h_idx, h_arr, color="red", linewidth=2, label="Exact Hessian (H)")

    prec_arr, prec_idx = _prep("prec_h")
    if _has_positive_finite(prec_arr):
        ax.plot(prec_idx, prec_arr, color="purple", linestyle="--", linewidth=2,
                label=r"Precond. Hessian ($\tilde{H}$)")

    gn_arr, gn_idx = _prep("gn")
    if _has_positive_finite(gn_arr):
        ax.plot(gn_idx, gn_arr, color="brown", linestyle="--", linewidth=2,
                label=r"Gauss-Newton ($H^{GN}$)")

    vv_arr, vv_idx = _prep("hessian_vv")
    if _has_positive_finite(vv_arr):
        ax.plot(vv_idx, vv_arr, color="magenta", linestyle=":", linewidth=2,
                label=r"Value Subspace ($H_{VV}$)")

    diag_arr, diag_idx = _prep("diag_h")
    if _has_positive_finite(diag_arr):
        ax.plot(diag_idx, diag_arr, color="teal", linestyle="-.", linewidth=2,
                label="Diag Hessian")

    fisher_arr, fisher_idx = _prep("fisher")
    if _has_positive_finite(fisher_arr):
        ax.plot(fisher_idx, fisher_arr, color="olive", linestyle="-.", linewidth=2,
                label="Empirical Fisher")

    if compute_fd:
        bfgs_arr, bfgs_idx = _prep("bfgs")
        if _has_positive_finite(bfgs_arr):
            ax.plot(bfgs_idx, bfgs_arr, color="navy", linestyle=":", linewidth=2,
                    label="BFGS")
        fd_arr, fd_idx = _prep("fd")
        if _has_positive_finite(fd_arr):
            ax.plot(fd_idx, fd_arr, color="cyan", linestyle="-.", linewidth=2,
                    label="FD")
    else:
        bfgs_arr, bfgs_idx = np.array([]), np.array([], dtype=int)
        fd_arr, fd_idx = np.array([]), np.array([], dtype=int)

    kfac_arr, kfac_idx = _prep("kfac")
    if _has_positive_finite(kfac_arr):
        ax.plot(kfac_idx, kfac_arr, color="darkgreen", linestyle=":", linewidth=2,
                label="K-FAC")

    _smooth_lam  = lam
    _smooth_alpha = 0.45
    _smooth_lw   = 5

    def _overlay_smooth(arr, idx, color):
        if arr.size >= 3:
            trend, _, _ = smooth_log_trend(arr, lam=_smooth_lam, use_abs=True)
            ax.plot(idx, trend, color=color, linewidth=_smooth_lw, alpha=_smooth_alpha)

    if _has_positive_finite(h_arr):
        _overlay_smooth(h_arr, h_idx, "red")
    if _has_positive_finite(prec_arr):
        _overlay_smooth(prec_arr, prec_idx, "purple")
    if _has_positive_finite(gn_arr):
        _overlay_smooth(gn_arr, gn_idx, "brown")
    if _has_positive_finite(vv_arr):
        _overlay_smooth(vv_arr, vv_idx, "magenta")
    if _has_positive_finite(diag_arr):
        _overlay_smooth(diag_arr, diag_idx, "teal")
    if _has_positive_finite(fisher_arr):
        _overlay_smooth(fisher_arr, fisher_idx, "olive")
    if compute_fd and _has_positive_finite(bfgs_arr):
        _overlay_smooth(bfgs_arr, bfgs_idx, "navy")
    if compute_fd and _has_positive_finite(fd_arr):
        _overlay_smooth(fd_arr, fd_idx, "cyan")
    if _has_positive_finite(kfac_arr):
        _overlay_smooth(kfac_arr, kfac_idx, "darkgreen")

    ax.set_yscale("log")
    ax.set_title(
        f"All Curvature Metrics (λ_max) — Raw vs Smoothed (λ={lam})", fontsize=13
    )
    _xlabel = (
        f"Iteration (every {hessian_freq})"
        if skip_intv and hessian_freq > 1 else "Iteration"
    )
    ax.set_xlabel(_xlabel, fontsize=12)
    ax.set_ylabel("Spectral Norm (λ_max)", fontsize=12)
    ax.legend(fontsize="small", loc="best")
    ax.grid(True, alpha=0.3, linestyle="--")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ======================================================================
# MAD-based spike co-occurrence
# ======================================================================


def plot_spike_cooccurrence(
    x: np.ndarray | list,
    y: np.ndarray | list,
    x_name: str = "X",
    y_name: str = "Y",
    window: int = 15,
    z_score: float = 10.0,
    log_scale: bool = True,
    save_path: str | None = None,
) -> tuple[plt.Figure, dict]:
    """
    Detect and plot spike co-occurrence between two metric time-series.

    Identical to ``ViT/src/plotting.py:plot_spike_cooccurrence``.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    orig_indices = np.arange(len(x))
    finite = np.isfinite(x) & np.isfinite(y)
    placeholder = (x == 0.0) & (y == 0.0)
    finite = finite & ~placeholder

    use_log_scale = bool(log_scale)
    if use_log_scale:
        valid = finite & (x > 0) & (y > 0)
        if valid.sum() < 3:
            use_log_scale = False
            valid = finite
    else:
        valid = finite

    if valid.sum() == 0:
        results = {
            "P(Y_spike | X_spike)": float("nan"),
            "baseline_P(Y_spike)": float("nan"),
            "n_X_spikes": 0, "n_Y_spikes": 0,
            "n_joint_spikes": 0, "n_points": 0,
        }
        fig = plt.figure(figsize=(12, 2.5))
        plt.text(0.5, 0.5, "No valid data points",
                 ha="center", va="center", transform=plt.gca().transAxes)
        plt.yticks([1, 1.5, 2], [f"{y_name} Only", "Joint", f"{x_name} Only"])
        plt.xlabel("Iteration")
        plt.title(f"Local Spike Co-occurrence: {x_name} vs {y_name} (Z > {z_score})")
        plt.grid(axis="x", linestyle="--", alpha=0.5)
        plt.xlim(0, len(x))
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig, results

    x_c = x[valid]
    y_c = y[valid]
    idx = orig_indices[valid]

    if use_log_scale:
        x_c = np.log(x_c)
        y_c = np.log(y_c)

    x_base = median_filter(x_c, size=window)
    y_base = median_filter(y_c, size=window)
    x_res = x_c - x_base
    y_res = y_c - y_base

    x_mad = max(np.median(np.abs(x_res - np.median(x_res))) * _MAD_TO_STD, 1e-6)
    y_mad = max(np.median(np.abs(y_res - np.median(y_res))) * _MAD_TO_STD, 1e-6)

    x_spike = x_res > z_score * x_mad
    y_spike = y_res > z_score * y_mad
    joint_spike = x_spike & y_spike

    x_spike_iters   = idx[x_spike]
    y_spike_iters   = idx[y_spike]
    joint_spike_iters = idx[joint_spike]

    n_x   = len(x_spike_iters)
    n_y   = len(y_spike_iters)
    n_both = len(joint_spike_iters)
    n_pts = len(x_c)

    cond_prob = n_both / n_x if n_x > 0 else float("nan")
    baseline  = n_y / n_pts if n_pts > 0 else float("nan")

    results = {
        "P(Y_spike | X_spike)": cond_prob,
        "baseline_P(Y_spike)":  baseline,
        "n_X_spikes":           n_x,
        "n_Y_spikes":           n_y,
        "n_joint_spikes":       n_both,
        "n_points":             n_pts,
    }

    fig = plt.figure(figsize=(12, 2.5))
    x_only = np.setdiff1d(x_spike_iters, joint_spike_iters)
    y_only = np.setdiff1d(y_spike_iters, joint_spike_iters)

    if len(x_only):
        plt.scatter(x_only, np.ones_like(x_only) * 2,
                    color="blue", marker="|", s=200, label=f"{x_name} Spike Only")
    if len(joint_spike_iters):
        plt.scatter(joint_spike_iters, np.ones_like(joint_spike_iters) * 1.5,
                    color="red", marker="x", s=100, label="Joint Spike")
    if len(y_only):
        plt.scatter(y_only, np.ones_like(y_only) * 1,
                    color="orange", marker="|", s=200, label=f"{y_name} Spike Only")

    plt.yticks([1, 1.5, 2], [f"{y_name} Only", "Joint", f"{x_name} Only"])
    plt.xlabel("Iteration")
    plt.title(f"Local Spike Co-occurrence: {x_name} vs {y_name} (Z > {z_score})")
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.xlim(0, len(x))
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, results


def plot_all_spike_cooccurrences(
    history: dict,
    window: int = 15,
    z_score: float = 2.0,
    log_scale: bool = True,
    save_dir: str | None = None,
    skip_intv: bool = True,
    hessian_freq: int = 1,
    compute_fd: bool = False,
) -> tuple[dict[str, plt.Figure], dict[str, dict]]:
    """
    Compute MAD spike co-occurrence for λ_max(H) vs each curvature proxy.

    Identical to ``ViT/src/plotting.py:plot_all_spike_cooccurrences``.
    """
    h_raw = np.asarray(history.get("hessian", []), dtype=float).ravel()
    if skip_intv:
        h, h_idx = _extract_positive(h_raw)
    else:
        h = h_raw
        h_idx = np.arange(len(h))

    proxy_specs = [
        ("prec_h",     "Prec_H",  "hessian_prec"),
        ("hessian_vv", "H_VV",    "hessian_vv"),
        ("gn",         "GN",      "hessian_gn"),
        ("diag_h",     "Diag_H",  "hessian_diag"),
        ("fisher",     "Fisher",  "hessian_fisher"),
        ("kfac",       "KFAC",    "hessian_kfac"),
    ]
    if compute_fd:
        proxy_specs.extend([
            ("fd",   "FD",   "hessian_fd"),
            ("bfgs", "BFGS", "hessian_bfgs"),
        ])

    figs: dict[str, plt.Figure] = {}
    results: dict[str, dict] = {}

    for key, label, suffix in proxy_specs:
        y_raw = np.asarray(history.get(key, []), dtype=float).ravel()
        if skip_intv:
            y, y_idx = _extract_positive(y_raw)
        else:
            y = y_raw
        if h.size == 0 or y.size == 0:
            continue
        if not (_has_positive_finite(h) and _has_positive_finite(y)):
            continue
        n = min(h.size, y.size)
        save_path = None
        if save_dir is not None:
            save_path = f"{save_dir}/spike_cooccurrence_H_vs_{suffix}_z{z_score}.png"

        fig, res = plot_spike_cooccurrence(
            h[:n], y[:n],
            x_name="Exact H", y_name=label,
            window=window, z_score=z_score,
            log_scale=log_scale, save_path=save_path,
        )
        figs[key] = fig
        results[key] = res

    return figs, results


# ======================================================================
# Correlation analysis
# ======================================================================


def print_correlations(
    history: dict,
    name: str,
    sample_every: int = 1,
    lam: float = 10.0,
    include_smooth: bool = True,
    skip_intv: bool = True,
    hessian_freq: int = 1,
    compute_fd: bool = False,
) -> dict:
    """
    Print raw and smoothed correlations for curvature metrics.

    Identical to ``ViT/src/plotting.py:print_correlations``.
    """
    def _gk(d, *keys):
        for k in keys:
            if k in d:
                raw = np.array(d[k])
                if skip_intv:
                    vals, _ = _extract_positive(raw)
                    return vals[::sample_every]
                return raw[::sample_every]
        return np.asarray([])

    h      = _gk(history, "hessian")
    prec_h = _gk(history, "prec_h")
    vv     = _gk(history, "hessian_vv", "vv")
    gn     = _gk(history, "gn")
    diag_h = _gk(history, "diag_h")
    fisher = _gk(history, "fisher")
    kfac   = _gk(history, "kfac")
    if compute_fd:
        fd   = _gk(history, "fd")
        bfgs = _gk(history, "bfgs")
        if not _has_positive_finite(fd) and not _has_positive_finite(bfgs):
            print("[warn] compute_fd=True but FD/BFGS data has no positive values")

    results: dict = {"raw": {}, "smoothed": {}, "entropy": {}}

    def _corr(a, b, label, store: dict | None = None):
        n = min(len(a), len(b))
        if n == 0:
            print(f"  {label}: no data")
            return
        a_, b_ = a[:n], b[:n]
        mask = np.isfinite(a_) & np.isfinite(b_)
        if mask.sum() < 3:
            print(f"  {label}: insufficient data")
            return
        a_m, b_m = a_[mask], b_[mask]
        if np.std(a_m) < 1e-12 or np.std(b_m) < 1e-12:
            print(f"  {label}: constant/near-constant series")
            return
        sp, _ = sp_stats.spearmanr(a_m, b_m)
        pe, _ = sp_stats.pearsonr(a_m, b_m)
        print(f"  {label}: Spearman {sp:.4f} | Pearson {pe:.4f}")
        if store is not None:
            store[label.strip()] = {"spearman": float(sp), "pearson": float(pe)}

    print(f"\n--- {name} Raw Correlations (H as reference) ---")
    _corr(h, prec_h, "H vs Prec_H  ", results["raw"])
    _corr(h, vv,     "H vs H_VV    ", results["raw"])
    _corr(h, gn,     "H vs GN      ", results["raw"])
    _corr(h, diag_h, "H vs Diag_H  ", results["raw"])
    _corr(h, fisher, "H vs Fisher  ", results["raw"])
    _corr(h, kfac,   "H vs KFAC    ", results["raw"])
    if compute_fd:
        _corr(h, fd,   "H vs FD      ", results["raw"])
        _corr(h, bfgs, "H vs BFGS    ", results["raw"])

    if _has_positive_finite(prec_h):
        print(f"\n--- {name} Raw Correlations (Prec_H as reference) ---")
        _corr(prec_h, vv,     "Prec_H vs H_VV   ", results["raw"])
        _corr(prec_h, gn,     "Prec_H vs GN     ", results["raw"])
        _corr(prec_h, diag_h, "Prec_H vs Diag_H ", results["raw"])
        _corr(prec_h, fisher, "Prec_H vs Fisher ", results["raw"])
        _corr(prec_h, kfac,   "Prec_H vs KFAC   ", results["raw"])
        if compute_fd:
            _corr(prec_h, fd,   "Prec_H vs FD     ", results["raw"])
            _corr(prec_h, bfgs, "Prec_H vs BFGS   ", results["raw"])

    if not include_smooth:
        return results

    def _smooth(arr):
        if arr.size < 3:
            return arr
        trend, _, _ = smooth_log_trend(arr, lam=lam, use_abs=True)
        return trend

    h_s      = _smooth(h)
    prec_h_s = _smooth(prec_h)
    vv_s     = _smooth(vv)
    gn_s     = _smooth(gn)
    diag_h_s = _smooth(diag_h)
    fisher_s = _smooth(fisher)
    kfac_s   = _smooth(kfac)

    print(f"\n--- {name} Smoothed Correlations, H reference (λ={lam}) ---")
    _corr(h_s, prec_h_s, "H vs Prec_H  ", results["smoothed"])
    _corr(h_s, vv_s,     "H vs H_VV    ", results["smoothed"])
    _corr(h_s, gn_s,     "H vs GN      ", results["smoothed"])
    _corr(h_s, diag_h_s, "H vs Diag_H  ", results["smoothed"])
    _corr(h_s, fisher_s, "H vs Fisher  ", results["smoothed"])
    _corr(h_s, kfac_s,   "H vs KFAC    ", results["smoothed"])
    if compute_fd:
        fd_s   = _smooth(fd)
        bfgs_s = _smooth(bfgs)
        _corr(h_s, fd_s,   "H vs FD      ", results["smoothed"])
        _corr(h_s, bfgs_s, "H vs BFGS    ", results["smoothed"])

    if _has_positive_finite(prec_h_s):
        print(f"\n--- {name} Smoothed Correlations, Prec_H reference (λ={lam}) ---")
        _corr(prec_h_s, vv_s,     "Prec_H vs H_VV   ", results["smoothed"])
        _corr(prec_h_s, gn_s,     "Prec_H vs GN     ", results["smoothed"])
        _corr(prec_h_s, diag_h_s, "Prec_H vs Diag_H ", results["smoothed"])
        _corr(prec_h_s, fisher_s, "Prec_H vs Fisher ", results["smoothed"])
        _corr(prec_h_s, kfac_s,   "Prec_H vs KFAC   ", results["smoothed"])
        if compute_fd:
            _corr(prec_h_s, fd_s,   "Prec_H vs FD     ", results["smoothed"])
            _corr(prec_h_s, bfgs_s, "Prec_H vs BFGS   ", results["smoothed"])

    # --- Proxy vs entropy ---
    raw_ent = history.get("entropy", [])
    if raw_ent and len(raw_ent) > 0:
        ent_arr = np.array(raw_ent[::sample_every])
        if ent_arr.ndim == 2 and ent_arr.shape[1] > 0:
            ent_first = ent_arr[:, 0]
            ent_avg   = ent_arr.mean(axis=1)
        else:
            ent_first = ent_avg = np.asarray([])
    else:
        ent_first = ent_avg = np.asarray([])

    if ent_first.size >= 3 and h_s.size >= 3:
        n = min(len(h_s), len(ent_first))
        print(f"\n--- {name} Smoothed Proxy vs Entropy (λ={lam}) ---")
        _corr(h_s[:n],      ent_first[:n], "H vs Entropy(L0)      ", results["entropy"])
        _corr(h_s[:n],      ent_avg[:n],   "H vs Entropy(avg)     ", results["entropy"])
        _corr(prec_h_s[:n], ent_first[:n], "Prec_H vs Entropy(L0) ", results["entropy"])
        _corr(prec_h_s[:n], ent_avg[:n],   "Prec_H vs Entropy(avg)", results["entropy"])
        _corr(vv_s[:n],     ent_first[:n], "H_VV vs Entropy(L0)   ", results["entropy"])
        _corr(gn_s[:n],     ent_first[:n], "GN vs Entropy(L0)     ", results["entropy"])

    return results


def print_smooth_correlations(
    history: dict,
    name: str,
    lam: float = 10.0,
    sample_every: int = 1,
) -> dict:
    """Backward-compatible wrapper — delegates to print_correlations."""
    return print_correlations(
        history, name, sample_every=sample_every, lam=lam, include_smooth=True,
    )


# ======================================================================
# Training dynamics — depth-specific multi-panel grid
# ======================================================================


def plot_training_dynamics(
    histories: dict[str, dict],
    lrs: dict[str, float],
    save_path: str | None = None,
    skip_intv: bool = True,
    entropy_freq: int = 1,
) -> plt.Figure:
    """
    Plot compact training dynamics for depth estimation.

    Panels per run (3 columns):
        * Col 0 — train / val SILog loss
        * Col 1 — val RMSE (log scale) + val δ<1.25 (secondary axis)
        * Col 2 — per-layer attention entropy

    History dict keys used:
        ``loss``        — dense train SILog per iteration
        ``val_loss``    — sparse list of (iter, val_silog) tuples
        ``val_rmse``    — sparse list of (iter, rmse) tuples
        ``val_delta1``  — sparse list of (iter, delta1) tuples
        ``entropy``     — list of per-layer entropy floats

    Args:
        histories:    Dict mapping run name to history dict.
        lrs:          Dict mapping run name to peak LR (API compatibility).
        save_path:    If provided, save the figure to this path.
        skip_intv:    Skip zero-placeholder entropy rows.
        entropy_freq: Entropy computation frequency for x-axis label.

    Returns:
        The matplotlib ``Figure`` object.
    """
    opt_names = list(histories.keys())
    n_rows = len(opt_names)
    fig, axs = plt.subplots(n_rows, 3, figsize=(21, 4.8 * n_rows))
    if n_rows == 1:
        axs = axs[np.newaxis, :]

    for row, name in enumerate(opt_names):
        h = histories[name]
        color = "blue" if "adam" in name.lower() else "orange"

        def _as1d(key, alt=None):
            val = h.get(key, h.get(alt) if alt is not None else None)
            if val is None:
                return np.asarray([])
            arr = np.asarray(val)
            if arr.ndim == 0:
                return arr.reshape(-1)
            return arr.ravel()

        def _sparse_or_dense(series, dense_color, dense_label, ax,
                              sparse_color=None, sparse_label=None,
                              linestyle="--"):
            if not isinstance(series, list) or len(series) == 0:
                return False
            first = series[0]
            if isinstance(first, (tuple, list)) and len(first) == 2:
                xs = [int(t[0]) for t in series]
                ys = [float(t[1]) for t in series]
                lbl = sparse_label or dense_label
                clr = sparse_color or dense_color
                ax.plot(xs, ys, color=clr, linestyle=linestyle,
                        linewidth=2, label=lbl)
            else:
                arr = np.asarray(series, dtype=float).ravel()
                if arr.size == 0:
                    return False
                ax.plot(arr, color=dense_color, linestyle=linestyle,
                        linewidth=2, label=dense_label)
            return True

        # --- Col 0: train/val SILog loss ---
        ax_loss = axs[row, 0]
        loss_arr = _as1d("loss")
        ax_loss.plot(loss_arr, color=color, linewidth=2, label=f"{name} train SILog")
        _sparse_or_dense(
            h.get("val_loss", []),
            dense_color="crimson", dense_label=f"{name} val SILog",
            ax=ax_loss,
        )
        ax_loss.set_title(f"{name} Scale-Invariant Log Loss")
        ax_loss.set_xlabel("Iteration")
        ax_loss.set_ylabel("SILog loss")
        ax_loss.legend(loc="upper right")
        ax_loss.grid(True, alpha=0.25, linestyle="--")

        # --- Col 1: val RMSE + δ<1.25 ---
        ax_rmse = axs[row, 1]
        _sparse_or_dense(
            h.get("val_rmse", []),
            dense_color=color, dense_label=f"{name} val RMSE (log)",
            ax=ax_rmse,
        )
        ax_rmse.set_title(f"{name} Depth Metrics")
        ax_rmse.set_xlabel("Iteration")
        ax_rmse.set_ylabel("RMSE (log)", color=color)
        ax_rmse.tick_params(axis="y", labelcolor=color)

        # Secondary axis: δ<1.25
        ax_d1 = ax_rmse.twinx()
        _sparse_or_dense(
            h.get("val_delta1", []),
            dense_color="crimson", dense_label=f"{name} δ<1.25",
            ax=ax_d1,
        )
        ax_d1.set_ylabel("δ<1.25 (%)", color="crimson")
        ax_d1.tick_params(axis="y", labelcolor="crimson")

        # Combine legends
        lines1, labels1 = ax_rmse.get_legend_handles_labels()
        lines2, labels2 = ax_d1.get_legend_handles_labels()
        ax_rmse.legend(lines1 + lines2, labels1 + labels2, loc="lower right",
                       fontsize="small")
        ax_rmse.grid(True, alpha=0.25, linestyle="--")

        # --- Col 2: per-layer attention entropy ---
        ax_ent = axs[row, 2]
        raw_entropy = np.array(h.get("entropy", []))
        if skip_intv:
            entropies, ent_idx = _extract_positive_2d(raw_entropy)
        else:
            entropies = _carry_forward_positive_2d(raw_entropy)
            ent_idx = np.arange(len(entropies))
        if entropies.ndim == 2 and entropies.shape[1] > 0:
            n_layers = entropies.shape[1]
            colors_entropy = plt.cm.viridis(np.linspace(0, 1, n_layers))
            for layer_idx in range(n_layers):
                ax_ent.plot(
                    ent_idx, entropies[:, layer_idx],
                    color=colors_entropy[layer_idx],
                    label=f"Layer {layer_idx + 1}",
                )
        ax_ent.set_title(f"{name} Attention Entropy (Per Layer)")
        _ent_xlabel = (
            f"Iteration (every {entropy_freq})"
            if skip_intv and entropy_freq > 1 else "Iteration"
        )
        ax_ent.set_xlabel(_ent_xlabel)
        ax_ent.set_ylabel("Entropy (nats)")
        ax_ent.legend(fontsize="small", ncol=2)
        ax_ent.grid(True, alpha=0.25, linestyle="--")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
