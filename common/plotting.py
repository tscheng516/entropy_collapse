"""
Shared plotting utilities for all entropy-collapse sub-projects.

All five function families are provided here and imported by per-project
``src/plotting.py`` thin wrappers so each folder stays self-contained for
existing callers.

Families
--------
1. ``plot_curvature_smoothed_comparison`` — raw vs smoothed for all nine metrics.
2. ``plot_spike_cooccurrence``            — MAD-based joint/disjoint spike timeline.
3. ``plot_all_spike_cooccurrences``       — λ_max(H) vs each other proxy.
4. ``print_correlations``                 — unified raw + smoothed correlation report.
5. ``plot_training_dynamics``             — compact training figure with ``task=``
                                           dispatch:
                                           * ``"classification"`` — 3 cols: CE loss /
                                             accuracy / entropy  (ViT, ViT5)
                                           * ``"depth"``          — 3 cols: SILog /
                                             RMSE+δ<1.25 / entropy  (ViT_depth)
                                           * ``"lm"``             — 2 cols: CE loss /
                                             entropy  (nanochat)
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Path bootstrap — ensure the project root is on sys.path so that
# ``from common.helpers import ...`` resolves when this file is imported
# from any sub-project's src/ directory.
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy import stats as sp_stats

from common.helpers import smooth_log_trend

# Consistency factor: sigma ≈ 1.4826 × MAD for a normal distribution.
_MAD_TO_STD = 1.4826


# ======================================================================
# Private helpers — carry-forward / extraction
# ======================================================================


def _carry_forward_positive(series: np.ndarray | list) -> np.ndarray:
    """Fill non-positive / non-finite samples with the most recent positive value."""
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
    """Apply positive carry-forward independently to each column."""
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return arr
    out = arr.copy()
    for j in range(out.shape[1]):
        out[:, j] = _carry_forward_positive(out[:, j])
    return out


def _has_positive_finite(arr: np.ndarray | list) -> bool:
    """Return True when an array contains at least one finite value > 0."""
    a = np.asarray(arr, dtype=float).ravel()
    if a.size == 0:
        return False
    return bool(np.any(np.isfinite(a) & (a > 0)))


def _extract_positive(series: np.ndarray | list) -> tuple[np.ndarray, np.ndarray]:
    """Extract positive finite values and their original indices."""
    arr = np.asarray(series, dtype=float).ravel()
    if arr.size == 0:
        return np.array([]), np.array([], dtype=int)
    valid = np.isfinite(arr) & (arr > 0)
    indices = np.where(valid)[0]
    return arr[valid], indices


def _extract_positive_2d(matrix: np.ndarray | list) -> tuple[np.ndarray, np.ndarray]:
    """Extract rows where any column has a positive finite value."""
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
    hessian_intv: int = 1,
    compute_fd: bool = False,
) -> plt.Figure:
    """
    1×3 figure: curvature traces | rolling Spearman | rolling Pearson.

    Left panel
        All spectral-norm curvature metrics plotted raw (thin) with a
        log-space Whittaker–Henderson smoothed overlay (thick, semi-transparent).

    Middle / Right panels
        Rolling-window Spearman (middle) and Pearson (right) correlation of
        each proxy against the exact Hessian H, computed in a centred window
        of 5 000 iterations (half-width = 2 500).  Both raw-scale (solid) and
        log-scale (dashed) rolling correlations are overlaid on the same panel.
        A horizontal line at the same style marks the corresponding whole-run
        baseline for each proxy.

    Args:
        history:      Training history dict with curvature metric keys.
        lam:          Smoothing strength for ``smooth_log_trend``.
        save_path:    If provided, save the figure to this path.
        skip_intv:    If True (default), skip zero-placeholder intervals and
                      plot only actually-measured values against their true
                      iteration indices.  If False, use the legacy
                      ``_carry_forward_positive`` step-function fill.
        hessian_intv: Hessian computation frequency (used for x-axis label).
        compute_fd:   If False (default), skip BFGS and FD metrics entirely.

    Returns:
        The matplotlib ``Figure`` object.
    """
    _ROLLING_HALF = 2500  # half-window width in iteration-index space

    fig, (ax_left, ax_mid, ax_right) = plt.subplots(1, 3, figsize=(21, 8))

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

    h_arr,     h_idx     = _prep("hessian")
    prec_arr,  prec_idx  = _prep("prec_h")
    gn_arr,    gn_idx    = _prep("gn")
    vv_arr,    vv_idx    = _prep("hessian_vv")
    diag_arr,  diag_idx  = _prep("diag_h")
    fisher_arr, fisher_idx = _prep("fisher")
    kfac_arr,  kfac_idx  = _prep("kfac")
    if compute_fd:
        bfgs_arr, bfgs_idx = _prep("bfgs")
        fd_arr,   fd_idx   = _prep("fd")
    else:
        bfgs_arr, bfgs_idx = np.array([]), np.array([], dtype=int)
        fd_arr,   fd_idx   = np.array([]), np.array([], dtype=int)

    # ------------------------------------------------------------------
    # Left panel — raw traces + smoothed overlays
    # ------------------------------------------------------------------
    ax = ax_left
    if _has_positive_finite(h_arr):
        ax.plot(h_idx, h_arr, color="red", linewidth=2, label="Exact Hessian (H)")
    if _has_positive_finite(prec_arr):
        ax.plot(prec_idx, prec_arr, color="purple", linestyle="--", linewidth=2,
                label=r"Precond. Hessian ($\tilde{H}$)")
    if _has_positive_finite(gn_arr):
        ax.plot(gn_idx, gn_arr, color="brown", linestyle="--", linewidth=2,
                label=r"Gauss-Newton ($H^{GN}$)")
    if _has_positive_finite(vv_arr):
        ax.plot(vv_idx, vv_arr, color="magenta", linestyle=":", linewidth=2,
                label=r"Value Subspace ($H_{VV}$)")
    if _has_positive_finite(diag_arr):
        ax.plot(diag_idx, diag_arr, color="teal", linestyle="-.", linewidth=2,
                label="Diag Hessian")
    if _has_positive_finite(fisher_arr):
        ax.plot(fisher_idx, fisher_arr, color="olive", linestyle="-.", linewidth=2,
                label="Empirical Fisher")
    if compute_fd:
        if _has_positive_finite(bfgs_arr):
            ax.plot(bfgs_idx, bfgs_arr, color="navy", linestyle=":", linewidth=2,
                    label="BFGS")
        elif _has_positive_finite(_as1d("bfgs")):
            print("[warn] compute_fd=True but BFGS data has no positive values")
        if _has_positive_finite(fd_arr):
            ax.plot(fd_idx, fd_arr, color="cyan", linestyle="-.", linewidth=2,
                    label="FD")
        elif _has_positive_finite(_as1d("fd")):
            print("[warn] compute_fd=True but FD data has no positive values")
    if _has_positive_finite(kfac_arr):
        ax.plot(kfac_idx, kfac_arr, color="darkgreen", linestyle=":", linewidth=2,
                label="K-FAC")

    _smooth_alpha = 0.45
    _smooth_lw = 5

    def _overlay_smooth(arr, idx, color):
        if arr.size >= 3:
            trend, _, _ = smooth_log_trend(arr, lam=lam, use_abs=True)
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

    _xlabel = f"Iteration (every {hessian_intv})" if skip_intv and hessian_intv > 1 else "Iteration"
    ax.set_yscale("log")
    ax.set_title(f"Curvature Metrics — Raw vs Smoothed (λ={lam})", fontsize=12)
    ax.set_xlabel(_xlabel, fontsize=11)
    ax.set_ylabel("Spectral Norm (λ_max)", fontsize=11)
    ax.legend(fontsize="x-small", loc="best")
    ax.grid(True, alpha=0.3, linestyle="--")

    # ------------------------------------------------------------------
    # Rolling-window correlation helper  (raw scale or log scale)
    # ------------------------------------------------------------------
    def _rolling_corr_vs_h(p_arr, p_idx, log_space: bool = False):
        """
        Rolling Spearman & Pearson of H vs a proxy over 5k-iter windows.

        If log_space=True, applies log to both series before computing
        (only positions where both series are strictly positive are kept).
        Returns (iters, spearman, pearson, whole_spearman, whole_pearson).
        """
        if not _has_positive_finite(h_arr) or not _has_positive_finite(p_arr):
            return np.array([]), np.array([]), np.array([]), float("nan"), float("nan")
        n = min(h_arr.size, p_arr.size)
        if n < 3:
            return np.array([]), np.array([]), np.array([]), float("nan"), float("nan")
        h_use   = h_arr[:n].astype(float)
        p_use   = p_arr[:n].astype(float)
        idx_use = h_idx[:n]

        if log_space:
            valid = (h_use > 0) & (p_use > 0)
            if valid.sum() < 3:
                return np.array([]), np.array([]), np.array([]), float("nan"), float("nan")
            h_use = np.where(valid, np.log(np.where(valid, h_use, 1.0)), np.nan)
            p_use = np.where(valid, np.log(np.where(valid, p_use, 1.0)), np.nan)

        # Whole-run correlation
        mask_all = np.isfinite(h_use) & np.isfinite(p_use)
        if mask_all.sum() >= 3:
            sp_whole, _ = sp_stats.spearmanr(h_use[mask_all], p_use[mask_all])
            pe_whole, _ = sp_stats.pearsonr(h_use[mask_all], p_use[mask_all])
        else:
            sp_whole = pe_whole = float("nan")

        if idx_use.size == 0 or idx_use[-1] <= 2 * _ROLLING_HALF:
            return np.array([]), np.array([]), np.array([]), sp_whole, pe_whole

        roll_iters, roll_sp, roll_pe = [], [], []
        for i in range(n):
            center = idx_use[i]
            if center < _ROLLING_HALF or center > idx_use[-1] - _ROLLING_HALF:
                continue
            lo = int(np.searchsorted(idx_use, center - _ROLLING_HALF))
            hi = int(np.searchsorted(idx_use, center + _ROLLING_HALF, side="right"))
            if hi - lo < 3:
                continue
            h_win = h_use[lo:hi]
            p_win = p_use[lo:hi]
            mask = np.isfinite(h_win) & np.isfinite(p_win)
            if mask.sum() < 3:
                continue
            sp, _ = sp_stats.spearmanr(h_win[mask], p_win[mask])
            pe, _ = sp_stats.pearsonr(h_win[mask], p_win[mask])
            roll_iters.append(center)
            roll_sp.append(sp)
            roll_pe.append(pe)

        return (np.asarray(roll_iters, dtype=float),
                np.asarray(roll_sp, dtype=float),
                np.asarray(roll_pe, dtype=float),
                sp_whole, pe_whole)

    # ------------------------------------------------------------------
    # Middle and right panels — rolling correlations (H vs each proxy).
    # Solid lines = raw scale; dashed lines = log scale (same colour).
    # Thin horizontal lines mark the corresponding whole-run baseline.
    # ------------------------------------------------------------------
    _proxy_rolling = [
        (prec_arr,   prec_idx,   "purple",    r"$\tilde{H}$"),
        (gn_arr,     gn_idx,     "brown",     r"$H^{GN}$"),
        (vv_arr,     vv_idx,     "magenta",   r"$H_{VV}$"),
        (diag_arr,   diag_idx,   "teal",      "Diag H"),
        (fisher_arr, fisher_idx, "olive",     "Fisher"),
        (kfac_arr,   kfac_idx,   "darkgreen", "K-FAC"),
    ]
    if compute_fd:
        _proxy_rolling += [
            (bfgs_arr, bfgs_idx, "navy", "BFGS"),
            (fd_arr,   fd_idx,   "cyan", "FD"),
        ]

    for p_arr_r, p_idx_r, color_r, label_r in _proxy_rolling:
        # Raw-scale rolling
        iters_r, sp_r, pe_r, sp_w, pe_w = _rolling_corr_vs_h(
            p_arr_r, p_idx_r, log_space=False
        )
        if iters_r.size > 0:
            ax_mid.plot(iters_r, sp_r, color=color_r, linewidth=1.5,
                        linestyle="-", label=label_r)
            if not np.isnan(sp_w):
                ax_mid.axhline(sp_w, color=color_r, linewidth=0.8,
                               linestyle="-", alpha=0.4)
            ax_right.plot(iters_r, pe_r, color=color_r, linewidth=1.5,
                          linestyle="-", label=label_r)
            if not np.isnan(pe_w):
                ax_right.axhline(pe_w, color=color_r, linewidth=0.8,
                                 linestyle="-", alpha=0.4)

        # Log-scale rolling (same colour, dashed)
        iters_l, sp_l, pe_l, sp_wl, pe_wl = _rolling_corr_vs_h(
            p_arr_r, p_idx_r, log_space=True
        )
        if iters_l.size > 0:
            ax_mid.plot(iters_l, sp_l, color=color_r, linewidth=1.5,
                        linestyle="--", label=f"{label_r} (log)")
            if not np.isnan(sp_wl):
                ax_mid.axhline(sp_wl, color=color_r, linewidth=0.8,
                               linestyle="--", alpha=0.4)
            ax_right.plot(iters_l, pe_l, color=color_r, linewidth=1.5,
                          linestyle="--", label=f"{label_r} (log)")
            if not np.isnan(pe_wl):
                ax_right.axhline(pe_wl, color=color_r, linewidth=0.8,
                                 linestyle="--", alpha=0.4)

    for _ax, _title, _ylabel in [
        (ax_mid,   "Rolling Spearman ρ  (solid=raw, dashed=log)", "Spearman ρ"),
        (ax_right, "Rolling Pearson r   (solid=raw, dashed=log)", "Pearson r"),
    ]:
        _ax.set_title(_title, fontsize=12)
        _ax.set_xlabel(_xlabel, fontsize=11)
        _ax.set_ylabel(_ylabel, fontsize=11)
        _ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
        _ax.set_ylim(-1.05, 1.05)
        _ax.legend(fontsize="x-small", loc="best")
        _ax.grid(True, alpha=0.3, linestyle="--")

    fig.suptitle(f"Curvature Analysis (λ_max) — λ={lam}", fontsize=13)
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
    Detect and plot spike co-occurrence between two metric time-series using
    the Median Absolute Deviation (MAD) method.

    Spikes are local anomalies — points that deviate from a rolling-median
    baseline by more than ``z_score`` MADs.  The timeline strip shows:
      * Blue  ``|``  — spikes in X only
      * Red   ``×``  — joint spikes (both X and Y spike together)
      * Orange ``|`` — spikes in Y only

    Args:
        x, y:       1-D arrays of the same length.
        x_name:     Label for the X series.
        y_name:     Label for the Y series.
        window:     Median-filter window size.
        z_score:    MAD multiplier for spike detection.
        log_scale:  If True, take log of the series before residuals.
        save_path:  If provided, save the figure to this path.

    Returns:
        fig:     The matplotlib ``Figure``.
        results: Dict with keys ``P(Y_spike | X_spike)``,
                 ``baseline_P(Y_spike)``, ``n_X_spikes``, ``n_Y_spikes``,
                 ``n_joint_spikes``, ``n_points``.
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
            "n_X_spikes": 0,
            "n_Y_spikes": 0,
            "n_joint_spikes": 0,
            "n_points": 0,
        }
        fig = plt.figure(figsize=(12, 2.5))
        plt.text(0.5, 0.5, "No valid data points", ha="center", va="center",
                 transform=plt.gca().transAxes)
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

    x_spike_iters = idx[x_spike]
    y_spike_iters = idx[y_spike]
    joint_spike_iters = idx[joint_spike]

    n_x = len(x_spike_iters)
    n_y = len(y_spike_iters)
    n_both = len(joint_spike_iters)
    n_pts = len(x_c)

    cond_prob = n_both / n_x if n_x > 0 else float("nan")
    baseline = n_y / n_pts if n_pts > 0 else float("nan")

    results = {
        "P(Y_spike | X_spike)": cond_prob,
        "baseline_P(Y_spike)": baseline,
        "n_X_spikes": n_x,
        "n_Y_spikes": n_y,
        "n_joint_spikes": n_both,
        "n_points": n_pts,
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
    hessian_intv: int = 1,
    compute_fd: bool = False,
) -> tuple[dict[str, plt.Figure], dict[str, dict]]:
    """
    Compute MAD spike co-occurrence for H vs every proxy AND Prec_H vs every
    other proxy.

    Returns
    -------
    figs : dict keyed by ``"H.<proxy>"`` and ``"Prec_H.<proxy>"``.
    results : nested dict ``{"H": {proxy_key: stats_dict, ...},
                             "Prec_H": {proxy_key: stats_dict, ...}}``.
        Each ``stats_dict`` has the same keys as ``plot_spike_cooccurrence``.

    Args:
        skip_intv:    If True (default), extract only positive finite values.
        hessian_intv: Hessian computation frequency (informational).
        compute_fd:   If False (default), skip BFGS and FD proxies.
    """
    def _get_arr(key: str) -> np.ndarray:
        raw = np.asarray(history.get(key, []), dtype=float).ravel()
        if skip_intv:
            arr, _ = _extract_positive(raw)
        else:
            arr = raw
        return arr

    h      = _get_arr("hessian")
    prec_h = _get_arr("prec_h")

    # Proxy specs shared between H-ref and Prec_H-ref comparisons
    _shared_proxies = [
        ("hessian_vv", "H_VV",   "hessian_vv"),
        ("gn",         "GN",     "hessian_gn"),
        ("diag_h",     "Diag_H", "hessian_diag"),
        ("fisher",     "Fisher", "hessian_fisher"),
        ("kfac",       "KFAC",   "hessian_kfac"),
    ]
    if compute_fd:
        _shared_proxies.extend([
            ("fd",   "FD",   "hessian_fd"),
            ("bfgs", "BFGS", "hessian_bfgs"),
        ])

    figs: dict[str, plt.Figure] = {}
    results: dict[str, dict] = {"H": {}, "Prec_H": {}}

    # ------------------------------------------------------------------
    # H as reference — compare against Prec_H + all shared proxies
    # ------------------------------------------------------------------
    h_proxy_specs = [("prec_h", "Prec_H", "hessian_prec")] + _shared_proxies
    if _has_positive_finite(h):
        for key, label, suffix in h_proxy_specs:
            y = _get_arr(key)
            if y.size == 0 or not _has_positive_finite(y):
                continue
            n = min(h.size, y.size)
            save_path = (
                f"{save_dir}/spike_cooccurrence_H_vs_{suffix}_z{z_score}.png"
                if save_dir is not None else None
            )
            fig, res = plot_spike_cooccurrence(
                h[:n], y[:n],
                x_name="H", y_name=label,
                window=window, z_score=z_score,
                log_scale=log_scale, save_path=save_path,
            )
            figs[f"H.{key}"] = fig
            results["H"][key] = res

    # ------------------------------------------------------------------
    # Prec_H as reference — compare against all shared proxies
    # ------------------------------------------------------------------
    if _has_positive_finite(prec_h):
        for key, label, suffix in _shared_proxies:
            y = _get_arr(key)
            if y.size == 0 or not _has_positive_finite(y):
                continue
            n = min(prec_h.size, y.size)
            save_path = (
                f"{save_dir}/spike_cooccurrence_PrecH_vs_{suffix}_z{z_score}.png"
                if save_dir is not None else None
            )
            fig, res = plot_spike_cooccurrence(
                prec_h[:n], y[:n],
                x_name="Prec_H", y_name=label,
                window=window, z_score=z_score,
                log_scale=log_scale, save_path=save_path,
            )
            figs[f"Prec_H.{key}"] = fig
            results["Prec_H"][key] = res

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
    hessian_intv: int = 1,
    compute_fd: bool = False,
    start: int = 0,
    end: int = -1,
) -> dict:
    """
    Print raw and smoothed correlations for curvature metrics and return a
    structured results dict for downstream reporting.

    Args:
        history:        Training history dict.
        name:           Experiment name (printed as a header).
        sample_every:   Subsample stride applied after extraction.
        lam:            Whittaker–Henderson smoothing strength.
        include_smooth: If False, skip the smoothed-correlation block.
        skip_intv:      If True (default), use only positive finite values.
        hessian_intv:   Hessian computation frequency (informational).
        compute_fd:     If True, include BFGS and FD metrics.
        start:          First iteration index to include (default 0).
        end:            One-past-last iteration index; -1 means the full
                        history (default -1).  Slicing is applied to the raw
                        history arrays *before* positive-value extraction.

    Returns:
        dict with keys ``raw``, ``smoothed``, ``entropy`` — each mapping
        a pair label to ``{"spearman": float, "pearson": float}``.
    """
    _end = None if end == -1 else end

    def _gk(d, *keys):
        for k in keys:
            if k in d:
                raw = np.array(d[k])[start:_end]
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

    results: dict = {"raw": {}, "smoothed": {}, "log_raw": {}, "log_smoothed": {}, "entropy": {}}

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

    # --- Log-scale raw correlations ---
    def _to_log(arr: np.ndarray) -> np.ndarray:
        out = np.full(arr.shape, np.nan, dtype=float)
        valid = np.isfinite(arr) & (arr > 0)
        out[valid] = np.log(arr[valid])
        return out

    h_log      = _to_log(h)
    prec_h_log = _to_log(prec_h)
    vv_log     = _to_log(vv)
    gn_log     = _to_log(gn)
    diag_h_log = _to_log(diag_h)
    fisher_log = _to_log(fisher)
    kfac_log   = _to_log(kfac)

    print(f"\n--- {name} Log-scale Raw Correlations (log H as reference) ---")
    _corr(h_log, prec_h_log, "log H vs log Prec_H  ", results["log_raw"])
    _corr(h_log, vv_log,     "log H vs log H_VV    ", results["log_raw"])
    _corr(h_log, gn_log,     "log H vs log GN      ", results["log_raw"])
    _corr(h_log, diag_h_log, "log H vs log Diag_H  ", results["log_raw"])
    _corr(h_log, fisher_log, "log H vs log Fisher  ", results["log_raw"])
    _corr(h_log, kfac_log,   "log H vs log KFAC    ", results["log_raw"])
    if compute_fd:
        fd_log   = _to_log(fd)
        bfgs_log = _to_log(bfgs)
        _corr(h_log, fd_log,   "log H vs log FD      ", results["log_raw"])
        _corr(h_log, bfgs_log, "log H vs log BFGS    ", results["log_raw"])

    if _has_positive_finite(prec_h):
        print(f"\n--- {name} Log-scale Raw Correlations (log Prec_H as reference) ---")
        _corr(prec_h_log, vv_log,     "log Prec_H vs log H_VV   ", results["log_raw"])
        _corr(prec_h_log, gn_log,     "log Prec_H vs log GN     ", results["log_raw"])
        _corr(prec_h_log, diag_h_log, "log Prec_H vs log Diag_H ", results["log_raw"])
        _corr(prec_h_log, fisher_log, "log Prec_H vs log Fisher ", results["log_raw"])
        _corr(prec_h_log, kfac_log,   "log Prec_H vs log KFAC   ", results["log_raw"])
        if compute_fd:
            _corr(prec_h_log, fd_log,   "log Prec_H vs log FD     ", results["log_raw"])
            _corr(prec_h_log, bfgs_log, "log Prec_H vs log BFGS   ", results["log_raw"])

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

    # --- Log-scale smoothed correlations ---
    h_s_log      = np.log(np.maximum(h_s,      1e-300)) if h_s.size > 0      else h_s
    prec_h_s_log = np.log(np.maximum(prec_h_s, 1e-300)) if prec_h_s.size > 0 else prec_h_s
    vv_s_log     = np.log(np.maximum(vv_s,     1e-300)) if vv_s.size > 0     else vv_s
    gn_s_log     = np.log(np.maximum(gn_s,     1e-300)) if gn_s.size > 0     else gn_s
    diag_h_s_log = np.log(np.maximum(diag_h_s, 1e-300)) if diag_h_s.size > 0 else diag_h_s
    fisher_s_log = np.log(np.maximum(fisher_s, 1e-300)) if fisher_s.size > 0 else fisher_s
    kfac_s_log   = np.log(np.maximum(kfac_s,   1e-300)) if kfac_s.size > 0   else kfac_s

    print(f"\n--- {name} Log-scale Smoothed Correlations, log H reference (λ={lam}) ---")
    _corr(h_s_log, prec_h_s_log, "log H vs log Prec_H  ", results["log_smoothed"])
    _corr(h_s_log, vv_s_log,     "log H vs log H_VV    ", results["log_smoothed"])
    _corr(h_s_log, gn_s_log,     "log H vs log GN      ", results["log_smoothed"])
    _corr(h_s_log, diag_h_s_log, "log H vs log Diag_H  ", results["log_smoothed"])
    _corr(h_s_log, fisher_s_log, "log H vs log Fisher  ", results["log_smoothed"])
    _corr(h_s_log, kfac_s_log,   "log H vs log KFAC    ", results["log_smoothed"])
    if compute_fd:
        fd_s_log   = np.log(np.maximum(fd_s,   1e-300)) if fd_s.size > 0   else fd_s
        bfgs_s_log = np.log(np.maximum(bfgs_s, 1e-300)) if bfgs_s.size > 0 else bfgs_s
        _corr(h_s_log, fd_s_log,   "log H vs log FD      ", results["log_smoothed"])
        _corr(h_s_log, bfgs_s_log, "log H vs log BFGS    ", results["log_smoothed"])

    if _has_positive_finite(prec_h_s):
        print(f"\n--- {name} Log-scale Smoothed Correlations, log Prec_H reference (λ={lam}) ---")
        _corr(prec_h_s_log, vv_s_log,     "log Prec_H vs log H_VV   ", results["log_smoothed"])
        _corr(prec_h_s_log, gn_s_log,     "log Prec_H vs log GN     ", results["log_smoothed"])
        _corr(prec_h_s_log, diag_h_s_log, "log Prec_H vs log Diag_H ", results["log_smoothed"])
        _corr(prec_h_s_log, fisher_s_log, "log Prec_H vs log Fisher ", results["log_smoothed"])
        _corr(prec_h_s_log, kfac_s_log,   "log Prec_H vs log KFAC   ", results["log_smoothed"])
        if compute_fd:
            _corr(prec_h_s_log, fd_s_log,   "log Prec_H vs log FD     ", results["log_smoothed"])
            _corr(prec_h_s_log, bfgs_s_log, "log Prec_H vs log BFGS   ", results["log_smoothed"])

    raw_ent = history.get("entropy", [])
    if raw_ent and len(raw_ent) > 0:
        ent_arr = np.array(raw_ent)[start:_end][::sample_every]
        if ent_arr.ndim == 2 and ent_arr.shape[1] > 0:
            ent_first = ent_arr[:, 0]
            ent_avg   = ent_arr.mean(axis=1)
        else:
            ent_first = ent_avg = np.asarray([])
    else:
        ent_first = ent_avg = np.asarray([])

    if ent_first.size >= 3 and h_s.size >= 3:
        n = min(len(h_s), len(ent_first))
        _h_s_log = np.log(np.maximum(h_s, 1e-300))
        print(f"\n--- {name} Log Proxy vs Entropy (smoothed log proxy, λ={lam}) ---")
        _corr(_h_s_log[:n],      ent_first[:n], "log H vs Entropy(L0)      ", results["entropy"])
        _corr(_h_s_log[:n],      ent_avg[:n],   "log H vs Entropy(avg)     ", results["entropy"])
        if prec_h_s.size > 0:
            _prec_h_s_log = np.log(np.maximum(prec_h_s, 1e-300))
            _corr(_prec_h_s_log[:n], ent_first[:n], "log Prec_H vs Entropy(L0) ", results["entropy"])
            _corr(_prec_h_s_log[:n], ent_avg[:n],   "log Prec_H vs Entropy(avg)", results["entropy"])
        if vv_s.size > 0:
            _vv_s_log = np.log(np.maximum(vv_s, 1e-300))
            _corr(_vv_s_log[:n],     ent_first[:n], "log H_VV vs Entropy(L0)   ", results["entropy"])
        if gn_s.size > 0:
            _gn_s_log = np.log(np.maximum(gn_s, 1e-300))
            _corr(_gn_s_log[:n],     ent_first[:n], "log GN vs Entropy(L0)     ", results["entropy"])

    return results


def print_smooth_correlations(
    history: dict,
    name: str,
    lam: float = 10.0,
    sample_every: int = 1,
) -> dict:
    """Backward-compatible wrapper — delegates to ``print_correlations``."""
    return print_correlations(
        history, name, sample_every=sample_every, lam=lam, include_smooth=True,
    )


# ======================================================================
# Training dynamics — multi-panel grid
# ======================================================================


def plot_training_dynamics(
    histories: dict[str, dict],
    lrs: dict[str, float],
    save_path: str | None = None,
    skip_intv: bool = True,
    entropy_intv: int = 1,
    task: str = "classification",
) -> plt.Figure:
    """
    Plot compact training dynamics for one or more runs.

    Layout is driven by the ``task`` argument:

    * ``"classification"`` — 3 cols: CE loss / accuracy / entropy  (ViT, ViT5)
    * ``"depth"``          — 3 cols: SILog loss / RMSE+δ<1.25 / entropy (ViT_depth)
    * ``"lm"``             — 2 cols: CE loss / entropy  (nanochat)

    Args:
        histories:    Dict mapping run name to a history dict.  Keys used:
                      ``loss``, ``val_loss``, ``acc``/``train_acc``,
                      ``val_acc``/``test_acc``, ``val_rmse``, ``val_delta1``,
                      ``entropy``.
        lrs:          Dict mapping run name to peak LR (API compatibility).
        save_path:    If provided, save the figure to this path.
        skip_intv:    If True (default), skip zero-placeholder entropy rows.
        entropy_intv: Entropy computation frequency (used for x-axis label).
        task:         One of ``"classification"``, ``"depth"``, ``"lm"``.

    Returns:
        The matplotlib ``Figure`` object.
    """
    opt_names = list(histories.keys())
    n_rows = len(opt_names)
    n_cols = 2 if task == "lm" else 3
    fw = 14 if task == "lm" else 21
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(fw, 4.8 * n_rows))
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

        # --- Col 0: loss (all tasks) ---
        ax_loss = axs[row, 0]
        loss_arr = _as1d("loss")
        if task == "depth":
            train_label = f"{name} train SILog"
            val_label   = f"{name} val SILog"
            loss_ylabel = "SILog loss"
            loss_title  = f"{name} Scale-Invariant Log Loss"
        else:
            train_label = "train loss"
            val_label   = "val loss"
            loss_ylabel = "Cross-entropy loss"
            loss_title  = "Loss"
        ax_loss.plot(loss_arr, color=color, linewidth=2, label=train_label)
        _sparse_or_dense(h.get("val_loss", []), dense_color="crimson",
                         dense_label=val_label, ax=ax_loss)
        ax_loss.set_title(loss_title)
        ax_loss.set_xlabel("Iteration")
        ax_loss.set_ylabel(loss_ylabel)
        ax_loss.legend(loc="upper right")
        ax_loss.grid(True, alpha=0.25, linestyle="--")

        # --- Col 1 / middle panel (classification or depth only) ---
        ent_col = 1  # default for lm
        if task == "classification":
            ax_acc = axs[row, 1]
            train_acc = _as1d("acc", "train_acc")
            if train_acc.size:
                ax_acc.plot(train_acc, color=color, linewidth=2, label="train acc")
            _sparse_or_dense(
                h.get("val_acc", h.get("test_acc", [])),
                dense_color="crimson", dense_label="val acc", ax=ax_acc,
            )
            ax_acc.set_title("Accuracy")
            ax_acc.set_xlabel("Iteration")
            ax_acc.set_ylabel("Accuracy")
            ax_acc.yaxis.set_major_formatter(
                plt.matplotlib.ticker.PercentFormatter(xmax=1.0)
                if (train_acc.size and train_acc.max() <= 1.0) else
                plt.matplotlib.ticker.ScalarFormatter()
            )
            ax_acc.legend(loc="lower right")
            ax_acc.grid(True, alpha=0.25, linestyle="--")
            ent_col = 2

        elif task == "depth":
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

            ax_d1 = ax_rmse.twinx()
            _sparse_or_dense(
                h.get("val_delta1", []),
                dense_color="crimson", dense_label=f"{name} δ<1.25",
                ax=ax_d1,
            )
            ax_d1.set_ylabel("δ<1.25 (%)", color="crimson")
            ax_d1.tick_params(axis="y", labelcolor="crimson")

            lines1, labels1 = ax_rmse.get_legend_handles_labels()
            lines2, labels2 = ax_d1.get_legend_handles_labels()
            ax_rmse.legend(lines1 + lines2, labels1 + labels2,
                           loc="lower right", fontsize="small")
            ax_rmse.grid(True, alpha=0.25, linestyle="--")
            ent_col = 2

        # --- Entropy panel ---
        ax_ent = axs[row, ent_col]
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
            f"Iteration (every {entropy_intv})"
            if skip_intv and entropy_intv > 1 else "Iteration"
        )
        ax_ent.set_xlabel(_ent_xlabel)
        ax_ent.set_ylabel("Entropy (nats)")
        ax_ent.legend(fontsize="small", ncol=2)
        ax_ent.grid(True, alpha=0.25, linestyle="--")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
