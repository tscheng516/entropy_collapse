"""
Shared plotting utilities for all entropy-collapse sub-projects.

All five function families are provided here and imported by per-project
``src/plotting.py`` thin wrappers so each folder stays self-contained for
existing callers.

Families
--------
1. ``plot_curvature_smoothed_comparison`` — raw vs smoothed for all nine metrics.
2. ``plot_spike_cooccurrence``            — called from plot_all_spike_cooccurrences.
3. ``plot_all_spike_cooccurrences``       — λ_max(H) vs each other proxy
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
    lam: float = 10.0,
    save_path: str | None = None,
    skip_intv: bool = True,
    hessian_intv: int = 1,
    entropy_intv: int = 1,
    compute_fd: bool = False,
    vs_prec_H: bool = False,
) -> tuple[plt.Figure, plt.Figure]:
    """
    2×3 figure: attention entropy | curvature traces | rolling Spearman
    (row 0 = raw / row 1 = smoothed).

    Row 0 — raw:
        Col 0  Per-layer attention entropy (raw).
        Col 1  Raw curvature metric traces (log y-scale).
        Col 2  Rolling Spearman ρ — H (or Prec_H) vs each proxy (raw scale).

    Row 1 — smoothed:
        Col 0  Per-layer attention entropy smoothed with WH smoother.
        Col 1  Smoothed curvature traces (log y-scale).
        Col 2  Rolling Spearman ρ — smoothed H (or Prec_H) vs smoothed proxy.

    Dashed horizontal lines mark the corresponding whole-run correlation baseline.

    Args:
        history:      Training history dict with curvature metric keys.
        lam:          Smoothing strength for ``smooth_log_trend``.
        save_path:    If provided, save the figure to this path.
        skip_intv:    If True (default), skip zero-placeholder intervals and
                      plot only actually-measured values against their true
                      iteration indices.  If False, use the legacy
                      ``_carry_forward_positive`` step-function fill.
        hessian_intv: Hessian computation frequency (used for x-axis label).
        entropy_intv: Entropy computation frequency (used for x-axis label).
        compute_fd:   If False (default), skip BFGS and FD metrics entirely.
        vs_prec_H:    If True, use Prec_H as reference in col 2 rolling Spearman.
                      If False (default), use exact Hessian (H) as reference.

    Returns:
        (fig, fig_simple): the 2×3 figure and a simplified 1×3 thumbnail
                           (second-row content with minimal styling).
    """
    import matplotlib.gridspec as gridspec
    _ROLLING_HALF = 2500  # half-window width in iteration-index space

    fig = plt.figure(figsize=(21, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)
    ax_ent_raw    = fig.add_subplot(gs[0, 0])   # raw entropy (per layer)
    ax_raw        = fig.add_subplot(gs[0, 1])   # raw curvature traces
    ax_sp         = fig.add_subplot(gs[0, 2])   # H/Prec_H vs proxy: raw Spearman
    ax_ent_smooth = fig.add_subplot(gs[1, 0])   # smoothed entropy (per layer)
    ax_smooth     = fig.add_subplot(gs[1, 1])   # smoothed curvature traces
    ax_sp_smooth  = fig.add_subplot(gs[1, 2])   # smoothed H/Prec_H vs proxy: Spearman

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

    # Shared x-axis upper bound — derived from actual measured iterations so
    # that all panels (entropy, curvature, Spearman) share the same x range
    # even though the rolling-window panels have a shorter plotted extent.
    _x_max_candidates = [
        idx[-1] for idx in [h_idx, prec_idx, gn_idx, vv_idx,
                             diag_idx, fisher_idx, kfac_idx]
        if idx.size > 0
    ]
    if compute_fd:
        _x_max_candidates += [idx[-1] for idx in [bfgs_idx, fd_idx] if idx.size > 0]
    _x_max: int | None = int(max(_x_max_candidates)) if _x_max_candidates else None

    _xlabel = (
        f"Iteration (every {hessian_intv})"
        if skip_intv and hessian_intv > 1 else "Iteration"
    )
    _ent_xlabel = (
        f"Iteration (every {entropy_intv})"
        if skip_intv and entropy_intv > 1 else "Iteration"
    )

    # ------------------------------------------------------------------
    # Entropy panels — raw (row 0) and smoothed (row 1)
    # ------------------------------------------------------------------
    raw_ent = history.get("entropy", [])
    ent_arr_raw = np.asarray(raw_ent, dtype=float) if len(raw_ent) > 0 else np.zeros((0, 0))
    if ent_arr_raw.ndim == 2 and ent_arr_raw.size > 0:
        if skip_intv:
            entropies, ent_idx_arr = _extract_positive_2d(ent_arr_raw)
        else:
            entropies = _carry_forward_positive_2d(ent_arr_raw)
            ent_idx_arr = np.arange(len(entropies))
    else:
        entropies = np.zeros((0, 0))
        ent_idx_arr = np.array([], dtype=int)

    if entropies.ndim == 2 and entropies.shape[1] > 0:
        n_layers = entropies.shape[1]
        colors_ent = plt.cm.viridis(np.linspace(0, 1, n_layers))
        for li in range(n_layers):
            ax_ent_raw.plot(ent_idx_arr, entropies[:, li],
                            color=colors_ent[li], label=f"Layer {li + 1}")
            if entropies[:, li].size >= 3:
                trend_ent, _, _ = smooth_log_trend(entropies[:, li], lam=lam, use_abs=True)
                ax_ent_smooth.plot(ent_idx_arr, trend_ent,
                                   color=colors_ent[li], label=f"Layer {li + 1}")

    for _ax_ent, _title_ent in [
        (ax_ent_raw,    "Attention Entropy (Raw)"),
        (ax_ent_smooth, f"Attention Entropy (Smoothed, λ={lam})"),
    ]:
        _ax_ent.set_title(_title_ent, fontsize=11)
        _ax_ent.set_xlabel(_ent_xlabel, fontsize=10)
        _ax_ent.set_ylabel("Entropy (nats)", fontsize=10)
        _ax_ent.legend(fontsize="x-small", loc="best", ncol=2)
        _ax_ent.grid(True, alpha=0.3, linestyle="--")

    # --- colour / style spec shared by both curvature panels ---
    _metric_specs = [
        (h_arr,      h_idx,      "red",       "-",   "Exact Hessian (H)"),
        (prec_arr,   prec_idx,   "purple",    "--",  r"Precond. Hessian ($\tilde{H}$)"),
        (gn_arr,     gn_idx,     "brown",     "--",  r"Gauss-Newton ($H^{GN}$)"),
        (vv_arr,     vv_idx,     "magenta",   ":",   r"Value Subspace ($H_{VV}$)"),
        (diag_arr,   diag_idx,   "teal",      "-.",  "Diag Hessian"),
        (fisher_arr, fisher_idx, "olive",     "-.",  "Empirical Fisher"),
        (kfac_arr,   kfac_idx,   "darkgreen", ":",   "K-FAC"),
    ]
    if compute_fd:
        _metric_specs += [
            (bfgs_arr, bfgs_idx, "navy", ":",   "BFGS"),
            (fd_arr,   fd_idx,   "cyan", "-.",  "FD"),
        ]

    # ------------------------------------------------------------------
    # Col 1 row 0 — raw curvature traces
    # ------------------------------------------------------------------
    for arr, idx, color, ls, label in _metric_specs:
        if _has_positive_finite(arr):
            ax_raw.plot(idx, arr, color=color, linestyle=ls, linewidth=1.5, label=label)
        elif compute_fd and label in ("BFGS", "FD") and _has_positive_finite(
                _as1d("bfgs" if label == "BFGS" else "fd")):
            print(f"[warn] compute_fd=True but {label} data has no positive values")

    ax_raw.set_yscale("log")
    ax_raw.set_title("Raw Curvature Metrics", fontsize=11)
    ax_raw.set_xlabel(_xlabel, fontsize=10)
    ax_raw.set_ylabel("Spectral Norm (λ_max)", fontsize=10)
    ax_raw.legend(fontsize="x-small", loc="best")
    ax_raw.grid(True, alpha=0.3, linestyle="--")

    # ------------------------------------------------------------------
    # Col 1 row 1 — smoothed curvature traces
    # ------------------------------------------------------------------
    for arr, idx, color, ls, label in _metric_specs:
        if _has_positive_finite(arr) and arr.size >= 3:
            trend, _, _ = smooth_log_trend(arr, lam=lam, use_abs=True)
            ax_smooth.plot(idx, trend, color=color, linestyle=ls,
                           linewidth=2, label=label)

    ax_smooth.set_yscale("log")
    ax_smooth.set_title(f"Smoothed Curvature Metrics (λ={lam})", fontsize=11)
    ax_smooth.set_xlabel(_xlabel, fontsize=10)
    ax_smooth.set_ylabel("Spectral Norm (λ_max)", fontsize=10)
    ax_smooth.legend(fontsize="x-small", loc="best")
    ax_smooth.grid(True, alpha=0.3, linestyle="--")

    # ------------------------------------------------------------------
    # Rolling-window correlation helper — general reference
    # ------------------------------------------------------------------
    def _rolling_corr(ref_arr_rc, ref_idx_rc, p_arr_rc, p_idx_rc,
                      log_space: bool = False):
        """
        Rolling Spearman & Pearson of ref vs proxy over 5k-iter windows.
        If log_space=True, applies log to both series first.
        Returns (iters, spearman, pearson, whole_spearman, whole_pearson).
        """
        if not _has_positive_finite(ref_arr_rc) or not _has_positive_finite(p_arr_rc):
            return np.array([]), np.array([]), np.array([]), float("nan"), float("nan")
        n = min(ref_arr_rc.size, p_arr_rc.size)
        if n < 3:
            return np.array([]), np.array([]), np.array([]), float("nan"), float("nan")
        ref_use = ref_arr_rc[:n].astype(float)
        p_use   = p_arr_rc[:n].astype(float)
        idx_use = ref_idx_rc[:n]

        if log_space:
            valid = (ref_use > 0) & (p_use > 0)
            if valid.sum() < 3:
                return np.array([]), np.array([]), np.array([]), float("nan"), float("nan")
            ref_use = np.where(valid, np.log(np.where(valid, ref_use, 1.0)), np.nan)
            p_use   = np.where(valid, np.log(np.where(valid, p_use,   1.0)), np.nan)

        mask_all = np.isfinite(ref_use) & np.isfinite(p_use)
        if mask_all.sum() >= 3:
            sp_whole, _ = sp_stats.spearmanr(ref_use[mask_all], p_use[mask_all])
            pe_whole, _ = sp_stats.pearsonr(ref_use[mask_all],  p_use[mask_all])
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
            r_win = ref_use[lo:hi]
            p_win = p_use[lo:hi]
            mask = np.isfinite(r_win) & np.isfinite(p_win)
            if mask.sum() < 3:
                continue
            sp, _ = sp_stats.spearmanr(r_win[mask], p_win[mask])
            pe, _ = sp_stats.pearsonr(r_win[mask],  p_win[mask])
            roll_iters.append(center)
            roll_sp.append(sp)
            roll_pe.append(pe)

        return (np.asarray(roll_iters, dtype=float),
                np.asarray(roll_sp,    dtype=float),
                np.asarray(roll_pe,    dtype=float),
                sp_whole, pe_whole)

    # Reference series for col 2 panels
    ref_arr_raw = prec_arr if vs_prec_H else h_arr
    ref_idx_raw = prec_idx if vs_prec_H else h_idx
    _ref_label  = r"$\tilde{H}$" if vs_prec_H else "H"

    # Smoothed reference
    if _has_positive_finite(ref_arr_raw) and ref_arr_raw.size >= 3:
        ref_arr_smooth, _, _ = smooth_log_trend(ref_arr_raw, lam=lam, use_abs=True)
    else:
        ref_arr_smooth = ref_arr_raw.copy()

    # Proxy list — excludes the reference itself
    if vs_prec_H:
        _proxy_rolling = [
            (h_arr,      h_idx,      "red",       "H"),
            (gn_arr,     gn_idx,     "brown",     r"$H^{GN}$"),
            (vv_arr,     vv_idx,     "magenta",   r"$H_{VV}$"),
            (diag_arr,   diag_idx,   "teal",      "Diag H"),
            (fisher_arr, fisher_idx, "olive",     "Fisher"),
            (kfac_arr,   kfac_idx,   "darkgreen", "K-FAC"),
        ]
    else:
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

    # Collect smoothed rolling Spearman results for reuse in fig_simple
    _smooth_rolling_results: list[tuple] = []

    # ------------------------------------------------------------------
    # Col 2 row 0 — raw rolling Spearman (reference vs proxies, raw scale)
    # Col 2 row 1 — smoothed rolling Spearman (smoothed ref vs smoothed proxies)
    # ------------------------------------------------------------------
    for p_arr_r, p_idx_r, color_r, label_r in _proxy_rolling:
        # Raw rolling → top row
        iters_r, sp_r, _, sp_w, _ = _rolling_corr(
            ref_arr_raw, ref_idx_raw, p_arr_r, p_idx_r, log_space=False
        )
        if iters_r.size > 0:
            ax_sp.plot(iters_r, sp_r, color=color_r, linewidth=1.5, label=label_r)
            if not np.isnan(sp_w):
                ax_sp.axhline(sp_w, color=color_r, linewidth=0.8,
                              linestyle="--", alpha=0.45)

        # Smoothed rolling → bottom row
        if _has_positive_finite(p_arr_r) and p_arr_r.size >= 3:
            p_arr_smooth, _, _ = smooth_log_trend(p_arr_r, lam=lam, use_abs=True)
        else:
            p_arr_smooth = p_arr_r.copy()
        iters_s, sp_s, _, sp_ws, _ = _rolling_corr(
            ref_arr_smooth, ref_idx_raw, p_arr_smooth, p_idx_r, log_space=False
        )
        if iters_s.size > 0:
            ax_sp_smooth.plot(iters_s, sp_s, color=color_r, linewidth=1.5, label=label_r)
            if not np.isnan(sp_ws):
                ax_sp_smooth.axhline(sp_ws, color=color_r, linewidth=0.8,
                                     linestyle="--", alpha=0.45)
        _smooth_rolling_results.append((iters_s, sp_s, color_r))

    for _ax, _title, _ylabel in [
        (ax_sp,        rf"Rolling Spearman ρ — {_ref_label} vs Proxy (raw)",      "Spearman ρ"),
        (ax_sp_smooth, rf"Rolling Spearman ρ — {_ref_label} vs Proxy (smoothed)", "Spearman ρ"),
    ]:
        _ax.set_title(_title, fontsize=11)
        _ax.set_xlabel(_xlabel, fontsize=10)
        _ax.set_ylabel(_ylabel, fontsize=10)
        _ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
        _ax.legend(fontsize="x-small", loc="best")
        _ax.grid(True, alpha=0.3, linestyle="--")

    # Shared x-axis limit for both fig and fig_simple
    _ent_x_max = int(ent_idx_arr[-1]) if ent_idx_arr.size > 0 else (_x_max or 0)
    _shared_x_max: int | None = max(_x_max, _ent_x_max) if _x_max is not None else None

    # Apply uniform x limits to all 6 panels
    if _shared_x_max is not None:
        for _ax_align in [ax_ent_raw, ax_raw, ax_sp,
                          ax_ent_smooth, ax_smooth, ax_sp_smooth]:
            _ax_align.set_xlim(0, _shared_x_max)

    fig.suptitle(
        f"Curvature Analysis (λ_max) — λ={lam}  |  5k-iter rolling window",
        fontsize=13,
    )
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    # ------------------------------------------------------------------
    # Simple comparison thumbnail (1×3) — second row of the 2×3 figure
    # ------------------------------------------------------------------
    fig_simple, axs_s = plt.subplots(1, 3, figsize=(20, 5))

    # Panel 0: smoothed average attention entropy
    if entropies.ndim == 2 and entropies.shape[1] > 0:
        ent_avg_s = entropies.mean(axis=1)
        if ent_avg_s.size >= 3:
            trend_ent_avg, _, _ = smooth_log_trend(ent_avg_s, lam=lam, use_abs=True)
            axs_s[0].plot(ent_idx_arr, trend_ent_avg, color="steelblue", linewidth=3)

    # Panel 1: smoothed curvature metrics — reference (H or Prec_H) and H_VV only
    _simple_metric_specs = [
        (prec_arr, prec_idx, "purple", "--") if vs_prec_H else (h_arr, h_idx, "red", "-"),
        (vv_arr,   vv_idx,   "magenta", ":"),
    ]
    for arr, idx, color, ls in _simple_metric_specs:
        if _has_positive_finite(arr) and arr.size >= 3:
            trend, _, _ = smooth_log_trend(arr, lam=lam, use_abs=True)
            axs_s[1].plot(idx, trend, color=color, linestyle=ls, linewidth=3)
    axs_s[1].set_yscale("log")
    axs_s[1].minorticks_off()

    # Panel 2: rolling Spearman — H_VV vs reference (smoothed) only
    if len(_smooth_rolling_results) > 2:
        iters_vv, sp_vv, color_vv = _smooth_rolling_results[2]
        if iters_vv.size > 0:
            axs_s[2].plot(iters_vv, sp_vv, color=color_vv, linewidth=3)
            axs_s[2].set_ylim(0, 1)
            # axs_s[2].axhline(0, color="black", linewidth=0.8,
            #                   linestyle="--", alpha=0.45)
            # axs_s[2].axhline(1, color="black", linewidth=0.8,
            #                   linestyle="--", alpha=0.45)

    _ref_title = r"$\tilde{H}$" if vs_prec_H else "H"
    _vv_label  = r"$H_{VV}$"
    for _ax_s, _t_s in zip(axs_s, [
        "Avg. Attention Entropy",
        f"{_ref_title} and {_vv_label}",
        f"Rolling Spearman\n({_vv_label} vs {_ref_title})",
    ]):
        _ax_s.set_title(_t_s, fontsize=16, fontweight="bold")
        _ax_s.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for _sp in _ax_s.spines.values():
            _sp.set_visible(True)

    if _shared_x_max is not None:
        for _ax_s in axs_s:
            _ax_s.set_xlim(0, _shared_x_max)

    fig_simple.tight_layout()
    return fig, fig_simple


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

    # --- Proxy pair lists (H as ref includes prec_h; Prec_H as ref excludes it) ---
    _h_proxies = [
        (prec_h, "Prec_H"), (vv, "H_VV"), (gn, "GN"),
        (diag_h, "Diag_H"), (fisher, "Fisher"), (kfac, "KFAC"),
    ]
    if compute_fd:
        _h_proxies += [(fd, "FD"), (bfgs, "BFGS")]
    _prec_proxies = _h_proxies[1:]

    def _corr_block(header, ref, ref_name, pairs, store):
        print(f"\n--- {name} {header} ---")
        for arr, plabel in pairs:
            _corr(ref, arr, f"{ref_name} vs {plabel}", store)

    # --- Raw ---
    _corr_block("Raw Correlations (H as reference)", h, "H", _h_proxies, results["raw"])
    if _has_positive_finite(prec_h):
        _corr_block("Raw Correlations (Prec_H as reference)", prec_h, "Prec_H", _prec_proxies, results["raw"])

    # --- Log-scale raw ---
    def _to_log(arr: np.ndarray) -> np.ndarray:
        out = np.full(arr.shape, np.nan, dtype=float)
        valid = np.isfinite(arr) & (arr > 0)
        out[valid] = np.log(arr[valid])
        return out

    h_log             = _to_log(h)
    _h_proxies_log    = [(_to_log(a), f"log {lab}") for a, lab in _h_proxies]
    _prec_proxies_log = _h_proxies_log[1:]
    prec_h_log        = _h_proxies_log[0][0]

    _corr_block("Log-scale Raw Correlations (log H as reference)", h_log, "log H", _h_proxies_log, results["log_raw"])
    if _has_positive_finite(prec_h):
        _corr_block("Log-scale Raw Correlations (log Prec_H as reference)", prec_h_log, "log Prec_H", _prec_proxies_log, results["log_raw"])

    if not include_smooth:
        return results

    # --- Smoothed ---
    def _smooth(arr):
        if arr.size < 3:
            return arr
        trend, _, _ = smooth_log_trend(arr, lam=lam, use_abs=True)
        return trend

    def _slog(arr: np.ndarray) -> np.ndarray:
        return np.log(np.maximum(arr, 1e-300)) if arr.size > 0 else arr

    h_s             = _smooth(h)
    _h_proxies_s    = [(_smooth(a), lab) for a, lab in _h_proxies]
    _prec_proxies_s = _h_proxies_s[1:]
    prec_h_s        = _h_proxies_s[0][0]

    _corr_block(f"Smoothed Correlations, H reference (λ={lam})", h_s, "H", _h_proxies_s, results["smoothed"])
    if _has_positive_finite(prec_h_s):
        _corr_block(f"Smoothed Correlations, Prec_H reference (λ={lam})", prec_h_s, "Prec_H", _prec_proxies_s, results["smoothed"])

    # --- Log-scale smoothed ---
    _h_proxies_sl    = [(_slog(a), f"log {lab}") for a, lab in _h_proxies_s]
    _prec_proxies_sl = _h_proxies_sl[1:]

    _corr_block(f"Log-scale Smoothed Correlations, log H reference (λ={lam})", _slog(h_s), "log H", _h_proxies_sl, results["log_smoothed"])
    if _has_positive_finite(prec_h_s):
        _corr_block(f"Log-scale Smoothed Correlations, log Prec_H reference (λ={lam})", _slog(prec_h_s), "log Prec_H", _prec_proxies_sl, results["log_smoothed"])

    # --- Entropy correlations ---
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
        _h_s_log = _slog(h_s)
        print(f"\n--- {name} Log Proxy vs Entropy (smoothed log proxy, λ={lam}) ---")
        _corr(_h_s_log[:n], ent_first[:n], "log H vs Entropy(L0)",       results["entropy"])
        _corr(_h_s_log[:n], ent_avg[:n],   "log H vs Entropy(avg)",      results["entropy"])
        if prec_h_s.size > 0:
            _corr(_slog(prec_h_s)[:n], ent_first[:n], "log Prec_H vs Entropy(L0)", results["entropy"])
            _corr(_slog(prec_h_s)[:n], ent_avg[:n],   "log Prec_H vs Entropy(avg)", results["entropy"])
        vv_s = _h_proxies_s[1][0]
        gn_s = _h_proxies_s[2][0]
        if vv_s.size > 0:
            _corr(_slog(vv_s)[:n], ent_first[:n], "log H_VV vs Entropy(L0)", results["entropy"])
        if gn_s.size > 0:
            _corr(_slog(gn_s)[:n], ent_first[:n], "log GN vs Entropy(L0)",   results["entropy"])

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

    # Auto-downgrade task when no task-specific metric data is present.
    # Prevents an empty middle panel when task='classification'/'depth' is
    # passed but the history dict contains no accuracy/RMSE keys (e.g. LM runs).
    if task == "classification":
        has_acc = any(
            len(h.get("acc", h.get("train_acc", h.get("val_acc", h.get("test_acc", []))))) > 0
            for h in histories.values()
        )
        if not has_acc:
            task = "lm"
    elif task == "depth":
        has_depth = any(
            len(h.get("val_rmse", h.get("val_delta1", []))) > 0
            for h in histories.values()
        )
        if not has_depth:
            task = "lm"

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
