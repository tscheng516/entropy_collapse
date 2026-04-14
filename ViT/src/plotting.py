"""
Plotting utilities for the ViT entropy-collapse experiments.

This module is self-contained: all spike-detection and correlation helpers
live here so no shared ``common/`` package is required.

Families of utilities provided:

1. ``plot_curvature_smoothed_comparison`` — raw vs smoothed comparison for all
                                 nine spectral-norm curvature metrics.
2. ``plot_spike_cooccurrence`` — MAD-based joint/disjoint spike timeline
                                for two concurrent metric series.
3. ``plot_all_spike_cooccurrences`` — lambda_max(H) vs each other proxy.
4. ``print_correlations`` — unified raw + smoothed correlation report.
5. ``plot_training_dynamics`` — compact training figure (loss + entropy).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy import stats as sp_stats

from src.helpers import smooth_log_trend

# Consistency factor: sigma ≈ 1.4826 × MAD for a normal distribution.
# Reference: Rousseeuw & Croux (1993), JASA 88(424):1273-1283.
_MAD_TO_STD = 1.4826


def _carry_forward_positive(series: np.ndarray | list) -> np.ndarray:
    """
    Fill non-positive / non-finite samples with the most recent positive value.

    Training scripts often log curvature metrics every N steps and append
    placeholders otherwise. The notebook visual style keeps the last measured
    value between refreshes; this helper reproduces that behavior.
    """
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


# ======================================================================
# Curvature metrics — raw vs smoothed comparison
# ======================================================================


def plot_curvature_smoothed_comparison(
    history: dict,
    lam: float = 100.0,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot all nine spectral-norm curvature metrics (raw + smoothed) side-by-side.

    Uses the same visual style as the training dynamics plot. Raw curves are
    thin and semi-transparent; smoothed trends (via log-space Whittaker–Henderson
    smoother) are thick and opaque overlaid on top.

    The nine metrics are:
      * hessian (H) — exact Hessian power iteration
      * prec_h — Adam-preconditioned Hessian
      * hessian_vv (H_VV) — value-projection subspace
      * gn — Gauss-Newton
      * bfgs — central-difference FD
      * fd — forward-difference FD
      * diag_h — max diagonal Hessian
      * fisher — empirical Fisher
      * kfac — K-FAC proxy

    Args:
        history:   Training history dict with curvature metric keys.
        lam:       Smoothing strength for ``smooth_log_trend``.
        save_path: If provided, save the figure to this path.

    Returns:
        The matplotlib ``Figure`` object.
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

    # --- Plot all nine metrics (raw + smoothed) ---
    h_arr = _carry_forward_positive(_as1d("hessian"))
    ax.plot(h_arr, color="red", linewidth=2, label="Exact Hessian (H)")

    prec_arr = _carry_forward_positive(_as1d("prec_h"))
    if _has_positive_finite(prec_arr):
        ax.plot(
            prec_arr,
            color="purple",
            linestyle="--",
            linewidth=2,
            label=r"Precond. Hessian ($\tilde{H}$)",
        )

    gn_arr = _carry_forward_positive(_as1d("gn"))
    if _has_positive_finite(gn_arr):
        ax.plot(
            gn_arr,
            color="brown",
            linestyle="--",
            linewidth=2,
            label=r"Gauss-Newton ($H^{GN}$)",
        )

    vv_arr = _carry_forward_positive(_as1d("hessian_vv"))
    if _has_positive_finite(vv_arr):
        ax.plot(
            vv_arr,
            color="magenta",
            linestyle=":",
            linewidth=2,
            label=r"Value Subspace ($H_{VV}$)",
        )

    diag_arr = _carry_forward_positive(_as1d("diag_h"))
    if _has_positive_finite(diag_arr):
        ax.plot(
            diag_arr,
            color="teal",
            linestyle="-.",
            linewidth=2,
            label="Diag Hessian",
        )

    fisher_arr = _carry_forward_positive(_as1d("fisher"))
    if _has_positive_finite(fisher_arr):
        ax.plot(
            fisher_arr,
            color="olive",
            linestyle="-.",
            linewidth=2,
            label="Empirical Fisher",
        )

    bfgs_arr = _carry_forward_positive(_as1d("bfgs"))
    if _has_positive_finite(bfgs_arr):
        ax.plot(
            bfgs_arr,
            color="navy",
            linestyle=":",
            linewidth=2,
            label="BFGS",
        )

    fd_arr = _carry_forward_positive(_as1d("fd"))
    if _has_positive_finite(fd_arr):
        ax.plot(
            fd_arr,
            color="cyan",
            linestyle="-.",
            linewidth=2,
            label="FD",
        )

    kfac_arr = _carry_forward_positive(_as1d("kfac"))
    if _has_positive_finite(kfac_arr):
        ax.plot(
            kfac_arr,
            color="darkgreen",
            linestyle=":",
            linewidth=2,
            label="K-FAC",
        )

    # --- Smoothed trend overlays (Whittaker–Henderson in log-space) ---
    _smooth_lam = lam
    _smooth_alpha = 0.45
    _smooth_lw = 5

    def _overlay_smooth(arr, color):
        if arr.size >= 3:
            trend, _, _ = smooth_log_trend(arr, lam=_smooth_lam, use_abs=True)
            ax.plot(trend, color=color, linewidth=_smooth_lw, alpha=_smooth_alpha)

    _overlay_smooth(h_arr, "red")
    if _has_positive_finite(prec_arr):
        _overlay_smooth(prec_arr, "purple")
    if _has_positive_finite(gn_arr):
        _overlay_smooth(gn_arr, "brown")
    if _has_positive_finite(vv_arr):
        _overlay_smooth(vv_arr, "magenta")
    if _has_positive_finite(diag_arr):
        _overlay_smooth(diag_arr, "teal")
    if _has_positive_finite(fisher_arr):
        _overlay_smooth(fisher_arr, "olive")
    if _has_positive_finite(bfgs_arr):
        _overlay_smooth(bfgs_arr, "navy")
    if _has_positive_finite(fd_arr):
        _overlay_smooth(fd_arr, "cyan")
    if _has_positive_finite(kfac_arr):
        _overlay_smooth(kfac_arr, "darkgreen")

    ax.set_yscale("log")
    ax.set_title(f"All Curvature Metrics (λ_max) — Raw vs Smoothed (λ={lam})", fontsize=13)
    ax.set_xlabel("Iteration", fontsize=12)
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
    Detect and plot spike co-occurrence between two metric time-series
    using the Median Absolute Deviation (MAD) method.

    Spikes are local anomalies — points that deviate from a rolling-median
    baseline by more than ``z_score`` MADs.  The timeline strip shows:
      * Blue  ``|``  — spikes in X only
      * Red   ``×``  — joint spikes (both X and Y spike together)
      * Orange ``|`` — spikes in Y only

    This implementation matches the ``conditional_exceedance_local``
    function in ``notebook.ipynb``.

        Implementation details:
                * Uses raw logged values for spike detection (no carry-forward) to
                    avoid staircase artifacts.
                * Drops placeholder points where both series are zero/non-finite.
                * If ``log_scale=True`` but too few positive samples exist, detection
                    automatically falls back to linear scale.
                * If no valid finite samples exist, the function returns an annotated
                    "No valid data points" figure and a results dict with zero counts.

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

    # Prefer log-scale spike detection, but fall back to linear scale when
    # too few positive samples are available (prevents empty plots).
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
        plt.text(0.5, 0.5, "No valid data points", ha="center", va="center", transform=plt.gca().transAxes)
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

    # --- 1. Rolling-median baselines (robust to spike contamination) ---
    x_base = median_filter(x_c, size=window)
    y_base = median_filter(y_c, size=window)

    # --- 2. Residuals above baseline ---
    x_res = x_c - x_base
    y_res = y_c - y_base

    # --- 3. MAD-based robust thresholds (1.4826 × MAD ≈ σ) ---
    x_mad = max(np.median(np.abs(x_res - np.median(x_res))) * _MAD_TO_STD, 1e-6)
    y_mad = max(np.median(np.abs(y_res - np.median(y_res))) * _MAD_TO_STD, 1e-6)

    # --- 4. Spike masks ---
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

    # --- 5. Plot spike timeline ---
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
) -> tuple[dict[str, plt.Figure], dict[str, dict]]:
    """
    Compute MAD spike co-occurrence for lambda_max(H) versus each of the
    other eight curvature proxies.

    Returns figures/results keyed by proxy name.
    """
    h = np.asarray(history.get("hessian", []), dtype=float).ravel()
    proxy_specs = [
        ("prec_h", "Prec_H", "hessian_prec"),
        ("hessian_vv", "H_VV", "hessian_vv"),
        ("gn", "GN", "hessian_gn"),
        ("fd", "FD", "hessian_fd"),
        ("diag_h", "Diag_H", "hessian_diag"),
        ("fisher", "Fisher", "hessian_fisher"),
        ("bfgs", "BFGS", "hessian_bfgs"),
        ("kfac", "KFAC", "hessian_kfac"),
    ]

    figs: dict[str, plt.Figure] = {}
    results: dict[str, dict] = {}

    for key, label, suffix in proxy_specs:
        y = np.asarray(history.get(key, []), dtype=float).ravel()
        if h.size == 0 or y.size == 0:
            continue
        if not (_has_positive_finite(h) and _has_positive_finite(y)):
            continue
        n = min(h.size, y.size)
        save_path = None
        if save_dir is not None:
            save_path = f"{save_dir}/spike_cooccurrence_H_vs_{suffix}_z{z_score}.png"

        fig, res = plot_spike_cooccurrence(
            h[:n],
            y[:n],
            x_name="Exact H",
            y_name=label,
            window=window,
            z_score=z_score,
            log_scale=log_scale,
            save_path=save_path,
        )
        figs[key] = fig
        results[key] = res

    return figs, results


# ======================================================================
# Correlation analysis (matches notebook.ipynb Cell 3)
# ======================================================================


def print_correlations(
    history: dict,
    name: str,
    sample_every: int = 1,
    lam: float = 100.0,
    include_smooth: bool = True,
) -> None:
    """
    Print raw and (optionally) smoothed correlations for curvature metrics.

    Args:
        history:      History dict from the training loop.  Expected keys:
                      ``hessian``, ``prec_h``, ``gn``, ``hessian_vv``
                      (or ``vv``).
        name:         Experiment name (printed as a header).
        sample_every: Subsample stride (e.g. 3 if metrics were recorded
                      every 3rd iteration, matching the notebook's
                      ``[::3]`` slicing).
    """
    def _gk(d, *keys):
        for k in keys:
            if k in d:
                return np.array(d[k][::sample_every])
        return np.asarray([])

    h = _gk(history, "hessian")
    prec_h = _gk(history, "prec_h")
    vv = _gk(history, "hessian_vv", "vv")
    gn = _gk(history, "gn")
    fd = _gk(history, "fd")
    diag_h = _gk(history, "diag_h")
    fisher = _gk(history, "fisher")
    bfgs = _gk(history, "bfgs")
    kfac = _gk(history, "kfac")

    def _corr(a, b, label):
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 3:
            print(f"  {label}: insufficient data")
            return
        a_m = a[mask]
        b_m = b[mask]
        if np.std(a_m) < 1e-12 or np.std(b_m) < 1e-12:
            print(f"  {label}: constant/near-constant series")
            return
        sp, _ = sp_stats.spearmanr(a_m, b_m)
        pe, _ = sp_stats.pearsonr(a_m, b_m)
        print(f"  {label}: Spearman {sp:.4f} | Pearson {pe:.4f}")

    print(f"\n--- {name} Correlation Results ---")
    _corr(h, prec_h,  "H vs Prec_H ")
    _corr(h, vv,      "H vs H_VV   ")
    _corr(h, gn,      "H vs GN     ")
    _corr(h, fd,      "H vs FD     ")
    _corr(h, diag_h,  "H vs Diag_H ")
    _corr(h, fisher,  "H vs Fisher ")
    _corr(h, bfgs,    "H vs BFGS   ")
    _corr(h, kfac,    "H vs KFAC   ")

    if not include_smooth:
        return

    def _smooth(arr):
        if arr.size < 3:
            return arr
        trend, _, _ = smooth_log_trend(arr, lam=lam, use_abs=True)
        return trend

    h_s = _smooth(h)
    prec_h_s = _smooth(prec_h)
    vv_s = _smooth(vv)
    gn_s = _smooth(gn)
    fd_s = _smooth(fd)
    diag_h_s = _smooth(diag_h)
    fisher_s = _smooth(fisher)
    bfgs_s = _smooth(bfgs)
    kfac_s = _smooth(kfac)

    print(f"\n--- {name} Smoothed Correlation Results (λ={lam}) ---")
    _corr(h_s, prec_h_s, "H vs Prec_H ")
    _corr(h_s, vv_s,     "H vs H_VV   ")
    _corr(h_s, gn_s,     "H vs GN     ")
    _corr(h_s, fd_s,     "H vs FD     ")
    _corr(h_s, diag_h_s, "H vs Diag_H ")
    _corr(h_s, fisher_s, "H vs Fisher ")
    _corr(h_s, bfgs_s,   "H vs BFGS   ")
    _corr(h_s, kfac_s,   "H vs KFAC   ")

    raw_ent = history.get("entropy", [])
    if raw_ent and len(raw_ent) > 0:
        ent_arr = np.array(raw_ent[::sample_every])
        if ent_arr.ndim == 2 and ent_arr.shape[1] > 0:
            ent_first = ent_arr[:, 0]
            ent_avg = ent_arr.mean(axis=1)
        else:
            ent_first = ent_avg = np.asarray([])
    else:
        ent_first = ent_avg = np.asarray([])

    if ent_first.size >= 3 and h_s.size >= 3:
        n = min(len(h_s), len(ent_first))
        print(f"\n--- {name} Smoothed Proxy vs Entropy (λ={lam}) ---")
        _corr(h_s[:n],      ent_first[:n], "H vs Entropy(L0) ")
        _corr(h_s[:n],      ent_avg[:n],   "H vs Entropy(avg)")
        _corr(vv_s[:n],     ent_first[:n], "H_VV vs Entropy  ")
        _corr(prec_h_s[:n], ent_first[:n], "Prec_H vs Entropy")


def print_smooth_correlations(
    history: dict,
    name: str,
    lam: float = 100.0,
    sample_every: int = 1,
) -> None:
    """Backward-compatible wrapper for legacy call sites."""
    print_correlations(
        history,
        name,
        sample_every=sample_every,
        lam=lam,
        include_smooth=True,
    )


# ======================================================================
# Training dynamics — multi-panel grid
# ======================================================================


def plot_training_dynamics(
    histories: dict[str, dict],
    lrs: dict[str, float],
    save_path: str | None = None,
) -> plt.Figure:
    """
        Plot compact training dynamics for one or two runs.

        Panels per run:
            * train/eval loss
            * per-layer attention entropy

    Args:
        histories:  Dict mapping optimizer name (e.g. ``"AdamW"``) to a
                    history dict with keys: ``loss``, ``val_loss``,
                    and ``entropy``.
                    ``entropy`` values are lists of per-layer floats.
        lrs:        Dict mapping optimizer name to the peak LR used
                    (kept for API compatibility).
        save_path:  If provided, save the figure to this path.

    Returns:
        The matplotlib ``Figure`` object.
    """
    opt_names = list(histories.keys())
    n_rows = len(opt_names)
    fig, axs = plt.subplots(n_rows, 2, figsize=(14, 4.8 * n_rows))
    if n_rows == 1:
        axs = axs[np.newaxis, :]  # ensure 2-D indexing works

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

        # --- Col 0: train/eval loss ---
        ax_loss = axs[row, 0]
        loss_arr = _as1d("loss")
        ax_loss.plot(loss_arr, color=color, linewidth=2, label=f"{name} train loss")
        val_series = h.get("val_loss", [])
        if isinstance(val_series, list) and len(val_series) > 0:
            first = val_series[0]
            if isinstance(first, tuple) and len(first) == 2:
                xs = [int(t[0]) for t in val_series]
                ys = [float(t[1]) for t in val_series]
                ax_loss.plot(xs, ys, color="crimson", linestyle="--", linewidth=2, label=f"{name} val loss")
            else:
                arr = np.asarray(val_series, dtype=float).ravel()
                if arr.size:
                    ax_loss.plot(arr, color="crimson", linestyle="--", linewidth=2, label=f"{name} val loss")
        ax_loss.set_title(f"{name} Loss")
        ax_loss.set_xlabel("Iteration")
        ax_loss.set_ylabel("Cross-entropy loss")
        ax_loss.legend(loc="upper left")
        ax_loss.grid(True, alpha=0.25, linestyle="--")

        # --- Col 1: per-layer attention entropy ---
        ax_ent = axs[row, 1]
        entropies = _carry_forward_positive_2d(np.array(h.get("entropy", [])))
        if entropies.ndim == 2 and entropies.shape[1] > 0:
            n_layers = entropies.shape[1]
            colors_entropy = plt.cm.viridis(np.linspace(0, 1, n_layers))
            for layer_idx in range(n_layers):
                ax_ent.plot(
                    entropies[:, layer_idx],
                    color=colors_entropy[layer_idx],
                    label=f"Layer {layer_idx + 1}",
                )
        ax_ent.set_title(f"{name} Attention Entropy (Per Layer)")
        ax_ent.set_xlabel("Iteration")
        ax_ent.set_ylabel("Entropy (nats)")
        ax_ent.legend(fontsize="small", ncol=2)
        ax_ent.grid(True, alpha=0.25, linestyle="--")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
