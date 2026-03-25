"""
Shared spike-detection and correlation utilities for sharpness proxy analysis.

These functions are model-agnostic and can be reused across ViT, LLM, and
any other experiment that records curvature metric time-series.

Two sets of utilities are provided:

1. ``plot_spike_cooccurrence``
   A spike-timeline strip for two concurrent time-series (e.g. exact
   Hessian vs. value-subspace Hessian).  Spikes are detected via the
   Median Absolute Deviation (MAD) method — a standard robust anomaly
   detector — and the plot highlights:
     - X-only spikes  (blue  |)
     - Y-only spikes  (orange|)
     - Joint spikes   (red   ×)

   The function also returns conditional-probability statistics so the
   caller can test whether spikes in one metric coincide with spikes in
   another beyond chance.

2. ``print_correlations``
   Prints Spearman and Pearson correlations between curvature metric
   pairs recorded in a history dict.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy import stats as sp_stats

# Consistency factor to convert MAD into an equivalent standard deviation
# for a normally-distributed variable: sigma ≈ 1.4826 × MAD.
# Reference: Rousseeuw & Croux (1993), "Alternatives to the Median Absolute
# Deviation", JASA 88(424):1273-1283.
_MAD_TO_STD = 1.4826


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

    Spikes are defined as local anomalies — points that deviate from a
    rolling-median baseline by more than ``z_score`` Median Absolute
    Deviations (MADs).  This is a robust alternative to global quantile
    thresholds and works well for heavy-tailed curvature dynamics.

    The timeline strip shows:
      * Blue  ``|``  — spikes in X only
      * Orange ``|`` — spikes in Y only
      * Red   ``×``  — joint spikes (both X and Y spike together)

    Args:
        x, y:       1-D arrays of the same length (e.g. H and H_VV
                    histories recorded at the same iterations).
        x_name:     Label for the X series (shown in legend & title).
        y_name:     Label for the Y series.
        window:     Median-filter window size (should roughly span one
                    spike width in iterations).
        z_score:    MAD multiplier for spike detection (2–3 typical;
                    higher = stricter).
        log_scale:  If True, take log of the series before computing
                    residuals (recommended for heavy-tailed curvature
                    metrics).
        save_path:  If provided, save the figure to this path.

    Returns:
        fig:     The matplotlib ``Figure``.
        results: Dict with spike statistics:
                 ``P(Y_spike | X_spike)``, ``baseline_P(Y_spike)``,
                 ``n_X_spikes``, ``n_Y_spikes``, ``n_joint_spikes``,
                 ``n_points``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    orig_indices = np.arange(len(x))
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)

    x_c = x[valid]
    y_c = y[valid]
    idx = orig_indices[valid]

    if log_scale:
        x_c = np.log(x_c)
        y_c = np.log(y_c)

    # Rolling-median baselines
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

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(14, 2.5))

    x_only = np.setdiff1d(x_spike_iters, joint_spike_iters)
    y_only = np.setdiff1d(y_spike_iters, joint_spike_iters)

    if len(x_only):
        ax.scatter(
            x_only,
            np.ones_like(x_only) * 2,
            color="blue",
            marker="|",
            s=200,
            label=f"{x_name} Spike Only",
        )
    if len(joint_spike_iters):
        ax.scatter(
            joint_spike_iters,
            np.ones_like(joint_spike_iters) * 1.5,
            color="red",
            marker="x",
            s=100,
            label="Joint Spike",
        )
    if len(y_only):
        ax.scatter(
            y_only,
            np.ones_like(y_only) * 1,
            color="orange",
            marker="|",
            s=200,
            label=f"{y_name} Spike Only",
        )

    ax.set_yticks([1, 1.5, 2])
    ax.set_yticklabels([f"{y_name} Only", "Joint", f"{x_name} Only"])
    ax.set_xlabel("Iteration")
    ax.set_title(
        f"Local Spike Co-occurrence: {x_name} vs {y_name} "
        f"(Z > {z_score})  |  "
        f"P(Y|X) = {cond_prob:.3f}  baseline = {baseline:.3f}"
    )
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.set_xlim(0, len(x))
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, results


def print_correlations(history: dict, name: str, sample_every: int = 1) -> None:
    """
    Print Spearman and Pearson correlations between all curvature metric
    pairs recorded in *history*.

    Args:
        history:      History dict as returned by the training loop.
                      Expected keys: ``hessian``, ``prec_h``, ``gn``,
                      ``hessian_vv`` (or ``vv``).
        name:         Experiment name (printed as a header).
        sample_every: Subsample stride (e.g. 3 if metrics were recorded
                      every 3rd iteration).
    """
    def _gk(d, *keys):
        for k in keys:
            if k in d:
                return np.array(d[k][::sample_every])
        return np.asarray([])

    h = _gk(history, "hessian")
    prec_h = _gk(history, "prec_h")
    gn = _gk(history, "gn")
    vv = _gk(history, "hessian_vv", "vv")

    def _corr(a, b, label):
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 3:
            print(f"  {label}: insufficient data")
            return
        sp, _ = sp_stats.spearmanr(a[mask], b[mask])
        pe, _ = sp_stats.pearsonr(a[mask], b[mask])
        print(f"  {label}: Spearman {sp:.4f} | Pearson {pe:.4f}")

    print(f"\n--- {name} Correlation Results ---")
    _corr(h, prec_h, "H vs Prec_H ")
    _corr(h, gn, "H vs GN     ")
    _corr(h, vv, "H vs H_VV   ")
