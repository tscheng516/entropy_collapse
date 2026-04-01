"""
Plotting utilities for the ViT entropy-collapse experiments.

This module is self-contained: all spike-detection and correlation helpers
live here so no shared ``common/`` package is required.

Two families of utilities are provided:

1. ``plot_training_dynamics``   — 2×6-panel training-dynamics grid (loss +
                                  accuracy, Hessian proxies, per-layer
                                  entropy, and three pairwise proxy scatter
                                  plots).
2. ``plot_spike_cooccurrence``  — MAD-based joint/disjoint spike timeline
                                  for two concurrent metric series.
                                  Matches the ``conditional_exceedance_local``
                                  format from ``Tin_Sum.ipynb``.
3. ``print_correlations``       — Spearman & Pearson correlations between
                                  curvature metric pairs.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy import stats as sp_stats

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


# ======================================================================
# MAD-based spike co-occurrence (matches Tin_Sum.ipynb Cell 5)
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
    function in ``Tin_Sum.ipynb``.

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
    x = _carry_forward_positive(x)
    y = _carry_forward_positive(y)

    orig_indices = np.arange(len(x))
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)

    x_c = x[valid]
    y_c = y[valid]
    idx = orig_indices[valid]

    if log_scale:
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


# ======================================================================
# Correlation analysis (matches Tin_Sum.ipynb Cell 3)
# ======================================================================


def print_correlations(history: dict, name: str, sample_every: int = 1) -> None:
    """
    Print Spearman and Pearson correlations between all curvature metric
    pairs recorded in *history*.

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
    _corr(h, gn,     "H vs GN     ")
    _corr(h, vv,     "H vs H_VV   ")


# ======================================================================
# Training dynamics — multi-panel grid
# ======================================================================


def plot_training_dynamics(
    histories: dict[str, dict],
    lrs: dict[str, float],
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot training dynamics for one or two training runs (e.g. AdamW vs SGD).

    Args:
        histories:  Dict mapping optimizer name (e.g. ``"AdamW"``) to a
                    history dict with keys:
                    ``loss``, ``val_loss``, ``acc``, ``val_acc``,
                    ``hessian``, ``prec_h``, ``hessian_vv``,
                    ``gn``, ``fd``, ``entropy``.
                    ``entropy`` values are lists of per-layer floats.
        lrs:        Dict mapping optimizer name to the peak LR used
                    (kept for API compatibility).
        save_path:  If provided, save the figure to this path.

    Returns:
        The matplotlib ``Figure`` object.
    """
    opt_names = list(histories.keys())
    n_rows = len(opt_names)
    fig, axs = plt.subplots(n_rows, 6, figsize=(36, 5 * n_rows))
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

        def _comp_plot(ax_cmp, x_arr, y_arr, x_label, y_label):
            x = np.asarray(x_arr, dtype=float)
            y = np.asarray(y_arr, dtype=float)
            valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
            if valid.sum() == 0:
                ax_cmp.text(0.5, 0.5, "no data", ha="center", va="center")
                return
            x_c = x[valid]
            y_c = y[valid]
            ax_cmp.scatter(x_c, y_c, s=8, alpha=0.6)
            mn = min(x_c.min(), y_c.min())
            mx = max(x_c.max(), y_c.max())
            ax_cmp.plot([mn, mx], [mn, mx], color="gray", linestyle="--", linewidth=1)
            ax_cmp.set_xscale("log")
            ax_cmp.set_yscale("log")
            try:
                sp, _ = sp_stats.spearmanr(x_c, y_c)
                pe, _ = sp_stats.pearsonr(x_c, y_c)
                stats_txt = f"Spearman {sp:.2f} | Pearson {pe:.2f}"
            except Exception:
                stats_txt = "corr: n/a"
            ax_cmp.set_title(f"{y_label} vs {x_label}\n{stats_txt}")
            ax_cmp.set_xlabel(x_label)
            ax_cmp.set_ylabel(y_label)

        # --- Col 0: loss (+ accuracy on twin axis) ---
        ax_loss = axs[row, 0]
        loss_arr = _as1d("loss")
        ax_loss.plot(loss_arr, color=color, label=f"{name} Loss")
        acc_arr = _as1d("acc")
        if acc_arr.size:
            ax_acc = ax_loss.twinx()
            ax_acc.plot(acc_arr, color="green", linestyle="--", alpha=0.7, label="Train Acc")
            ax_acc.set_ylabel("Accuracy (%)", color="green")
            ax_acc.tick_params(axis="y", labelcolor="green")
        ax_loss.set_title(f"{name} Training Loss")
        ax_loss.set_xlabel("Iteration")
        ax_loss.set_ylabel("Cross-entropy loss")
        ax_loss.legend(loc="upper left")

        # --- Col 1: Hessian proxies ---
        ax = axs[row, 1]
        h_arr = _carry_forward_positive(_as1d("hessian"))
        ax.plot(h_arr, color="red", linewidth=2, label="Exact Hessian (H)")
        if "adam" in name.lower():
            prec_arr = _carry_forward_positive(_as1d("prec_h"))
            if prec_arr.size:
                ax.plot(
                    prec_arr,
                    color="purple",
                    linestyle="--",
                    linewidth=2,
                    label=r"Precond. Hessian ($\tilde{H}$)",
                )
        gn_arr = _carry_forward_positive(_as1d("gn"))
        if gn_arr.size:
            ax.plot(
                gn_arr,
                color="brown",
                linestyle="--",
                linewidth=2,
                label=r"Gauss-Newton ($H^{GN}$)",
            )
        vv_arr = _carry_forward_positive(_as1d("hessian_vv", alt="vv"))
        if vv_arr.size:
            ax.plot(
                vv_arr,
                color="magenta",
                linestyle=":",
                linewidth=2,
                label=r"Value Subspace ($H_{VV}$)",
            )
        ax.set_yscale("log")
        ax.set_title(f"{name} Hessian proxies")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("λ_max estimate")
        ax.legend(fontsize="small")

        # --- Col 2: per-layer attention entropy ---
        entropies = np.array(h.get("entropy", []))  # (T, n_layer)
        if entropies.ndim == 2 and entropies.shape[1] > 0:
            n_layers = entropies.shape[1]
            colors_entropy = plt.cm.viridis(np.linspace(0, 1, n_layers))
            for layer_idx in range(n_layers):
                axs[row, 2].plot(
                    entropies[:, layer_idx],
                    color=colors_entropy[layer_idx],
                    label=f"Layer {layer_idx + 1}",
                )
        axs[row, 2].set_title(f"{name} Attention Entropy (Per Layer)")
        axs[row, 2].set_xlabel("Iteration")
        axs[row, 2].set_ylabel("Entropy (nats)")
        axs[row, 2].legend(fontsize="small", ncol=2)

        # --- Cols 3–5: three comparisons vs exact Hessian ---
        h_arr = _carry_forward_positive(_as1d("hessian"))
        prec_arr = _carry_forward_positive(_as1d("prec_h"))
        gn_arr = _carry_forward_positive(_as1d("gn"))
        vv_arr = _carry_forward_positive(_as1d("hessian_vv", alt="vv"))

        _comp_plot(axs[row, 3], h_arr, prec_arr, "H (exact)", "Precond H")
        _comp_plot(axs[row, 4], h_arr, gn_arr, "H (exact)", "Gauss-Newton")
        _comp_plot(axs[row, 5], h_arr, vv_arr, "H (exact)", "Value Subspace H")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
