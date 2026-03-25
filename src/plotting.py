"""
Plotting utilities for the NanoGPT entropy-collapse experiments.

Two sets of plots are provided:

1. ``plot_training_dynamics``
   A 2×3 grid showing, for two optimisers side-by-side:
     - Training loss
     - All Hessian proxy metrics (H, H_tilde, H_VV, H_GN) on a log scale
     - Per-layer attention entropy over training

2. ``plot_spike_cooccurrence``
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
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

# Consistency factor to convert MAD into an equivalent standard deviation
# for a normally-distributed variable: sigma ≈ 1.4826 × MAD.
# Reference: Rousseeuw & Croux (1993), "Alternatives to the Median Absolute
# Deviation", JASA 88(424):1273-1283.
_MAD_TO_STD = 1.4826


# ======================================================================
# 1. Training dynamics — 2×3 grid
# ======================================================================


def plot_training_dynamics(
    histories: dict[str, dict],
    lrs: dict[str, float],
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot training dynamics for two training runs (e.g. AdamW vs SGD).

    Args:
        histories:  Dict mapping optimizer name (e.g. ``"AdamW"``) to a
                    history dict with keys:
                    ``loss``, ``hessian``, ``prec_h``, ``hessian_vv``,
                    ``gn``, ``fd``, ``entropy``.
                    ``entropy`` values are lists of per-layer floats.
        lrs:        Dict mapping optimizer name to the peak LR used
                    (for annotating the EoS ceiling lines).
        save_path:  If provided, save the figure to this path.

    Returns:
        The matplotlib ``Figure`` object.
    """
    opt_names = list(histories.keys())
    n_rows = len(opt_names)
    fig, axs = plt.subplots(n_rows, 3, figsize=(24, 5 * n_rows))
    if n_rows == 1:
        axs = axs[np.newaxis, :]  # ensure 2-D indexing works

    for row, name in enumerate(opt_names):
        h = histories[name]
        lr = lrs.get(name, 1e-3)
        color = "blue" if "adam" in name.lower() else "orange"

        # --- Col 0: loss ---
        axs[row, 0].plot(h["loss"], color=color, label=f"{name} Loss")
        axs[row, 0].set_title(f"{name} Training Loss")
        axs[row, 0].set_xlabel("Iteration")
        axs[row, 0].set_ylabel("Cross-entropy loss")
        axs[row, 0].legend()

        # --- Col 1: Hessian proxies ---
        ax = axs[row, 1]
        ax.plot(h["hessian"], color="red", linewidth=2, label="Exact Hessian (H)")
        if "adam" in name.lower():
            ax.plot(
                h["prec_h"],
                color="purple",
                linestyle="--",
                linewidth=2,
                label=r"Precond. Hessian ($\tilde{H}$)",
            )
            ceiling = 38.0 / lr
            ax.axhline(
                y=ceiling,
                color="gray",
                linestyle=":",
                label=f"AEoS ceiling (38/η = {ceiling:.1f})",
            )
        else:
            ceiling = 2.0 / lr
            ax.axhline(
                y=ceiling,
                color="gray",
                linestyle=":",
                label=f"EoS ceiling (2/η = {ceiling:.1f})",
            )
        ax.plot(
            h["gn"],
            color="brown",
            linestyle="--",
            linewidth=2,
            label=r"Gauss-Newton ($H^{GN}$)",
        )
        ax.plot(
            h["hessian_vv"],
            color="magenta",
            linestyle=":",
            linewidth=2,
            label=r"Value Subspace ($H_{VV}$)",
        )
        ax.set_yscale("log")
        ax.set_title(f"{name} Hessian Proxies")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("λ_max estimate")
        ax.legend(fontsize="small")

        # --- Col 2: per-layer attention entropy ---
        entropies = np.array(h["entropy"])  # (T, n_layer)
        if entropies.ndim == 2:
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

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ======================================================================
# 2. Spike co-occurrence plot
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


# ======================================================================
# 3. Correlation summary
# ======================================================================


def print_correlations(history: dict, name: str, sample_every: int = 1) -> None:
    """
    Print Spearman and Pearson correlations between all curvature metric
    pairs recorded in *history*.

    Args:
        history:      History dict as returned by the training loop.
        name:         Experiment name (printed as a header).
        sample_every: Subsample stride (e.g. 3 if metrics were recorded
                      every 3rd iteration).
    """
    from scipy import stats as sp_stats

    h = np.array(history["hessian"][::sample_every])
    prec_h = np.array(history["prec_h"][::sample_every])
    gn = np.array(history["gn"][::sample_every])
    vv = np.array(history["hessian_vv"][::sample_every])

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
