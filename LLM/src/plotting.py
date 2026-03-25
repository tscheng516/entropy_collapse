"""
Plotting utilities for the NanoGPT entropy-collapse experiments.

Generic spike co-occurrence and correlation helpers are provided by
``common.spike_analysis`` and re-exported here for convenience.

``plot_training_dynamics`` is NanoGPT-specific and lives in this module:
it renders a multi-panel grid showing training loss, all Hessian proxy
metrics, per-layer attention entropy, and pairwise proxy comparisons.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

# Re-export shared utilities so callers can import from a single place.
from common.spike_analysis import plot_spike_cooccurrence, print_correlations  # noqa: F401


# ======================================================================
# Training dynamics — multi-panel grid
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
                    (kept for API compatibility; no longer used for
                    ceiling-line annotation).
        save_path:  If provided, save the figure to this path.

    Returns:
        The matplotlib ``Figure`` object.
    """
    opt_names = list(histories.keys())
    n_rows = len(opt_names)
    # Expand layout to include three comparison scatter panels
    fig, axs = plt.subplots(n_rows, 6, figsize=(36, 5 * n_rows))
    if n_rows == 1:
        axs = axs[np.newaxis, :]  # ensure 2-D indexing works

    for row, name in enumerate(opt_names):
        h = histories[name]
        color = "blue" if "adam" in name.lower() else "orange"

        # Normalize supported key names and ensure numpy arrays for plotting
        def _as1d(key, alt=None):
            val = h.get(key, h.get(alt) if alt is not None else None)
            if val is None:
                return np.asarray([])
            arr = np.asarray(val)
            if arr.ndim == 0:
                return arr.reshape(-1)
            return arr.ravel()

        # Small helper to draw log-log scatter comparisons and show correlations
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

        # --- Col 0: loss ---
        axs[row, 0].plot(_as1d("loss"), color=color, label=f"{name} Loss")
        axs[row, 0].set_title(f"{name} Training Loss")
        axs[row, 0].set_xlabel("Iteration")
        axs[row, 0].set_ylabel("Cross-entropy loss")
        axs[row, 0].legend()

        # --- Col 1: Hessian proxies ---
        ax = axs[row, 1]
        ax.plot(_as1d("hessian"), color="red", linewidth=2, label="Exact Hessian (H)")
        if "adam" in name.lower():
            prec_arr = _as1d("prec_h")
            if prec_arr.size:
                ax.plot(
                    prec_arr,
                    color="purple",
                    linestyle="--",
                    linewidth=2,
                    label=r"Precond. Hessian ($\tilde{H}$)",
                )
        gn_arr = _as1d("gn")
        if gn_arr.size:
            ax.plot(
                gn_arr,
                color="brown",
                linestyle="--",
                linewidth=2,
                label=r"Gauss-Newton ($H^{GN}$)",
            )

        vv_arr = _as1d("hessian_vv", alt="vv")
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

        # --- Col 3..5: three comparisons vs exact Hessian ---
        h_arr = _as1d("hessian")
        prec_arr = _as1d("prec_h")
        gn_arr = _as1d("gn")
        vv_arr = _as1d("hessian_vv", alt="vv")

        _comp_plot(axs[row, 3], h_arr, prec_arr, "H (exact)", "Precond H")
        _comp_plot(axs[row, 4], h_arr, gn_arr, "H (exact)", "Gauss-Newton")
        _comp_plot(axs[row, 5], h_arr, vv_arr, "H (exact)", "Value Subspace H")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
