"""
thumbnail.py — Produce a compact 1×2 training summary figure from a
``history.pkl`` file.

Layout
------
Left  — Smoothed average attention entropy across all layers.
Right — Smoothed curvature metrics: H, Prec_H (Ĥ), H_VV, and GN (H^GN).

Styling mirrors ``fig_simple`` in ``plot_curvature_smoothed_comparison``:
no axis ticks or grid lines, bold titles, legend shown.

Usage
-----
Standalone::

    python common/thumbnail.py path/to/history.pkl
    python common/thumbnail.py path/to/history.pkl -o out/thumbnail.pdf

As a library::

    from common.thumbnail import plot_thumbnail
    fig = plot_thumbnail("path/to/history.pkl", save_path="thumb.pdf")
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.helpers import smooth_log_trend


# ---------------------------------------------------------------------------
# Helpers (minimal, self-contained copies)
# ---------------------------------------------------------------------------

def _extract_positive(series: np.ndarray | list) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(series, dtype=float).ravel()
    if arr.size == 0:
        return np.array([]), np.array([], dtype=int)
    valid = np.isfinite(arr) & (arr > 0)
    return arr[valid], np.where(valid)[0]


def _extract_positive_2d(matrix: np.ndarray | list) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return arr, np.arange(max(len(arr), 0), dtype=int)
    valid_rows = np.any(np.isfinite(arr) & (arr > 0), axis=1)
    return arr[valid_rows], np.where(valid_rows)[0]


def _has_positive_finite(arr: np.ndarray | list) -> bool:
    a = np.asarray(arr, dtype=float).ravel()
    return bool(a.size > 0 and np.any(np.isfinite(a) & (a > 0)))


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def plot_thumbnail(
    pkl_path: str,
    lam: float = 10.0,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Load ``history.pkl`` and produce a 1×2 thumbnail figure.

    Args:
        pkl_path:   Path to a ``history.pkl`` produced by any base_train script.
        lam:        Whittaker–Henderson smoothing strength.
        save_path:  If provided, save the figure to this path.

    Returns:
        The matplotlib ``Figure``.
    """
    with open(pkl_path, "rb") as f:
        history = pickle.load(f)

    fig, (ax_ent, ax_curv) = plt.subplots(1, 2, figsize=(14, 5))

    # ------------------------------------------------------------------ #
    # Left panel — smoothed average attention entropy
    # ------------------------------------------------------------------ #
    raw_ent = history.get("entropy", [])
    if raw_ent and len(raw_ent) > 0:
        ent_arr = np.asarray(raw_ent, dtype=float)
        if ent_arr.ndim == 2 and ent_arr.size > 0:
            entropies, ent_idx = _extract_positive_2d(ent_arr)
            if entropies.shape[0] >= 3:
                ent_avg = entropies.mean(axis=1)
                trend_avg, _, _ = smooth_log_trend(ent_avg, lam=lam, use_abs=True)
                ax_ent.plot(ent_idx, trend_avg, color="steelblue", linewidth=3,
                            label="Avg. entropy")
                ax_ent.legend(fontsize="small", loc="best")

    # ------------------------------------------------------------------ #
    # Right panel — smoothed curvature metrics: H, Prec_H, H_VV, GN
    # ------------------------------------------------------------------ #
    _metric_specs = [
        ("hessian",    "red",     "-",   "H"),
        ("prec_h",     "purple",  "--",  r"$\tilde{H}$"),
        ("hessian_vv", "magenta", ":",   r"$H_{VV}$"),
        ("gn",         "brown",   "--",  r"$H^{GN}$"),
    ]

    all_idx: list[np.ndarray] = []
    for key, color, ls, label in _metric_specs:
        raw = np.asarray(history.get(key, []), dtype=float).ravel()
        arr, idx = _extract_positive(raw)
        if not _has_positive_finite(arr) or arr.size < 3:
            continue
        trend, _, _ = smooth_log_trend(arr, lam=lam, use_abs=True)
        ax_curv.plot(idx, trend, color=color, linestyle=ls, linewidth=3, label=label)
        all_idx.append(idx)

    ax_curv.set_yscale("log")
    ax_curv.minorticks_off()
    ax_curv.legend(fontsize="small", loc="best")

    # ------------------------------------------------------------------ #
    # Shared x limit
    # ------------------------------------------------------------------ #
    x_max_candidates: list[int] = []
    if 'ent_idx' in dir() and ent_idx.size > 0:   # type: ignore[name-defined]
        x_max_candidates.append(int(ent_idx[-1]))  # type: ignore[name-defined]
    x_max_candidates += [int(idx[-1]) for idx in all_idx if idx.size > 0]
    if x_max_candidates:
        x_max = max(x_max_candidates)
        ax_ent.set_xlim(0, x_max)
        ax_curv.set_xlim(0, x_max)

    # ------------------------------------------------------------------ #
    # Minimal styling (no ticks, no grid)
    # ------------------------------------------------------------------ #
    for ax, title in [(ax_ent, "Avg. Attention Entropy"),
                      (ax_curv, "Curvature Metrics")]:
        ax.set_title(title, fontsize=16, fontweight="bold")
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for sp in ax.spines.values():
            sp.set_visible(True)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Produce a 1×2 training summary thumbnail from history.pkl."
    )
    parser.add_argument("pkl_path", help="Path to history.pkl")
    parser.add_argument(
        "-o", "--out", default=None,
        help="Output file path (e.g. thumbnail.pdf). "
             "Defaults to thumbnail.pdf next to the pkl.",
    )
    parser.add_argument(
        "--lam", type=float, default=10.0,
        help="Smoothing strength (default: 10)",
    )
    args = parser.parse_args()

    save_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.pkl_path)), "thumbnail.pdf"
    )
    fig = plot_thumbnail(args.pkl_path, lam=args.lam, save_path=save_path)
    plt.close(fig)
    print(f"[thumbnail] saved → {save_path}")


if __name__ == "__main__":
    main()
