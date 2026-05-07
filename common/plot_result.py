"""
Produce a single subplot summary for one or more history.pkl files found under a given folder.

Usage
-----
  # Single file
  python common/plot_results.py path/to/history.pkl

  # Entire project folder — walks <folder>/out/ recursively
  python common/plot_results.py ViT/
  python common/plot_results.py nanochat/
  
  # smart path handling: absolute or workspace-prefixed paths can be pasted directly
  python common/plot_result.py /Users/foo/projects/entropy_collapse/ViT/

  # Override output directory
  python common/plot_results.py ViT/ -o /tmp/figures
  
  Called by base_train.py: output result.{fmt}
  Called by plot_results.py: output results_{layout}.{fmt}
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

# Half-window width in iteration-index space (matches plot_curvature_smoothed_comparison)
_ROLLING_HALF = 2500


# ======================================================================
# Core plotting function
# ======================================================================

def plot_results(
    pkl_path: str,
    save_path: str | None = None,
    skip_intv: bool = True,
    hessian_intv: int = 50,
    entropy_intv: int = 50,
    compute_fd: bool = False,
    vs_H_prec: bool = False,
    layout: str = "22",
    fmt: str = "png",
) -> plt.Figure:
    """
    Layout modes
    ------------
    "12" — 1×2: avg entropy (thumbnail) | curvature subset (H, Prec_H, H_VV, GN)
    "15" — 1×5: loss | entropy | curvature | ref vs proxies | ref vs entropy
    "22" — 2×2: entropy (top-left) | curvature (top-right)
                ref vs entropy (bottom-left) | ref vs proxies (bottom-right)
    "13" — 1×3: entropy | curvature | ref vs proxies

    Args:
        pkl_path:     Path to the ``history.pkl`` file.
        save_path:    If provided, save the figure to this path.
        skip_intv:    If True (default) keep only positive-finite samples and
                      plot them against their true iteration indices.
        hessian_intv: Hessian computation frequency (x-axis label).
        entropy_intv: Entropy computation frequency (x-axis label).
        compute_fd:   If True, include BFGS and FD metrics.

    Returns:
        The matplotlib ``Figure``.
    """
    with open(pkl_path, "rb") as fh:
        history = pickle.load(fh)

    if layout == "12":
        fig, (_a0, _a1) = plt.subplots(1, 2, figsize=(14, 6))
        # avg entropy | curvature subset (H, Prec_H, H_VV, GN)
        axs = [None, _a0, _a1, None, None]
    elif layout == "22":
        fig, ((_sq00, _sq01), (_sq10, _sq11)) = plt.subplots(2, 2, figsize=(14, 12))
        # (0,0)=entropy  (0,1)=curvature
        # (1,0)=ref vs entropy  (1,1)=ref vs proxies
        axs = [None, _sq00, _sq01, _sq11, _sq10]
    elif layout == "13":
        fig, (_a0, _a1, _a2) = plt.subplots(1, 3, figsize=(21, 6))
        # entropy | curvature | ref vs proxies  (ref vs entropy omitted)
        axs = [None, _a0, _a1, _a2, None]
    else:  # "15"
        fig, _axs_raw = plt.subplots(1, 5, figsize=(35, 6))
        axs = list(_axs_raw)

    # ------------------------------------------------------------------
    # Scalar / 1-D extraction helpers
    # ------------------------------------------------------------------
    def _as1d(key: str) -> np.ndarray:
        val = history.get(key)
        if val is None:
            return np.asarray([])
        arr = np.asarray(val)
        return arr.reshape(-1) if arr.ndim == 0 else arr.ravel()

    def _prep(key: str) -> tuple[np.ndarray, np.ndarray]:
        raw = _as1d(key)
        if skip_intv:
            return _extract_positive(raw)
        return raw.copy(), np.arange(len(raw))

    # ------------------------------------------------------------------
    # Extract curvature metrics
    # ------------------------------------------------------------------
    h_arr,      h_idx      = _prep("hessian")
    prec_arr,   prec_idx   = _prep("prec_h")
    gn_arr,     gn_idx     = _prep("gn")
    vv_arr,     vv_idx     = _prep("hessian_vv")
    diag_arr,   diag_idx   = _prep("diag_h")
    fisher_arr, fisher_idx = _prep("fisher")
    if compute_fd:
        kfac_arr,   kfac_idx   = _prep("kfac")
        bfgs_arr, bfgs_idx = _prep("bfgs")
        fd_arr,   fd_idx   = _prep("fd")
    else:
        kfac_arr = kfac_idx = np.array([], dtype=int)
        bfgs_arr = bfgs_idx = np.array([], dtype=int)
        fd_arr   = fd_idx   = np.array([], dtype=int)

    _xlabel = (
        f"Iteration (every {hessian_intv})"
        if skip_intv and hessian_intv > 1 else "Iteration"
    )
    _ent_xlabel = (
        f"Iteration (every {entropy_intv})"
        if skip_intv and entropy_intv > 1 else "Iteration"
    )

    # Shared x-axis upper bound from actual measured iteration indices
    _x_max_cands = [
        idx[-1] for idx in [h_idx, prec_idx, gn_idx, vv_idx,
                             diag_idx, fisher_idx, kfac_idx]
        if len(idx) > 0
    ]
    if compute_fd:
        _x_max_cands += [idx[-1] for idx in [bfgs_idx, fd_idx] if len(idx) > 0]
    _x_max: int | None = int(max(_x_max_cands)) if _x_max_cands else None

    # ------------------------------------------------------------------
    # Extract entropy
    # ------------------------------------------------------------------
    raw_ent = history.get("entropy", [])
    ent_raw = np.asarray(raw_ent, dtype=float) if len(raw_ent) > 0 else np.zeros((0, 0))
    if ent_raw.ndim == 2 and ent_raw.size > 0:
        if skip_intv:
            entropies, ent_idx_arr = _extract_positive_2d(ent_raw)
        else:
            entropies = ent_raw.copy()
            ent_idx_arr = np.arange(len(ent_raw))
    else:
        entropies = np.zeros((0, 0))
        ent_idx_arr = np.array([], dtype=int)

    _ent_x_max = int(ent_idx_arr[-1]) if ent_idx_arr.size > 0 else (_x_max or 0)
    _shared_x_max: int | None = (
        max(_x_max, _ent_x_max) if _x_max is not None else (_ent_x_max or None)
    )

    # ------------------------------------------------------------------
    # Panel 0 — Train + val loss 
    # ------------------------------------------------------------------
    ax_loss = axs[0]  # None when square_plot=True

    def _plot_series(series, color: str, label: str, ax, linestyle: str = "--") -> None:
        """Plot a possibly sparse (list-of-(step, val) tuples) or dense series."""
        if not series or len(series) == 0:
            return
        first = series[0]
        if isinstance(first, (tuple, list)) and len(first) == 2:
            xs = [int(t[0]) for t in series]
            ys = [float(t[1]) for t in series]
            ax.plot(xs, ys, color=color, linestyle=linestyle,
                    linewidth=2, label=label)
        else:
            arr = np.asarray(series, dtype=float).ravel()
            if arr.size > 0:
                ax.plot(arr, color=color, linestyle=linestyle,
                        linewidth=2, label=label)

    if ax_loss is not None:
        loss_arr = _as1d("loss")
        if loss_arr.size > 0:
            ax_loss.plot(loss_arr, color="orange", linewidth=2, label="train loss")
        _plot_series(history.get("val_loss", []), "crimson", "val loss", ax_loss)
        ax_loss.set_title("Loss", fontsize=24)
        ax_loss.set_xlabel("Iteration", fontsize=20)
        ax_loss.set_ylabel("Cross-entropy loss", fontsize=20)
        ax_loss.legend(loc="upper right", fontsize="medium")
        ax_loss.grid(True, alpha=0.3, linestyle="--")

    # ------------------------------------------------------------------
    # Panel 1 — Raw per-layer attention entropy
    # ------------------------------------------------------------------
    ax_ent = axs[1]
    if entropies.ndim == 2 and entropies.shape[1] > 0:
        n_layers = entropies.shape[1]
        if layout == "12":
            # Thumbnail mode: plot average entropy as a single bold line
            ent_avg = entropies.mean(axis=1)
            ax_ent.plot(ent_idx_arr, ent_avg, color="steelblue", linewidth=3,
                        label="Avg. entropy")
        else:
            colors_ent = plt.cm.viridis(np.linspace(0, 1, n_layers))
            for li in range(n_layers):
                ax_ent.plot(ent_idx_arr, entropies[:, li],
                            color=colors_ent[li], label=f"Layer {li + 1}")
    ax_ent.set_title("Avg. Attention Entropy over all layers" if layout == "12" else "Attention Entropy", fontsize=24)
    ax_ent.set_xlabel(_ent_xlabel, fontsize=20)
    # ax_ent.set_ylabel("Entropy (nats)", fontsize=20)
    ax_ent.legend(fontsize="medium", loc="best", ncol=2)
    ax_ent.grid(True, alpha=0.3, linestyle="--")

    # ------------------------------------------------------------------
    # Panel 2 — Raw curvature metric traces (log y)
    # ------------------------------------------------------------------
    ax_curv = axs[2]
    if layout == "12":
        # Thumbnail mode: show only the key subset of proxies
        _metric_specs = [
            (h_arr,    h_idx,    "red",     "-",  "H"),
            (prec_arr, prec_idx, "purple",  "--", r"$\tilde{H}$"),
            (vv_arr,   vv_idx,   "magenta", ":",  r"$H_{VV}$"),
            (gn_arr,   gn_idx,   "brown",   "--", r"$H^{GN}$"),
        ]
    else:
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
                (bfgs_arr, bfgs_idx, "navy", ":",  "BFGS"),
                (fd_arr,   fd_idx,   "cyan", "-.", "FD"),
            ]
    for arr, idx, color, ls, label in _metric_specs:
        if _has_positive_finite(arr):
            ax_curv.plot(idx, arr, color=color, linestyle=ls,
                         linewidth=1.5, label=label)
    ax_curv.set_yscale("log")
    ax_curv.set_title("Spectral Norm of Hessian and Proxies", fontsize=24)
    ax_curv.set_xlabel(_xlabel, fontsize=20)
    # ax_curv.set_ylabel("Spectral Norm (λ_max)", fontsize=20)
    ax_curv.legend(fontsize="medium", loc="best")
    ax_curv.grid(True, alpha=0.3, linestyle="--")

    # ------------------------------------------------------------------
    # Panel 3 — Rolling Spearman: reference (H or Prec_H) vs curvature proxies
    # ------------------------------------------------------------------
    ref_v3   = prec_arr if vs_H_prec else h_arr
    ref_i3   = prec_idx if vs_H_prec else h_idx
    ref_lbl3 = r"$\tilde{H}$" if vs_H_prec else "H"

    if vs_H_prec:
        _proxies3 = [
            (h_arr,      h_idx,      "red",       "H"),
            (gn_arr,     gn_idx,     "brown",     r"$H^{GN}$"),
            (vv_arr,     vv_idx,     "magenta",   r"$H_{VV}$"),
            (diag_arr,   diag_idx,   "teal",      "Diag H"),
            (fisher_arr, fisher_idx, "olive",     "Fisher"),
            (kfac_arr,   kfac_idx,   "darkgreen", "K-FAC"),
        ]
    else:
        _proxies3 = [
            (prec_arr,   prec_idx,   "purple",    r"$\tilde{H}$"),
            (gn_arr,     gn_idx,     "brown",     r"$H^{GN}$"),
            (vv_arr,     vv_idx,     "magenta",   r"$H_{VV}$"),
            (diag_arr,   diag_idx,   "teal",      "Diag H"),
            (fisher_arr, fisher_idx, "olive",     "Fisher"),
            (kfac_arr,   kfac_idx,   "darkgreen", "K-FAC"),
        ]
    if compute_fd:
        _proxies3 += [
            (bfgs_arr, bfgs_idx, "navy", "BFGS"),
            (fd_arr,   fd_idx,   "cyan", "FD"),
        ]

    ax_sp_ref = axs[3]
    for p_v, p_i, color_r, label_r in _proxies3:
        iters_r, sp_r, sp_w = _rolling_corr(ref_v3, ref_i3, p_v, p_i)
        if iters_r.size > 0:
            ax_sp_ref.plot(iters_r, sp_r, color=color_r, linewidth=1.5, label=label_r)
            if not np.isnan(sp_w):
                ax_sp_ref.axhline(sp_w, color=color_r, linewidth=0.8,
                                  linestyle="--", alpha=0.45)
    ax_sp_ref.set_title(f"Rolling Spearman ρ — {ref_lbl3} vs Proxy", fontsize=24)
    ax_sp_ref.set_xlabel(_xlabel, fontsize=20)
    # ax_sp_ref.set_ylabel("Spearman ρ", fontsize=20)
    # ax_sp_ref.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax_sp_ref.legend(fontsize="medium", loc="best")
    ax_sp_ref.grid(True, alpha=0.3, linestyle="--")

    # ------------------------------------------------------------------
    # Panel 4 — Rolling Spearman: reference metric vs per-layer attention entropy
    # Each layer's entropy is interpolated onto the curvature reference's index
    # grid so both series share the same iteration positions for the rolling windows.
    # ------------------------------------------------------------------
    ax_sp_ent = axs[4]
    if ax_sp_ent is not None and entropies.ndim == 2 and entropies.shape[1] > 0 and ref_v3.size >= 2 and ent_idx_arr.size >= 2:
        n_layers = entropies.shape[1]
        colors_ent = plt.cm.viridis(np.linspace(0, 1, n_layers))
        for li in range(n_layers):
            layer_ent = entropies[:, li]
            ent_on_ref_grid = np.interp(
                ref_i3.astype(float),
                ent_idx_arr.astype(float),
                layer_ent,
            )
            iters_e, sp_e, sp_w_e = _rolling_corr(
                ref_v3, ref_i3, ent_on_ref_grid, ref_i3
            )
            if iters_e.size > 0:
                ax_sp_ent.plot(iters_e, sp_e, color=colors_ent[li],
                               linewidth=1.5, label=f"Layer {li + 1}")
                if not np.isnan(sp_w_e):
                    ax_sp_ent.axhline(sp_w_e, color=colors_ent[li], linewidth=0.8,
                                      linestyle="--", alpha=0.45)
    if ax_sp_ent is not None:
        ax_sp_ent.set_title(
            f"Rolling Spearman ρ — {ref_lbl3} vs Entropy", fontsize=24
        )
        ax_sp_ent.set_xlabel(_xlabel, fontsize=20)
        ax_sp_ent.legend(fontsize="large", loc="best", ncol=2)
        ax_sp_ent.grid(True, alpha=0.3, linestyle="--")

    # ------------------------------------------------------------------
    # Apply shared x limits to all panels
    # ------------------------------------------------------------------
    if _shared_x_max is not None:
        for ax in axs:
            if ax is not None:
                ax.set_xlim(0, _shared_x_max)

    # run_name = os.path.basename(os.path.dirname(pkl_path))
    # fig.suptitle(f"Raw Analysis — {run_name}", fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, format=fmt, dpi=150, bbox_inches="tight")

    return fig


# ======================================================================
# extraction helper
# ======================================================================

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
# rolling correlation helper
# ======================================================================
def _rolling_corr(
    ref_v: np.ndarray, ref_i: np.ndarray,
    p_v: np.ndarray,   p_i: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Rolling Spearman ρ of ref vs proxy over ±_ROLLING_HALF windows.
    Returns (iters, spearman_values, whole_run_spearman).
    """
    if not _has_positive_finite(ref_v) or not _has_positive_finite(p_v):
        return np.array([]), np.array([]), float("nan")
    n = min(ref_v.size, p_v.size)
    if n < 3:
        return np.array([]), np.array([]), float("nan")
    ref_u = ref_v[:n].astype(float)
    p_u   = p_v[:n].astype(float)
    idx_u = ref_i[:n]
    mask_all = np.isfinite(ref_u) & np.isfinite(p_u)
    sp_whole = (
        sp_stats.spearmanr(ref_u[mask_all], p_u[mask_all])[0]
        if mask_all.sum() >= 3 else float("nan")
    )
    if idx_u.size == 0 or idx_u[-1] <= 2 * _ROLLING_HALF:
        return np.array([]), np.array([]), sp_whole
    roll_iters, roll_sp = [], []
    for i in range(n):
        center = idx_u[i]
        if center < _ROLLING_HALF or center > idx_u[-1] - _ROLLING_HALF:
            continue
        lo = int(np.searchsorted(idx_u, center - _ROLLING_HALF))
        hi = int(np.searchsorted(idx_u, center + _ROLLING_HALF, side="right"))
        if hi - lo < 3:
            continue
        r_w = ref_u[lo:hi]
        p_w = p_u[lo:hi]
        mask = np.isfinite(r_w) & np.isfinite(p_w)
        if mask.sum() < 3:
            continue
        roll_iters.append(center)
        roll_sp.append(sp_stats.spearmanr(r_w[mask], p_w[mask])[0])
    return (
        np.asarray(roll_iters, dtype=float),
        np.asarray(roll_sp,    dtype=float),
        sp_whole,
    )

# ======================================================================
# Batch helper
# ======================================================================

def _find_pkl_files(path: str) -> list[str]:
    """
    Return a sorted list of ``history.pkl`` paths to process.

    * If *path* is a ``.pkl`` file, return ``[path]``.
    * If *path* is a directory, walk ``<path>/out/`` recursively.
    * If no ``out/`` sub-dir exists, walk *path* itself.

    If the path contains ``entropy_collapse/``, everything up to and including
    that token is stripped so users can paste absolute or workspace-prefixed
    paths directly (e.g. from VS Code or a terminal copy-paste).
    """
    _MARKER = "entropy_collapse/"
    if _MARKER in path:
        path = os.path.join(_PROJECT_ROOT, path.split(_MARKER, 1)[1])
    path = os.path.abspath(path)
    if os.path.isfile(path):
        return [path]
    search_root = os.path.join(path, "out")
    if not os.path.isdir(search_root):
        search_root = path
    pkls: list[str] = []
    for root, _dirs, files in os.walk(search_root):
        if "history.pkl" in files:
            pkls.append(os.path.join(root, "history.pkl"))
    return sorted(pkls)


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a subplot summary for each history.pkl found "
            "under the given path.  Accepts a single .pkl file or a project "
            "folder whose out/ sub-tree is scanned recursively."
        )
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to a history.pkl file or a project folder (e.g. ViT/).",
    )
    parser.add_argument(
        "-o", "--out",
        type=str, default=None,
        help=(
            "Output file path when processing a single pkl, or output directory "
            "when processing multiple pkls.  Defaults to saving plot_results.png "
            "next to each history.pkl."
        ),
    )
    parser.add_argument(
        "--hessian-intv", type=int, default=50,
        help="Hessian computation frequency (x-axis label, default 50).",
    )
    parser.add_argument(
        "--entropy-intv", type=int, default=50,
        help="Entropy computation frequency (x-axis label, default 50).",
    )
    parser.add_argument(
        "--no-skip-intv", action="store_true",
        help="Use legacy dense / carry-forward mode instead of interval-skipping.",
    )
    parser.add_argument(
        "--compute-fd", action="store_true",
        help="Include k-Fac, BFGS and FD metrics (only if computed during training).",
    )
    parser.add_argument(
        "--vs-H-prec", action="store_true",
        help="Use Prec_H as reference in correlation (default: H).",
    )
    parser.add_argument(
        "--layout", type=str, default="15",
        choices=["12", "15", "22", "13"],
        help=(
            "Figure layout: "
            "'12' = 1×2 thumbnail (avg-entropy|curvature-subset), "
            "'15' = 1×5 detailed (loss|entropy|curvature|ref-vs-proxy|ref-vs-entropy), "
            "'22' = 2×2 square (entropy|curvature||ref-vs-entropy|ref-vs-proxy), "
            "'13' = 1×3 compact (entropy|curvature|ref-vs-proxy)."
        ),
    )
    parser.add_argument(
        "--fmt", type=str, default="png",
        choices=["png", "pdf", "svg", "eps"],
        help="Output image format (default: png).",
    )
    args = parser.parse_args()

    pkl_paths = _find_pkl_files(args.path)
    if not pkl_paths:
        print(f"[plot_results] no history.pkl found under '{args.path}'")
        return

    single = len(pkl_paths) == 1

    for i, pkl in enumerate(pkl_paths, 1):
        print(f"[plot_results] [{i}/{len(pkl_paths)}] {pkl}")

        _suffix = f"results_{args.layout}.{args.fmt}"
        if single and args.out:
            save_path = args.out
        elif args.out:
            os.makedirs(args.out, exist_ok=True)
            run_id = os.path.basename(os.path.dirname(pkl))
            save_path = os.path.join(args.out, f"{run_id}_{_suffix}")
        else:
            save_path = os.path.join(os.path.dirname(pkl), _suffix)

        fig = plot_results(
            pkl_path=pkl,
            save_path=save_path,
            skip_intv=not args.no_skip_intv,
            hessian_intv=args.hessian_intv,
            entropy_intv=args.entropy_intv,
            compute_fd=args.compute_fd,
            vs_H_prec=args.vs_H_prec,
            layout=args.layout,
            fmt=args.fmt,
        )
        plt.close(fig)
        print(f"           → saved to {save_path}")


if __name__ == "__main__":
    main()
