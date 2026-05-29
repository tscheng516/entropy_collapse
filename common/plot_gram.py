"""
Generate centered Gram-matrix heatmaps (head=0) sampled evenly across
Hessian evaluation points, one figure per snapshot.

Usage
-----
  # Single pkl file
  python common/plot_gram.py path/to/history.pkl

  # Entire project folder — walks <folder>/out/ recursively
  python common/plot_gram.py ViT/
  python common/plot_gram.py nanochat/

  # Smart path handling: absolute or workspace-prefixed paths can be pasted directly
  python common/plot_gram.py /Users/foo/projects/entropy_collapse/ViT/

  # Override output directory / sampling
  python common/plot_gram.py ViT/ -o /tmp/figures --freq 8 --layer 2

History keys expected
---------------------
  history["gram_hessian"]       — list of snapshots; each snapshot is a list
                                  of (B, B) float32 np.ndarray, one per layer
                                  (centred Gram matrix of attention maps for
                                  head=0 computed on the Hessian batch).
  history["gram_hessian_iters"] — optional 1-D array of training-step indices
                                  corresponding to the snapshot axis.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

# ---------------------------------------------------------------------------
# Path bootstrap (mirrors plot_result.py)
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
# Core plotting function
# ======================================================================

def plot_gram(
    pkl_path: str,
    out_dir: str | None = None,
    freq: int = 6,
    layer: int = 0,
    fmt: str = "png",
) -> list[plt.Figure]:
    """
    Generate *freq* separate Gram-matrix heatmap figures evenly sampled
    across Hessian evaluation snapshots, one figure per snapshot.

    Each figure shows the ``(B, B)`` centred Gram matrix of the attention-
    weight maps (head=0) on the Hessian batch for the specified *layer*.

    Args:
        pkl_path:  Path to the ``history.pkl`` file.
        out_dir:   Directory to save the figures in.  Defaults to the directory
                   containing *pkl_path*.  Each figure is saved as
                   ``gram_L{layer}_iter{step:07d}.{fmt}``.
        freq:      Number of snapshots (and therefore figures) to generate.
        layer:     Layer index to visualise (0-based).
        fmt:       Output image format.

    Returns:
        List of *freq* matplotlib ``Figure`` objects (one per snapshot).
    """
    with open(pkl_path, "rb") as fh:
        history = pickle.load(fh)

    # ------------------------------------------------------------------
    # Load Gram snapshots
    # ------------------------------------------------------------------
    grams_raw = history.get("gram_hessian")
    iters_raw = history.get("gram_hessian_iters")

    if not grams_raw:
        raise KeyError(
            "'gram_hessian' not found (or empty) in history. "
            "Re-run training with att_sim=True to record Gram matrices."
        )

    n_snapshots = len(grams_raw)
    iters = (
        np.asarray(iters_raw, dtype=int)
        if iters_raw is not None
        else np.arange(n_snapshots)
    )

    # Validate layer index against first snapshot
    n_layers_avail = len(grams_raw[0])
    if layer >= n_layers_avail:
        raise IndexError(
            f"Requested layer={layer} but only {n_layers_avail} layer(s) "
            "stored in gram_hessian."
        )

    # ------------------------------------------------------------------
    # Sample freq indices evenly from [0, n_snapshots)
    # ------------------------------------------------------------------
    freq = min(freq, n_snapshots)
    if freq == 1:
        sample_idx = np.array([0], dtype=int)
    else:
        sample_idx = np.round(np.linspace(0, n_snapshots - 1, freq)).astype(int)

    # ------------------------------------------------------------------
    # Build one figure per snapshot
    # ------------------------------------------------------------------
    run_name = os.path.basename(os.path.dirname(pkl_path))
    save_dir = out_dir if out_dir is not None else os.path.dirname(pkl_path)

    figures: list[plt.Figure] = []
    for si in sample_idx:
        step = int(iters[si])
        K = np.asarray(grams_raw[si][layer], dtype=float)

        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(K, aspect="auto", cmap="RdBu_r",
                       vmin=-np.abs(K).max(), vmax=np.abs(K).max())
        fig.colorbar(im, ax=ax)
        ax.set_title(
            f"{run_name}  |  layer={layer}  head=0  |  iter {step}",
            fontsize=10,
        )
        ax.set_xlabel("sample j", fontsize=9)
        ax.set_ylabel("sample i", fontsize=9)
        fig.tight_layout()

        fname = f"gram_L{layer}_iter{step:07d}.{fmt}"
        fig.savefig(
            os.path.join(save_dir, fname),
            format=fmt, dpi=150, bbox_inches="tight",
        )
        figures.append(fig)

    return figures


# ======================================================================
# Address resolution helper (reused from plot_result.py)
# ======================================================================

def _find_pkl_files(path: str) -> list[str]:
    """
    Return a sorted list of ``history.pkl`` paths to process.

    * If *path* is a ``.pkl`` file, return ``[path]``.
    * If *path* is a directory, walk ``<path>/out/`` recursively.
    * If no ``out/`` sub-dir exists, walk *path* itself.

    If the path contains ``entropy_collapse/``, everything up to and including
    that token is stripped so users can paste absolute or workspace-prefixed
    paths directly.
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
            "Generate centred Gram-matrix heatmaps (head=0) sampled evenly "
            "across Hessian evaluation points. "
            "Accepts a single .pkl file or a project folder whose out/ "
            "sub-tree is scanned recursively."
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
            "Output directory for heatmap images. "
            "Defaults to the directory containing each history.pkl."
        ),
    )
    parser.add_argument(
        "--freq", type=int, default=6,
        help="Number of snapshots to generate, evenly spaced across all Hessian eval points (default: 6).",
    )
    parser.add_argument(
        "--layer", type=int, default=0,
        help="Layer index to visualise (0-based, default: 0).",
    )
    parser.add_argument(
        "--fmt", type=str, default="png",
        choices=["png", "pdf", "svg", "eps"],
        help="Output image format (default: png).",
    )
    args = parser.parse_args()

    pkl_paths = _find_pkl_files(args.path)
    if not pkl_paths:
        print(f"[plot_gram] no history.pkl found under '{args.path}'")
        return

    for i, pkl in enumerate(pkl_paths, 1):
        print(f"[plot_gram] [{i}/{len(pkl_paths)}] {pkl}")

        if args.out:
            out_dir = args.out
            os.makedirs(out_dir, exist_ok=True)
        else:
            out_dir = os.path.dirname(pkl)

        figs = plot_gram(
            pkl_path=pkl,
            out_dir=out_dir,
            freq=args.freq,
            layer=args.layer,
            fmt=args.fmt,
        )
        for fig in figs:
            plt.close(fig)
        print(f"           → {len(figs)} Gram heatmap(s) saved to {out_dir}/")


if __name__ == "__main__":
    main()
