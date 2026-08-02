"""
Generate attention heatmaps sampled evenly across training for a specific layer/head.

Usage
-----
  # Single pkl file
  python common/plot_heatmap.py path/to/history.pkl

  # Entire project folder — walks <folder>/out/ recursively
  python common/plot_heatmap.py ViT/
  python common/plot_heatmap.py nanochat/

  # Smart path handling: absolute or workspace-prefixed paths can be pasted directly
  python common/plot_heatmap.py /Users/foo/projects/entropy_collapse/ViT/

  # Override output directory
  python common/plot_heatmap.py ViT/ -o /tmp/figures

  # Custom sampling
  python common/plot_heatmap.py ViT/ --freq 8 --layer 2 --head 3

History key expected
--------------------
  history["att_heatmaps"]  — array-like of shape
                             (n_snapshots, n_layers, n_heads, seq_len, seq_len)
                             OR (n_snapshots, seq_len, seq_len) for a single head/layer.
  history["att_heatmap_iters"] — optional 1-D array of training-step indices
                                 corresponding to the n_snapshots axis.

Falls back to history["att_heatmap"] (a single 2-D snapshot) when the
time-series key is absent.
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

def plot_heatmap(
    pkl_path: str,
    out_dir: str | None = None,
    freq: int = 6,
    layer: int = 0,
    head: int = 0,
    fmt: str = "png",
) -> list[plt.Figure]:
    """
    Generate *freq* separate attention heatmap figures evenly sampled across
    training snapshots, one figure per snapshot.

    Args:
        pkl_path:  Path to the ``history.pkl`` file.
        out_dir:   Directory to save the figures in.  Defaults to the directory
                   containing *pkl_path*.  Each figure is saved as
                   ``att_heatmap_L{layer}H{head}_iter{step:07d}.{fmt}``.
        freq:      Number of snapshots (and therefore figures) to generate.
        layer:     Layer index to visualise (0-based).
        head:      Attention-head index to visualise (0-based).
        fmt:       Output image format.

    Returns:
        List of *freq* matplotlib ``Figure`` objects (one per snapshot).
    """
    with open(pkl_path, "rb") as fh:
        history = pickle.load(fh)

    # ------------------------------------------------------------------
    # Load attention snapshots
    # ------------------------------------------------------------------
    maps_raw = history.get("att_heatmaps")
    iters_raw = history.get("att_heatmap_iters")

    if maps_raw is not None:
        maps = np.asarray(maps_raw, dtype=float)
        # Expected shapes:
        #   (T, n_layers, n_heads, S, S)  — full per-layer/head series
        #   (T, S, S)                     — pre-extracted single head/layer
        if maps.ndim not in (3, 5):
            raise ValueError(
                f"history['att_heatmaps'] has unexpected shape {maps.shape}. "
                "Expected (T, S, S) or (T, n_layers, n_heads, S, S)."
            )
        n_snapshots = maps.shape[0]
        iters = (
            np.asarray(iters_raw, dtype=int)
            if iters_raw is not None
            else np.arange(n_snapshots)
        )
    else:
        # Fallback: single final heatmap stored as 2-D array
        single = history.get("att_heatmap")
        if single is None:
            raise KeyError(
                "Neither 'att_heatmaps' nor 'att_heatmap' found in history. "
                "Re-run training with compute_spectrum=True to record attention maps."
            )
        maps = np.asarray(single, dtype=float)[np.newaxis]  # (1, S, S)
        iters = np.array([0])
        n_snapshots = 1
        if freq > 1:
            print(
                f"[plot_heatmap] Only one snapshot available in history "
                f"(att_heatmap). Showing it once instead of freq={freq}."
            )
        freq = 1

    # ------------------------------------------------------------------
    # Sample freq indices evenly from [0, n_snapshots)
    # ------------------------------------------------------------------
    freq = min(freq, n_snapshots)
    if freq == 1:
        sample_idx = np.array([0], dtype=int)
    else:
        sample_idx = np.round(np.linspace(0, n_snapshots - 1, freq)).astype(int)

    # ------------------------------------------------------------------
    # Extract 2-D heatmap for each selected snapshot
    # ------------------------------------------------------------------
    frames: list[tuple[int, np.ndarray]] = []  # (iter_step, 2d_map)
    for si in sample_idx:
        m = maps[si]
        if m.ndim == 4:
            # shape: (n_layers, n_heads, S, S)
            n_layers_avail, n_heads_avail = m.shape[0], m.shape[1]
            if layer >= n_layers_avail or head >= n_heads_avail:
                raise IndexError(
                    f"Requested layer={layer}, head={head} but snapshot has "
                    f"{n_layers_avail} layers and {n_heads_avail} heads."
                )
            m2d = m[layer, head]
        elif m.ndim == 2:
            m2d = m
        else:
            raise ValueError(
                f"Unexpected per-snapshot shape {m.shape}. "
                "Expected (S, S) or (n_layers, n_heads, S, S)."
            )
        frames.append((int(iters[si]), m2d))

    # ------------------------------------------------------------------
    # Build one figure per snapshot
    # ------------------------------------------------------------------
    run_name = os.path.basename(os.path.dirname(pkl_path))
    save_dir = out_dir if out_dir is not None else os.path.dirname(pkl_path)

    figures: list[plt.Figure] = []
    for step, m2d in frames:
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(m2d, aspect="auto", cmap="viridis")
        fig.colorbar(im, ax=ax)
        ax.set_title(
            f"{run_name}  |  layer={layer}  head={head}  |  iter {step}",
            fontsize=10,
        )
        ax.set_xlabel("key pos", fontsize=9)
        ax.set_ylabel("query pos", fontsize=9)
        fig.tight_layout()

        fname = f"att_heatmap_L{layer}H{head}_iter{step:07d}.{fmt}"
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
            "Generate attention heatmaps sampled evenly across training. "
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
        help="Number of heatmaps to generate, evenly spaced across all snapshots (default: 6).",
    )
    parser.add_argument(
        "--layer", type=int, default=0,
        help="Layer index to visualise (0-based, default: 0).",
    )
    parser.add_argument(
        "--head", type=int, default=0,
        help="Attention-head index to visualise (0-based, default: 0).",
    )
    parser.add_argument(
        "--fmt", type=str, default="png",
        choices=["png", "pdf", "svg", "eps"],
        help="Output image format (default: png).",
    )
    args = parser.parse_args()

    pkl_paths = _find_pkl_files(args.path)
    if not pkl_paths:
        print(f"[plot_heatmap] no history.pkl found under '{args.path}'")
        return

    for i, pkl in enumerate(pkl_paths, 1):
        print(f"[plot_heatmap] [{i}/{len(pkl_paths)}] {pkl}")

        if args.out:
            out_dir = args.out
            os.makedirs(out_dir, exist_ok=True)
        else:
            out_dir = os.path.dirname(pkl)

        figs = plot_heatmap(
            pkl_path=pkl,
            out_dir=out_dir,
            freq=args.freq,
            layer=args.layer,
            head=args.head,
            fmt=args.fmt,
        )
        for fig in figs:
            plt.close(fig)
        print(f"           → {len(figs)} heatmap(s) saved to {out_dir}/")


if __name__ == "__main__":
    main()
