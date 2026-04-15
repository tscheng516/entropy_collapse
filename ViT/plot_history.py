"""
plot_history.py — Re-run all post-training plots and analysis from a saved
history.pkl file.

This is useful for re-analysing previous runs with the new ``skip_intv``
mode (which strips zero-placeholder intervals instead of carry-forwarding).

Usage
-----
Basic (uses the directory containing the .pkl as output)::

    python plot_history.py path/to/history.pkl

Specify an output directory::

    python plot_history.py path/to/history.pkl -o reanalysis_output/

Override hessian/entropy frequencies (defaults: 500)::

    python plot_history.py path/to/history.pkl --hessian_freq 100 --entropy_freq 100

Use legacy carry-forward mode::

    python plot_history.py path/to/history.pkl --no-skip-intv
"""

from __future__ import annotations

import argparse
import io
import os
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup — same as base_train.py so ``src`` sub-package resolves.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from src.plotting import (  # noqa: E402
    plot_training_dynamics,
    plot_all_spike_cooccurrences,
    plot_curvature_smoothed_comparison,
    print_correlations,
)


class _Tee:
    """Write to both a stream and a string buffer simultaneously."""

    def __init__(self, stream):
        self._stream = stream
        self._buf = io.StringIO()

    def write(self, s):
        self._stream.write(s)
        self._buf.write(s)

    def flush(self):
        self._stream.flush()

    def getvalue(self) -> str:
        return self._buf.getvalue()


def plot_history(
    pkl_path: str,
    out_dir: str | None = None,
    hessian_freq: int = 500,
    entropy_freq: int = 500,
    skip_intv: bool = True,
    lam: float = 100.0,
) -> None:
    """
    Load a ``history.pkl`` and reproduce every post-training plot and
    analysis from ``base_train.py`` section 11.

    All printed analysis is also saved to ``analysis.txt`` inside the
    output directory.

    Args:
        pkl_path:     Path to a ``history.pkl`` file.
        out_dir:      Directory to write plots and analysis text.
                      Defaults to the directory containing *pkl_path*.
        hessian_freq: Hessian computation frequency used during training.
        entropy_freq: Entropy computation frequency used during training.
        skip_intv:    If True (default), use the new interval-skipping mode.
                      If False, use legacy carry-forward step-function.
        lam:          Smoothing strength for Whittaker–Henderson smoother.
    """
    # --- Load history ---
    with open(pkl_path, "rb") as f:
        history = pickle.load(f)
    print(f"[plot_history] loaded {pkl_path}  ({len(history.get('loss', []))} iterations)")

    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(pkl_path))
    os.makedirs(out_dir, exist_ok=True)

    # --- Tee stdout so we capture all printed analysis ---
    tee = _Tee(sys.stdout)
    old_stdout = sys.stdout
    sys.stdout = tee

    try:
        # --- Correlations ---
        print_correlations(
            history, "Run", lam=lam, include_smooth=True,
            skip_intv=skip_intv, hessian_freq=hessian_freq,
        )

        # --- Training dynamics (loss + entropy) ---
        lr_value = history.get("lr", [0.0])
        lr_peak = max(lr_value) if lr_value else 0.0
        fig = plot_training_dynamics(
            histories={"Run": history},
            lrs={"Run": lr_peak},
            save_path=os.path.join(out_dir, "training_dynamics.png"),
            skip_intv=skip_intv,
            entropy_freq=entropy_freq,
        )
        plt.close(fig)
        print(f"[plot] training dynamics → {os.path.join(out_dir, 'training_dynamics.png')}")

        # --- Smoothed curvature comparison ---
        fig_smooth = plot_curvature_smoothed_comparison(
            history,
            lam=lam,
            save_path=os.path.join(out_dir, "curvature_smoothed_comparison.png"),
            skip_intv=skip_intv,
            hessian_freq=hessian_freq,
        )
        plt.close(fig_smooth)
        print(f"[plot] smoothed curvature comparison → {os.path.join(out_dir, 'curvature_smoothed_comparison.png')}")

        # --- Spike co-occurrence ---
        proxy_label = {
            "prec_h": "Prec_H",
            "hessian_vv": "H_VV",
            "gn": "GN",
            "fd": "FD",
            "diag_h": "Diag_H",
            "fisher": "Fisher",
            "bfgs": "BFGS",
            "kfac": "KFAC",
        }
        for z in (1.5, 2):
            spike_figs, spike_results = plot_all_spike_cooccurrences(
                history,
                window=15,
                z_score=z,
                log_scale=True,
                save_dir=out_dir,
                skip_intv=skip_intv,
                hessian_freq=hessian_freq,
            )
            for fig_spike in spike_figs.values():
                plt.close(fig_spike)

            for key, res in spike_results.items():
                label = proxy_label.get(key, key)
                print(
                    f"[plot] z={z} spike co-occurrence: "
                    f"P({label} spike | H spike) = {res['P(Y_spike | X_spike)']:.3f}"
                )

        print(f"\n[plot_history] all outputs saved to {out_dir}")

    finally:
        sys.stdout = old_stdout

    # --- Write analysis text ---
    txt_path = os.path.join(out_dir, "analysis.txt")
    with open(txt_path, "w") as f:
        f.write(tee.getvalue())
    print(f"[plot_history] analysis log → {txt_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Re-run post-training plots and analysis from a history.pkl file."
    )
    parser.add_argument("pkl_path", type=str, help="Path to history.pkl")
    parser.add_argument(
        "-o", "--out-dir", type=str, default=None,
        help="Output directory (default: same directory as pkl_path)",
    )
    parser.add_argument(
        "--hessian_freq", type=int, default=500,
        help="Hessian computation frequency used during training (default: 500)",
    )
    parser.add_argument(
        "--entropy_freq", type=int, default=500,
        help="Entropy computation frequency used during training (default: 500)",
    )
    parser.add_argument(
        "--no-skip-intv", action="store_true",
        help="Use legacy carry-forward mode instead of interval-skipping",
    )
    parser.add_argument(
        "--lam", type=float, default=100.0,
        help="Smoothing strength for Whittaker–Henderson smoother (default: 100)",
    )
    args = parser.parse_args()

    plot_history(
        pkl_path=args.pkl_path,
        out_dir=args.out_dir,
        hessian_freq=args.hessian_freq,
        entropy_freq=args.entropy_freq,
        skip_intv=not args.no_skip_intv,
        lam=args.lam,
    )


if __name__ == "__main__":
    main()
