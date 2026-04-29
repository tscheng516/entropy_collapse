"""
plot_history.py — Re-run all post-training plots and analysis from a saved
history.pkl file.

Usage
-----
Basic (output goes next to the .pkl file)::

    python plot_history.py path/to/history.pkl

Custom output directory::

    python plot_history.py path/to/history.pkl -o reanalysis/

Override frequencies (default 500)::

    python plot_history.py out/cifar100_vitb16/.../history.pkl \\
        --hessian_freq 100 --entropy_freq 100

Legacy carry-forward mode::

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
# Path setup — mirror base_train.py so ``src`` sub-package resolves.
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
    """Write to both stdout and an in-memory buffer simultaneously."""

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


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a simple Markdown table string."""
    col_w = [max(len(h), max((len(str(r[i])) for r in rows), default=0))
             for i, h in enumerate(headers)]
    sep = "| " + " | ".join("-" * w for w in col_w) + " |"
    header = "| " + " | ".join(h.ljust(col_w[i]) for i, h in enumerate(headers)) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(row[i]).ljust(col_w[i]) for i in range(len(headers))) + " |")
    return "\n".join(lines)


def _corr_rows(corr_dict: dict) -> list[list[str]]:
    rows = []
    for label, vals in corr_dict.items():
        sp = f"{vals['spearman']:.4f}" if isinstance(vals, dict) else "n/a"
        pe = f"{vals['pearson']:.4f}"  if isinstance(vals, dict) else "n/a"
        rows.append([label, sp, pe])
    return rows


def _write_analysis_md(
    out_dir: str,
    run_label: str,
    pkl_path: str,
    corr_results: dict,
    spike_all: dict[float, dict],   # z → {proxy: result_dict}
    lam: float,
    hessian_freq: int,
    entropy_freq: int,
    train_config: dict | None = None,
) -> None:
    """Write a Markdown analysis report to analysis.md."""
    lines = [
        f"# Analysis: {run_label}",
        "",
        f"- **Source**: `{pkl_path}`",
        f"- **Smoothing λ**: {lam}",
        f"- **Hessian freq**: {hessian_freq}",
        f"- **Entropy freq**: {entropy_freq}",
        "",
    ]

    if train_config:
        lines += ["## Training Configuration", ""]
        # Group config fields by category for readability
        _cfg_groups = [
            ("Model", ["model_name", "pretrained", "num_classes", "img_size",
                        "depth", "num_heads", "embed_dim", "patch_size",
                        "init_std", "use_scaled_init", "qk_norm", "label_smoothing"]),
            ("Data", ["dataset", "data_dir", "batch_size", "num_workers"]),
            ("Optimiser", ["optimizer", "learning_rate", "max_iters", "weight_decay",
                           "beta1", "beta2", "grad_clip", "eps"]),
            ("LR Schedule", ["decay_lr", "warmup_iters", "lr_decay_iters", "min_lr"]),
            ("Hessian", ["hessian_freq", "hessian_max_iter", "hessian_batch_size", "compute_fd"]),
            ("Entropy", ["entropy_freq"]),
            ("Intervention", ["temp_shift_step", "temp_shift_factor"]),
            ("Compute", ["device", "compile", "dtype", "seed"]),
            ("I/O", ["out_dir", "eval_interval", "log_interval",
                      "checkpoint_interval", "save_checkpoint", "init_from"]),
            ("W&B", ["wandb_log", "wandb_project", "wandb_run_name"]),
        ]
        for group_name, keys in _cfg_groups:
            group_rows = [[k, str(train_config[k])] for k in keys if k in train_config]
            if not group_rows:
                continue
            lines.append(f"### {group_name}")
            lines.append("")
            lines.append(_md_table(["Parameter", "Value"], group_rows))
            lines.append("")
        # Any remaining keys not covered by the groups above
        covered = {k for _, keys in _cfg_groups for k in keys}
        extra_rows = [[k, str(v)] for k, v in sorted(train_config.items()) if k not in covered]
        if extra_rows:
            lines.append("### Other")
            lines.append("")
            lines.append(_md_table(["Parameter", "Value"], extra_rows))
            lines.append("")

    proxy_display = {
        "H vs Prec_H":           "H vs Prec_H",
        "H vs H_VV":             "H vs H_VV",
        "H vs GN":               "H vs GN",
        "H vs Diag_H":           "H vs Diag_H",
        "H vs Fisher":           "H vs Fisher",
        "H vs KFAC":             "H vs KFAC",
        "H vs FD":               "H vs FD",
        "H vs BFGS":             "H vs BFGS",
        "Prec_H vs H_VV":        "Prec_H vs H_VV",
        "Prec_H vs GN":          "Prec_H vs GN",
        "Prec_H vs Diag_H":      "Prec_H vs Diag_H",
        "Prec_H vs Fisher":      "Prec_H vs Fisher",
        "Prec_H vs KFAC":        "Prec_H vs KFAC",
        "H vs Entropy(L0)":      "H vs Entropy(L0)",
        "H vs Entropy(avg)":     "H vs Entropy(avg)",
        "Prec_H vs Entropy(L0)": "Prec_H vs Entropy(L0)",
        "Prec_H vs Entropy(avg)":"Prec_H vs Entropy(avg)",
        "H_VV vs Entropy(L0)":   "H_VV vs Entropy(L0)",
        "GN vs Entropy(L0)":     "GN vs Entropy(L0)",
    }

    if corr_results.get("raw"):
        lines += ["## Raw Correlations", ""]
        rows = _corr_rows(corr_results["raw"])
        if rows:
            lines.append(_md_table(["Pair", "Spearman", "Pearson"], rows))
        lines.append("")

    if corr_results.get("smoothed"):
        lines += [f"## Smoothed Correlations (λ={lam})", ""]
        rows = _corr_rows(corr_results["smoothed"])
        if rows:
            lines.append(_md_table(["Pair", "Spearman", "Pearson"], rows))
        lines.append("")

    if corr_results.get("entropy"):
        lines += [f"## Proxy vs Entropy (smoothed, λ={lam})", ""]
        rows = _corr_rows(corr_results["entropy"])
        if rows:
            lines.append(_md_table(["Pair", "Spearman", "Pearson"], rows))
        lines.append("")

    proxy_label_map = {
        "prec_h": "Prec_H", "hessian_vv": "H_VV", "gn": "GN",
        "diag_h": "Diag_H", "fisher": "Fisher", "bfgs": "BFGS",
        "fd": "FD", "kfac": "KFAC",
    }
    for z, spike_results in sorted(spike_all.items()):
        lines += [f"## Spike Co-occurrence (z={z})", ""]
        rows = []
        for key, res in spike_results.items():
            label = proxy_label_map.get(key, key)
            p = res["P(Y_spike | X_spike)"]
            p_str = f"{p:.3f}" if not (p != p) else "nan"   # nan check
            rows.append([f"H vs {label}", str(res["n_X_spikes"]),
                         str(res["n_joint_spikes"]), p_str])
        if rows:
            lines.append(_md_table(["Pair", "n_H_spikes", "n_joint", "P(Y|X spike)"], rows))
        lines.append("")

    md_path = os.path.join(out_dir, "analysis.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[plot_history] analysis report → {md_path}")


# ---------------------------------------------------------------------------

def plot_history(
    pkl_path: str,
    out_dir: str | None = None,
    hessian_freq: int = 500,
    entropy_freq: int = 500,
    skip_intv: bool = True,
    lam: float = 10.0,
    compute_fd: bool = False,
) -> None:
    """
    Load a ``history.pkl`` and reproduce every post-training plot and analysis.

    Outputs:
        * ``training_dynamics.png``
        * ``curvature_smoothed_comparison.png``
        * Spike co-occurrence PNGs for z=1.5 and z=2.0
        * ``analysis.txt`` — raw stdout capture
        * ``analysis.md``  — structured Markdown report with tables

    Args:
        pkl_path:     Path to a ``history.pkl`` file.
        out_dir:      Output directory; defaults to same dir as pkl_path.
        hessian_freq: Hessian frequency used during training.
        entropy_freq: Entropy frequency used during training.
        skip_intv:    True (default) = interval-skipping; False = carry-forward.
        lam:          Whittaker–Henderson smoothing strength.
        compute_fd:   Include BFGS/FD metrics (must match training config).
    """
    with open(pkl_path, "rb") as f:
        history = pickle.load(f)
    n_iters = len(history.get("loss", []))
    print(f"[plot_history] loaded {pkl_path}  ({n_iters} iterations)")

    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(pkl_path))
    os.makedirs(out_dir, exist_ok=True)

    run_label = os.path.basename(os.path.dirname(os.path.abspath(pkl_path)))

    tee = _Tee(sys.stdout)
    old_stdout = sys.stdout
    sys.stdout = tee

    corr_results: dict = {}
    spike_all: dict[float, dict] = {}

    try:
        # --- Correlations ---
        corr_results = print_correlations(
            history, run_label, lam=lam, include_smooth=True,
            skip_intv=skip_intv, hessian_freq=hessian_freq,
            compute_fd=compute_fd,
        )

        # --- Training dynamics ---
        lr_value = history.get("lr", [0.0])
        lr_peak = max(lr_value) if lr_value else 0.0
        fig = plot_training_dynamics(
            histories={run_label: history},
            lrs={run_label: lr_peak},
            save_path=os.path.join(out_dir, "training_dynamics.png"),
            skip_intv=skip_intv,
            entropy_freq=entropy_freq,
        )
        plt.close(fig)
        print(f"[plot] training_dynamics.png")

        # --- Smoothed curvature comparison ---
        fig_smooth = plot_curvature_smoothed_comparison(
            history, lam=lam,
            save_path=os.path.join(out_dir, "curvature_smoothed_comparison.png"),
            skip_intv=skip_intv,
            hessian_freq=hessian_freq,
            compute_fd=compute_fd,
        )
        plt.close(fig_smooth)
        print(f"[plot] curvature_smoothed_comparison.png")

        # --- Spike co-occurrence ---
        proxy_label = {
            "prec_h": "Prec_H", "hessian_vv": "H_VV", "gn": "GN",
            "fd": "FD", "diag_h": "Diag_H", "fisher": "Fisher",
            "bfgs": "BFGS", "kfac": "KFAC",
        }
        for z in (1.5, 2.0):
            spike_figs, spike_results = plot_all_spike_cooccurrences(
                history, window=15, z_score=z, log_scale=True,
                save_dir=out_dir, skip_intv=skip_intv,
                hessian_freq=hessian_freq, compute_fd=compute_fd,
            )
            for fig_spike in spike_figs.values():
                plt.close(fig_spike)
            spike_all[z] = spike_results
            for key, res in spike_results.items():
                label = proxy_label.get(key, key)
                p = res["P(Y_spike | X_spike)"]
                print(
                    f"[spike z={z}] P({label} spike | H spike) = "
                    f"{'nan' if p != p else f'{p:.3f}'}"
                )

        print(f"\n[plot_history] outputs saved to {out_dir}")

    finally:
        sys.stdout = old_stdout

    # --- Write plain-text log ---
    txt_path = os.path.join(out_dir, "analysis.txt")
    with open(txt_path, "w") as f:
        f.write(tee.getvalue())
    print(f"[plot_history] analysis.txt  → {txt_path}")

    # --- Write Markdown report ---
    _write_analysis_md(
        out_dir=out_dir,
        run_label=run_label,
        pkl_path=pkl_path,
        corr_results=corr_results,
        spike_all=spike_all,
        lam=lam,
        hessian_freq=hessian_freq,
        entropy_freq=entropy_freq,
        train_config=history.get("config"),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Re-run post-training plots and analysis from a history.pkl file."
    )
    parser.add_argument("pkl_path", type=str, help="Path to history.pkl")
    parser.add_argument(
        "-o", "--out-dir", type=str, default=None,
        help="Output directory (default: same dir as pkl_path)",
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
        "--lam", type=float, default=10.0,
        help="Smoothing strength for Whittaker–Henderson smoother (default: 10)",
    )
    parser.add_argument(
        "--compute-fd", action="store_true",
        help="Include BFGS and FD metrics (only if computed during training)",
    )
    args = parser.parse_args()

    plot_history(
        pkl_path=args.pkl_path,
        out_dir=args.out_dir,
        hessian_freq=args.hessian_freq,
        entropy_freq=args.entropy_freq,
        skip_intv=not args.no_skip_intv,
        lam=args.lam,
        compute_fd=args.compute_fd,
    )


if __name__ == "__main__":
    main()
