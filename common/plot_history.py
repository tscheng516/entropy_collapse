"""
plot_history.py — Re-run all post-training plots and analysis from a saved
history.pkl file.

This module is the single canonical implementation used by every sub-project
(ViT/, ViT5/, ViT_depth/, nanochat/).  The per-folder ``plot_history.py`` files
are deprecated; all base_train scripts now import directly from here.

Usage (standalone re-analysis)::

    # from the project root or from common/
    python common/plot_history.py  path/to/history.pkl
    python common/plot_history.py  path/to/history.pkl --task depth
    python common/plot_history.py  path/to/history.pkl --task lm --lam 10

Outputs are written to the same directory as ``history.pkl`` by default.
Use ``-o / --out-dir`` to redirect them elsewhere.
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
import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap — ensure the project root is on sys.path so that
# ``from common.plotting import ...`` resolves when this module is imported
# from any sub-project directory.
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from common.plotting import (  # noqa: E402
    plot_training_dynamics,
    plot_all_spike_cooccurrences,
    plot_curvature_smoothed_comparison,
    print_correlations,
)


# ======================================================================
# I/O helpers
# ======================================================================


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


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a simple Markdown table string."""
    col_w = [
        max(len(h), max((len(str(r[i])) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]
    sep    = "| " + " | ".join("-" * w for w in col_w) + " |"
    header = "| " + " | ".join(h.ljust(col_w[i]) for i, h in enumerate(headers)) + " |"
    lines  = [header, sep]
    for row in rows:
        lines.append(
            "| " + " | ".join(str(row[i]).ljust(col_w[i]) for i in range(len(headers))) + " |"
        )
    return "\n".join(lines)


def _corr_rows(corr_dict: dict) -> list[list[str]]:
    rows = []
    for label, vals in corr_dict.items():
        sp = f"{vals['spearman']:.4f}" if isinstance(vals, dict) else "n/a"
        pe = f"{vals['pearson']:.4f}"  if isinstance(vals, dict) else "n/a"
        rows.append([label, sp, pe])
    return rows


_WINDOW_LABELS = ["Whole", "Q1", "Q2", "Q3", "Q4"]


def _corr_table_by_window(corr_by_window: dict[str, dict], section: str) -> str:
    """Build a Markdown table with Spearman and Pearson columns for each window."""
    all_pairs: list[str] = []
    seen: set[str] = set()
    for lbl in _WINDOW_LABELS:
        cd = corr_by_window.get(lbl, {}).get(section, {})
        for pair in cd:
            pair_clean = pair.strip()
            if pair_clean not in seen:
                seen.add(pair_clean)
                all_pairs.append(pair_clean)
    if not all_pairs:
        return ""
    headers = ["Pair"]
    for lbl in _WINDOW_LABELS:
        headers += [f"{lbl} Sp", f"{lbl} Pe"]
    rows = []
    for pair in all_pairs:
        row = [pair]
        for lbl in _WINDOW_LABELS:
            cd = corr_by_window.get(lbl, {}).get(section, {})
            val = next((v for k, v in cd.items() if k.strip() == pair), None)
            if isinstance(val, dict):
                row.append(f"{val['spearman']:.4f}")
                row.append(f"{val['pearson']:.4f}")
            else:
                row.append("n/a")
                row.append("n/a")
        rows.append(row)
    return _md_table(headers, rows)


# ======================================================================
# Per-task config groups
# ======================================================================

_CFG_GROUPS_CLASSIFICATION = [
    ("Model",        ["model_name", "pretrained", "num_classes", "img_size",
                      "depth", "num_heads", "embed_dim", "patch_size",
                      "init_std", "use_scaled_init", "qk_norm", "label_smoothing"]),
    ("Data",         ["dataset", "data_dir", "batch_size", "num_workers"]),
    ("Optimiser",    ["optimizer", "learning_rate", "max_iters", "weight_decay",
                      "beta1", "beta2", "grad_clip", "eps"]),
    ("LR Schedule",  ["decay_lr", "warmup_iters", "lr_decay_iters", "min_lr"]),
    ("Hessian",      ["hessian_intv", "hessian_max_iter", "hessian_batch_size",
                      "compute_fd"]),
    ("Entropy",      ["entropy_intv"]),
    ("Intervention", ["temp_shift_step", "temp_shift_factor"]),
    ("Compute",      ["device", "compile", "dtype", "seed"]),
    ("I/O",          ["out_dir", "eval_interval", "log_interval",
                      "checkpoint_interval", "save_checkpoint", "init_from"]),
    ("W&B",          ["wandb_log", "wandb_project", "wandb_run_name"]),
]

_CFG_GROUPS_DEPTH = [
    ("Model",        ["model_name", "img_size", "patch_size", "depth", "num_heads",
                      "embed_dim", "init_std", "use_scaled_init", "qk_norm"]),
    ("Loss",         ["silog_lambda"]),
    ("Data",         ["data_dir", "batch_size", "num_workers"]),
    ("Optimiser",    ["optimizer", "learning_rate", "max_iters", "weight_decay",
                      "beta1", "beta2", "grad_clip", "eps"]),
    ("LR Schedule",  ["decay_lr", "warmup_iters", "lr_decay_iters", "min_lr"]),
    ("Hessian",      ["hessian_intv", "hessian_max_iter", "compute_fd"]),
    ("Entropy",      ["entropy_intv"]),
    ("Intervention", ["temp_shift_step", "temp_shift_factor"]),
    ("Compute",      ["device", "compile", "dtype", "seed"]),
    ("I/O",          ["out_dir", "eval_interval", "log_interval",
                      "checkpoint_interval", "save_checkpoint", "init_from"]),
    ("W&B",          ["wandb_log", "wandb_project", "wandb_run_name"]),
]

_CFG_GROUPS_LM = [
    ("Model",        ["n_layer", "n_head", "n_kv_head", "n_embd", "sequence_len",
                      "vocab_size", "window_pattern", "init_std", "use_scaled_init"]),
    ("Data",         ["nanochat_dir", "batch_size", "num_workers"]),
    ("Optimiser",    ["optimizer", "learning_rate", "max_iters", "weight_decay",
                      "beta1", "beta2", "grad_clip", "eps"]),
    ("MuonAdamW",    ["muon_matrix_lr", "muon_embedding_lr", "muon_unembedding_lr",
                      "muon_scalar_lr", "muon_ns_steps"]),
    ("LR Schedule",  ["warmup_iters", "warmdown_ratio", "min_lr_frac"]),
    ("Hessian",      ["hessian_intv", "hessian_max_iter", "hessian_batch_size",
                      "compute_fd", "label_smoothing"]),
    ("Entropy",      ["entropy_intv"]),
    ("Intervention", ["temp_shift_step", "temp_shift_factor"]),
    ("Compute",      ["device", "compile", "compute_dtype", "seed"]),
    ("I/O",          ["out_dir", "eval_interval", "log_interval",
                      "checkpoint_interval", "save_checkpoint", "init_from"]),
    ("W&B",          ["wandb_log", "wandb_project", "wandb_run_name"]),
]

_CFG_GROUPS_BY_TASK = {
    "classification": _CFG_GROUPS_CLASSIFICATION,
    "depth":          _CFG_GROUPS_DEPTH,
    "lm":             _CFG_GROUPS_LM,
}


# ======================================================================
# Markdown report writer
# ======================================================================


def _write_analysis_md(
    out_dir: str,
    run_label: str,
    pkl_path: str,
    corr_by_window: dict[str, dict],
    spike_all: dict[float, dict],
    lam: float,
    hessian_intv: int,
    entropy_intv: int,
    train_config: dict | None = None,
    task: str = "classification",
) -> None:
    """Write a Markdown analysis report to ``analysis.md``."""
    lines = [
        f"# Analysis: {run_label}",
        "",
        f"- **Source**: `{pkl_path}`",
        f"- **Smoothing λ**: {lam}",
        f"- **Hessian freq**: {hessian_intv}",
        f"- **Entropy freq**: {entropy_intv}",
        "",
    ]

    if train_config:
        lines += ["## Training Configuration", ""]
        cfg_groups = _CFG_GROUPS_BY_TASK.get(task, _CFG_GROUPS_CLASSIFICATION)
        covered: set[str] = set()
        for group_name, keys in cfg_groups:
            group_rows = [[k, str(train_config[k])] for k in keys if k in train_config]
            covered.update(k for k in keys if k in train_config)
            if not group_rows:
                continue
            lines.append(f"### {group_name}")
            lines.append("")
            lines.append(_md_table(["Parameter", "Value"], group_rows))
            lines.append("")
        extra_rows = [
            [k, str(v)] for k, v in sorted(train_config.items()) if k not in covered
        ]
        if extra_rows:
            lines.append("### Other")
            lines.append("")
            lines.append(_md_table(["Parameter", "Value"], extra_rows))
            lines.append("")

    raw_table = _corr_table_by_window(corr_by_window, "raw")
    if raw_table:
        lines += ["## Raw Correlations", "", raw_table, ""]

    log_raw_table = _corr_table_by_window(corr_by_window, "log_raw")
    if log_raw_table:
        lines += ["## Log-scale Raw Correlations", "", log_raw_table, ""]

    smoothed_table = _corr_table_by_window(corr_by_window, "smoothed")
    if smoothed_table:
        lines += [f"## Smoothed Correlations (λ={lam})", "", smoothed_table, ""]

    log_smoothed_table = _corr_table_by_window(corr_by_window, "log_smoothed")
    if log_smoothed_table:
        lines += [f"## Log-scale Smoothed Correlations (λ={lam})", "", log_smoothed_table, ""]

    entropy_table = _corr_table_by_window(corr_by_window, "entropy")
    if entropy_table:
        lines += [f"## Log Proxy vs Entropy (λ={lam})", "", entropy_table, ""]

    proxy_label_map = {
        "prec_h": "Prec_H", "hessian_vv": "H_VV", "gn": "GN",
        "diag_h": "Diag_H", "fisher": "Fisher", "bfgs": "BFGS",
        "fd": "FD", "kfac": "KFAC",
    }

    def _spike_rows(ref_dict: dict, ref_name: str) -> list[list[str]]:
        rows = []
        for key, res in ref_dict.items():
            label = proxy_label_map.get(key, key)
            p     = res["P(Y_spike | X_spike)"]
            base  = res["baseline_P(Y_spike)"]
            p_str    = f"{p:.3f}"    if not np.isnan(p)    else "nan"
            base_str = f"{base:.3f}" if not np.isnan(base) else "nan"
            rows.append([
                f"{ref_name} vs {label}",
                str(res["n_X_spikes"]),
                str(res["n_joint_spikes"]),
                p_str,
                base_str,
            ])
        return rows

    for z, spike_results in sorted(spike_all.items()):
        lines += [f"## Spike Co-occurrence (z={z})", ""]

        h_rows = _spike_rows(spike_results.get("H", {}), "H")
        if h_rows:
            lines.append("### H as Reference")
            lines.append("")
            lines.append(_md_table(
                ["Pair", "n_X_spikes", "n_joint", "P(Y|X spike)", "baseline P(Y spike)"],
                h_rows,
            ))
            lines.append("")

        prec_h_rows = _spike_rows(spike_results.get("Prec_H", {}), "Prec_H")
        if prec_h_rows:
            lines.append("### Prec_H as Reference")
            lines.append("")
            lines.append(_md_table(
                ["Pair", "n_X_spikes", "n_joint", "P(Y|X spike)", "baseline P(Y spike)"],
                prec_h_rows,
            ))
            lines.append("")

    md_path = os.path.join(out_dir, "analysis.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[plot_history] analysis report → {md_path}")


# ======================================================================
# Main API
# ======================================================================


def plot_history(
    pkl_path: str,
    out_dir: str | None = None,
    hessian_intv: int = 50,
    entropy_intv: int = 50,
    skip_intv: bool = True,
    lam: float = 10.0,
    compute_fd: bool = False,
    task: str = "classification",
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
        hessian_intv: Hessian frequency used during training.
        entropy_intv: Entropy frequency used during training.
        skip_intv:    True (default) = interval-skipping; False = carry-forward.
        lam:          Whittaker–Henderson smoothing strength.
        compute_fd:   Include BFGS/FD metrics (must match training config).
        task:         One of ``"classification"`` (ViT/ViT5), ``"depth"``
                      (ViT_depth), or ``"lm"`` (nanochat).
    """
    with open(pkl_path, "rb") as f:
        history = pickle.load(f)
    n_iters = len(history.get("loss", []))
    print(f"[plot_history] loaded {pkl_path}  ({n_iters} iterations)")

    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(pkl_path))
    os.makedirs(out_dir, exist_ok=True)

    run_label = os.path.basename(os.path.dirname(os.path.abspath(pkl_path)))

    # Determine iteration windows for quarter-wise correlation analysis.
    cfg = history.get("config") or {}
    max_iter = int(cfg.get("max_iters", n_iters) or n_iters)
    q = max(max_iter // 4, 1)
    _windows = [
        ("Whole", 0,       -1),
        ("Q1",    0,        q),
        ("Q2",    q,    2 * q),
        ("Q3",    2 * q, 3 * q),
        ("Q4",    3 * q,   -1),
    ]

    tee = _Tee(sys.stdout)
    old_stdout = sys.stdout
    sys.stdout = tee

    corr_by_window: dict[str, dict] = {}
    spike_all: dict[float, dict] = {}

    try:
        # --- Correlations (whole history + four quarters) ---
        for win_label, win_start, win_end in _windows:
            corr_by_window[win_label] = print_correlations(
                history, f"{run_label} [{win_label}]", lam=lam,
                include_smooth=True, skip_intv=skip_intv,
                hessian_intv=hessian_intv, compute_fd=compute_fd,
                start=win_start, end=win_end,
            )

        # --- Training dynamics ---
        lr_value = history.get("lr", [0.0])
        lr_peak  = max(lr_value) if lr_value else 0.0
        fig = plot_training_dynamics(
            histories={run_label: history},
            lrs={run_label: lr_peak},
            save_path=os.path.join(out_dir, "training_dynamics.png"),
            skip_intv=skip_intv,
            entropy_intv=entropy_intv,
            task=task,
        )
        plt.close(fig)
        print("[plot] training_dynamics.png")

        # --- Smoothed curvature comparison ---
        fig_smooth = plot_curvature_smoothed_comparison(
            history, lam=lam,
            save_path=os.path.join(out_dir, "curvature_smoothed_comparison.png"),
            skip_intv=skip_intv,
            hessian_intv=hessian_intv,
            entropy_intv=entropy_intv,
            compute_fd=compute_fd,
        )
        plt.close(fig_smooth)
        print("[plot] curvature_smoothed_comparison.png")

        # --- Spike co-occurrence (computation kept; PNGs disabled) ---
        proxy_label = {
            "prec_h": "Prec_H", "hessian_vv": "H_VV", "gn": "GN",
            "fd": "FD", "diag_h": "Diag_H", "fisher": "Fisher",
            "bfgs": "BFGS", "kfac": "KFAC",
        }
        for z in (1.5, 2.0):
            spike_figs, spike_results = plot_all_spike_cooccurrences(
                history, window=15, z_score=z, log_scale=True,
                save_dir=None,  # PNG saving disabled; stats still computed
                skip_intv=skip_intv,
                hessian_intv=hessian_intv, compute_fd=compute_fd,
            )
            for fig_spike in spike_figs.values():
                plt.close(fig_spike)
            spike_all[z] = spike_results
            for ref_name, ref_results in spike_results.items():
                for key, res in ref_results.items():
                    label = proxy_label.get(key, key)
                    p = res["P(Y_spike | X_spike)"]
                    print(
                        f"[spike z={z}] P({label} spike | {ref_name} spike) = "
                        f"{'nan' if np.isnan(p) else f'{p:.3f}'}"
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
        corr_by_window=corr_by_window,
        spike_all=spike_all,
        lam=lam,
        hessian_intv=hessian_intv,
        entropy_intv=entropy_intv,
        train_config=history.get("config"),
        task=task,
    )


# ======================================================================
# CLI entry point (used by per-folder thin wrappers)
# ======================================================================


def build_arg_parser(description: str = "Re-run post-training analysis from history.pkl.") -> argparse.ArgumentParser:
    """Return a pre-populated ArgumentParser for the plot_history CLI."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("pkl_path", type=str, help="Path to history.pkl")
    parser.add_argument(
        "-o", "--out-dir", type=str, default=None,
        help="Output directory (default: same dir as pkl_path)",
    )
    parser.add_argument(
        "--hessian_intv", type=int, default=50,
        help="Hessian computation frequency used during training (default: 500)",
    )
    parser.add_argument(
        "--entropy_intv", type=int, default=50,
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
    parser.add_argument(
        "--task", type=str, default="classification",
        choices=["classification", "depth", "lm"],
        help="Task type: classification (ViT/ViT5), depth (ViT_depth), lm (nanochat). "
             "Controls which config groups appear in the Markdown report. "
             "(default: classification)",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    plot_history(
        pkl_path=args.pkl_path,
        out_dir=args.out_dir,
        hessian_intv=args.hessian_intv,
        entropy_intv=args.entropy_intv,
        skip_intv=not args.no_skip_intv,
        lam=args.lam,
        compute_fd=args.compute_fd,
        task=args.task,
    )


if __name__ == "__main__":
    main()
