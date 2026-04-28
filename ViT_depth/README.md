# depth — Entropy Collapse Experiments on NYU Depth V2

Entropy-collapse analysis for Vision Transformers trained from scratch on
**monocular depth estimation** (NYU Depth V2).  Extends the analysis from
`ViT/` to a dense-prediction task with the scale-invariant log loss (SILog).

---

## Task Overview

| Item | Value |
|------|-------|
| Dataset | NYU Depth V2 (Silberman et al., 2012) — 47 584 train / 654 test |
| Backbone | ViT-B/16 (86 M params), trained **from scratch** |
| Input | RGB frames, 448×448 (28×28 = 784 patches) |
| Output | Dense depth map (B, 1, 448, 448), metres |
| Loss | Scale-invariant log loss (SILog, λ=0.5, Eigen et al., 2014) |
| Metrics | SILog, RMSE (log-scale), δ<1.25 accuracy |

---

## Quick Setup

### 1. Install dependencies

```bash
# venv
python3.10 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r depth/requirements.txt
```

```bash
# Conda
conda create -n entropy-collapse-depth python=3.10 -y
conda activate entropy-collapse-depth
pip install -r depth/requirements.txt
```

### 2. Train (default: ViT-B/16 on NYU Depth V2)

```bash
python depth/base_train.py
```

NYU Depth V2 is downloaded automatically from Hugging Face on first run:

```bash
# Optional: set a custom cache directory
python depth/base_train.py data_dir=/data/nyu_depth_v2
```

---

## Config Presets

| Preset | Model | Batch | LR | Max Iters |
|---|---|---|---|---|
| `default` / `nyudepth_base` | ViT-B/16 (86 M) | 16 | 5e-4 | 50 000 |
| `nyudepth_large` | ViT-L/16 (307 M) | 8 | 2e-4 | 50 000 |

```bash
python depth/base_train.py config=nyudepth_base
bash  depth/train.sh        config=nyudepth_large   # multi-GPU via torchrun
```

---

## Advanced Experiments

### Override individual flags

Any `DepthTrainConfig` field can be overridden on the command line:

```bash
python depth/base_train.py config=nyudepth_base \
    learning_rate=2e-4 \
    max_iters=100000 \
    warmup_iters=5000 \
    hessian_freq=10 \
    entropy_freq=5 \
    wandb_log=true \
    wandb_run_name=vitb16-nyudepth-run
```

### Multi-GPU training

`train.sh` wraps `torchrun`. Set `GPUS` to select devices:

```bash
GPUS=0,1,2,3 bash depth/train.sh config=nyudepth_base
```

### Temperature-shift intervention

```bash
python depth/base_train.py temp_shift_step=15000 temp_shift_factor=2.0
```

---

## Loss Function: Scale-Invariant Log Loss (SILog)

Following [Eigen et al., 2014](https://arxiv.org/abs/1406.2283):

```
L(pred, gt) = (1/n) Σ d_i²  −  (λ/n²)(Σ d_i)²
```

where `d_i = log(pred_i) − log(gt_i)` and `λ=0.5`.

Only valid (positive, finite) ground-truth pixels contribute; zero pixels
(invalid / out-of-range) are masked out before loss computation.

---

## Architecture

```
Input RGB (B, 3, 448, 448)
      ↓
ViT-B/16 backbone  (12 blocks, 12 heads, 768 embed_dim)
  [ attention hooks: cache last_att for entropy ]
      ↓
Patch tokens (B, 784, 768)   ← drop CLS
      ↓ reshape to (B, 768, 28, 28)
Depth head: LayerNorm → Conv1×1(768→256) → GELU → Conv1×1(256→1)
      ↓
Bilinear upsample → (B, 1, 448, 448)
      ↓
Softplus + 1e-3  → depth in metres (strictly positive)
```

---

## Post-Training Analysis

`plot_history.py` replays a saved `history.pkl` and writes all figures plus
a structured analysis report.

```bash
python depth/plot_history.py out/nyudepth_vitb16/.../history.pkl
```

Outputs written next to the pickle:

| File | Contents |
|---|---|
| `training_dynamics.png` | SILog loss, RMSE / δ<1.25, attention entropy |
| `curvature_smoothed_comparison.png` | Smoothed proxy traces (λ=100) |
| Spike co-occurrence PNGs | MAD-spike timelines (z=1.5, z=2.0) |
| `analysis.txt` | Full correlation report (plain text) |
| `analysis.md` | Markdown tables: raw/smoothed correlations, spike co-occurrence |

---

## Key Source Files

| File | Description |
|---|---|
| `base_train.py` | Training entry-point; accepts `config=<preset> key=value …` |
| `configs/train_config.py` | All hyperparameter presets (DepthTrainConfig) |
| `src/model.py` | HookedViTDepth: ViT-B/16 + lightweight depth head |
| `src/helpers.py` | Nine curvature proxies adapted for SILog; attention entropy |
| `src/data_utils.py` | NYU Depth V2 via Hugging Face with auto-download |
| `src/plotting.py` | Dynamics plots (SILog/RMSE), spike detection, correlations |
| `plot_history.py` | Post-training CLI: `history.pkl → figures + analysis.{txt,md}` |
