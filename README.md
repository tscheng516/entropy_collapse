# Entropy Collapse — Loss Landscape Sharpness Validation

This repository validates loss landscape sharpness (via the spectral norm
of the Hessian and its proxies) and confirms that the proxies track each
other closely.  Experiments are organised by model family, each with its
own self-contained analysis helpers.

---

## Repository Layout

```
entropy_collapse/
├── LLM/                          # NanoGPT entropy-collapse experiments
│   ├── base_train.py             # Main training + checkpointing entry-point
│   ├── configs/
│   │   └── train_config.py       # All experiment flags (LR, init, wandb, …)
│   ├── src/
│   │   ├── helpers.py            # Curvature helpers & attention entropy
│   │   ├── model.py              # HookedGPT — NanoGPT with attention caching
│   │   ├── data_utils.py         # Data loading & batch sampling
│   │   └── plotting.py           # Training-dynamics plot, MAD spike detection,
│   │                             #   and Spearman/Pearson correlation helpers
│   └── Tin_Sum.ipynb             # Original exploratory notebook
├── ViT/                          # Vision Transformer experiments
│   ├── base_train.py             # Main training + checkpointing entry-point
│   ├── configs/
│   │   └── train_config.py       # All experiment flags (qk_norm, dataset, …)
│   └── src/
│       ├── helpers.py            # Curvature helpers & attention entropy
│       ├── model.py              # HookedViT — timm ViT with attention caching
│       ├── data_utils.py         # CIFAR-10/100 / ImageNet data loaders
│       └── plotting.py           # Training-dynamics plot, MAD spike detection,
│                                 #   and Spearman/Pearson correlation helpers
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

Each `src/plotting.py` is fully self-contained — spike detection and
correlation utilities are inlined directly rather than imported from a
shared package.  This keeps the LLM and ViT experiments independent and
easy to adapt.

---

## Clean Workspace Setup (Recommended)

The steps below are intended for a fresh machine / fresh clone and are
tested for both LLM (NanoGPT-based) and ViT experiments.

### 1. Create and activate a Python virtual environment

Use Python 3.10 or 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 1b. Alternative: create a Conda environment

If you prefer Conda, use this instead of `venv`.

CPU / MPS (macOS) example:

```bash
conda create -n entropy-collapse python=3.11 -y
conda activate entropy-collapse
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

CUDA 11.8 (Linux) example:

```bash
conda create -n entropy-collapse python=3.11 -y
conda activate entropy-collapse
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.1.2 torchvision==0.16.2
python -m pip install -r requirements.txt
```

Note: for CUDA environments, install `torch`/`torchvision` with the CUDA
index first (as above), then install the rest from `requirements.txt`.

### 2. Install a stable, pinned package stack

For best reproducibility, install pinned versions first, then install the
project requirements.

CPU / MPS (macOS) example:

```bash
pip install \
    torch==2.1.2 torchvision==0.16.2 \
    numpy==1.26.4 scipy==1.11.4 matplotlib==3.8.2 \
    timm==0.9.12 wandb==0.16.6
pip install -r requirements.txt
```

CUDA 11.8 example:

```bash
pip install --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.1.2 torchvision==0.16.2
pip install \
    numpy==1.26.4 scipy==1.11.4 matplotlib==3.8.2 \
    timm==0.9.12 wandb==0.16.6
pip install -r requirements.txt
```

### 3. Clone NanoGPT inside `LLM/` and prepare Shakespeare data

All commands below assume you are in the repo root.

```bash
git clone https://github.com/karpathy/nanoGPT.git LLM/nanoGPT
cd LLM/nanoGPT
# Optional: pin to the exact commit you used in prior runs for strict reproducibility
# git checkout <your-known-good-commit>
python data/shakespeare_char/prepare.py
cd ../..
```

### 4. Smoke-test both experiment entry points

```bash
python LLM/base_train.py --max_it 2 --hessian_freq 1 --entropy_freq 1 data_dir=LLM/nanoGPT/data/shakespeare_char
python ViT/base_train.py --max_it 2 --hessian_freq 1 --entropy_freq 1
```

## Quick Start (LLM / NanoGPT)

### 1. Train with default config

```bash
python LLM/base_train.py data_dir=LLM/nanoGPT/data/shakespeare_char
```

### 2. Override flags from the command line

Short argparse aliases:
- `--cp` maps to `init_from`
- `--optim` maps to `optimizer`
- `--lr` maps to `learning_rate`
- `--max_it` maps to `max_iters`
- `--wandb` maps to `wandb_log`
- `--z` maps to `z_score`

```bash
python LLM/base_train.py \
        data_dir=LLM/nanoGPT/data/shakespeare_char \
        --lr 5e-4 \
        --optim adamw \
        --max_it 2000 \
        hessian_freq=5 \
        entropy_freq=10 \
        --wandb true \
        wandb_run_name=small-lr-run
```

### 3. Resume from a checkpoint

```bash
python LLM/base_train.py --cp resume out_dir=out
```

## Quick Start (ViT)

### 1. Train with default config (CIFAR-10)

```bash
python ViT/base_train.py
```

### 2. Override core flags

```bash
python ViT/base_train.py \
    --optim adamw \
    --lr 1e-3 \
    --max_it 5000 \
    hessian_freq=5 \
    entropy_freq=10 \
    --wandb true \
    wandb_run_name=vit-cifar10-run
```

---

## Key Components

### `LLM/src/plotting.py` / `ViT/src/plotting.py`

Each module is self-contained and provides three utilities:

#### `plot_spike_cooccurrence(x, y, x_name, y_name, window, z_score, …)`
Spike-timeline strip that answers: *do spikes in metric X coincide with
spikes in metric Y?*

Spikes are detected via the **MAD (Median Absolute Deviation)** method —
matching the `conditional_exceedance_local` function in `Tin_Sum.ipynb`:
a point is a spike if its residual from the local rolling median exceeds
`z_score × MAD`.  The plot shows:
- Blue `|`  — X-only spikes
- Red `×`   — joint spikes (both X and Y spike together)
- Orange `|` — Y-only spikes

Returns `P(Y spike | X spike)` and the marginal baseline for statistical
comparison.

#### `print_correlations(history, name)`
Prints Spearman and Pearson correlations between all curvature metric
pairs (H vs H̃, H vs H_GN, H vs H_VV).

#### `plot_training_dynamics(histories, lrs, save_path)`
2×6-panel grid: training loss, Hessian proxies (H, H̃, H_GN, H_VV),
per-layer attention entropy, and three pairwise proxy scatter plots.

To match notebook behavior, sparse metric logging is smoothed with
carry-forward preprocessing before plotting:
- Curvature proxies carry forward the last positive finite value.
- Per-layer entropy applies the same carry-forward rule per layer.

---

### `LLM/configs/train_config.py` / `ViT/configs/train_config.py` — `TrainConfig`

Single dataclass containing every configurable flag.  Annotated groups:

| Group | Key flags |
|---|---|
| **I/O** | `out_dir`, `checkpoint_interval`, `init_from` |
| **W&B** | `wandb_log`, `wandb_project`, `wandb_run_name` |
| **Data** | `data_dir`, `batch_size`, `block_size` (LLM) / `dataset`, `img_size` (ViT) |
| **Architecture** | `n_layer`, `n_head`, `n_embd` (LLM) / `model_name`, `qk_norm` (ViT) |
| **Init** | `init_std`, `use_scaled_init` |
| **Optimiser** | `optimizer`, `learning_rate`, `beta1/2`, `eps`, `grad_clip` |
| **LR schedule** | `decay_lr`, `warmup_iters`, `lr_decay_iters`, `min_lr` |
| **Hessian** | `hessian_freq`, `hessian_max_iter`, `compute_fd` |
| **Entropy** | `entropy_freq` |
| **Compute** | `device`, `dtype`, `compile`, `seed` |

---

### `LLM/src/helpers.py` / `ViT/src/helpers.py`

#### `get_VV_subspace_mask(model)`
Returns a flat binary tensor selecting only the **value-projection
parameters** (`W_V` slice) of every fused attention layer.
Used to restrict power iteration to the value subspace (H_VV).

#### `get_curvature_metrics(model, optimizer, X, Y, loss, vv_mask, …)`
Computes **five sharpness proxies** in a single call:

| Symbol | Description |
|---|---|
| **H** | λ_max of the full Hessian (power iteration on HVPs) |
| **H̃ (prec_h)** | λ_max of Adam-preconditioned Hessian D⁻½ H D⁻½ |
| **H_VV** | λ_max of H restricted to the value-projection subspace |
| **H_GN** | λ_max of the Gauss-Newton matrix J^T H_L J |
| **FD** | Finite-difference proxy ‖Δg‖/‖Δw‖ between consecutive steps |

#### `get_attention_entropy(model)`
Computes the mean **Shannon entropy** (in nats) of the attention
distribution for each transformer layer, using the `last_att` cache
populated by the patched attention forward pass.

---

### `LLM/src/model.py` — `build_hooked_gpt`

Builds a standard NanoGPT `GPT` model and applies two modifications:

1. **Custom weight init** — all Linear/Embedding weights drawn from
   `N(0, init_std)` with optional residual-depth scaling for c_proj
   (NanoGPT scaled-init).

2. **Attention caching** — each attention block's `forward` method is
   monkey-patched to store the explicit softmax attention matrix as
   `block.attn.last_att` after every forward pass.  Flash-attention is
   disabled to enable this and to support second-order autograd.

---

### `ViT/src/model.py` — `build_hooked_vit`

Builds a timm ViT model with the same two modifications plus:

3. **Optional QK normalisation** — controlled by `qk_norm` (default
   `False`, matching the NanoGPT / LLM experiment setup where no QK-norm
   is applied).

---

### `LLM/src/data_utils.py`

| Function | Description |
|---|---|
| `load_data(data_dir)` | Memory-map `train.bin` / `val.bin` |
| `get_batch(data, batch_size, block_size, device)` | Sample a random (x, y) batch |

---

### `LLM/base_train.py` / `ViT/base_train.py`

End-to-end training script that:
1. Parses `TrainConfig` with optional CLI overrides
2. Loads training/validation data
3. Builds the model (fresh / resume / fine-tune)
4. Sets up AdamW or SGD with a cosine LR schedule
5. Runs the training loop, emitting metrics to stdout and W&B
6. Saves periodic checkpoints to `out_dir/`
7. After training: saves `history.pkl` and generates plots

---

## Weights & Biases Integration

Enable W&B with either `--wandb true` or `wandb_log=True`. Logged metrics:

| W&B key | Logged when |
|---|---|
| `train/loss` | every `log_interval` iters |
| `train/lr` | every `log_interval` iters |
| `val/loss` | every `eval_interval` iters |
| `hessian/lambda_max` | every `hessian_freq` iters |
| `hessian/prec_H` | every `hessian_freq` iters |
| `hessian/H_VV` | every `hessian_freq` iters |
| `hessian/GN` | every `hessian_freq` iters |
| `hessian/FD` | every `hessian_freq` iters |
| `entropy/layer_<k>` | every `entropy_freq` iters |

---

## Reproducing the Notebook Experiments

### Experiment A — Smooth dynamics (Spearman correlation)

```bash
python LLM/base_train.py \
    data_dir=LLM/nanoGPT/data/shakespeare_char \
    --optim adamw --lr 1e-5 --max_it 800 \
    hessian_freq=3 entropy_freq=10 \
    --wandb true wandb_run_name=exp-A-adamw

python LLM/base_train.py \
    data_dir=LLM/nanoGPT/data/shakespeare_char \
    --optim sgd --lr 0.002 --max_it 800 \
    hessian_freq=3 entropy_freq=10 \
    --wandb true wandb_run_name=exp-A-sgd
```

### Experiment B — Large-LR instability (spike co-occurrence)

```bash
python LLM/base_train.py \
    data_dir=LLM/nanoGPT/data/shakespeare_char \
    --optim adamw --lr 5e-3 --max_it 100 \
    hessian_freq=1 entropy_freq=1 \
    --wandb true wandb_run_name=exp-B-adamw

python LLM/base_train.py \
    data_dir=LLM/nanoGPT/data/shakespeare_char \
    --optim sgd --lr 0.5 --max_it 100 \
    hessian_freq=1 entropy_freq=1 \
    --wandb true wandb_run_name=exp-B-sgd
```

After training, each `base_train.py` saves six spike co-occurrence plots
for `H` vs `{H_VV, Prec_H, GN}` at `z=3` and `z=10`.

---

## NanoGPT Parts Reused

| NanoGPT component | How it is reused |
|---|---|
| `model.py` — `GPT`, `GPTConfig` | Base architecture; `HookedGPT` extends it |
| `model.py` — `configure_optimizers` | AdamW with correct weight-decay splits |
| `train.py` — cosine LR schedule | `get_lr()` function in `LLM/base_train.py` |
| `train.py` — checkpoint format | Same `ckpt.pt` dict structure |
| `data/*/prepare.py` | Data preparation (run once; not modified) |
| `configurator.py` — CLI override pattern | `key=value` CLI arg parsing in `LLM/base_train.py` |
