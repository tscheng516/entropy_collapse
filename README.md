# Entropy Collapse — Loss Landscape Sharpness Validation

This repository validates loss landscape sharpness (via the spectral norm
of the Hessian and its proxies) and confirms that the proxies track each
other closely.  Experiments are organised by model family so that shared
analysis code can be reused cleanly.

---

## Repository Layout

```
entropy_collapse/
├── common/                       # Shared analysis utilities (model-agnostic)
│   ├── power_iteration.py        # λ_max estimation via Hessian-vector products
│   └── spike_analysis.py         # MAD spike detection, co-occurrence plot,
│                                 #   Pearson / Spearman correlation helpers
├── LLM/                          # NanoGPT entropy-collapse experiments
│   ├── base_train.py             # Main training + checkpointing entry-point
│   ├── configs/
│   │   └── train_config.py       # All experiment flags (LR, init, wandb, …)
│   ├── src/
│   │   ├── helpers.py            # Curvature helpers & attention entropy
│   │   ├── model.py              # HookedGPT — NanoGPT with attention caching
│   │   ├── data_utils.py         # Data loading & batch sampling
│   │   └── plotting.py           # Training-dynamics plot (imports from common)
│   └── Tin_Sum.ipynb             # Original exploratory notebook
├── ViT/                          # Vision Transformer experiments (coming soon)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

### `common/` — shared utilities

| Module | Contents |
|---|---|
| `power_iteration.py` | `power_iteration()` — λ_max of the Hessian (or a subspace / preconditioned variant) via HVP power iteration |
| `spike_analysis.py` | `plot_spike_cooccurrence()` — MAD-based spike detection & co-occurrence timeline; `print_correlations()` — Pearson & Spearman correlations between proxy pairs |

---

## Quick Start (LLM / NanoGPT)

### 1. Clone NanoGPT inside `LLM/` and prepare Shakespeare data

All subsequent commands assume you are in the **repo root**.

```bash
# From the repo root:
git clone https://github.com/karpathy/nanoGPT.git LLM/nanoGPT
cd LLM/nanoGPT && python data/shakespeare_char/prepare.py && cd ../..
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train with default config

```bash
python LLM/base_train.py data_dir=LLM/nanoGPT/data/shakespeare_char
```

### 4. Override flags from the command line

```bash
python LLM/base_train.py \
    data_dir=LLM/nanoGPT/data/shakespeare_char \
    learning_rate=5e-4 \
    optimizer=adamw \
    max_iters=2000 \
    hessian_freq=5 \
    entropy_freq=10 \
    wandb_log=True \
    wandb_run_name=small-lr-run
```

### 5. Resume from a checkpoint

```bash
python LLM/base_train.py init_from=resume out_dir=out
```

---

## Key Components

### `common/power_iteration.py` — `power_iteration`

Estimates **λ_max of the full Hessian** (or a projected / preconditioned
variant) via iterated Hessian-vector products (HVPs) using
`torch.autograd.grad` with `create_graph=True`.  No full Hessian matrix
is materialised.  Reused by both the LLM and ViT experiment modules.

### `common/spike_analysis.py`

#### `plot_spike_cooccurrence(x, y, x_name, y_name, window, z_score, …)`
Spike-timeline strip that answers: *do spikes in metric X coincide with
spikes in metric Y?*

Spikes are detected via the **MAD (Median Absolute Deviation)** method:
a point is a spike if its residual from the local rolling median exceeds
`z_score × MAD`.  The plot shows:
- Blue `|`  — X-only spikes
- Orange `|` — Y-only spikes
- Red `×`   — joint spikes (both X and Y spike together)

Returns `P(Y spike | X spike)` and the marginal baseline for statistical
comparison.

#### `print_correlations(history, name)`
Prints Spearman and Pearson correlations between all curvature metric
pairs (H vs H̃, H vs H_GN, H vs H_VV).

---

### `LLM/configs/train_config.py` — `TrainConfig`

Single dataclass containing every configurable flag.  Annotated groups:

| Group | Key flags |
|---|---|
| **I/O** | `out_dir`, `checkpoint_interval`, `init_from` |
| **W&B** | `wandb_log`, `wandb_project`, `wandb_run_name` |
| **Data** | `data_dir`, `batch_size`, `block_size` |
| **Architecture** | `n_layer`, `n_head`, `n_embd`, `vocab_size` |
| **Init** | `init_std`, `use_scaled_init` |
| **Optimiser** | `optimizer`, `learning_rate`, `beta1/2`, `eps`, `grad_clip` |
| **LR schedule** | `decay_lr`, `warmup_iters`, `lr_decay_iters`, `min_lr` |
| **Hessian** | `hessian_freq`, `hessian_max_iter`, `compute_fd` |
| **Entropy** | `entropy_freq` |
| **Compute** | `device`, `dtype`, `compile`, `seed` |

---

### `LLM/src/helpers.py`

#### `get_VV_subspace_mask(model)`
Returns a flat binary tensor selecting only the **value-projection
parameters** (`W_V` slice) of every fused `c_attn` attention layer.
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

### `LLM/src/data_utils.py`

| Function | Description |
|---|---|
| `load_data(data_dir)` | Memory-map `train.bin` / `val.bin` |
| `get_batch(data, batch_size, block_size, device)` | Sample a random (x, y) batch |

---

### `LLM/base_train.py`

End-to-end training script that:
1. Parses `TrainConfig` with optional CLI overrides
2. Memory-maps training/validation data
3. Builds `HookedGPT` (fresh / resume / fine-tune)
4. Sets up AdamW or SGD with a cosine LR schedule
5. Runs the training loop, emitting metrics to stdout and W&B
6. Saves periodic checkpoints to `out_dir/`
7. After training: saves `history.pkl` and generates plots

---

## Weights & Biases Integration

Set `wandb_log=True` to enable.  Logged metrics:

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
    optimizer=adamw learning_rate=1e-5 max_iters=800 \
    hessian_freq=3 entropy_freq=10 \
    wandb_log=True wandb_run_name=exp-A-adamw

python LLM/base_train.py \
    data_dir=LLM/nanoGPT/data/shakespeare_char \
    optimizer=sgd learning_rate=0.002 max_iters=800 \
    hessian_freq=3 entropy_freq=10 \
    wandb_log=True wandb_run_name=exp-A-sgd
```

### Experiment B — Large-LR instability (spike co-occurrence)

```bash
python LLM/base_train.py \
    data_dir=LLM/nanoGPT/data/shakespeare_char \
    optimizer=adamw learning_rate=5e-3 max_iters=100 \
    hessian_freq=1 entropy_freq=1 \
    wandb_log=True wandb_run_name=exp-B-adamw

python LLM/base_train.py \
    data_dir=LLM/nanoGPT/data/shakespeare_char \
    optimizer=sgd learning_rate=0.5 max_iters=100 \
    hessian_freq=1 entropy_freq=1 \
    wandb_log=True wandb_run_name=exp-B-sgd
```

After training, load `out/history.pkl` and call
`common.spike_analysis.plot_spike_cooccurrence` to reproduce the
joint/disjoint spike figures.

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
