# nanochat — Entropy-Collapse Research Harness

A thin research harness around [karpathy/nanochat](https://github.com/karpathy/nanochat)
for studying **attention entropy collapse** and **second-order curvature dynamics**
in transformer language models.

The harness adds:
- **HookedGPT** — monkey-patches `CausalSelfAttention.forward` with an explicit
  softmax path so attention weights are cacheable and the autograd graph stays
  fully differentiable (required for Hessian-vector products).
- **Nine curvature proxies** — λ_max of H, Prec_H, H_VV (value subspace), GN,
  Diag_H, Fisher, KFAC; plus optional FD / BFGS.
- **Per-layer attention entropy** logged every `entropy_freq` steps.
- **Temperature-shift intervention** — one-shot sharpening/softening of all
  attention heads at a configurable step.

Data loading is delegated entirely to nanochat's own ClimbMix loader; no custom
dataset code is needed.

---

## Installation

### Step 1 — Clone the nanochat repo at the pinned SHA

```bash
# From the repo root (entropy_collapse/)
git clone https://github.com/karpathy/nanochat nanochat/nanochat_repo
cd nanochat/nanochat_repo
git checkout 0aaca56      # pinned SHA — tested with this harness
```

> **Why pin?**  nanochat is under active development.  The harness was written
> against commit `0aaca56`.  Later commits may change `GPTConfig` fields,
> `CausalSelfAttention` signatures, or the dataloader API and break the harness.

### Step 2 — Create the nanochat environment

#### Option A — uv (recommended, faster)

```bash
cd nanochat/nanochat_repo
uv sync --extra gpu          # installs PyTorch 2.5+, triton, flash-attn …
uv pip install -r ../../nanochat/requirements.txt   # research extras
```

#### Option B — conda

```bash
conda create -n entropy-nanochat python=3.11 -y
conda activate entropy-nanochat
cd nanochat/nanochat_repo
pip install -e ".[gpu]"
pip install -r ../../nanochat/requirements.txt
```

The `requirements.txt` adds:

| Package | Pin | Purpose |
|---------|-----|---------|
| numpy | 1.26.4 | Array utilities |
| scipy | 1.11.4 | Smoothing, correlations |
| matplotlib | 3.8.2 | Plotting |
| wandb | 0.19.11 | Experiment tracking |

### Step 3 — Download ClimbMix data shards

Run this inside the nanochat clone; it downloads parquet shards to
`nanochat/nanochat_repo/data/`.

```bash
cd nanochat/nanochat_repo
python -m nanochat.dataset -n 170
```
Approximately 150 shards are needed for GPT-2 capability pretraining, add 20 for padding.
The maximum total number of shards available in the entire dataset is 6542, which requires ~50 GB of disk space and a stable internet connection.

---

## Training

### Single-GPU

```bash
# Activate the uv venv first
source nanochat/nanochat_repo/.venv/bin/activate

# d12 preset (~85 M params, default)
python nanochat/base_train.py config=d12 \
    nanochat_dir=nanochat/nanochat_repo

# d8 pilot (~30 M params, fast iteration)
python nanochat/base_train.py config=d8 \
    nanochat_dir=nanochat/nanochat_repo
```

### Multi-GPU (torchrun)

```bash
# Uses train.sh — reads GPUS env var (default: 0,1,2,3)
GPUS=0,1,2,3 NANOCHAT_DIR=nanochat/nanochat_repo bash nanochat/train.sh config=d12

# or directly
torchrun --nproc_per_node=4 nanochat/base_train.py \
    config=d12 \
    nanochat_dir=nanochat/nanochat_repo
```

### CLI overrides (NanoGPT-style)

Any `TrainConfig` field can be overridden as `key=value`:

```bash
python nanochat/base_train.py config=d12 \
    learning_rate=1e-4 \
    max_iters=20000 \
    hessian_freq=200 \
    entropy_freq=50 \
    temp_shift_step=5000 \
    temp_shift_factor=0.25
```

Shorthand flags: `--lr`, `--optim`, `--max_it`, `--wandb`, `--bs`.

---

## Config presets

| Key | n_layer | n_embd | n_head / n_kv_head | ~params | seq_len | max_iters |
|-----|---------|--------|-------------------|---------|---------|-----------|
| `d8` | 8 | 512 | 4 / 4 | ~30 M | 512 | 2 000 |
| `d12` | 12 | 768 | 6 / 6 | ~85 M | 512 | 10 000 |
| `d24` | 24 | 1 024 | 8 / 4 (GQA) | ~350 M | 2 048 | 50 000 |

Set `config=d8` for fast iteration, `config=d12` for the default research run,
and `config=d24` to replicate GPT-2 scale dynamics.

---

## Outputs

Each run writes to `out/nanochat/<config>/<timestamp>/`:

```
history.pkl                   # full training history
final_ckpt.pt                 # model + optimizer state
training_dynamics.png         # loss + entropy panels
curvature_smoothed_comparison.png
spike_cooccurrence_H_vs_*.png
analysis.txt / analysis.md
```

### Re-run plots from a saved history

```bash
python nanochat/plot_history.py out/nanochat/d12/<timestamp>/history.pkl \
    --hessian_freq 500 --entropy_freq 100
```

---

## Architecture notes

### Patched attention

`src/model.py` replaces Flash Attention 3 / SDPA with an explicit
`(q @ k.T).softmax()` computation.  This enables:
1. `torch.autograd` Hessian-vector products (`create_graph=True`).
2. Attention weight caching (`block.attn.last_att`).

Sliding window masking is **ignored** in the patched forward — the patched
path always uses full causal context.  The `window_pattern` field in
`GPTConfig` is preserved for round-trip compatibility but has no effect.

### Hessian precision with bf16

nanochat's `Linear` layers cast their weights to `x.dtype` (typically bf16)
on each forward pass.  HVP power iteration still flows through these casts,
so second-order gradients are computed in a mixed-precision graph.  This is
sufficient for monitoring λ_max trends; absolute values will differ slightly
from a pure fp32 baseline.

### GQA (d24 preset)

The d24 preset uses `n_kv_head=4 < n_head=8`.  The patched attention handles
GQA via `repeat_interleave` before the attention matmul.

---

## Tracked metrics

| Key | Description |
|-----|-------------|
| `loss` | Per-step training CE loss |
| `val_loss` | Validation CE loss (sparse — every `eval_interval`) |
| `entropy` | Per-layer attention entropy in nats (sparse) |
| `hessian` | λ_max(H) — exact Hessian power iteration |
| `prec_h` | λ_max(D⁻½ H D⁻½) — Adam-preconditioned Hessian |
| `hessian_vv` | λ_max(H_VV) — value-projection subspace |
| `gn` | λ_max(H_GN) — Gauss-Newton |
| `diag_h` | max(diag(H)) — Hutchinson diagonal estimator |
| `fisher` | λ_max(F) — empirical Fisher |
| `kfac` | max λ_A · λ_G — K-FAC Kronecker proxy |
| `fd` | λ_max(H) — forward-difference FD (opt-in) |
| `bfgs` | λ_max(H) — central-difference FD (opt-in) |

Enable FD metrics with `compute_fd=True` (slow; for cross-validation only).
