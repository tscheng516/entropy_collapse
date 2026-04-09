# ViT — Entropy Collapse Experiments

This folder contains entropy-collapse experiments for Vision Transformers (ViT)
and is designed to reproduce and extend the results from
[apple/ml-sigma-reparam](https://github.com/apple/ml-sigma-reparam).

The environment closely follows the
[`vision/environment.yaml`](https://github.com/apple/ml-sigma-reparam/blob/main/vision/environment.yaml)
from that repository, pinning the same `timm` commit for reproducibility.

---

## Environment Setup

**Python 3.10** is required (matches the apple/ml-sigma-reparam environment).

### Option A — venv

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r ViT/requirements.txt
```

### Option B — Conda

```bash
conda create -n entropy-vit python=3.10 -y
conda activate entropy-vit
pip install --upgrade pip setuptools wheel
pip install -r ViT/requirements.txt
```

---

## Quick Start

### Train with default config (CIFAR-10)

```bash
python ViT/base_train.py
```

### Train on CIFAR-100

`torchvision` auto-downloads CIFAR-100 — no manual download needed.

```bash
python ViT/base_train.py \
    config=cifar100_small \
    dataset=cifar100 \
    data_dir=ViT/data/ \
    num_classes=100 \
    --optim adamw \
    --lr 1e-3 \
    --max_it 1000
```

### Train on ImageNet-1k

ImageNet-1k is a **gated dataset** on Hugging Face — you need to:

1. Accept the licence at <https://huggingface.co/datasets/imagenet-1k>
2. Create a [Hugging Face access token](https://huggingface.co/settings/tokens)
3. Export it before running:

```bash
export HF_TOKEN=hf_...
```

#### Option A — Local ImageFolder data

If you already have ImageNet-1k on disk in `ImageFolder` layout
(`train/` and `val/` sub-directories), point `data_dir` at it:

```bash
python ViT/base_train.py \
    config=imagenet1k_base \
    data_dir=/path/to/imagenet
```

#### Option B — Automatic Hugging Face download

When `data_dir` does **not** contain `train/` and `val/` folders, the
dataset is automatically downloaded from Hugging Face and cached locally.
Subsequent runs reuse the cache — no re-download needed:

```bash
python ViT/base_train.py \
    config=imagenet1k_base \
    data_dir=ViT/data/imagenet1k
```

### Override core flags

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

## Common CLI Flags

| Short flag | Full name | Default |
|---|---|---|
| `--lr` | `learning_rate` | `1e-3` |
| `--optim` | `optimizer` | `adamw` |
| `--max_it` | `max_iters` | `5000` |
| `--wandb` | `wandb_log` | `false` |
| `--z` | `z_score` | `3` |

---

## Key Source Files

| File | Description |
|---|---|
| `base_train.py` | Training entry-point |
| `configs/train_config.py` | All configurable flags |
| `src/model.py` | `build_hooked_vit` — timm ViT with attention caching |
| `src/helpers.py` | Curvature helpers (H, H̃, H_GN, H_VV, FD) |
| `src/data_utils.py` | CIFAR-10/100 / ImageNet data loaders |
| `src/plotting.py` | Training-dynamics plots, spike detection, correlations |
