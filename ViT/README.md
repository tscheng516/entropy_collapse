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

### CUDA variants

For **CUDA 11.8**, install the GPU-enabled wheels first, then the rest:

```bash
pip install --index-url https://download.pytorch.org/whl/cu118 \
    torch==2.1.2 torchvision==0.16.2
pip install -r ViT/requirements.txt
```

For **CUDA 12.1**, replace `cu118` with `cu121`.

> **Note on `timm`:** `ViT/requirements.txt` installs `timm` directly from the
> exact git commit used by the apple/ml-sigma-reparam paper
> (`9ee846ff0cbbc05a99b45140aa6d84083bcf6488`).  This ensures identical model
> definitions and avoids API drift in newer timm releases.

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
    dataset=cifar100 \
    data_dir=ViT/data/cifar100 \
    num_classes=100 \
    --optim adamw \
    --lr 1e-3 \
    --max_it 5000
```

### Train on ImageNet-1k (Hugging Face streaming)

```bash
python ViT/base_train.py \
    dataset=imagenet_hf \
    data_dir=ViT/data/imagenet1k_hf \
    num_classes=1000 \
    --optim adamw \
    --lr 1e-3 \
    --max_it 5000
```

If you already have local ImageNet in `ImageFolder` layout:

```bash
python ViT/base_train.py \
    dataset=imagenet \
    data_dir=/path/to/imagenet \
    num_classes=1000
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

## Smoke Test

```bash
python ViT/base_train.py --max_it 2 --hessian_freq 1 --entropy_freq 1
```

---

## Reproducing apple/ml-sigma-reparam Results

The ViT model is built with `timm` and supports optional QK normalisation
(controlled by the `qk_norm` flag, default `False`).

To replicate the sigma-reparametrisation experiments from the apple paper,
enable QK normalisation:

```bash
python ViT/base_train.py \
    qk_norm=True \
    --optim adamw \
    --lr 1e-3 \
    --max_it 5000 \
    --wandb true \
    wandb_run_name=vit-qknorm
```

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
