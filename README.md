# Entropy Collapse — Loss Landscape Sharpness Validation

This repository validates loss landscape sharpness (via the spectral norm
of the Hessian and its proxies) and confirms that the proxies track each
other closely. 

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
│   ├── notebook.ipynb            # Original exploratory notebook
│   ├── requirements.txt          # LLM-specific dependencies (Python 3.10/3.11)
│   └── README.md                 # LLM setup and usage
├── ViT/                          # Vision Transformer experiments
│   ├── base_train.py             # Main training + checkpointing entry-point
│   ├── configs/
│   │   └── train_config.py       # All experiment flags (qk_norm, dataset, …)
│   ├── src/
│   │   ├── helpers.py            # Curvature helpers & attention entropy
│   │   ├── model.py              # HookedViT — timm ViT with attention caching
│   │   ├── data_utils.py         # CIFAR-10/100 / ImageNet data loaders
│   │   └── plotting.py           # Training-dynamics plot, MAD spike detection,
│   │                             #   and Spearman/Pearson correlation helpers
│   ├── requirements.txt          # ViT-specific dependencies (Python 3.10)
│   └── README.md                 # ViT setup and usage
└── README.md                     # This file
```

---

## Environment Setup

The two experiment families have different Python and `timm` version
requirements, so each has its own `requirements.txt`.  See the dedicated
READMEs for full instructions:

- **LLM / NanoGPT** — [`LLM/README.md`](LLM/README.md)  (Python 3.10 or 3.11)
- **ViT** — [`ViT/README.md`](ViT/README.md)  (Python 3.10, mirrors
  [apple/ml-sigma-reparam](https://github.com/apple/ml-sigma-reparam/blob/main/vision/environment.yaml))

### LLM quick setup (venv, CPU/MPS)
Please refer to the [.md](LLM/README.md).

### ViT quick setup (venv, CPU/MPS)
Please refer to the [.md](ViT/README.md).

### NanoGPT data preparation (LLM only)

```bash
git clone https://github.com/karpathy/nanoGPT.git LLM/nanoGPT
cd LLM/nanoGPT
python data/shakespeare_char/prepare.py
cd ../..
```

### Smoke-test both experiment entry points

```bash
python LLM/base_train.py --max_it 2 --hessian_freq 1 --entropy_freq 1 data_dir=LLM/nanoGPT/data/shakespeare_char
python ViT/base_train.py --max_it 2 --hessian_intv 1 --entropy_intv 1
```

## Quick Start (LLM / NanoGPT)

### 1. Train with default config

```bash
python LLM/base_train.py data_dir=LLM/nanoGPT/data/shakespeare_char
```

### 2. Override flags from the command line

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

### 2. Train on CIFAR-100

No manual download is needed; `torchvision` auto-downloads CIFAR-100.

```bash
python ViT/base_train.py \
    dataset=cifar100 \
    data_dir=ViT/data/cifar100 \
    num_classes=100 \
    --optim adamw \
    --lr 1e-3 \
    --max_it 5000
```

### 3. Train on ImageNet-1k directly from Hugging Face

No manual export script is required. The loader can pull
`imagenet-1k` directly via the `datasets` package.

```bash
python ViT/base_train.py \
    dataset=imagenet_hf \
    data_dir=ViT/data/imagenet1k_hf \
    num_classes=1000 \
    --optim adamw \
    --lr 1e-3 \
    --max_it 5000
```

If you already have local ImageNet in ImageFolder layout, use:

```bash
python ViT/base_train.py dataset=imagenet data_dir=/path/to/imagenet num_classes=1000
```

### 4. Override core flags

```bash
python ViT/base_train.py \
    --optim adamw \
    --lr 1e-3 \
    --max_it 5000 \
    hessian_intv=5 \
    entropy_intv=10 \
    --wandb true \
    wandb_run_name=vit-cifar10-run
```
