# ViT — Entropy Collapse Experiments

This folder contains entropy-collapse experiments for Vision Transformers (ViT)
and is designed to reproduce and extend the results from
[apple/ml-sigma-reparam](https://github.com/apple/ml-sigma-reparam).


---

## Result

### imagenet1k (with temperature shift at 15k-th epoch)

#### Raw Correlations

| Pair        | Spearman | Pearson  |
|-------------|----------|----------|
| H vs Prec_H | 0.9363   | 0.9530   |
| H vs H_VV   | 0.8710   | 0.6869   |
| H vs GN     | 0.6761   | 0.2271   |
| H vs Diag_H | 0.8747   | 0.9030   |
| H vs Fisher | 0.6614   | 0.5349   |
| H vs KFAC   | 0.5420   | −0.0394  |

#### Smoothed Correlations (λ = 100.0)

| Pair        | Spearman | Pearson  |
|-------------|----------|----------|
| H vs Prec_H | 0.9633   | 0.9775   |
| H vs H_VV   | 0.9495   | 0.9486   |
| H vs GN     | 0.7019   | 0.3186   |
| H vs Diag_H | 0.9409   | 0.9849   |
| H vs Fisher | 0.7568   | 0.4688   |
| H vs KFAC   | 0.6608   | −0.0767  |

#### Spike Co-occurrence — P(X spike | H spike)

| Pair        | z = 1.5 | z = 2.0 |
|-------------|---------|---------|
| H vs Prec_H | 0.636   | 0.812   |
| H vs H_VV   | 0.318   | 0.250   |
| H vs GN     | 0.318   | 0.125   |
| H vs Diag_H | 0.682   | 0.688   |
| H vs Fisher | 0.318   | 0.250   |
| H vs KFAC   | 0.227   | 0.188   |

![Curvature metrics](figure/imagenet1k_curvature_smoothed_comparison.png)

![training dynamics](figure/imagenet1k_training_dynamics.png)

### cifar100

#### Raw Correlations

| Pair        | Spearman | Pearson |
|-------------|----------|---------|
| H vs Prec_H | 0.6723   | 0.8185  |
| H vs H_VV   | 0.7211   | 0.6583  |
| H vs GN     | 0.3915   | 0.5124  |
| H vs Diag_H | 0.6639   | 0.7933  |
| H vs Fisher | 0.2038   | 0.1238  |
| H vs KFAC   | 0.2296   | 0.3463  |

#### Smoothed Correlations (λ = 100.0)

| Pair        | Spearman | Pearson  |
|-------------|----------|----------|
| H vs Prec_H | 0.8002   | 0.9456   |
| H vs H_VV   | 0.8588   | 0.9743   |
| H vs GN     | 0.4528   | 0.1784   |
| H vs Diag_H | 0.8381   | 0.8974   |
| H vs Fisher | 0.2817   | −0.0057  |
| H vs KFAC   | 0.0764   | 0.6479   |

#### Spike Co-occurrence — P(X spike | H spike)

| Pair        | z = 1.5 | z = 2.0 |
|-------------|---------|---------|
| H vs Prec_H | 0.339   | 0.306   |
| H vs H_VV   | 0.321   | 0.306   |
| H vs GN     | 0.339   | 0.306   |
| H vs Diag_H | 0.571   | 0.611   |
| H vs Fisher | 0.250   | 0.222   |
| H vs KFAC   | 0.089   | 0.083   |


![Curvature metrics](figure/cifar100_curvature_smoothed_comparison.png)

![training dynamics](figure/cifar100_training_dynamics.png)


---

## Environment Setup

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

In which case you need to:

1. Accept the licence at <https://huggingface.co/datasets/imagenet-1k>
2. Create a [Hugging Face access token](https://huggingface.co/settings/tokens)
3. Export it before running:

```bash
export HF_TOKEN=hf_...
```

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

---

## Key Source Files

| File | Description |
|---|---|
| `base_train.py` | Training entry-point |
| `configs/train_config.py` | All configurable flags |
| `src/model.py`  — timm ViT with attention caching |
| `src/helpers.py` | Curvature metric helpers |
| `src/data_utils.py` | CIFAR-10/100 / ImageNet data loaders |
| `src/plotting.py` | Training-dynamics plots, spike detection, correlations |
