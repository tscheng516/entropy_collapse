# LLM / NanoGPT — Entropy Collapse Experiments

This folder contains entropy-collapse experiments for character-level language
models built on top of [NanoGPT](https://github.com/karpathy/nanoGPT).

---

## Environment Setup

**Python 3.10 or 3.11** is recommended.

### Option A — venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r LLM/requirements.txt
```

### Option B — Conda

```bash
conda create -n entropy-llm python=3.11 -y
conda activate entropy-llm
pip install --upgrade pip setuptools wheel
pip install -r LLM/requirements.txt
```

### CUDA variants

For **CUDA 11.8**, install the GPU-enabled wheel first, then the rest:

```bash
pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.6.0
pip install -r LLM/requirements.txt
```

For **CUDA 12.1**, replace `cu118` with `cu121`.

---

## One-time Data Preparation

Clone NanoGPT inside this folder and prepare the Shakespeare dataset:

```bash
# Run from the repo root
git clone https://github.com/karpathy/nanoGPT.git LLM/nanoGPT
cd LLM/nanoGPT
python data/shakespeare_char/prepare.py
cd ../..
```

---

## Quick Start

### Train with default config

```bash
python LLM/base_train.py data_dir=LLM/nanoGPT/data/shakespeare_char
```

### Common CLI flags

| Short flag | Full name | Default |
|---|---|---|
| `--lr` | `learning_rate` | `1e-3` |
| `--optim` | `optimizer` | `adamw` |
| `--max_it` | `max_iters` | `5000` |
| `--cp` | `init_from` | `scratch` |
| `--wandb` | `wandb_log` | `false` |

### Override flags from the command line

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

### Resume from a checkpoint

```bash
python LLM/base_train.py --cp resume out_dir=out
```

---

## Reproducing the Paper Experiments

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

---

## Smoke Test

```bash
python LLM/base_train.py --max_it 2 --hessian_freq 1 --entropy_freq 1 \
    data_dir=LLM/nanoGPT/data/shakespeare_char
```

---

## NanoGPT Components Reused

| NanoGPT component | How it is reused |
|---|---|
| `model.py` — `GPT`, `GPTConfig` | Base architecture; `HookedGPT` extends it |
| `model.py` — `configure_optimizers` | AdamW with correct weight-decay splits |
| `train.py` — cosine LR schedule | `get_lr()` function in `base_train.py` |
| `train.py` — checkpoint format | Same `ckpt.pt` dict structure |
| `data/*/prepare.py` | Data preparation (run once; not modified) |
| `configurator.py` — CLI override pattern | `key=value` CLI arg parsing in `base_train.py` |
