# LLM / NanoGPT — Entropy Collapse Experiments

This folder contains entropy-collapse experiments for character-level and
subword-level language models built on top of
[NanoGPT](https://github.com/karpathy/nanoGPT).

Three datasets are supported out of the box:

| Dataset | Tokeniser | Vocab size | Auto-download |
|---|---|---|---|
| `shakespeare_char` | char-level (65 unique chars) | 65 | manual (see below) |
| `fineweb_edu` | GPT-2 BPE (tiktoken) | 50 257 | ✅ HuggingFace |
| `climbmix` | GPT-2 BPE (tiktoken) | 50 257 | ✅ HuggingFace |

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

## Datasets

### Shakespeare (char-level)

Clone NanoGPT and run the preparation script once:

```bash
git clone https://github.com/karpathy/nanoGPT.git LLM/nanoGPT
cd LLM/nanoGPT && python data/shakespeare_char/prepare.py && cd ../..
```

### FineWeb-Edu and ClimbMix (HuggingFace auto-download)

Data is downloaded and tokenised automatically when you run `base_train.py`
with the matching preset.  The encoded `.bin` files are cached in `data/`:

```bash
# FineWeb-Edu (10 BT sample, ~20 GB download the first time)
python LLM/base_train.py config=fineweb_edu

# NVIDIA ClimbMix
python LLM/base_train.py config=climbmix
```

You can control how many tokens are prepared via `max_tokens` and the number
of worker processes via `num_proc`:

```bash
python LLM/base_train.py config=fineweb_edu num_proc=8
```

---

## Tokeniser

`src/tokenizer.py` provides a unified interface:

```python
from src.tokenizer import get_tokenizer

tok = get_tokenizer("shakespeare_char")  # or "fineweb_edu" / "climbmix"
ids = tok["encode"]("Hello, world!")
text = tok["decode"](ids)
vocab_size = tok["vocab_size"]   # 65 for shakespeare, 50257 for the rest
```

---

## Quick Start

### Train with a preset config

```bash
# Shakespeare (default small-model config)
python LLM/base_train.py config=shakespeare

# FineWeb-Edu (12-layer GPT-2 scale)
python LLM/base_train.py config=fineweb_edu

# ClimbMix
python LLM/base_train.py config=climbmix
```

### Common CLI flags

| Short flag | Full name | Default |
|---|---|---|
| `--lr` | `learning_rate` | `1e-3` |
| `--optim` | `optimizer` | `adamw` |
| `--max_it` | `max_iters` | `1000` (shakespeare) / `600000` (fw/cm) |
| `--cp` | `init_from` | `scratch` |
| `--wandb` | `wandb_log` | `false` |
| `--dataset` | `dataset` | (from preset) |

### Override flags from the command line

```bash
python LLM/base_train.py config=shakespeare \
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

## Re-running Plots from a Saved History

After training, `plot_history.py` regenerates all figures and a Markdown
analysis report from a saved `history.pkl` without re-running training:

```bash
python LLM/plot_history.py out/<run_id>/history.pkl
python LLM/plot_history.py out/<run_id>/history.pkl -o reanalysis/ --lam 20
```

---

## Reproducing the Paper Experiments

### Experiment A — Smooth dynamics (Spearman correlation)

```bash
python LLM/base_train.py config=shakespeare \
    --optim adamw --lr 1e-5 --max_it 800 \
    hessian_freq=3 entropy_freq=10 \
    --wandb true wandb_run_name=exp-A-adamw

python LLM/base_train.py config=shakespeare \
    --optim sgd --lr 0.002 --max_it 800 \
    hessian_freq=3 entropy_freq=10 \
    --wandb true wandb_run_name=exp-A-sgd
```

### Experiment B — Large-LR instability (spike co-occurrence)

```bash
python LLM/base_train.py config=shakespeare \
    --optim adamw --lr 5e-3 --max_it 100 \
    hessian_freq=1 entropy_freq=1 \
    --wandb true wandb_run_name=exp-B-adamw

python LLM/base_train.py config=shakespeare \
    --optim sgd --lr 0.5 --max_it 100 \
    hessian_freq=1 entropy_freq=1 \
    --wandb true wandb_run_name=exp-B-sgd
```

---

## Smoke Test

```bash
# Shakespeare (no download required after nanoGPT setup)
python LLM/base_train.py config=shakespeare --max_it 2 hessian_freq=1 entropy_freq=1
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
