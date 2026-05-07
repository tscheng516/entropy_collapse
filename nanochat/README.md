# nanochat — Entropy Collapse Experiments

Entropy-collapse analysis for a GPT-style language model (nanochat).

---

## Quick Setup

### 1. Clone the nanochat repo at the pinned SHA

```bash
# From the repo root
git clone https://github.com/karpathy/nanochat nanochat/nanochat_repo
cd nanochat/nanochat_repo
git checkout 0aaca56
```

### 2. Install dependencies

```bash
cd nanochat/nanochat_repo
conda create -n entropy-nanochat python=3.11 -y && conda activate entropy-nanochat
# change to cu121 for newer CUDA
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
# Patch pyproject.toml so setuptools only finds the nanochat package
echo -e '\n[tool.setuptools.packages.find]\ninclude = ["nanochat*"]' >> pyproject.toml
pip install -e ".[gpu]"
pip install -r ../../nanochat/requirements.txt
```

### 3. Download ClimbMix data shards

```bash
python -m nanochat.dataset -n 170
```

~150 shards are needed for pretraining; 170 adds a small buffer.

### 4. Train the BPE tokenizer (once)

```bash
python -m scripts.tok_train
```

The tokenizer is saved to `tokenizer/tokenizer.pkl` (~5 min on a modern CPU).

---

### 5. Train

For a quick pilot run use the smaller d6 model:

```bash
python base_train.py 
```


---

