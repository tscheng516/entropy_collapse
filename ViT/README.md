# ViT — Entropy Collapse Experiments

Entropy-collapse analysis for Vision Transformers (ViT).

---

## Quick Setup

### 1. Install dependencies


```bash
# Conda
cd ViT/
conda create -n entropy-vit python=3.10 -y && conda activate entropy-vit
# change to cu121 for newer CUDA
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```


### 2. Train (default: ViT-B/16 on CIFAR-100)

```bash
python base_train.py
```

CIFAR-100 is downloaded automatically on first run.

For the base case, override the config:

```bash
torchrun --nproc_per_node=2 base_train.py \
        config=cifar100_base \
        temp_shift_step=15000 
```


---

### ImageNet-1k via Hugging Face

When `data_dir` does not contain `train/` and `val/` sub-directories the
dataset is downloaded from Hugging Face automatically.

```bash
# Accept the licence at https://huggingface.co/datasets/imagenet-1k first.
export HF_TOKEN=hf_...
python base_train.py config=imagenet1k_base 
```

Subsequent runs reuse the local cache.


---
