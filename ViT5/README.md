# ViT5 — Entropy Collapse Experiments (ViT-5)

Entropy-collapse analysis for **ViT-5** (Wang et al., 2026), a modernised
Vision Transformer backbone.

Paper: https://arxiv.org/abs/2602.08071
Official implementation: https://github.com/wangf3014/ViT-5

## Architecture

ViT-5 upgrades the canonical ViT backbone with, all active by default:

* **RMSNorm** instead of LayerNorm.
* **2-D Rotary Position Embedding (RoPE)** on patch tokens (`theta=10000`),
  plus a separate low-theta RoPE (`theta=100`) for register tokens.
* **Register tokens** (4 by default, appended after the patch sequence).
* **QK-normalisation** — per-head RMSNorm on queries and keys.
* **Layer-scale** — learnable per-channel residual scaling (init `1e-4`).
* Standard Transformer `Mlp` (GELU, `mlp_ratio=4`), no QKV bias.
* Absolute position embedding (APE) kept alongside RoPE, as in the official
  implementation.

Three sizes are available (see `src/model.py`'s `MODEL_SIZES`):

| `model_name`  | embed_dim | depth | num_heads | params (approx.) |
|---------------|-----------|-------|-----------|-------------------|
| `vit5_small`  | 384       | 12    | 6         | ~22M              |
| `vit5_base`   | 768       | 12    | 12        | ~87M              |
| `vit5_large`  | 1024      | 24    | 16        | ~304M             |

### Deviations from the official (ImageNet-pretraining) repo

Required by the entropy-collapse framework, mirroring `ViT/`'s conventions:

* **No flash-attention** — the explicit softmax path is always used, since it
  is required for attention-matrix caching and second-order gradient
  computation (Hessian / curvature metrics).
* **Attention caching**: `block.attn._cache_attn = True` populates
  `block.attn.last_att` after a forward pass, read by
  `common.helpers.get_attention_entropy()` / `get_attention_similarity()`.
* **Runtime attention temperature**: `set_attention_temperature(model, T)`
  scales attention logits for entropy-collapse intervention experiments.

---

## Quick Setup

### 1. Install dependencies

```bash
# Conda
cd ViT5/
conda create -n entropy-vit5 python=3.10 -y && conda activate entropy-vit5
# change to cu121 for newer CUDA
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 2. Train (default: ViT-5-Base on CIFAR-100)

```bash
python base_train.py
```

CIFAR-100 is downloaded automatically on first run.

Named presets:

```bash
python base_train.py config=cifar100_small
python base_train.py config=cifar100_base
python base_train.py config=cifar100_large

python base_train.py config=imagenet1k_small
python base_train.py config=imagenet1k_base
python base_train.py config=imagenet1k_large
```

Multi-GPU via `torchrun`:

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

## Citation

```bibtex
@article{wang2026vit5,
  title={ViT-5: Vision Transformers for The Mid-2020s},
  author={Wang, Feng and Ren, Sucheng and Zhang, Tiezheng and Neskovic, Predrag and Bhattad, Anand and Xie, Cihang and Yuille, Alan},
  journal={arXiv preprint arXiv:2602.08071},
  year={2026}
}
```
