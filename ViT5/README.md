# ViT-5 Entropy-Collapse Experiments

Entropy-collapse experiments using **ViT-5** (Wang et al., 2025 —
[arXiv:2602.08071](https://arxiv.org/abs/2602.08071)), adapted from the `ViT/`
folder in this repository.

ViT-5-Base uses RMSNorm, per-head QK-normalisation, 2-D RoPE, register tokens,
and layer-scale — all enabled by default.  Only the Base variant (87 M
parameters) is included here.

---

## Installation

### 1. Install PyTorch

Follow [pytorch.org](https://pytorch.org/get-started/locally/) for your CUDA
version.  The codebase is tested with PyTorch 2.4.1:

```bash
pip install torch==2.4.1 torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Install ViT-5 dependencies

ViT-5 requires **timm 0.4.12** (older API) and **einops**:

```bash
pip install timm==0.4.12 einops
```

> **Why timm 0.4.12?**  ViT-5's model code imports
> `timm.models.vision_transformer.{Mlp,PatchEmbed}` and
> `timm.models.layers.{DropPath,trunc_normal_}`, which were removed in later
> timm versions.

### 3. (Optional) Install Flash Attention

For faster training on Ampere+ GPUs:

```bash
pip install flash-attn --no-build-isolation
```

The default ViT-5-Base config in this repo runs with `flash=False` (standard
PyTorch attention) to keep Hessian computation stable.

### 4. Install the rest of the requirements

```bash
pip install -r ViT5/requirements.txt
```

---

## Training

### CIFAR-100 (default)

```bash
python ViT5/base_train.py config=cifar100_base
```

### ImageNet-1k

```bash
python ViT5/base_train.py config=imagenet1k_base data_dir=/path/to/imagenet
```

ImageNet should be in `ImageFolder` layout (`train/` and `val/` sub-folders).
HuggingFace streaming (`imagenet_hf`) is also supported if a local copy is not
available — see `ViT5/src/data_utils.py`.

### Multi-GPU (torchrun)

```bash
bash ViT5/train.sh config=imagenet1k_base data_dir=/data/imagenet
# or set GPUS explicitly:
GPUS=0,1,2,3 bash ViT5/train.sh config=cifar100_base
```

### Override individual fields

```bash
python ViT5/base_train.py config=cifar100_base --lr 1e-3 max_iters=20000
```

---

## Config presets

| Key                 | Dataset      | img_size | patch | Sequence length | Notes                       |
|---------------------|-------------|----------|-------|-----------------|-----------------------------|
| `default`           | CIFAR-100   | 32       | 4     | 69 (64+1+4)     | Base TrainConfig            |
| `cifar100_base`     | CIFAR-100   | 32       | 4     | 69              | 50 k iters, lr=3e-3         |
| `imagenet1k_base`   | ImageNet-1k | 192      | 16    | 149 (144+1+4)   | 50 k iters, lr=3e-3         |

Sequence length = patches + CLS token + 4 register tokens.

---

## Architecture — ViT-5-Base

| Parameter       | Value         |
|-----------------|---------------|
| `embed_dim`     | 768           |
| `depth`         | 12            |
| `num_heads`     | 12            |
| `mlp_ratio`     | 4             |
| `patch_size`    | 16 (ImageNet) / 4 (CIFAR) |
| `qkv_bias`      | False         |
| Norm            | RMSNorm       |
| QK-norm         | Yes (per-head) |
| Position embed  | APE + 2-D RoPE |
| Registers       | 4 (√4 = 2 × 2 grid) |
| Layer-scale     | γ₁/γ₂, init=1e-4 |
| Parameters      | ~87 M         |

---

## Differences from `ViT/`

| Aspect             | `ViT/`                          | `ViT5/`                          |
|--------------------|----------------------------------|-----------------------------------|
| Model              | timm ViT (various sizes)        | ViT-5-Base (fixed)                |
| Norm               | LayerNorm                        | RMSNorm                           |
| QK-norm            | Optional                         | Always on (ViT-5 default)         |
| Position encoding  | APE only                         | APE + 2-D RoPE                    |
| Register tokens    | None                             | 4 (must be a perfect square)      |
| Layer-scale        | None                             | γ₁/γ₂ (init 1e-4)                |
| Default lr         | 1e-3                             | 3e-3 (ViT-5-Base paper)           |
| Default img_size   | 32 / 224                         | 32 / 192                          |
| Attention caching  | Monkey-patched                   | Built-in (`_cache_attn` flag)     |
| Data dir           | `./data`                         | `./data` (shared cache)           |

---

## Shared data cache

Both `ViT/` and `ViT5/` use `data_dir="./data"` as the default root for
torchvision downloads.  When both training scripts are run from the repository
root, they share the same CIFAR-100 download without duplicating it.

---

## Citation

```bibtex
@article{wang2025vit5,
  title   = {ViT-5: Vision Transformer with 5 Innovations},
  author  = {Wang, Fei and others},
  journal = {arXiv preprint arXiv:2602.08071},
  year    = {2025},
}
```

Official repo: <https://github.com/wangf3014/ViT-5>
