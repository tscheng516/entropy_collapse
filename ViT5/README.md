# ViT-5 — Entropy Collapse Experiments

Entropy-collapse analysis for **ViT-5-Base** (Wang et al., 2025 —
[arXiv:2602.08071](https://arxiv.org/abs/2602.08071)), extending the `ViT/`
experiments with RMSNorm, per-head QK-normalisation, 2-D RoPE, register tokens,
and layer-scale.  Only the Base variant (~87 M parameters) is included here.

---

## Quick Setup

### 1. Check your CUDA version (servers)

Before installing PyTorch on a GPU server, confirm which CUDA version the
driver exposes.  The driver's reported CUDA version is the **maximum** toolkit
version it supports; choose a PyTorch wheel compiled against that version or
earlier.

```bash
# Most reliable — shows driver version and max supported CUDA
nvidia-smi

# Shows the installed CUDA toolkit version (may differ from the driver's max)
nvcc --version

# After PyTorch is installed, confirm it sees the GPU
python -c "import torch; print('PyTorch:', torch.__version__, \
    '| CUDA build:', torch.version.cuda, \
    '| GPU available:', torch.cuda.is_available())"
```

Match the `nvidia-smi` CUDA version to the correct wheel tag **for `torch==2.4.1`**:

| `nvidia-smi` CUDA Version | Wheel tag to use | Why |
|---|---|---|
| 11.8 | `cu118` | Native build |
| 12.1 – 13.x (incl. 13.0) | `cu121` | `torch==2.4.1` has **no** `cu124`/`cu126`/`cu130` wheels; use `cu121` — CUDA drivers are backward compatible |

> **`torch==2.4.1` wheel availability**: this release only ships `cu118` and
> `cu121` builds.  Attempting `--index-url .../cu124`, `cu126`, or `cu130`
> will produce `ERROR: Could not find a version that satisfies the requirement
> torch==2.4.1`.  Always use `cu121` for any CUDA 12.x or 13.x driver.

> If `nvidia-smi` is not found, the server may have no GPU or the NVIDIA driver
> is not installed — use the CPU-only PyTorch wheel (no `--index-url` needed).

---

### 2. Install dependencies

```bash
# venv
cd ViT5/
python3.10 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

```bash
# Conda
cd ViT5/
conda create -n entropy-vit5 python=3.10 -y && conda activate entropy-vit5
pip install -r requirements.txt
```

> **CUDA variants** — `requirements.txt` pins `torch==2.4.1`, which only has
> `cu118` and `cu121` wheels.  There are **no** `cu124`, `cu126`, or `cu130`
> builds for this version — requesting them will raise a "Could not find a
> version" error.  For any CUDA 12.x or 13.x driver, use the `cu121` wheel:
> CUDA drivers are backward compatible, so a CUDA 13.0 driver runs `cu121`
> compiled code without issue.
>
> ```bash
> # CUDA 11.8
> pip install torch==2.4.1 torchvision==0.19.1 \
>     --index-url https://download.pytorch.org/whl/cu118
> pip install -r requirements.txt
>
> # CUDA 12.1 / 12.4 / 12.6 / 13.x  — all use cu121
> pip install torch==2.4.1 torchvision==0.19.1 \
>     --index-url https://download.pytorch.org/whl/cu121
> pip install -r requirements.txt
> ```
>
> `pip install -r requirements.txt` is safe to run after the CUDA-specific
> torch install: pip sees `torch==2.4.1` is already satisfied and skips it,
> so the CUDA build is not overwritten.  Do **not** pass `--no-deps` here —
> that would suppress the transitive dependencies of `timm`, `einops`,
> `datasets`, and `wandb`.

> **`timm==0.4.12` + Python 3.10 compatibility** — Python 3.10 removed
> `collections.Callable` (use `collections.abc.Callable` instead).
> timm 0.4.12 triggers this in some code paths.  If you hit a
> `AttributeError: module 'collections' has no attribute 'Callable'` error,
> apply the one-line patch before importing timm:
> ```python
> import collections, collections.abc
> collections.Callable = collections.abc.Callable      # timm 0.4.12 shim
> import timm
> ```
> or run:
> ```bash
> python -c "import collections, collections.abc; \
>     collections.Callable = collections.abc.Callable; import timm" \
>     && echo "timm OK"
> ```

> **Why timm 0.4.12?**  ViT-5's model code imports
> `timm.models.vision_transformer.{Mlp,PatchEmbed}` and
> `timm.models.layers.{DropPath,trunc_normal_}`, which were removed in later
> timm versions.

### 3. Verify the install

```bash
python -c "import torch; print('PyTorch:', torch.__version__, \
    '| CUDA build:', torch.version.cuda, \
    '| GPU available:', torch.cuda.is_available())"
```

Expected output (CUDA 12.x / 13.x):
```
PyTorch: 2.4.1+cu121 | CUDA build: 12.1 | GPU available: True
```

### 4. (Optional) Flash Attention

For faster training on Ampere+ GPUs:

```bash
pip install flash-attn --no-build-isolation
```

The default config runs with `flash=False` to keep Hessian computation stable.

### 4. Train (default: ViT-5-Base on CIFAR-100)

```bash
python base_train.py
```

CIFAR-100 is downloaded automatically on first run.

---

## Config Presets

Select a preset with `config=<name>` on the command line.
Each preset is a `@dataclass` defined in `configs/train_config.py`.

| Preset | Dataset | img\_size | Seq len | Batch | LR | Max Iters |
|---|---|---|---|---|---|---|
| `default` | CIFAR-100 | 32 | 69 (64 + CLS + 4 reg) | 32 | 3e-3 | 200 |
| `cifar100_base` | CIFAR-100 | 32 | 69 | 256 | 3e-3 | 50 000 |
| `imagenet1k_base` | ImageNet-1k | 192 | 149 (144 + CLS + 4 reg) | 256 | 3e-3 | 50 000 |

Sequence length = patches + CLS token + 4 register tokens.

```bash
python base_train.py config=imagenet1k_base
bash  train.sh        config=cifar100_base   # multi-GPU via torchrun
```

---

## Advanced Experiments

### Override individual flags

Any `TrainConfig` field can be overridden on the command line after the preset:

```bash
python base_train.py config=cifar100_base \
    learning_rate=1e-3 \
    max_iters=50000 \
    hessian_intv=500 \
    entropy_intv=500 \
    wandb_log=true \
    wandb_run_name=vit5-base-cifar100-run1
```

### Multi-GPU training

`train.sh` wraps `torchrun`. Set `GPUS` to select devices:

```bash
GPUS=0,1,2,3 bash train.sh config=cifar100_base
```

### ImageNet-1k via Hugging Face

When `data_dir` does not contain `train/` and `val/` sub-directories the
dataset is downloaded from Hugging Face automatically.

```bash
# Accept the licence at https://huggingface.co/datasets/imagenet-1k first.
export HF_TOKEN=hf_...
python base_train.py config=imagenet1k_base data_dir=data/imagenet1k
```

Subsequent runs reuse the local cache.

### Local ImageNet in ImageFolder layout

```bash
python base_train.py config=imagenet1k_base data_dir=/path/to/imagenet
```

---

## Post-Training Analysis

`plot_history.py` replays a saved `history.pkl` and writes all figures plus
a structured analysis report.

```bash
python plot_history.py outputs/cifar100_base/history.pkl
```

Outputs written next to the pickle:

| File | Contents |
|---|---|
| `*_curvature_smoothed_comparison.png` | Smoothed proxy traces (λ = 10) |
| `*_training_dynamics.png` | Loss, accuracy, LR schedule |
| `analysis.txt` | Full correlation report (plain text) |
| `analysis.md` | Markdown tables: raw/smoothed correlations, spike co-occurrence |

Override the smoothing strength: `python plot_history.py history.pkl --lam 20`

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

## Troubleshooting

### `ImportError: libnvJitLink.so.12: cannot open shared object file`

This happens with `torch==2.4.1+cu121` on servers where the full CUDA 12
toolkit is not installed system-wide.  The PyTorch wheel does not bundle
`libnvJitLink.so.12`, and it is not pulled in automatically.

Try the fixes below in order:

**1. Check if the library already exists on the system** (most servers have it):
```bash
find /usr/local/cuda* /usr/lib/x86_64-linux-gnu -name "libnvJitLink.so.12" 2>/dev/null
ldconfig -p | grep nvJitLink
```
If found, add its directory to `LD_LIBRARY_PATH`:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH  # adjust path
# Add to ~/.bashrc or the top of train.sh to make it permanent
```

**2. Conda (recommended if using a conda env)**:
```bash
conda install -y -c nvidia cuda-nvjitlink
```

**3. pip via NVIDIA's package index** (standard PyPI may not carry this package on all platforms):
```bash
pip install nvidia-cuda-nvjitlink-cu12 \
    --extra-index-url https://pypi.nvidia.com
```

**4. Switch to the cu118 wheel** (avoids the issue entirely — cu118 does not
need `libnvJitLink`).  Only viable if your driver supports CUDA ≥ 11.8:
```bash
pip install torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu118
```

> Note: this error does **not** occur with the `cu118` wheel.  If none of the
> above fixes work on your server, switching to `cu118` is the most portable
> option — the CUDA 11.8 runtime is fully self-contained in the wheel.

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
