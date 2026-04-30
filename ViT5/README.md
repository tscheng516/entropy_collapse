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

| `nvidia-smi` CUDA Version | Recommended wheel | Why |
|---|---|---|
| 11.8 | `cu118` | Native build; fully self-contained |
| 12.x – 13.x (incl. 13.0, RTX 4090) | **`cu118`** (recommended) | `cu121` exists but requires `libnvJitLink.so.12` as an external system lib that is often missing; `cu118` bundles everything and runs on any CUDA ≥ 11.8 driver |

> **RTX 4090 + CUDA 13.0**: use the `cu118` wheel.  RTX 4090 (sm_89) is
> supported from CUDA 11.8, and a CUDA 13.0 driver runs `cu118` compiled
> code without restriction.

> **`cu121` wheel caveat**: `torch==2.4.1+cu121` depends on
> `libnvJitLink.so.12` at runtime but does not bundle it.  On servers without
> a full CUDA 12 toolkit installed system-wide this causes
> `ImportError: libnvJitLink.so.12: cannot open shared object file`.
> The `cu118` wheel has no such external dependency.

> **`torch==2.4.1` wheel availability**: only `cu118` and `cu121` exist.
> Requesting `cu124`, `cu126`, or `cu130` raises
> `ERROR: Could not find a version that satisfies the requirement torch==2.4.1`.

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

> **CUDA variants** — `requirements.txt` pins `torch==2.4.1`.  **Always
> install PyTorch first** with the correct wheel, then run
> `pip install -r requirements.txt`.
>
> ```bash
> # Recommended for all CUDA drivers (11.8 through 13.x, RTX 4090).
> # cu118 is fully self-contained — no external CUDA libs required.
> pip install torch==2.4.1 torchvision==0.19.1 \
>     --index-url https://download.pytorch.org/whl/cu118
> pip install -r requirements.txt
>
> # cu121 — only if you are certain libnvJitLink.so.12 is present system-wide
> # (see Troubleshooting).  Not recommended for CUDA 13.x / RTX 4090.
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

Expected output (cu118 wheel on any CUDA 12.x / 13.x driver):
```
PyTorch: 2.4.1+cu118 | CUDA build: 11.8 | GPU available: True
```

### 4. (Optional) Flash Attention

For faster training on Ampere+ GPUs:

```bash
pip install flash-attn --no-build-isolation
```

The default config runs with `flash=False` to keep Hessian computation stable.

### 5. Train (default: ViT-5-Base on CIFAR-100)

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

This is caused by the `cu121` wheel, which depends on `libnvJitLink.so.12`
as an external system lib that is frequently absent on servers.

**Permanent fix — reinstall with the `cu118` wheel** (fully self-contained,
works on any CUDA ≥ 11.8 driver including 13.0, RTX 4090):

```bash
conda activate entropy-vit5
pip install torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Verify
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.4.1+cu118 True
```

If you cannot reinstall torch, try these workarounds in order:

**1. Check if the library exists on the server**:
```bash
find /usr/local/cuda* /usr/lib/x86_64-linux-gnu -name "libnvJitLink.so.12" 2>/dev/null
# If found:
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH  # adjust path
```

**2. Conda**:
```bash
conda install -y -c nvidia cuda-nvjitlink
```

**3. pip via NVIDIA's index** (standard PyPI may return "no matching distribution"):
```bash
pip install nvidia-cuda-nvjitlink-cu12 --extra-index-url https://pypi.nvidia.com
```

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
