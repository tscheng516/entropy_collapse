#!/usr/bin/env bash
# ViT entropy-collapse training launcher.
#
# Usage:
#   bash train.bash                          # default preset (cifar100)
#   bash train.bash config=cifar100_small    # named preset
#   bash train.bash config=imagenet1k_base   # ImageNet-1k ViT-B/16
#   bash train.bash --lr 5e-4 max_iters=2000 # override individual flags
set -euo pipefail

# ---- Configuration  ------------------------------------------------------
GPUS="${GPUS:-2,3,4,5}"
# --------------------------------------------------------------------------

IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NPROC=${#GPU_ARRAY[@]}

export CUDA_VISIBLE_DEVICES="$GPUS"

# Activate conda
eval "$(conda shell.bash hook)"
conda activate entropy-collapse-vit

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

torchrun \
    --nproc_per_node="$NPROC" \
    "$SCRIPT_DIR/base_train.py" \
    "$@"
