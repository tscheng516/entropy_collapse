#!/usr/bin/env bash
# ViT entropy-collapse training launcher.
#
# Usage:
#   bash train.sh                               # default: ViT-B/16 CIFAR-100
#   bash train.sh config=cifar100_base          # ViT-B/16 CIFAR-100 (named preset)
#   bash train.sh config=imagenet1k_base        # ViT-B/16 ImageNet-1k
#   bash train.sh config=cifar100_large         # ViT-L/16 CIFAR-100
#   bash train.sh config=imagenet1k_large       # ViT-L/16 ImageNet-1k
#   bash train.sh config=cifar100_huge          # ViT-H/14 CIFAR-100
#   bash train.sh config=imagenet1k_huge        # ViT-H/14 ImageNet-1k
#   bash train.sh config=imagenet1k_base --lr 5e-4 max_iters=100000
set -euo pipefail

# ---- GPU selection  -------------------------------------------------------
GPUS="${GPUS:-2,3,4,5}"
# --------------------------------------------------------------------------

IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NPROC=${#GPU_ARRAY[@]}

export CUDA_VISIBLE_DEVICES="$GPUS"

# Prevent DataLoader workers from spawning redundant OpenMP/MKL threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Activate conda
eval "$(conda shell.bash hook)"
conda activate entropy-collapse-vit

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

torchrun \
    --nproc_per_node="$NPROC" \
    "$SCRIPT_DIR/base_train.py" \
    "$@"
