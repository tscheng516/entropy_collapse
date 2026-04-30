#!/usr/bin/env bash
# ViT-5 entropy-collapse training launcher.
#
# Usage:
#   bash train.sh                               # default: ViT-5-Base CIFAR-100
#   bash train.sh config=cifar100_base          # ViT-5-Base CIFAR-100
#   bash train.sh config=imagenet1k_base        # ViT-5-Base ImageNet-1k
#   bash train.sh config=imagenet1k_base data_dir=/data/imagenet
#   bash train.sh config=cifar100_base --lr 1e-3 max_iters=100000
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
conda activate entropy-vit5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

torchrun \
    --nproc_per_node="$NPROC" \
    "$SCRIPT_DIR/base_train.py" \
    "$@"
