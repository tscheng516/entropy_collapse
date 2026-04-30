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

# -----------------------------------------------------------------------------
# 1. GPU selection
# -----------------------------------------------------------------------------
GPUS="${GPUS:-2,3,4,5}"
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NPROC=${#GPU_ARRAY[@]}
export CUDA_VISIBLE_DEVICES="$GPUS"


# -----------------------------------------------------------------------------
# 2. Environment setup
# -----------------------------------------------------------------------------
# Paths 
ROOT="$(pwd)" 
KEYS_DIR="$ROOT/../../.keys"
export WANDB_API_KEY="$(cat "$KEYS_DIR/wandb_api_key")"

eval "$(conda shell.bash hook)"
conda activate entropy-vit

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# -----------------------------------------------------------------------------
# 3. Script
# -----------------------------------------------------------------------------
torchrun --nproc_per_node="$NPROC" base_train.py \