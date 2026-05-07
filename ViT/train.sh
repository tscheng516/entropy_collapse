#!/usr/bin/env bash
# ViT entropy-collapse training launcher.
#
# Usage:
#   bash train.sh                               # default: ViT-B/16 CIFAR-100 pilot run
#   bash train.sh config=cifar100_base          # ViT-B/16 CIFAR-100 (named preset)
#   bash train.sh config=imagenet1k_base max_iters=100000 # config override 
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
# KEYS_DIR="$ROOT/../../.keys"
# export WANDB_API_KEY="$(cat "$KEYS_DIR/wandb_api_key")"

eval "$(conda shell.bash hook)"
conda activate entropy-vit

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# -----------------------------------------------------------------------------
# 3. Script
# -----------------------------------------------------------------------------
torchrun --nproc_per_node="$NPROC" base_train.py \
        # config=cifar100_base \
        # temp_shift_step=15000 \