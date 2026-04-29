#!/usr/bin/env bash
# nanochat entropy-collapse training launcher.
#
# Usage:
#   bash nanochat/train.sh                          # default: d12 (~85 M)
#   bash nanochat/train.sh config=d8                # 8-layer pilot (~30 M)
#   bash nanochat/train.sh config=d24               # GPT-2 scale (~350 M)
#   bash nanochat/train.sh config=d12 max_iters=20000 --lr 2e-4
#
# Environment:
#   GPUS          Comma-separated GPU indices  (default: "0,1,2,3")
#   WANDB_KEY     Path to file containing W&B API key (optional)
#   NANOCHAT_DIR  Absolute path to the nanochat repo clone
#                 (default: <repo_root>/nanochat/nanochat_repo)
set -euo pipefail


# -----------------------------------------------------------------------------
# 2. GPU selection
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
