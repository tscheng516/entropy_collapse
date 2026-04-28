#!/usr/bin/env bash
# depth/ entropy-collapse training launcher.
#
# Usage:
#   bash train.sh                         # default: ViT-B/16 NYU Depth V2
#   bash train.sh config=nyudepth_base    # ViT-B/16 NYU Depth V2 (named preset)
#   bash train.sh config=nyudepth_large   # ViT-L/16 NYU Depth V2
#   bash train.sh config=nyudepth_base --lr 2e-4 max_iters=100000
set -euo pipefail

# ---- GPU selection  -------------------------------------------------------
GPUS="${GPUS:-0,1}"
# --------------------------------------------------------------------------

IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NPROC=${#GPU_ARRAY[@]}

export CUDA_VISIBLE_DEVICES="$GPUS"

# Prevent DataLoader workers from spawning redundant OpenMP/MKL threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Activate conda (adjust environment name if needed).
eval "$(conda shell.bash hook)"
conda activate entropy-collapse-depth

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

torchrun \
    --nproc_per_node="$NPROC" \
    "$SCRIPT_DIR/base_train.py" \
    "$@"
