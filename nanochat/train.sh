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

# ---------------------------------------------------------------------------
# 1. GPU selection
# ---------------------------------------------------------------------------
GPUS="${GPUS:-0,1,2,3}"
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NPROC=${#GPU_ARRAY[@]}
export CUDA_VISIBLE_DEVICES="$GPUS"

# ---------------------------------------------------------------------------
# 2. Repository root — resolve from this script's location
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------------------------------------------------------------------------
# 3. Environment setup
# ---------------------------------------------------------------------------
# W&B API key (optional — training proceeds without W&B if omitted)
KEYS_DIR="${REPO_ROOT}/../.keys"
if [[ -f "${KEYS_DIR}/wandb_api_key" ]]; then
    export WANDB_API_KEY="$(cat "${KEYS_DIR}/wandb_api_key")"
fi

# Activate the nanochat uv environment.
# The venv is created by ``uv sync --extra gpu`` inside the nanochat clone.
NANOCHAT_DIR="${NANOCHAT_DIR:-${REPO_ROOT}/nanochat/nanochat_repo}"
NANOCHAT_VENV="${NANOCHAT_DIR}/.venv"

if [[ -f "${NANOCHAT_VENV}/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "${NANOCHAT_VENV}/bin/activate"
    echo "[env] activated uv venv: ${NANOCHAT_VENV}"
else
    echo "[warn] uv venv not found at ${NANOCHAT_VENV}."
    echo "       Run:  cd ${NANOCHAT_DIR} && uv sync --extra gpu"
    exit 1
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------------------------------------------------------------------------
# 4. Launch
# ---------------------------------------------------------------------------
cd "${REPO_ROOT}"

torchrun --nproc_per_node="$NPROC" nanochat/base_train.py \
    nanochat_dir="${NANOCHAT_DIR}" \
    "$@"
    # Uncomment and adapt as needed:
    # config=d12 \
    # max_iters=10000 \
    # --lr 3e-4 \
    # --bs 8 \
    # temp_shift_step=5000 \
    # temp_shift_factor=0.25 \
