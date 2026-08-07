#!/bin/bash
#SBATCH --job-name=waiv-pathorob-extract
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# PathoROB feature extraction (PLAN.md §4 phase 5 -- the Phase-2 gate).
#
#   sbatch scripts/extract_pathorob.sbatch phikonv2_clsmean_ours camelyon
#   sbatch scripts/extract_pathorob.sbatch phikonv2_clsmean_ours "camelyon tcga tolkach_esca"
#   sbatch scripts/extract_pathorob.sbatch waiv_step5000 camelyon /path/to/ckpt.pt
#
# One GPU is plenty: the whole benchmark is 151k patches of ViT-L/16 inference. The login
# node has no GPU, so this must go through SLURM.

set -euo pipefail

REPO="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
cd "$REPO"

MODEL_NAME="${1:-phikonv2_clsmean_ours}"
DATASETS="${2:-camelyon}"
CKPT="${3:-}"
ADAPTER="${4:-}"

export HF_HOME="${HF_HOME:-/data/ryan.kim/hf_home}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true

for DS in $DATASETS; do
  echo "--- $DS ---"
  ARGS=(--dataset "$DS" --model-name "$MODEL_NAME" --batch-size "${BATCH_SIZE:-256}" --num-workers 16)
  # Backbone is a parameter now. Unset => owkin/phikon-v2 (the Phase-2 gate model);
  # WAIV_BACKBONE=kaiko-ai/midnight for the ViT-g row. --pooling likewise.
  if [ -n "${WAIV_BACKBONE:-}" ]; then ARGS+=(--backbone "$WAIV_BACKBONE"); fi
  if [ -n "${WAIV_POOLING:-}" ]; then ARGS+=(--pooling "$WAIV_POOLING"); fi
  if [ -n "$CKPT" ]; then ARGS+=(--checkpoint "$CKPT"); fi
  if [ -n "$ADAPTER" ]; then
    ARGS+=(--adapter "$ADAPTER" --lora-rank "${LORA_RANK:-8}" --lora-alpha "${LORA_ALPHA:-32}" --proj-out-dim "${PROJ_OUT_DIM:-512}")
  fi
  # fp32 on purpose: the gate is a numeric comparison against PathoROB's committed
  # reference, so we do not want autocast rounding in the one run that validates the
  # harness. Add `--amp bfloat16` for throughput once the gate has passed.
  srun --unbuffered "$REPO/.venv/bin/python" scripts/extract_pathorob_features.py "${ARGS[@]}"
done

echo "=== done $(date) ==="
