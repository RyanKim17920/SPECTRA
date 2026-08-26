#!/bin/bash
#SBATCH --job-name=waiv-sizing
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
#   sbatch scripts/sizing_probe.sbatch kaiko-ai/midnight "2x32 2x48 2x64 2x96 2x128"
#
# One GPU. The probe is deliberately short (3 steps per shape) -- it measures peak memory
# and step time, not convergence.

set -euo pipefail

REPO="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
cd "$REPO"

BACKBONE="${1:-owkin/phikon-v2}"
SHAPES="${2:-2x64 2x128 2x192}"
CKPT_FLAG="${3:---grad-checkpointing}"

export HF_HOME="${HF_HOME:-/data/huggingface}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
# Fragmentation is what usually decides whether the largest shape fits at all; without
# this the probe reports a ceiling lower than the real one.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

SLUG="$(echo "$BACKBONE" | tr '/' '_')"
srun --unbuffered "$REPO/.venv/bin/python" scripts/sizing_probe.py \
  --backbone "$BACKBONE" --shapes $SHAPES $CKPT_FLAG \
  --out "runs/sizing_${SLUG}_${SLURM_JOB_ID:-local}.json"

echo "=== done $(date) ==="
