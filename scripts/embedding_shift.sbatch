#!/bin/bash
#SBATCH --job-name=waiv-embed-shift
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# Renders figures/embedding_shift.{pdf,png}: 2x2 (acquisition/tissue x base/tuned)
# embedding scatter grids, one 2x2 block per backbone (phikon-v2, midnight, Virchow2).
# See scripts/embedding_shift.py docstring for the full method (pseudo-tissue label
# provenance, sampling, projection). GPU needed to embed 10 conditions x 400 tiles x
# 2 models (base + LoRA step 200) x 3 backbones = 24,000 tile forward passes.
#
#   sbatch --account=max --qos=high scripts/embedding_shift.sbatch

set -euo pipefail

REPO="${SPECTRA_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
. "$REPO/scripts/_env.sh"
cd "$REPO"

PY="$REPO/.venv-hest/bin/python"

export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$SPECTRA_HF_HOME}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true

"$PY" scripts/embedding_shift.py

echo "=== done $(date) ==="
