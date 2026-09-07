#!/bin/bash
#SBATCH --job-name=waiv-retrieval-qual
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# Renders figures/retrieval_examples.{pdf,png}: base vs fine-tuned cross-stain
# retrieval on individual PLISM tiles. See scripts/retrieval_qualitative.py docstring
# for the condition-pair and query-selection rules. GPU only needed to embed 256 tiles
# x 2 conditions x 2 models (phikon-v2, base + LoRA step 200) -- a few minutes.
#
#   sbatch --account=max --qos=high scripts/retrieval_qualitative.sbatch

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

"$PY" scripts/retrieval_qualitative.py

echo "=== done $(date) ==="
