#!/bin/bash
#SBATCH --job-name=waiv-core-map
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail
REPO="${SPECTRA_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
. "$REPO/scripts/_env.sh"
cd "$REPO"
export HF_HOME="${HF_HOME:-$SPECTRA_HF_HOME}"
export HF_HUB_OFFLINE=0
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
PY="$REPO/.venv/bin/python"
"$PY" scripts/derive_core_map_v2.py \
  --packed-dir $SPECTRA_PLISM_PACKED \
  --out-labels runs/.plism_core_labels.npy \
  --batch-size 128
echo "=== done $? ==="
