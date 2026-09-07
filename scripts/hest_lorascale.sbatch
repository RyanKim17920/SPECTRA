#!/bin/bash
#SBATCH --job-name=waiv-hestls
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail
# WiSE-FT interpolation at eval: scaling the LoRA delta by s IS
# theta = (1-s)*theta_base + s*theta_ft, exactly, with no retraining.
# Tests whether an intermediate s is Pareto-better than s=1 (confounder-insensitivity
# saturates early while prediction_performance erodes; if those curves differ in
# curvature vs update MAGNITUDE, some s<1 wins).
#
# NOTE: Virchow2 adapters have base_model_name_or_path=null, so --backbone is MANDATORY
# or phikon-v2 is silently built and the adapter keys fail to load.
REPO="${SPECTRA_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
. "$REPO/scripts/_env.sh"
cd "$REPO"
export HF_HOME="${HF_HOME:-$SPECTRA_HF_HOME}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
ADAPTER="${SPECTRA_ADAPTER:?set SPECTRA_ADAPTER (checkpoint dir)}"
S="${SPECTRA_S:?set SPECTRA_S (lora scale)}"
BACKBONE="${SPECTRA_BACKBONE:?set SPECTRA_BACKBONE}"
POOL="${SPECTRA_POOL:?set SPECTRA_POOL (cls|clsmean)}"
BASETAG="${SPECTRA_TAG:?set SPECTRA_TAG (exp-code stem)}"
TAG=$("$REPO/.venv/bin/python" -c "print(f'ls{int(round(float('$S')*100)):03d}')")
EXP="${BASETAG}_${TAG}"
echo "=== $(date) | adapter=$ADAPTER s=$S pool=$POOL exp=$EXP ==="
"$REPO/.venv-hest/bin/python" scripts/run_hest.py \
  --backbone "$BACKBONE" --pooling "$POOL" \
  --lora-rank 32 --lora-alpha 64 \
  --adapter "$ADAPTER" \
  --lora-scale "$S" \
  --num-workers 0 \
  --exp-code "$EXP"
echo "=== done rc=$? exp=$EXP ==="
