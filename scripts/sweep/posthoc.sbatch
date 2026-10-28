#!/bin/bash
#SBATCH --job-name=q8-posthoc
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --requeue
# usage: SPECTRA_REPO=<repo> sbatch [--account=X --partition=Y] scripts/sweep/posthoc.sbatch \
#            [--steps 125,150] <run_dir> ...
# (re-runnable: already-scored steps are skipped). Same environment as run_one.sbatch.
REPO="${SPECTRA_REPO:?set SPECTRA_REPO}"
PY="${SPECTRA_PY:-$REPO/.venv/bin/python}"
export PYTHONPATH="$REPO/src" PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PY" "$REPO/scripts/sweep/posthoc_eval.py" "$@"
