#!/bin/bash
#SBATCH --job-name=waiv-svdfit-seed
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail
# Seed replicate of the svd-nuisance fit. SEED comes in via --export; no smoke step,
# the layout is already proven by job 389740. A different seed redraws WHICH 2000 of the
# 16278 tiles are used, so re-running the k-sweep on these tests whether the k-curve
# shape is a property of the encoder or of the particular tile draw.
WT="${WAIV_WT:-/admin/home/ryan.kim/waiv_worktrees/svd-nuisance}"
MAIN="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
cd "$WT"
export HF_HOME="${HF_HOME:-/data/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$WT/src"
"$MAIN/.venv/bin/python" scripts/fit_svd_nuisance.py \
  --backbone paige-ai/Virchow2 --pooling clsmean \
  --n-tiles 2000 --seed "$SEED" \
  --batch-size 128 --amp bfloat16 \
  --packed-dir /data/plism/repacked \
  --out "$WT/runs/svd_nuisance_virchow2_clsmean_s${SEED}.npz"
echo "=== done rc=$? seed=$SEED ==="
