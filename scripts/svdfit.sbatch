#!/bin/bash
#SBATCH --job-name=waiv-svdfit
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail
# Runs from the svd-nuisance worktree (which holds the scripts) but borrows the main
# checkout's venv and HF cache -- .venv/ and third_party/ are untracked and exist only there.
WT="${WAIV_WT:-/admin/home/ryan.kim/waiv_worktrees/svd-nuisance}"
MAIN="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
cd "$WT"
export HF_HOME="${HF_HOME:-/data/huggingface}"
export HF_HUB_OFFLINE=0
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$WT/src"
PY="$MAIN/.venv/bin/python"
BACKBONE=paige-ai/Virchow2
OUT="$WT/runs/svd_nuisance_virchow2_clsmean_s0.npz"

echo "=== SMOKE (4 conditions x 64 tiles) ==="
"$PY" scripts/fit_svd_nuisance.py \
  --backbone "$BACKBONE" --pooling clsmean \
  --n-tiles 64 --limit-conditions 4 --seed 0 \
  --batch-size 64 --amp bfloat16 \
  --packed-dir /data/plism/repacked \
  --out "$WT/runs/_smoke_svd.npz"

echo "=== FULL (91 conditions x 2000 tiles) ==="
"$PY" scripts/fit_svd_nuisance.py \
  --backbone "$BACKBONE" --pooling clsmean \
  --n-tiles 2000 --seed 0 \
  --batch-size 128 --amp bfloat16 \
  --packed-dir /data/plism/repacked \
  --out "$OUT"
echo "=== done rc=$? ==="
ls -la "$OUT"
