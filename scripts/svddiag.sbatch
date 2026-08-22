#!/bin/bash
#SBATCH --job-name=waiv-svddiag
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail
WT="${WAIV_WT:-/admin/home/ryan.kim/waiv_worktrees/svd-nuisance}"
MAIN="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
cd "$WT"
export HF_HOME="${HF_HOME:-/data/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$WT/src"
mkdir -p docs
"$MAIN/.venv/bin/python" scripts/diagnose_nuisance.py \
  --backbone paige-ai/Virchow2 --pooling clsmean \
  --n-tiles 2000 --seed 0 --batch-size 128 --amp bfloat16 \
  --packed-dir /data/plism/repacked \
  --shift-pixels 2 4 8 \
  --out docs/nuisance_diagnosis_virchow2_clsmean.json
echo "=== done rc=$? ==="
