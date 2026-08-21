#!/bin/bash
#SBATCH --job-name=waiv-svdhest
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm-%x-k%a-%j.out
#SBATCH --error=logs/slurm-%x-k%a-%j.err
set -euo pipefail
# HEST under the training-free nuisance projection, one array task per rank.
# Virchow2 takes clsmean for HEST (collect_final5.py), which is also the pooling the
# subspace was fitted in -- a cls-pooled readout would be 1280-d against a 2560-d basis.
# exp_code MUST carry the svdNNN marker: HEST's embedding cache is keyed on exp_code
# alone, so an unmarked code silently reuses the unprojected embeddings.
WT="${WAIV_WT:-/admin/home/ryan.kim/waiv_worktrees/svd-nuisance}"
MAIN="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
cd "$WT"
export HF_HOME="${HF_HOME:-/data/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$WT/src"
K="${SLURM_ARRAY_TASK_ID:-0}"
FIT="$WT/runs/svd_nuisance_virchow2_clsmean_s0.npz"
CODE="$(printf 'svd%03d_virchow2_clsmean' "$K")"
ARGS=(--pooling clsmean --backbone paige-ai/Virchow2
      --exp-code "$CODE" --bench-data /data/hest_bench)
if [ "$K" -gt 0 ]; then ARGS+=(--svd-fit "$FIT" --svd-k "$K"); fi
echo "=== HEST k=$K exp_code=$CODE ==="
"$MAIN/.venv-hest/bin/python" scripts/run_hest.py "${ARGS[@]}"
echo "=== done rc=$? k=$K ==="
