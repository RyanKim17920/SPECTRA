#!/bin/bash
#SBATCH --job-name=waiv-hest
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# HEST-Benchmark retention eval (PLAN.md §3 risk 1, §4 phase 11).
#
#   sbatch scripts/run_hest.sbatch cls     base_cls
#   sbatch scripts/run_hest.sbatch clsmean base_clsmean
#   sbatch scripts/run_hest.sbatch clsmean step5000 "" runs/xxx/step5000
#
# One GPU: ~1.5M tiles of ViT-L/16 inference across the 9 tasks, extracted once (HEST
# only embeds on fold 0 and reuses). The login node has no GPU.

set -euo pipefail

REPO="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
cd "$REPO"

POOLING="${1:-cls}"
EXP_CODE="${2:-base_${POOLING}}"
CKPT="${3:-}"
ADAPTER="${4:-}"

export HF_HOME="${HF_HOME:-/data/ryan.kim/hf_home}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
# A broken pandas in ~/.local shadowed a venv earlier on this box; isolate or the pins
# in .venv-hest mean nothing.
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true

ARGS=(--pooling "$POOLING" --exp-code "$EXP_CODE" --num-workers 8)
# Unset => owkin/phikon-v2, i.e. every existing invocation is unchanged.
if [ -n "${WAIV_BACKBONE:-}" ]; then ARGS+=(--backbone "$WAIV_BACKBONE"); fi
if [ -n "$CKPT" ];    then ARGS+=(--checkpoint "$CKPT"); fi
if [ -n "$ADAPTER" ]; then
  # Defaults are the real run's geometry (runs/waiv-real-369043 == r32/alpha64). build_model
  # cross-checks these against the saved adapter_config.json and hard-fails on a mismatch,
  # so a stale default is loud rather than a silently mis-scaled adapter.
  ARGS+=(--adapter "$ADAPTER" --lora-rank "${LORA_RANK:-32}" --lora-alpha "${LORA_ALPHA:-64}" --proj-out-dim "${PROJ_OUT_DIM:-512}")
fi

# .venv-hest, NOT .venv: trident hard-pins timm==0.9.16 and drags transformers back to
# 4.57, which we do not want anywhere near the training venv. Only torch 2.8.0+cu128 is
# shared, and that pin is fixed by the cluster's CUDA 12.8 driver.
srun --unbuffered "$REPO/.venv-hest/bin/python" scripts/run_hest.py "${ARGS[@]}"

echo "=== done $(date) ==="
