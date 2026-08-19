#!/bin/bash
#SBATCH --job-name=waiv-hest-phikon-split500-cls
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# HEST-Benchmark: phikon-v2, splitgrid final checkpoint, step 500 (RI 0.8335).
# Run: runs/splitgrid-386374/step_0000500
# Pooling: cls  (phikon-v2 HEST protocol -- verified base 0.37470 vs Waiv 0.3747, exact)
# LoRA r32 / alpha64 / proj_out_dim 512 / clsmean at train time.
#
# CACHE KEY: exp_code `splitgrid_phikon_ft500_cls_v1` verified absent from
#   /data/ryan.kim/hest_work/embeddings/ and /data/ryan.kim/hest_work/results/
#   before submission. A stale exp_code would silently return wrong features.
#
#   sbatch --account=max --qos=high scripts/hest_splitgrid_phikon_ft500_cls.sbatch

set -euo pipefail

REPO="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
cd "$REPO"

POOLING="cls"
EXP_CODE="splitgrid_phikon_ft500_cls_v1"
BACKBONE="owkin/phikon-v2"
ADAPTER="$REPO/runs/splitgrid-386374/step_0000500"

export HF_HOME="${HF_HOME:-/data/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} ==="
echo "exp_code=$EXP_CODE adapter=$ADAPTER"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true

# --tasks omitted => default H.LEADERBOARD_TASKS (9 tasks, no HCC)
# --num-workers 0: avoids c10::Error shared-memory teardown crash at 2 and 8
ARGS=(
  --pooling "$POOLING"
  --exp-code "$EXP_CODE"
  --backbone "$BACKBONE"
  --adapter "$ADAPTER"
  --lora-rank 32
  --lora-alpha 64
  --proj-out-dim 512
  --num-workers 0
)

srun --unbuffered "$REPO/.venv-hest/bin/python" scripts/run_hest.py "${ARGS[@]}"

echo "=== done $(date) ==="
