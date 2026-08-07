#!/bin/bash
#SBATCH --job-name=waiv-eval-backfill
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# Finish the RI-vs-step curve for a run whose in-job follower did not drain.
#
# The follower shares CPUs with the trainer's dataloader workers, and the PathoROB
# parquet load is CPU-bound -- under contention an eval stretches past the checkpoint
# interval and the follower falls behind. That costs nothing permanent: adapters are a
# few MB each and every step_* is on disk, so the curve can be completed afterwards on
# an idle GPU.
#
# eval_checkpoints.py is idempotent -- it reads the existing ri_curve.json, skips steps
# already in it, and reuses any probe_step_*.json already written. Re-running it is
# always safe.
#
#   sbatch scripts/eval_backfill.sbatch runs/waiv-real-369043
#   sbatch scripts/eval_backfill.sbatch runs/waiv-real-369043 32 64

set -euo pipefail

REPO="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
cd "$REPO"

RUN_DIR="${1:?usage: sbatch scripts/eval_backfill.sbatch <run-dir> [lora-rank] [lora-alpha]}"
LORA_RANK="${2:-32}"
LORA_ALPHA="${3:-64}"

export HF_HOME="${HF_HOME:-/data/ryan.kim/hf_home}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} | backfill $RUN_DIR ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

CONDS=""
if [ -f "$RUN_DIR/conditions_used.json" ]; then
  CONDS="--conditions-file $RUN_DIR/conditions_used.json"
fi

"$REPO/.venv/bin/python" scripts/eval_checkpoints.py \
  --run-dir "$RUN_DIR" \
  --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" \
  --num-workers 12 \
  $CONDS

echo "=== done $(date) ==="
