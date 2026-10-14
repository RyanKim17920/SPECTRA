#!/bin/bash
#SBATCH --job-name=hsub5
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:40:00
#
# HEST 5-task subset (SKCM COAD READ PAAD LUNG) for ONE checkpoint.
# usage: sbatch --output=... --error=... scripts/hest_sub5_arm.sbatch <exp_tag> <adapter_dir>
#
# NOTE: --output/--error/--chdir MUST be shared storage under /admin/home/ryan.kim.
# /tmp and the agent scratchpad are NODE-LOCAL and silently swallow output.
set -euo pipefail

REPO=/admin/home/ryan.kim/waiv
cd "$REPO"

TAG="$1"
ADAPTER="$2"
# 0 disables the DataLoader worker processes entirely. Use it on retry: the
# worker path passes tensors through /dev/shm, and shm contention on this box
# aborts the run with "could not unlink the shared memory file" (killed arm b
# on its first attempt). Serial loading costs a couple of minutes per arm.
WORKERS="${3:-2}"
EXP="sub5_${TAG}_clsmean"

export HF_HOME="${HF_HOME:-/data/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

echo "=== ARM START $TAG | $(date -u) | node $(hostname) | job ${SLURM_JOB_ID:-none} ==="
echo "=== adapter: $ADAPTER"
echo "=== exp_code: $EXP"

if [ ! -d "$ADAPTER/adapter" ]; then
  echo "FATAL: no adapter/ under $ADAPTER"; exit 2
fi

# HEST's embedding cache is keyed on FILE EXISTENCE, not integrity: a job killed
# mid-write leaves a truncated .h5 that later runs treat as a valid cache hit.
# Each arm owns its own embed dir, and we wipe it at the top of every attempt so
# a requeue can never inherit a half-written cache from its own previous attempt.
EMBED_DIR="/data/ryan.kim/hest_work/embeddings/$EXP"
if [ -d "$EMBED_DIR" ]; then
  echo "=== wiping stale embed cache $EMBED_DIR"
  rm -rf "$EMBED_DIR"
fi

srun --unbuffered "$REPO/.venv-hest/bin/python" scripts/run_hest.py \
  --pooling clsmean \
  --exp-code "$EXP" \
  --adapter "$ADAPTER" \
  --lora-rank 32 --lora-alpha 64 --proj-out-dim 512 \
  --tasks SKCM COAD READ PAAD LUNG \
  --num-workers "$WORKERS" \
  --bench-data /data/ryan.kim/hest_bench \
  --work-dir /data/ryan.kim/hest_work
RC=$?

# /data is 94% full and volatile -- mirror the summary to /admin immediately.
mkdir -p "$REPO/results_backup/hest_sub5"
SUM="/data/ryan.kim/hest_work/results/${EXP}_summary.json"
if [ -f "$SUM" ]; then
  cp -f "$SUM" "$REPO/results_backup/hest_sub5/"
  echo "=== mirrored $SUM -> results_backup/hest_sub5/"
else
  echo "WARN: no summary at $SUM"
fi

echo "=== ARM DONE $TAG rc=$RC $(date -u) ==="
exit $RC
