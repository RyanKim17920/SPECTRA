#!/bin/bash
#SBATCH --job-name=waiv-hest-openmidnight-base-cls
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# HEST-Benchmark, SophontAI/OpenMidnight BASE, cls pooling, 9-task leaderboard protocol.
#
# Backbone pinned in-file (NOT $WAIV_BACKBONE) because an unset env var silently
# falls back to owkin/phikon-v2 and would produce a phikon row wearing this label.
#
# Pooling is `cls`: neither H-Optimus-0 nor UNI2-h is on Waiv's clsmean list
# (H0-mini is, H-Optimus-0 is not).
#
# CACHE KEY: run_hest.py keys the embedding cache on --exp-code ALONE (pooling is NOT
# part of the key -- see scripts/run_hest.py:114-155). Fresh exp_code below, verified
# absent from /data/ryan.kim/hest_work/embeddings/ before submission.
#
# --num-workers 0 is REQUIRED: the HEST dataloader crashes in shm teardown with >0.

set -euo pipefail

REPO="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
cd "$REPO"

POOLING="cls"
EXP_CODE="openmidnight_base_cls_9task_v1"
BACKBONE="SophontAI/OpenMidnight"

export HF_HOME="${HF_HOME:-/data/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true

# Gated backbone: hard-fail BEFORE any inference if weights/normalisation are wrong.
"$REPO/.venv/bin/python" scripts/preflight_gated_backbone.py "$BACKBONE"

# No --checkpoint / --adapter: this is the BASE model.
ARGS=(--pooling "$POOLING" --exp-code "$EXP_CODE" --backbone "$BACKBONE" --num-workers 0 )

srun --unbuffered "$REPO/.venv-hest/bin/python" scripts/run_hest.py "${ARGS[@]}"

echo "=== done $(date) ==="
