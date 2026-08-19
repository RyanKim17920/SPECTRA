#!/bin/bash
#SBATCH --job-name=waiv-hest-vir-ft-cls
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# HEST-Benchmark, Midnight-12k BASE, CLS-ONLY pooling, 9-task leaderboard protocol.
#
# Why this exists as its own file rather than `sbatch scripts/run_hest.sbatch cls ...`:
# the generic wrapper takes the backbone from $WAIV_BACKBONE, and an env var that is
# silently unset falls back to owkin/phikon-v2 -- which would produce a phikon row
# labelled "midnight". The backbone is pinned in-file here so the label cannot drift.
#
# Head-to-head being run: our Midnight base HEST average is 0.41210 under `clsmean`
# (exp_code mbase_clsmean, job 370850, 26-08-06). Waiv publish 0.3952 for the same
# model and state no pooling protocol. Hypothesis: their number is CLS-only. This job
# produces the CLS-only Midnight base number on the identical 9-task protocol so the
# two can be compared on a common scale.
#
# CACHE KEY: run_hest.py keys the embedding cache on --exp-code ALONE (see the warning
# block at scripts/run_hest.py:114-155) -- pooling is NOT part of the key. Reusing
# `mbase_clsmean` here would silently score the existing 3072-d clsmean features and
# manufacture "cls == clsmean". Hence a fresh, never-used exp_code below; verified
# absent from /data/ryan.kim/hest_work/embeddings/ before submission.
#
#   sbatch --account=max --qos=high scripts/hest_midnight_cls.sbatch

set -euo pipefail

REPO="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
cd "$REPO"

POOLING="cls"
EXP_CODE="virchow2_ft250_cls_9task_v1"
BACKBONE="paige-ai/Virchow2"

export HF_HOME="${HF_HOME:-/data/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true

# No --checkpoint / --adapter: this is the BASE model.
# --tasks omitted => run_hest.py default = H.LEADERBOARD_TASKS, the 9 published tasks
# (IDC PRAD PAAD SKCM COAD READ CCRCC LUNG LYMPH_IDC), HCC excluded. Matches the
# mbase_clsmean run exactly. --precision omitted => float32, likewise matching.
ARGS=(--pooling "$POOLING" --exp-code "$EXP_CODE" --backbone "$BACKBONE" --num-workers 0 --adapter /admin/home/ryan.kim/waiv/runs/waiv-virchow2-375367/step_0000250 --lora-rank 32 --lora-alpha 64 --proj-out-dim 512)

srun --unbuffered "$REPO/.venv-hest/bin/python" scripts/run_hest.py "${ARGS[@]}"

echo "=== done $(date) ==="
