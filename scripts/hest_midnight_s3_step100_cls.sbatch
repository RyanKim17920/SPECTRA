#!/bin/bash
#SBATCH --job-name=spectra-hest-midnight-s3-100-cls
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# HEST-Benchmark, Midnight-12k seed 3 FINE-TUNED at step 100, CLS-only pooling.
#
# WHY: docs/seed_stats.md reports Midnight-12k HEST at n=2 while every other Midnight
# cell (PathoROB RI, all four THUNDER tasks, CPTAC) is n=3.  The third seed
# (genMASK-c50-ms500-midnight-s3-t900-436609) WAS scored on HEST, but at step 150 and
# under `clsmean` pooling, while the paper's cell for that seed is step 100 and the
# other two seeds (399166 s0, 407566 s1) are scored under `cls`.  Averaging a clsmean
# result into two cls results would manufacture a wrong mean rather than a third seed,
# so it was correctly dropped.  This job produces the missing cls/step-100 cell.
#
# Backbone pinned in-file, NOT via $SPECTRA_BACKBONE: an unset env var silently falls back
# to owkin/phikon-v2 and would emit a phikon row labelled "midnight".
#
# CACHE KEY: run_hest.py keys the embedding cache on --exp-code ALONE; pooling is NOT
# part of the key.  Reusing the existing `..._s0000150_clsmean` code would silently
# score 3072-d clsmean features and manufacture "cls == clsmean".  The exp_code below
# was verified absent from $SPECTRA_HEST_WORK/{embeddings,results}/ before
# submission.  The name matches the glob seed_stats.py uses:
#   f5_<run without .rN>_s<step>_*_summary.json
#
# Training config verified identical to seeds 0 and 1 (lora_rank 32, lora_alpha 64,
# proj_out_dim 512, C=2, T=900, ms500, lr 1e-4); only `seed` differs.  Training used
# clsmean pooling for all three seeds -- that is the TRAINING head, independent of the
# CLS-only readout protocol used for the HEST leaderboard.
#
#   sbatch --account=max --qos=high scripts/hest_midnight_s3_step100_cls.sbatch

set -euo pipefail

REPO="${SPECTRA_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
. "$REPO/scripts/_env.sh"
cd "$REPO"

POOLING="cls"
EXP_CODE="f5_genMASK-c50-ms500-midnight-s3-t900-436609_s0000100_cls"
BACKBONE="kaiko-ai/midnight"
ADAPTER="$REPO/runs/genMASK-c50-ms500-midnight-s3-t900-436609.r5/step_0000100"

export HF_HOME="${HF_HOME:-$SPECTRA_HF_HOME}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} ==="
echo "exp_code=$EXP_CODE"
echo "adapter=$ADAPTER"
test -d "$ADAPTER" || { echo "FATAL: adapter dir missing"; exit 2; }
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true

# --tasks omitted => run_hest.py default = the 9 published leaderboard tasks, matching
# how seeds 0 and 1 were scored.  --precision omitted => float32, likewise matching.
ARGS=(--pooling "$POOLING" --exp-code "$EXP_CODE" --backbone "$BACKBONE" --num-workers 0 \
      --adapter "$ADAPTER" --lora-rank 32 --lora-alpha 64 --proj-out-dim 512)

srun --unbuffered "$REPO/.venv-hest/bin/python" scripts/run_hest.py "${ARGS[@]}"

echo "=== done $(date) ==="
