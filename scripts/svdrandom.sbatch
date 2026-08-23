#!/bin/bash
#SBATCH --job-name=waiv-svdrandom
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%x-s%a-%j.out
#SBATCH --error=logs/slurm-%x-s%a-%j.err
set -euo pipefail
# NEGATIVE CONTROL. Same projection machinery, but the basis is a random orthonormal
# frame carrying zero information about PLISM. If this reproduces the RI curve, the
# fitted nuisance subspace is irrelevant and the effect is pure rank reduction --
# i.e. the headline result would be an artifact. This is the control that decides it.
WT="${WAIV_WT:-/admin/home/ryan.kim/waiv_worktrees/svd-nuisance}"
MAIN="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
cd "$WT"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$WT/src"
S="${SLURM_ARRAY_TASK_ID:-0}"
"$MAIN/.venv/bin/python" scripts/score_svd_nuisance.py \
  --fit "$WT/runs/svd_nuisance_virchow2_clsmean_RANDOM_s${S}.npz" \
  --ks 0 8 16 32 64 128 256 \
  --src-model virchow2_clsmean_base \
  --pathorob-root "$MAIN/third_party/PathoROB" \
  --python-exe "$MAIN/.venv-pathorob/bin/python" \
  --dst-tag "_rand${S}" \
  --out "$WT/runs/svd_nuisance_sweep_virchow2_RANDOM_s${S}.json"
echo "=== done rc=$? seed=$S ==="
