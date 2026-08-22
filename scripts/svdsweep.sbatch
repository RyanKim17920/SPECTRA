#!/bin/bash
#SBATCH --job-name=waiv-svdsweep
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail
# The k-sweep is pure CPU (project cached PathoROB features, run their kNN metric), but
# it is long enough that running it in an interactive/background shell loses it to any
# session event -- which is exactly how the first attempt died after only k=0.
WT="${WAIV_WT:-/admin/home/ryan.kim/waiv_worktrees/svd-nuisance}"
MAIN="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
cd "$WT"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$WT/src"
SEED="${SEED:-0}"
"$MAIN/.venv/bin/python" scripts/score_svd_nuisance.py \
  --fit "$WT/runs/svd_nuisance_virchow2_clsmean_s${SEED}.npz" \
  --ks 0 1 2 4 8 16 32 64 128 256 \
  --src-model virchow2_clsmean_base \
  --pathorob-root "$MAIN/third_party/PathoROB" \
  --python-exe "$MAIN/.venv-pathorob/bin/python" \
  --out "$WT/runs/svd_nuisance_sweep_virchow2_s${SEED}.json"
echo "=== done rc=$? seed=$SEED ==="
