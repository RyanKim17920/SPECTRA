#!/bin/bash
#SBATCH --job-name=waiv-svdsplit
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail
# Fidelity fix: the reference implementation fits a SEPARATE basis per readout site
# (rn_v_cls from cls deltas, rn_v_pm from patch-mean deltas). Our default 'joint' basis
# fits one eigendecomposition across the concatenated 2560-d vector, which lets a
# high-variance direction in one half claim rank the other half does not warrant.
# NOTE: under split, rank k removes k directions PER HALF = 2k total. The sweep records
# directions_removed per row so the table is not silently comparing k against 2k.
WT="${WAIV_WT:-/admin/home/ryan.kim/waiv_worktrees/svd-nuisance}"
MAIN="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
cd "$WT"
export HF_HOME="${HF_HOME:-/data/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$WT/src"
FIT="$WT/runs/svd_nuisance_virchow2_clsmean_split_s0.npz"
"$MAIN/.venv/bin/python" scripts/fit_svd_nuisance.py \
  --backbone paige-ai/Virchow2 --pooling clsmean --basis split \
  --n-tiles 2000 --seed 0 --batch-size 128 --amp bfloat16 \
  --packed-dir /data/plism/repacked --out "$FIT"
# k here is PER HALF, so this sweep spans 2..256 directions removed in total.
"$MAIN/.venv/bin/python" scripts/score_svd_nuisance.py \
  --fit "$FIT" --ks 0 1 2 4 8 16 32 64 128 \
  --src-model virchow2_clsmean_base \
  --pathorob-root "$MAIN/third_party/PathoROB" \
  --python-exe "$MAIN/.venv-pathorob/bin/python" \
  --dst-tag "_split" \
  --out "$WT/runs/svd_nuisance_sweep_virchow2_split_s0.json"
echo "=== done rc=$? ==="
