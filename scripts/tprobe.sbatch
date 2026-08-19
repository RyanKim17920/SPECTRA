#!/bin/bash
#SBATCH --job-name=waiv-tprobe
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=700G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# COMMON-T PROBE. Find the largest C=2 grid geometry each backbone can train at WITH
# activation offload, so all three backbones can share ONE T instead of the current
# 1800/600/450 split that confounds the three-backbone comparison.
#
# --mem=700G is load-bearing: save_on_cpu(pin_memory=True) parks saved activations in
# PINNED host RAM, which is not swappable. Under-requesting turns a GPU-memory win into a
# host OOM and reads as "offload does not work".
set -euo pipefail
REPO="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
cd "$REPO"
export HF_HOME="${HF_HOME:-/data/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="$REPO/.venv/bin/python"
ARM="${WAIV_ARM:?set WAIV_ARM=phikon|midnight|virchow2}"
case "$ARM" in
  phikon)   BB="owkin/phikon-v2" ;;
  midnight) BB="kaiko-ai/midnight" ;;
  virchow2) BB="paige-ai/Virchow2" ;;
  *) echo "bad WAIV_ARM=$ARM" >&2; exit 2 ;;
esac
GEOMS="${WAIV_GEOMS:-2x450 2x900 2x1800 2x2400}"
CHUNK="${WAIV_CHUNK:-300}"
echo "=== $(date) | $(hostname) | job ${SLURM_JOB_ID:-none} | arm=$ARM bb=$BB ==="
echo "geoms='$GEOMS' chunk=$CHUNK offload=ON gradckpt=ON"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
free -g | head -2 || true
"$PY" scripts/sizing_probe_grid.py \
  --backbone "$BB" \
  --geoms $GEOMS --chunk $CHUNK --steps 3 \
  --grad-checkpointing --activation-offload \
  --out "runs/tprobe_${ARM}_${SLURM_JOB_ID:-local}.json"
echo "=== done $(date) ==="
