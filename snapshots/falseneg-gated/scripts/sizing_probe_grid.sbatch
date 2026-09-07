#!/bin/bash
#SBATCH --job-name=waiv-gridsize
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
#   sbatch scripts/sizing_probe_grid.sbatch "2x1200 2x1400 2x1600" 600 4
#
# GRID geometries (CxT), one GPU. See scripts/sizing_probe_grid.py for why C=2 is the
# negatives-maximising choice and docs/RESULTS.md 8.6 for the resulting table.
#
# ALLOC_CONF is a first-class variable here, not boilerplate: at these batch sizes the
# ceiling is partly a FRAGMENTATION ceiling, and expandable_segments changes it. Pass it
# explicitly so a row in the results table always says which allocator produced it.
#     WAIV_ALLOC_CONF="" sbatch ... -> plain caching allocator
#     WAIV_ALLOC_CONF=expandable_segments:True sbatch ...

set -euo pipefail

REPO="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
cd "$REPO"

GEOMS="${1:-2x1200 2x1400 2x1600}"
CHUNK="${2:-600}"
STEPS="${3:-4}"
EXTRA="${4:-}"

export HF_HOME="${HF_HOME:-/data/huggingface}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${WAIV_ALLOC_CONF-expandable_segments:True}"

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} ==="
echo "geoms='${GEOMS}' chunk=${CHUNK} steps=${STEPS} alloc='${PYTORCH_CUDA_ALLOC_CONF}'"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv || true

TAG="${WAIV_TAG:-${SLURM_JOB_ID:-local}}"
srun --unbuffered "$REPO/.venv/bin/python" scripts/sizing_probe_grid.py \
  --geoms $GEOMS --chunk $CHUNK --steps "$STEPS" \
  --grad-checkpointing --loss-only $EXTRA \
  --out "runs/sizing_grid_${TAG}.json"

echo "=== done $(date) ==="
