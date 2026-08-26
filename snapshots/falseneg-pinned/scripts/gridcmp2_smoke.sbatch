#!/bin/bash
#SBATCH --job-name=waiv-gridcmp2-smoke-grid2
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --time=00:40:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# 20-step GPU smoke for the GRID2 arm ONLY (--grid -C 2 -T 1200). It is the untested
# extreme of the grid sampler: T=1200 means a 1200x1200 per-pair similarity block, and
# only 2 slides supply 1200 random reads each -- a different I/O pattern from C=49's
# 49 slides x 49 reads. GRID12 (C=12 T=200) interpolates between already-smoked points
# and does not need its own smoke.
#
# Reference from the completed grid arms: 2401 images measured 55.98 GiB of 79.2 GiB and
# ~395-405 img/s. Read tiles_per_s and gpu_mem_reserved_gib out of history.json.
#
# Training only -- no BEFORE probe and no eval follower, so this fits on one GPU.

set -euo pipefail

REPO="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
cd "$REPO"

export HF_HOME="${HF_HOME:-/data/huggingface}"
export HF_HUB_OFFLINE=0
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PACKED_DIR="${WAIV_PACKED_DIR:-/data/plism/repacked}"
OUT_DIR="$REPO/runs/gridcmp2-smoke-grid2-${SLURM_JOB_ID:-local}"
mkdir -p "$OUT_DIR"
PY="$REPO/.venv/bin/python"

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} | $OUT_DIR ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv

"$PY" scripts/train_lora.py \
  --packed-dir "$PACKED_DIR" \
  --out-dir "$OUT_DIR" \
  --heldout-scanners GT450 S210 \
  --heldout-stains HRH KR MY \
  --lora-rank 32 \
  --lora-alpha 64 \
  --temperature 0.07 \
  --lr 1e-4 \
  --max-steps 20 \
  --warmup-steps 200 \
  --grid --grid-conditions 2 --grid-tiles 1200 --grid-forward-chunk 600 \
  --grad-checkpointing \
  --workers 20 \
  --amp bfloat16 \
  --log-every 10 \
  --ckpt-every 100000 \
  --eval-every 100000 \
  --seed 0

echo "=== smoke history ==="
"$PY" - "$OUT_DIR" <<'EOF'
import json, sys
from pathlib import Path
h = json.loads((Path(sys.argv[1]) / "history.json").read_text())
for r in h:
    print({k: r[k] for k in ("step", "loss", "top1", "tiles_per_s",
                             "gpu_mem_alloc_gib", "gpu_mem_reserved_gib") if k in r})
EOF
echo "=== smoke done $(date) ==="
