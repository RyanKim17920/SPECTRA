#!/bin/bash
#SBATCH --job-name=waiv-offload-probe
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=700G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# ACTIVATION OFFLOAD: is it exact, and does it actually move the ceiling?
#
# Two phases, in this order, and the second only runs if the first passes.
#
#   1. scripts/offload_bitcheck.py -- matched control at a tiny geometry. Same seed, same
#      pixels, same weights, one flag flipped; loss AND every gradient must match. This is
#      a GATE, not a diagnostic: if offload is not exact then its ceiling is worthless,
#      because whatever it buys, it is no longer measuring the same objective. (Precedent:
#      chunking the projector was also "just a memory device" and moved the loss by 1.65.)
#
#   2. scripts/sizing_probe_grid.py --activation-offload -- the ladder, at the SAME chunk
#      the plain sweep used (150), so the two tables differ in exactly one variable and
#      the new ceiling can be read against the old one directly.
#
# --mem=700G is deliberate and load-bearing. save_on_cpu(pin_memory=True) parks the saved
# activations in PINNED host RAM, and pinned pages are not swappable -- at the ceiling
# geometries this is expected to be tens of GiB, and it grows with T just as the device
# copy would have. The GPU nodes carry ~1.5 TiB, so this is affordable; requesting the
# 120G the plain sizing sbatch uses would turn a memory win into a host OOM and read as
# "offload does not work".
#
# The 2x1975 rung is not padding: it is the SAME geometry the plain sweep topped out at,
# so it prices offload's throughput cost against a known-good point before the ladder
# climbs past it.
#
#   sbatch scripts/offload_probe.sbatch

set -euo pipefail

REPO="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
cd "$REPO"

export HF_HOME="${HF_HOME:-/data/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${WAIV_ALLOC_CONF-expandable_segments:True}"

PY="$REPO/.venv/bin/python"
# The rungs are CxT pairs, not a B ladder, and that is the whole point. Memory is NOT a
# function of B = C*T alone here: grid_info_nce's logit tensor is (C,C,T,T) = B^2 either
# way, but the QUERY ROW count is C*(C-1)*T, and that is 11x larger at C=12/T=800 (105,600
# rows) than at the same B=9600 spent as C=2/T=4800 (9,600 rows). So a C=2 ladder cannot
# be used to predict a C=12 ceiling; the target geometries get measured directly.
#
#   2x1975   the plain sweep's ceiling -- carried over ONLY as the reference rung, to
#            price what offload costs in throughput at a point whose plain number is known
#            exactly (77.60 GiB, 373.4 img/s)
#   12x600   arm F fallback   B=7200   599 neg  12 cond   (arm c's negatives, 2x its conds)
#   12x800   arm F preferred  B=9600   799 neg  12 cond   (arm a's conds, 2.7x its negs)
#   12x1000  headroom, to say where the new ceiling actually is rather than just clearing
#            the target
GEOMS="${WAIV_GEOMS:-2x1975 12x600 12x800 12x1000}"
CHUNK="${WAIV_CHUNK:-150}"
STEPS="${WAIV_STEPS:-3}"
TAG="${WAIV_TAG:-offload-${SLURM_JOB_ID:-local}}"

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} ==="
echo "geoms='${GEOMS}' chunk=${CHUNK} steps=${STEPS} alloc='${PYTORCH_CUDA_ALLOC_CONF}'"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv || true
free -g | head -2 || true

if [ "${WAIV_SKIP_BITCHECK:-0}" = "1" ]; then
  # Skippable ONLY because it already passed on this exact source (job 382004:
  # |dloss| = 0.0, worst |dgrad| = 0.0 over 294 gradients). It is not skippable
  # because it is optional -- if src/ changes again, run it again.
  echo "=== PHASE 1: bitcheck SKIPPED (WAIV_SKIP_BITCHECK=1) ==="
else
  echo "=== PHASE 1: exactness bitcheck (gate) ==="
  "$PY" scripts/offload_bitcheck.py --n-cond 2 --n-tiles 64 --chunk 16
fi

echo "=== PHASE 2: offload sizing ladder ==="
"$PY" scripts/sizing_probe_grid.py \
  --geoms $GEOMS --chunk $CHUNK --steps "$STEPS" \
  --grad-checkpointing --activation-offload \
  --out "runs/sizing_grid_${TAG}.json"

echo "=== done $(date) -> runs/sizing_grid_${TAG}.json ==="
