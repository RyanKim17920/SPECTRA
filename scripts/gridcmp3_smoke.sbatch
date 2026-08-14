#!/bin/bash
#SBATCH --job-name=waiv-gridcmp3-smoke
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# 20-step GPU smoke for the three GRIDCMP-3 arms, run back to back on ONE GPU.
#
# What it is actually checking is the GEOMETRY, not the loss. The grid sampler takes C
# and T as separate flags and the objective derives negatives from the collated batch, so
# a mis-set flag produces a run that trains perfectly happily at the WRONG number of
# negatives -- and nothing downstream would say so. Two things prove the geometry:
#
#   negatives_per_anchor in history.json  MUST equal T-1 exactly (299 / 149 / 599)
#   assert_grid_batch                     runs EVERY step inside the trainer
#                                         (contrastive.py, cfg.grid branch) and raises if
#                                         the group structure is not C groups x T tiles
#
# So a smoke that reaches step 20 has already passed 20 grid-invariant assertions; the
# printed negatives_per_anchor is the number to eyeball.
#
# Training only -- no BEFORE probe and no eval follower, so this fits on one GPU.
# Runs against the same PINNED source the real arms will use, so it smokes the code that
# will actually run, not the working tree.

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

PIN="${WAIV_PIN:-/admin/home/ryan.kim/waiv-snapshots/gridcmp3-pinned}"
export PYTHONPATH="$PIN/src${PYTHONPATH:+:$PYTHONPATH}"

PACKED_DIR="${WAIV_PACKED_DIR:-/data/plism/repacked}"
PY="$REPO/.venv/bin/python"
CHUNK="${WAIV_CHUNK:-300}"
ARMS="${WAIV_ARMS:-a b c}"

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} | chunk $CHUNK | arms '$ARMS' ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv

declare -A RESULT
for ARM in $ARMS; do
  case "$ARM" in
    a) COND=12; TILES=300 ;;
    b) COND=24; TILES=150 ;;
    c) COND=6;  TILES=600 ;;
    *) echo "unknown arm $ARM" >&2; exit 2 ;;
  esac
  OUT_DIR="$REPO/runs/gridcmp3-smoke-${ARM}-${SLURM_JOB_ID:-local}"
  mkdir -p "$OUT_DIR"
  echo "=== SMOKE arm $ARM | C=$COND T=$TILES | B=$((COND * TILES)) | expect neg/row $((TILES - 1)) ==="

  "$PY" "$PIN"/scripts/train_lora.py \
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
    --grid --grid-conditions "$COND" --grid-tiles "$TILES" --grid-forward-chunk "$CHUNK" \
    --grad-checkpointing \
    --workers 20 \
    --amp bfloat16 \
    --log-every 10 \
    --ckpt-every 100000 \
    --eval-every 100000 \
    --seed 0

  echo "=== smoke history: arm $ARM ==="
  # Asserts rather than prints: a smoke whose only output is a number nobody reads is
  # not a gate. Exits non-zero if negatives_per_anchor is not exactly T-1.
  "$PY" - "$OUT_DIR" "$TILES" "$COND" <<'EOF'
import json, sys
from pathlib import Path
h = json.loads((Path(sys.argv[1]) / "history.json").read_text())
tiles, cond = int(sys.argv[2]), int(sys.argv[3])
want_neg, want_b = float(tiles - 1), tiles * cond
for r in h:
    print({k: r[k] for k in ("step", "loss", "top1", "negatives_per_anchor",
                             "batch_negatives_per_anchor", "batch_distinct_conditions",
                             "batch_n_rows", "tiles_per_s",
                             "gpu_mem_alloc_gib", "gpu_mem_reserved_gib") if k in r})

# TWO independent witnesses, deliberately. `negatives_per_anchor` is what the OBJECTIVE
# thinks it scored (grid_info_nce derives it from the n_cond/n_tiles it was handed).
# `batch_*` comes from assert_grid_batch reading the COLLATED TENSORS. A flag threaded
# wrong would keep those two consistent with each other only if the sampler and the loss
# agree -- which is exactly the thing being verified -- so both are asserted.
def uniq(key):
    return {r[key] for r in h if key in r}

assert uniq("negatives_per_anchor") == {want_neg}, \
    f"objective negatives_per_anchor {uniq('negatives_per_anchor')} != {{{want_neg}}}"
got_batch = uniq("batch_negatives_per_anchor")
assert got_batch in ({want_neg}, set()), \
    f"batch negatives_per_anchor {got_batch} != {{{want_neg}}}"
got_cond = uniq("batch_distinct_conditions")
assert got_cond in ({float(cond)}, {cond}, set()), \
    f"batch_distinct_conditions {got_cond} != {{{cond}}} -- sampler drew the wrong C"
print(f"[smoke] OK negatives_per_anchor={want_neg:.0f} C={cond} B={want_b} "
      f"over {len(h)} logged steps")
EOF
done

echo "=== smoke done $(date) ==="
