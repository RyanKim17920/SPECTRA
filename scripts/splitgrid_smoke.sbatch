#!/bin/bash
#SBATCH --job-name=waiv-splitgrid-smoke
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# 20-step GPU smoke: SPLIT-HEADS x GRID at arm d's geometry (C=2, T=1800, B=3600).
#
# TWO things are being gated here, and only one of them is the loss:
#
#   1. DOES IT EVEN RUN. The grid path calls _chunked_forward_split unconditionally.
#      That function raises on a non-default pool_head -- which is what blocks GEM x grid.
#      This arm passes NO --pool-head, and WaivEncoder sets self.pool_head = None when
#      pool_head_name == "mean" (encoder.py:706-712), so the guard is not tripped. If that
#      reading is wrong, this smoke dies in the first step with the RuntimeError and the
#      combination is BLOCKED -- which is a real result, just not the one we want.
#
#   2. DOES IT FIT. Arm d ran single-head at B=3600 with chunk 300. Split-heads keeps two
#      projected views (cls and mean) of the full 3600-row batch alive for the backward,
#      plus a second InfoNCE over a (3600 x 3600)-ish score structure. The backbone
#      activations dominate and those are unchanged, so it should fit -- but "should fit"
#      is exactly the claim that has to be measured before spending a 10 h budget run.
#
# Geometry witnesses are asserted, not printed: negatives_per_anchor MUST equal T-1 = 1799
# and batch_distinct_conditions MUST equal C = 2. assert_grid_batch also runs every step
# inside the trainer, so reaching step 20 means 20 passed grid invariants.
#
# Training only -- no probe, no eval follower -- so this fits on one GPU.
# Runs against the SAME PINNED source the real run will use.

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

PIN="${WAIV_PIN:-/admin/home/ryan.kim/waiv-snapshots/finalgem-pinned}"
export PYTHONPATH="$PIN/src${PYTHONPATH:+:$PYTHONPATH}"

PACKED_DIR="${WAIV_PACKED_DIR:-/data/plism/repacked}"
PY="$REPO/.venv/bin/python"

COND=2
TILES=1800
CHUNK="${WAIV_CHUNK:-300}"

OUT_DIR="$REPO/runs/splitgrid-smoke-${SLURM_JOB_ID:-local}"
mkdir -p "$OUT_DIR"

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} ==="
echo "=== SMOKE split-heads x grid | C=$COND T=$TILES | B=$((COND * TILES)) | chunk $CHUNK | expect neg/row $((TILES - 1)) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv

"$PY" "$PIN"/scripts/train_lora.py \
  --packed-dir "$PACKED_DIR" \
  --out-dir "$OUT_DIR" \
  --heldout-scanners GT450 S210 \
  --heldout-stains HRH KR MY \
  --backbone owkin/phikon-v2 \
  --pooling clsmean \
  --proj-out-dim 512 \
  --lora-rank 32 \
  --lora-alpha 64 \
  --temperature 0.07 \
  --lr 1e-4 \
  --max-steps 20 \
  --warmup-steps 200 \
  --split-heads --cls-weight 0.5 --mean-weight 0.5 \
  --grid --grid-conditions "$COND" --grid-tiles "$TILES" --grid-forward-chunk "$CHUNK" \
  --grad-checkpointing \
  --workers 20 \
  --amp bfloat16 \
  --log-every 10 \
  --ckpt-every 100000 \
  --eval-every 100000 \
  --seed 0

echo "=== smoke history ==="
"$PY" - "$OUT_DIR" "$TILES" "$COND" <<'EOF'
import json, sys
from pathlib import Path
h = json.loads((Path(sys.argv[1]) / "history.json").read_text())
tiles, cond = int(sys.argv[2]), int(sys.argv[3])
want_neg, want_b = float(tiles - 1), tiles * cond
for r in h:
    print({k: r[k] for k in ("step", "loss", "top1", "loss_cls", "loss_mean",
                             "negatives_per_anchor", "batch_negatives_per_anchor",
                             "batch_distinct_conditions", "batch_n_rows", "tiles_per_s",
                             "gpu_mem_alloc_gib", "gpu_mem_reserved_gib") if k in r})

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

# The split has to be REAL, not a single head wearing two names. config.json is the
# witness the trainer itself wrote.
cfg = json.loads((Path(sys.argv[1]) / "config.json").read_text())
assert cfg["split_heads"] is True, "split_heads did not stick"
assert cfg["encoder"]["split_heads"] == ["cls", "mean"], \
    f"encoder split_heads {cfg['encoder']['split_heads']} != ['cls','mean']"
assert cfg["grid"] is True and cfg["grid_conditions"] == cond and cfg["grid_tiles"] == tiles
peak = max((r.get("gpu_mem_reserved_gib", 0) for r in h), default=0)
print(f"[smoke] OK split x grid neg/row={want_neg:.0f} C={cond} B={want_b} "
      f"over {len(h)} logged steps | peak reserved {peak:.1f} GiB")
EOF

echo "=== smoke done $(date) ==="
