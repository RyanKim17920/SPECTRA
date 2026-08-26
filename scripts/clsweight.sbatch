#!/bin/bash
#SBATCH --job-name=waiv-clsw
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=700G
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# ======================================================================================
# FINAL CONFIG, THREE BACKBONES, FIVE SEEDS -- ONE IDENTICAL RECIPE
# ======================================================================================
#
# WHY THIS EXISTS. The three-backbone table in docs/FINAL_RESULTS.md sec.2 is confounded:
# the three runs behind it do not share a recipe.
#
#   splitgrid-386374  phikon-v2   C=2 T=1800  1500 steps  pool_head=mean
#   splitgrid-386380  midnight    C=2 T=450   1500 steps  pool_head=mean
#   splitgrid-386375  Virchow2    C=2 T=600   1500 steps  pool_head=mean
#
# Two separate problems with citing that table as a result for "the final config":
#
#   1. T varies 4x across backbones, so "method helps phikon but not the others" is
#      inseparable from "phikon got 4x the negatives".
#   2. NONE of them is the config sec.1 calls final: all three are pool_head=mean, i.e.
#      split-heads WITHOUT GEM. The only GEM+grid runs that exist anywhere are
#      gemgrid-38605{2,6,7}, and those are phikon-only.
#
# This script fixes ONE recipe and moves ONLY --backbone and --seed.
#
# --- WHY GRID_CHUNK=0 IS NOT A TUNABLE HERE -------------------------------------------
# A learned pool head pools over the raw PATCH-TOKEN sequence, which embed_parts() has
# already reduced away. contrastive.py:896-912 therefore RAISES if asked to chunk with a
# non-default pool_head, and delegates to model.forward_split() when unchunked. So GEM
# forces chunk=0: the whole C*T forward is one graph. That, not the backbone, is what set
# gemgrid's T=900 ceiling.
#
# --- WHY --activation-offload ---------------------------------------------------------
# It is the ONLY lever left, and it is free of correctness risk. Gradient checkpointing is
# already on in every arm and chunking is unavailable (above), so the remaining memory is
# the saved activations themselves. save_on_cpu(pin_memory=True) parks them in pinned host
# RAM; contrastive.py:911-920 wraps the forward_split delegation in that same ctx, so it
# covers the GEM path including the projectors. It is an exact save/restore hook -- NOT an
# approximation, no batch boundaries move, no BatchNorm statistics change. Verified by
# scripts/offload_bitcheck.py on job 382004: |dloss| = 0.0, worst |dgrad| = 0.0 over 294
# gradients. It buys 2-3x T for roughly 40% throughput.
#
# --mem=700G is load-bearing, not padding. Pinned pages are NOT swappable, and the parked
# activations grow with T exactly as the device copy would have. Requesting the 250G that
# gemgrid.sbatch used would convert a GPU-memory win into a host OOM, which would then be
# misread as "offload does not work".
#
# --- PIN ------------------------------------------------------------------------------
# waiv-snapshots/gemgrid-pinned, which is byte-identical to HEAD for both
# src/waivphaet/train/contrastive.py and scripts/train_lora.py (verified by diff). It is
# the ONLY snapshot carrying the pool-head delegation; finalgem-pinned (commit 386283a)
# still has the old guard that raises unconditionally, so GEM+grid CANNOT run there.
#
#   WAIV_ARM=virchow2 WAIV_SEED=0 WAIV_T=900 sbatch scripts/final5.sbatch
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

PY="$REPO/.venv/bin/python"
PIN="${WAIV_PIN:-/admin/home/ryan.kim/waiv-snapshots/gemgrid-pinned}"
export PYTHONPATH="$PIN/src${PYTHONPATH:+:$PYTHONPATH}"
[ -d "$PIN/src/waivphaet" ]     || { echo "pin missing src/waivphaet: $PIN" >&2; exit 3; }
[ -f "$PIN/scripts/train_lora.py" ] || { echo "pin missing train_lora.py: $PIN" >&2; exit 3; }
# The pin MUST carry the pool-head delegation, or GEM+grid dies at step 0.
grep -q 'delegate rather than refuse' "$PIN/src/waivphaet/train/contrastive.py" \
  || { echo "PIN LACKS pool-head delegation -- GEM+grid cannot run: $PIN" >&2; exit 3; }

PACKED_DIR="${WAIV_PACKED_DIR:-/data/plism/repacked}"
ARM="${WAIV_ARM:?set WAIV_ARM=phikon|midnight|virchow2|hoptimus|uni2}"
SEED="${WAIV_SEED:?set WAIV_SEED}"
GRID_TILES="${WAIV_T:?set WAIV_T (common across backbones -- that is the point)}"

case "$ARM" in
  phikon)   BACKBONE="" ;;                        # unset => train_lora.py default owkin/phikon-v2
  midnight) BACKBONE="kaiko-ai/midnight" ;;
  virchow2) BACKBONE="paige-ai/Virchow2" ;;
  # Both hub repos are GATED (403).  encoder.BACKBONE_LOCAL_DIRS binds these ids to the
  # local checkpoints under /data; nothing here needs a path.  Override with
  # WAIV_BACKBONE_LOCAL_DIRS if /data is swept again.
  hoptimus) BACKBONE="bioptimus/H-optimus-0" ;;
  uni2)     BACKBONE="MahmoodLab/UNI2-h" ;;
  *) echo "bad WAIV_ARM=$ARM" >&2; exit 2 ;;
esac
BACKBONE_ARGS=()
[ -n "$BACKBONE" ] && BACKBONE_ARGS=(--backbone "$BACKBONE")

RUN_NAME="clsw${WAIV_CW}-${ARM}-s${SEED}-t${GRID_TILES}-${SLURM_JOB_ID:-local}${SLURM_RESTART_COUNT:+.r$SLURM_RESTART_COUNT}"
OUT_DIR="$REPO/runs/$RUN_NAME"
mkdir -p "$OUT_DIR"

MAX_STEPS="${WAIV_MAX_STEPS:-1500}"
CKPT_EVERY="${WAIV_CKPT_EVERY:-250}"
WORKERS="${WAIV_WORKERS:-10}"
LORA_RANK=32
LORA_ALPHA=64
LR=1e-4
TEMPERATURE=0.07
GRID_COND=2
GRID_CHUNK=0          # NOT tunable: GEM requires unchunked. See header.
EVAL_MAX_WAIT_S="${WAIV_EVAL_MAX_WAIT_S:-72000}"
CW="${WAIV_CW:-0.75}"; MW=$(python3 -c "print(1-$CW)")
# At mean-weight 0 the mean head is never built, so --pool-head has nothing to pool and
# train_lora.py refuses. cls-only is therefore run WITHOUT the pool head by construction.
if [ "$MW" = "0.0" ] || [ "$MW" = "0" ]; then
  HEAD_ARGS=(--split-heads --cls-weight "$CW" --mean-weight "$MW")
else
  HEAD_ARGS=(--split-heads --cls-weight "$CW" --mean-weight "$MW" --pool-head gem)
fi

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} restart ${SLURM_RESTART_COUNT:-0} ==="
echo "run       $RUN_NAME"
echo "arm       $ARM  backbone=${BACKBONE:-<default owkin/phikon-v2>}  seed=$SEED"
echo "geometry  C=$GRID_COND T=$GRID_TILES B=$((GRID_COND * GRID_TILES)) chunk=$GRID_CHUNK neg/row=$((GRID_TILES - 1))"
echo "head      ${HEAD_ARGS[*]}"
echo "steps     $MAX_STEPS  ckpt/eval $CKPT_EVERY"
echo "pin       $PIN"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
free -g | head -2 || true

"$PY" - "$PIN" <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / "src"))
import waivphaet
assert waivphaet.__file__.startswith(sys.argv[1]), f"PIN LEAK: {waivphaet.__file__}"
print(f"[pin] waivphaet={waivphaet.__file__}", flush=True)
EOF

"$PY" - "$PACKED_DIR" "$OUT_DIR" <<'EOF'
import json, sys
from pathlib import Path
packed, out = Path(sys.argv[1]), Path(sys.argv[2])
man = json.loads((packed / "manifest.json").read_text())
ok = {k: v for k, v in man.items() if v.get("verified")}
assert len(ok) == 91, len(ok)
(out / "conditions_used.json").write_text(json.dumps(sorted(ok), indent=2))
stains = sorted({v["stain"] for v in ok.values()})
scanners = sorted({v["scanner"] for v in ok.values()})
print(f"[preflight] {len(ok)} conditions, {len(stains)} stains, {len(scanners)} scanners")
assert len(stains) == 13 and len(scanners) == 7
EOF
md5sum "$OUT_DIR/conditions_used.json"

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
  --format=csv,noheader -l 5 > "$OUT_DIR/gpu.csv" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

rm -f "$OUT_DIR/TRAIN_DONE"
# WAIV_NO_FOLLOWER=1 skips the RI eval follower. ONLY for memory/geometry smokes, which
# write no checkpoints and so give the follower nothing to read. Never set it for a real
# run: the follower is where ri_curve.json comes from, and RI is the only readout in this
# project with the dynamic range to rank arms.
FOLLOWER=""
if [ "${WAIV_NO_FOLLOWER:-0}" = "1" ]; then
  echo "=== eval follower DISABLED (WAIV_NO_FOLLOWER=1) -- smoke only, no ri_curve.json ==="
else
CUDA_VISIBLE_DEVICES=1 "$PY" "$PIN"/scripts/eval_checkpoints.py \
  --run-dir "$OUT_DIR" \
  --packed-dir "$PACKED_DIR" \
  --conditions-file "$OUT_DIR/conditions_used.json" \
  --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" "${BACKBONE_ARGS[@]}" \
  --num-workers 8 \
  --max-wait-s "$EVAL_MAX_WAIT_S" \
  --follow --stop-file "$OUT_DIR/TRAIN_DONE" --poll-s 60 \
  > "$OUT_DIR/eval_follow.log" 2>&1 &
FOLLOWER=$!
trap 'kill $SAMPLER $FOLLOWER 2>/dev/null || true' EXIT
echo "=== eval follower pid $FOLLOWER -> $OUT_DIR/eval_follow.log ==="
fi

set +e
CUDA_VISIBLE_DEVICES=0 "$PY" "$PIN"/scripts/train_lora.py \
  --packed-dir "$PACKED_DIR" \
  --out-dir "$OUT_DIR" \
  --heldout-scanners GT450 S210 \
  --heldout-stains HRH KR MY \
  --pooling clsmean \
  --proj-out-dim 512 \
  "${BACKBONE_ARGS[@]}" \
  --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" \
  --temperature "$TEMPERATURE" \
  --lr "$LR" \
  --max-steps "$MAX_STEPS" \
  --warmup-steps 200 \
  --grid --grid-conditions "$GRID_COND" --grid-tiles "$GRID_TILES" \
  --grid-forward-chunk "$GRID_CHUNK" \
  "${HEAD_ARGS[@]}" \
  --grad-checkpointing \
  --activation-offload \
  --workers "$WORKERS" --amp bfloat16 --log-every 20 \
  --ckpt-every "$CKPT_EVERY" --eval-every "$CKPT_EVERY" \
  --seed "$SEED"
TRAIN_RC=$?
set -e

echo "rc=$TRAIN_RC" > "$OUT_DIR/TRAIN_DONE"
[ -n "$FOLLOWER" ] && wait $FOLLOWER 2>/dev/null || true
kill $SAMPLER 2>/dev/null || true

echo "=== RI curve ==="
"$PY" - "$OUT_DIR" <<'EOF'
import sys
from pathlib import Path
p = Path(sys.argv[1]) / "ri_curve.json"
print(p.read_text() if p.exists() else "no ri_curve.json")
EOF

echo "=== done $(date) | $OUT_DIR | arm=$ARM seed=$SEED T=$GRID_TILES rc=$TRAIN_RC ==="
exit $TRAIN_RC
