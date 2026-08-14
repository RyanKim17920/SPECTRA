#!/bin/bash
#SBATCH --job-name=waiv-gridcmp
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --time=05:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# GRID-vs-PAIR sampler comparison: ONE arm per job. Three separate 2-GPU jobs, not one
# 6-GPU job, so a node failure or a preemption on one arm does not take the other two
# with it.
#
#   WAIV_ARM=ctrl    --n-groups 6 --group-size 200      2400 img/step  199 neg/row  R=1,200
#   WAIV_ARM=grid24  --grid -C 24 -T 100                2400 img/step   99 neg/row  R=55,200
#   WAIV_ARM=grid49  --grid -C 49 -T 49                 2401 img/step   48 neg/row  R=115,248
#
# CTRL is the control and is NOT optional: it is the existing sampler at the SAME
# images/step, the same 1500 steps, the same backbone and the same seed. Without it, any
# difference in the downstream PathoROB RI curve is confounded with "1500 steps at
# B=2400", which no previous run in this repo has done either.
#
# Structure is lifted from train_real.sbatch: GPU 0 trains, GPU 1 runs
# eval_checkpoints.py --follow so each checkpoint gets its PathoROB RI live instead of
# doubling wall time inline. conditions_used.json is pinned from the preflight so the
# BEFORE probe and the follower score the SAME condition set the training run saw.
#
# Budget: 1500 x 5.85 s = 2.44 h of training plus the follower draining behind it.
#
#   WAIV_ARM=ctrl   sbatch --job-name=waiv-gridcmp-ctrl   scripts/gridcmp.sbatch

set -euo pipefail

REPO="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
cd "$REPO"

export HF_HOME="${HF_HOME:-/data/huggingface}"
export HF_HUB_OFFLINE=0               # the follower streams the PathoROB parquet
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
# The grid step keeps ~70 GiB live and then asks for one more large recompute
# buffer; the default caching allocator stranded 5.56 GiB in reserved-but-unused
# segments and OOMed. expandable_segments hands that back.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PACKED_DIR="${WAIV_PACKED_DIR:-/data/plism/repacked}"
ARM="${WAIV_ARM:?set WAIV_ARM to ctrl | grid24 | grid49}"
RUN_NAME="gridcmp-${ARM}-${SLURM_JOB_ID:-local}"
OUT_DIR="$REPO/runs/$RUN_NAME"
mkdir -p "$OUT_DIR"
PY="$REPO/.venv/bin/python"

LORA_RANK=32
LORA_ALPHA=64

# --- the ONE thing that differs between the three arms --------------------------------
case "$ARM" in
  ctrl)   SAMPLER_ARGS=(--n-groups 6 --group-size 200) ;;
  # --grid-forward-chunk is a MEMORY setting (one graph, one loss over all
  # C*T embeddings); 2400 images in a single forward OOMs an 80 GiB H100.
  grid24) SAMPLER_ARGS=(--grid --grid-conditions 24 --grid-tiles 100 --grid-forward-chunk 600) ;;
  grid49) SAMPLER_ARGS=(--grid --grid-conditions 49 --grid-tiles 49 --grid-forward-chunk 600) ;;
  *) echo "unknown WAIV_ARM=$ARM (want ctrl | grid24 | grid49)" >&2; exit 2 ;;
esac

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} | run $RUN_NAME ==="
echo "=== arm $ARM | sampler ${SAMPLER_ARGS[*]} ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

# --- preflight: all 91, verified, and pin the probe's condition set --------------------
"$PY" - "$PACKED_DIR" "$OUT_DIR" <<'EOF'
import json, sys
from pathlib import Path
packed, out = Path(sys.argv[1]), Path(sys.argv[2])
man = json.loads((packed / "manifest.json").read_text())
ok = {k: v for k, v in man.items() if v.get("verified")}
assert len(ok) == 91, f"expected 91 verified slides, manifest has {len(ok)} (repack incomplete)"
(out / "conditions_used.json").write_text(json.dumps(sorted(ok), indent=2))
stains = sorted({v["stain"] for v in ok.values()})
scanners = sorted({v["scanner"] for v in ok.values()})
print(f"[preflight] {len(ok)} verified slides | {len(stains)} stains | {len(scanners)} scanners")
assert len(stains) == 13 and len(scanners) == 7
EOF

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
  --format=csv,noheader -l 5 > "$OUT_DIR/gpu.csv" &
SAMPLER_PID=$!
trap 'kill $SAMPLER_PID 2>/dev/null || true' EXIT

# --- embedding probe, BEFORE (base phikon-v2, pinned condition set) --------------------
CUDA_VISIBLE_DEVICES=1 "$PY" scripts/embed_probe.py \
  --packed-dir "$PACKED_DIR" \
  --conditions-file "$OUT_DIR/conditions_used.json" \
  --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" \
  --out "$OUT_DIR/probe_before.json"

# --- eval follower on GPU 1 -----------------------------------------------------------
rm -f "$OUT_DIR/TRAIN_DONE"
CUDA_VISIBLE_DEVICES=1 "$PY" scripts/eval_checkpoints.py \
  --run-dir "$OUT_DIR" \
  --packed-dir "$PACKED_DIR" \
  --conditions-file "$OUT_DIR/conditions_used.json" \
  --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" \
  --num-workers 8 \
  --follow --stop-file "$OUT_DIR/TRAIN_DONE" --poll-s 60 \
  > "$OUT_DIR/eval_follow.log" 2>&1 &
FOLLOWER=$!
trap 'kill $SAMPLER_PID $FOLLOWER 2>/dev/null || true' EXIT
echo "=== eval follower pid $FOLLOWER -> $OUT_DIR/eval_follow.log ==="

# --- training on GPU 0 ----------------------------------------------------------------
set +e
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/train_lora.py \
  --packed-dir "$PACKED_DIR" \
  --out-dir "$OUT_DIR" \
  --heldout-scanners GT450 S210 \
  --heldout-stains HRH KR MY \
  --lora-rank "$LORA_RANK" \
  --lora-alpha "$LORA_ALPHA" \
  --temperature 0.07 \
  --lr 1e-4 \
  --max-steps 1500 \
  --warmup-steps 200 \
  "${SAMPLER_ARGS[@]}" \
  --grad-checkpointing \
  --workers 20 \
  --amp bfloat16 \
  --log-every 20 \
  --ckpt-every 250 \
  --eval-every 250 \
  --seed 0 \
  "$@"
TRAIN_RC=$?
set -e
echo "=== training exited rc=$TRAIN_RC ==="
touch "$OUT_DIR/TRAIN_DONE"

wait $FOLLOWER || true
kill $SAMPLER_PID 2>/dev/null || true

echo "=== RI vs step ==="
"$PY" - "$OUT_DIR" <<'EOF'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "ri_curve.json"
print(p.read_text() if p.exists() else "no ri_curve.json written")
EOF
echo "=== done $(date) -> $OUT_DIR (arm $ARM, train rc=$TRAIN_RC) ==="
exit $TRAIN_RC
