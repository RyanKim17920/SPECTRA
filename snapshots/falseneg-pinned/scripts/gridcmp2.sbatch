#!/bin/bash
#SBATCH --job-name=waiv-gridcmp2
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# GRIDCMP-2: the three follow-up arms to the completed gridcmp experiment
# (380777/380778/380779, docs/RESULTS.md 8.2). That experiment found PathoROB RI tracks
# negatives-per-row and runs INVERSELY to query rows:
#
#   CTRL   6x200      199 neg    1,200 rows   0.8127
#   GRID24 C=24 T=100  99 neg   55,200 rows   0.8043
#   GRID49 C=49 T=49   48 neg  115,248 rows   0.7973
#
# It has two limitations these arms exist to remove.
#
#   (a) At a fixed 2400 images/step, C and T are COUPLED: negatives and conditions moved
#       together in opposite directions, so the two cannot be separated (8.3).
#   (b) Every gap is n=1. The +-0.008 claims rest on differences the same size as an
#       unmeasured between-run seed variance.
#
#   WAIV_ARM=grid2     --grid -C 2  -T 1200   2400 img/step  1199 neg/row  R=2,400
#   WAIV_ARM=grid12    --grid -C 12 -T 200    2400 img/step   199 neg/row  R=26,400
#   WAIV_ARM=ctrlseed  --n-groups 6 --group-size 200 --seed 1
#
# GRID2 is the MAX-NEGATIVES point: 6x CTRL's negatives AND 2x its query rows at
# identical compute. GRID12 is the DISENTANGLING arm -- negatives identical to CTRL
# (199) but 2x the conditions and 22x the rows, so a tie with CTRL's 0.8127 says
# negatives is the only variable that matters and a win says rows help after all.
# CTRLSEED is the CTRL arm again with seed 1 and nothing else changed; it prices the
# between-run variance. Its config.json must differ from runs/gridcmp-ctrl-380777 in
# `seed` and `out_dir` ONLY -- anything else and it is not a seed repeat.
#
# Same structure as gridcmp.sbatch: GPU 0 trains, GPU 1 runs eval_checkpoints.py
# --follow so each checkpoint gets its PathoROB RI live. conditions_used.json is pinned
# from the preflight so the BEFORE probe and the follower score the SAME condition set
# the training run saw.
#
#   WAIV_ARM=grid12 sbatch --job-name=waiv-gridcmp2-grid12 scripts/gridcmp2.sbatch

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
ARM="${WAIV_ARM:?set WAIV_ARM to grid2 | grid12 | ctrlseed}"
# See gridcmp.sbatch: SLURM_RESTART_COUNT keeps a requeued attempt out of the previous
# attempt's run dir, so a preemption cannot silently mix two training trajectories.
RUN_NAME="gridcmp2-${ARM}-${SLURM_JOB_ID:-local}${SLURM_RESTART_COUNT:+.r$SLURM_RESTART_COUNT}"
OUT_DIR="$REPO/runs/$RUN_NAME"
mkdir -p "$OUT_DIR"
PY="$REPO/.venv/bin/python"

LORA_RANK=32
LORA_ALPHA=64

# --- the ONLY things that differ between the three arms -------------------------------
# --grid-forward-chunk 600 is a MEMORY setting, not a science one, and is held at the
# SAME value the completed grid arms used so the BatchNorm micro-batch (8.5) is
# unchanged. CTRLSEED takes chunk 0 because the completed CTRL arm did.
case "$ARM" in
  grid2)    SAMPLER_ARGS=(--grid --grid-conditions 2  --grid-tiles 1200 --grid-forward-chunk 600); SEED=0 ;;
  grid12)   SAMPLER_ARGS=(--grid --grid-conditions 12 --grid-tiles 200  --grid-forward-chunk 600); SEED=0 ;;
  ctrlseed) SAMPLER_ARGS=(--n-groups 6 --group-size 200);                                         SEED=1 ;;
  *) echo "unknown WAIV_ARM=$ARM (want grid2 | grid12 | ctrlseed)" >&2; exit 2 ;;
esac

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} | run $RUN_NAME ==="
echo "=== arm $ARM | seed $SEED | sampler ${SAMPLER_ARGS[*]} ==="
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
  --seed "$SEED" \
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
