#!/bin/bash
#SBATCH --job-name=waiv-headcmp
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# SPLIT-LOSS-HEAD comparison: ONE arm per job. Three separate 2-GPU jobs, not one 6-GPU
# job, so a node failure or a preemption on one arm does not take the other two with it.
# Structure lifted from gridcmp.sbatch.
#
#   WAIV_ARM=split     --split-heads --cls-weight 0.5 --mean-weight 0.5
#   WAIV_ARM=clsonly   --split-heads --cls-weight 1.0 --mean-weight 0.0
#   WAIV_ARM=meanonly  --split-heads --cls-weight 0.0 --mean-weight 1.0
#
# Everything else is IDENTICAL to the CTRL arm of the gridcmp experiment (6 x 200 pair
# path, 2400 images/step, phikon-v2, rank 32 / alpha 64, lr 1e-4, tau 0.07, 1500 steps,
# 200 warmup, seed 0), so that arm is this experiment's baseline: the only thing that
# moved is the loss head.
#
# The 0.5/0.5 default is SCALE NEUTRALITY, not a hedge: at 1.0/1.0 the total gradient
# magnitude would be ~2x the single-head baseline's, i.e. a silent LR change, and the
# comparison would measure "split PLUS 2x LR". A weight of 0.0 REMOVES its head rather
# than zero-multiplying it -- a built-but-unweighted head still updates its BatchNorm
# running stats every step, so the "only" arms would not actually be single-head.
#
# EVAL POOLING STAYS clsmean ON ALL THREE ARMS. It is a protocol constant (PathoROB's
# reference row is phikonv2_clsmean), not a model choice, and the training pooling must
# not leak into it. On clsonly/meanonly the follower will print
#   [build_model] skipping projector: trained with a 1024-d input, evaluating at 2048-d
# which is EXPECTED and benign: every eval path reads model.embed(), and the LoRA backbone
# weights are what carry the fine-tune. The projector is training-only.
#
# GPU 0 trains, GPU 1 runs eval_checkpoints.py --follow so each checkpoint gets its
# PathoROB RI live instead of doubling wall time inline. conditions_used.json is pinned
# from the preflight so the BEFORE probe and the follower score the SAME condition set the
# training run saw.
#
#   WAIV_ARM=split sbatch --job-name=waiv-headcmp-split scripts/headcmp.sbatch

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PACKED_DIR="${WAIV_PACKED_DIR:-/data/plism/repacked}"
ARM="${WAIV_ARM:?set WAIV_ARM to split | clsonly | meanonly}"
RUN_NAME="headcmp-${ARM}-${SLURM_JOB_ID:-local}"
OUT_DIR="$REPO/runs/$RUN_NAME"
mkdir -p "$OUT_DIR"
PY="$REPO/.venv/bin/python"

LORA_RANK=32
LORA_ALPHA=64

# --- the ONE thing that differs between the three arms --------------------------------
case "$ARM" in
  split)    HEAD_ARGS=(--split-heads --cls-weight 0.5 --mean-weight 0.5) ;;
  clsonly)  HEAD_ARGS=(--split-heads --cls-weight 1.0 --mean-weight 0.0) ;;
  meanonly) HEAD_ARGS=(--split-heads --cls-weight 0.0 --mean-weight 1.0) ;;
  *) echo "unknown WAIV_ARM=$ARM (want split | clsonly | meanonly)" >&2; exit 2 ;;
esac

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} | run $RUN_NAME ==="
echo "=== arm $ARM | heads ${HEAD_ARGS[*]} ==="
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
# NOTE: no --pooling here on purpose. eval_checkpoints.py defaults to clsmean, which is
# the protocol constant; passing the training head selection through would silently change
# what PathoROB's reference row is being compared against.
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
  --n-groups 6 --group-size 200 \
  "${HEAD_ARGS[@]}" \
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

echo "=== per-head loss curve ==="
"$PY" - "$OUT_DIR" <<'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
h = json.loads((d / "history.json").read_text()) if (d / "history.json").exists() else []
cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
heads = cfg.get("encoder", {}).get("split_heads", [])
for r in h[::10]:
    parts = " ".join(f"{k}={r.get('loss_' + k, float('nan')):.4f}" for k in heads)
    print(f"step {r['step']:5d} loss={r['loss']:.4f} top1={r['top1']:.3f} {parts}")
EOF

echo "=== RI vs step ==="
"$PY" - "$OUT_DIR" <<'EOF'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "ri_curve.json"
print(p.read_text() if p.exists() else "no ri_curve.json written")
EOF
echo "=== done $(date) -> $OUT_DIR (arm $ARM, train rc=$TRAIN_RC) ==="
exit $TRAIN_RC
