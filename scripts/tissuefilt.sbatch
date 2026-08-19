#!/bin/bash
#SBATCH --job-name=waiv-tissuefilt
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
# ARM 1: tissuefilt-phikon-s0-t900
# Background tile filtering only. Heldout split IDENTICAL to final5 recipe.
# Baseline for comparison: runs/final5-phikon-s0-t900*
#
# Differs from final5 recipe ONLY by: --min-tissue-frac 0.25
# Pin: waiv-snapshots/tissuefilt-pinned (fork of gemgrid-pinned with tissue filter wiring)
# ======================================================================================
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
PIN="/admin/home/ryan.kim/waiv-snapshots/tissuefilt-pinned"
export PYTHONPATH="$PIN/src${PYTHONPATH:+:$PYTHONPATH}"

[ -d "$PIN/src/waivphaet" ]         || { echo "pin missing src/waivphaet: $PIN" >&2; exit 3; }
[ -f "$PIN/scripts/train_lora.py" ] || { echo "pin missing train_lora.py: $PIN" >&2; exit 3; }
grep -q 'delegate rather than refuse' "$PIN/src/waivphaet/train/contrastive.py" \
  || { echo "PIN LACKS pool-head delegation -- GEM+grid cannot run: $PIN" >&2; exit 3; }
grep -q 'min_tissue_frac' "$PIN/src/waivphaet/train/contrastive.py" \
  || { echo "PIN LACKS min_tissue_frac field: $PIN" >&2; exit 3; }

PACKED_DIR="${WAIV_PACKED_DIR:-/data/plism/repacked}"
SEED=0
GRID_TILES=900

RUN_NAME="tissuefilt-phikon-s0-t${GRID_TILES}-${SLURM_JOB_ID:-local}${SLURM_RESTART_COUNT:+.r$SLURM_RESTART_COUNT}"
OUT_DIR="$REPO/runs/$RUN_NAME"
mkdir -p "$OUT_DIR"

MAX_STEPS=1500
CKPT_EVERY=250
WORKERS=10
LORA_RANK=32
LORA_ALPHA=64
LR=1e-4
TEMPERATURE=0.07
GRID_COND=2
GRID_CHUNK=0
EVAL_MAX_WAIT_S="${WAIV_EVAL_MAX_WAIT_S:-72000}"
HEAD_ARGS=(--split-heads --cls-weight 0.5 --mean-weight 0.5 --pool-head gem)
MIN_TISSUE_FRAC=0.25

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} restart ${SLURM_RESTART_COUNT:-0} ==="
echo "run         $RUN_NAME"
echo "arm         phikon (default owkin/phikon-v2)  seed=$SEED"
echo "geometry    C=$GRID_COND T=$GRID_TILES B=$((GRID_COND * GRID_TILES)) chunk=$GRID_CHUNK"
echo "head        ${HEAD_ARGS[*]}"
echo "tissue_frac min=$MIN_TISSUE_FRAC"
echo "steps       $MAX_STEPS  ckpt/eval $CKPT_EVERY"
echo "pin         $PIN"
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
CUDA_VISIBLE_DEVICES=1 "$PY" "$PIN"/scripts/eval_checkpoints.py \
  --run-dir "$OUT_DIR" \
  --packed-dir "$PACKED_DIR" \
  --conditions-file "$OUT_DIR/conditions_used.json" \
  --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" \
  --num-workers 8 \
  --max-wait-s "$EVAL_MAX_WAIT_S" \
  --follow --stop-file "$OUT_DIR/TRAIN_DONE" --poll-s 60 \
  > "$OUT_DIR/eval_follow.log" 2>&1 &
FOLLOWER=$!
trap 'kill $SAMPLER $FOLLOWER 2>/dev/null || true' EXIT
echo "=== eval follower pid $FOLLOWER -> $OUT_DIR/eval_follow.log ==="

set +e
CUDA_VISIBLE_DEVICES=0 "$PY" "$PIN"/scripts/train_lora.py \
  --packed-dir "$PACKED_DIR" \
  --out-dir "$OUT_DIR" \
  --heldout-scanners GT450 S210 \
  --heldout-stains HRH KR MY \
  --pooling clsmean \
  --proj-out-dim 512 \
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
  --seed "$SEED" \
  --min-tissue-frac "$MIN_TISSUE_FRAC"
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

echo "=== done $(date) | $OUT_DIR | seed=$SEED T=$GRID_TILES min_tissue_frac=$MIN_TISSUE_FRAC rc=$TRAIN_RC ==="
exit $TRAIN_RC
