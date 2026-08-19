#!/bin/bash
#SBATCH --job-name=negprobe
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
# 32 CPUs / 250G / 50 min, not the obvious 33/300G/2h: the partition runs with every GPU
# node alloc/mix, so the ask has to BACKFILL. The measured cost is 300 steps x ~5.9 s =
# ~30 min plus ~3 min of startup, and a 2-hour ask on 33 CPUs put job 380721 behind
# Priority with an estimated start a day out (it had already been PREEMPTED at step
# 295/300 once). 32 CPUs is also the idle-CPU count on the mixed nodes here; 30 of them
# are the three arms' 10 dataloader workers.
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --time=00:50:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# InfoNCE negatives-saturation probe. Three arms at IDENTICAL compute (2*C*T images per
# step), varying ONLY how those images are grouped -- i.e. only the number of negatives
# per anchor, which is group_size - 1:
#
#   arm A (control)  C=6  T=200 -> 199 negatives/row
#   arm B            C=12 T=100 ->  99 negatives/row
#   arm C            C=24 T=50  ->  49 negatives/row
#
# Arm A is the control: it should reproduce the ~191-negative reference behaviour
# (top-1 plateau ~0.97, loss floor ~0.09, <10% of steps above top1 0.98). If it does not,
# the probe setup is broken and arms B/C mean nothing.
#
# The three arms run CONCURRENTLY, one per GPU, so they see the same node, the same
# filesystem contention and the same wall clock. No eval follower, no embedding probe,
# no checkpoints (--ckpt-every/--eval-every are pushed past --max-steps): the only
# artifact this experiment needs is history.json.
#
# WAIV_NEG_C_SCALE halves/doubles the number of groups in ALL THREE arms at once -- the
# arms must stay compute-matched, so the scale is deliberately global rather than
# per-arm. Set it to 0.5 if 2400 images/step OOMs.
#
#   sbatch scripts/negprobe.sbatch
#   WAIV_NEG_C_SCALE=0.5 sbatch scripts/negprobe.sbatch    # 1200 images/step fallback

set -euo pipefail

REPO="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
cd "$REPO"

export HF_HOME="${HF_HOME:-/data/huggingface}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
# At 2400 images/step fragmentation, not raw footprint, is what decides whether the
# largest arm fits.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PACKED_DIR="${WAIV_PACKED_DIR:-/data/plism/repacked}"
PY="$REPO/.venv/bin/python"
JOB="${SLURM_JOB_ID:-local}"

SCALE="${WAIV_NEG_C_SCALE:-1}"
NG_A=$("$PY" -c "print(max(1,int(6*$SCALE)))")
NG_B=$("$PY" -c "print(max(1,int(12*$SCALE)))")
NG_C=$("$PY" -c "print(max(1,int(24*$SCALE)))")

echo "=== $(date) | node $(hostname) | job $JOB ==="
echo "=== arms: A ${NG_A}x200  B ${NG_B}x100  C ${NG_C}x50  (scale $SCALE) ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

# --- preflight: refuse to measure anything on a partial corpus -------------------------
"$PY" - "$PACKED_DIR" <<'EOF'
import json, sys
from pathlib import Path
man = json.loads((Path(sys.argv[1]) / "manifest.json").read_text())
ok = {k: v for k, v in man.items() if v.get("verified")}
assert len(ok) == 91, f"expected 91 verified slides, manifest has {len(ok)}"
print(f"[preflight] {len(ok)} verified slides")
EOF

run_arm () {  # gpu, n_groups, group_size
  local GPU="$1" NG="$2" GS="$3"
  local NEG=$((GS - 1))
  local OUT="$REPO/runs/negprobe-n${NEG}-${JOB}"
  mkdir -p "$OUT"
  echo "=== arm n=${NEG}: GPU $GPU, ${NG}x${GS} = $((2 * NG * GS)) images/step -> $OUT ==="
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/train_lora.py \
    --packed-dir "$PACKED_DIR" \
    --out-dir "$OUT" \
    --heldout-scanners GT450 S210 \
    --heldout-stains HRH KR MY \
    --backbone owkin/phikon-v2 \
    --lora-rank 32 --lora-alpha 64 \
    --lr 1e-4 --temperature 0.07 \
    --max-steps 300 --warmup-steps 20 \
    --n-groups "$NG" --group-size "$GS" \
    --grad-checkpointing --amp bfloat16 \
    --seed 0 \
    --log-every 5 \
    --ckpt-every 100000 --eval-every 100000 \
    --workers 10 \
    > "$REPO/logs/negprobe-n${NEG}-${JOB}.log" 2>&1
}

set +e
run_arm 0 "$NG_A" 200 & PID_A=$!
run_arm 1 "$NG_B" 100 & PID_B=$!
run_arm 2 "$NG_C" 50  & PID_C=$!

wait $PID_A; RC_A=$?
wait $PID_B; RC_B=$?
wait $PID_C; RC_C=$?
set -e

echo "=== rc: A(199)=$RC_A  B(99)=$RC_B  C(49)=$RC_C ==="
echo "=== done $(date) ==="
exit $(( RC_A | RC_B | RC_C ))
