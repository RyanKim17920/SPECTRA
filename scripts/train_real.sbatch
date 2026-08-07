#!/bin/bash
#SBATCH --job-name=waiv-real
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
# 32 CPUs / 250G / 6h, not "as much as possible": every 8xH100 node here is MIXED+PLANNED,
# so a 48-CPU 12-hour ask sits behind Priority for ~11 h. The nearest PLANNED start is
# 3 h 49 m out, so the box is sized to backfill INTO that window: 4000 x 1.91 s = 2.1 h
# of training with the eval follower draining alongside it, ~2.5 h total.
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --time=03:15:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#
# The first REAL run (PLAN.md 3 phase 7/8). Three deliberate changes from the 300-step
# rank-8 smoke checkpoint (job 369019, Avg RI 0.469 -> 0.736 against a 0.806 target):
#
# 1. FAR MORE NEGATIVES. The smoke ran 2 groups x 32, i.e. 31 negatives per anchor, and
#    top-1 hit 1.0 by step 70 with the loss at 0.038 -- the objective was solved, so the
#    gradient after that was noise. Here: 2 x 192 = 191 negatives, 6.2x. Because the
#    negative set is the anchor's own condition-homogeneous group (PLAN.md 2), the ONLY
#    way to buy negatives is a bigger forward batch: 384 anchors + 384 positives = 768
#    images/step. At the measured 0.21 GiB/image that is ~161 GiB, so this run turns on
#    gradient checkpointing (~0.02 GiB/image, ~+35% step time). Without it the ceiling
#    is ~340 images/step and 2 x 192 is simply not reachable.
#
# 2. ALL 91 SLIDES. The smoke started while the repack was still streaming and saw
#    59-72 of them; worse, its before/after probes scored *different* condition sets
#    (the pin landed in ffc6e4c, after that job's probes ran), so its "separation stayed
#    flat" readout mixed a real change with a set change. All 91 are repacked and
#    verified now, and the preflight below hard-fails on anything less.
#
# 3. LONGER + HIGHER RANK. Rank 32 (alpha 64) vs rank 8, and 4000 steps vs 300, with
#    log-every 20 so top-1 and loss are visible throughout and saturation cannot hide.
#
# Sized on a real 25-step probe (job 369039, not arithmetic): 2 x 192 + checkpointing =
# 21.77 GiB peak of 79.19, 1.91 s/step, 401 tiles/s, and top-1 0.74 at step 25 where the
# smoke run was already at 0.98 by step 30. 4000 x 1.91 s is ~2.1 h. 2 x 256 also fits,
# so `--group-size 256` is the escalation if top-1 still saturates.
#
# The second GPU runs eval_checkpoints.py as a follower: every checkpoint gets the
# embedding probe AND the full PathoROB RI on all three datasets, live, while GPU 0
# stays saturated. Gating is on embedding-space separation + top-1, never on training
# loss or matched cosine (matched cosine rises under collapse).
#
#   sbatch scripts/train_real.sbatch
#   sbatch scripts/train_real.sbatch --group-size 256 --max-steps 6000   # more negatives

set -euo pipefail

REPO=/admin/home/ryan.kim/waiv
cd "$REPO"

export HF_HOME=/data/ryan.kim/hf_home
export HF_HUB_OFFLINE=0               # the follower streams the PathoROB parquet
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1

PACKED_DIR=/data/ryan.kim/plism/repacked
RUN_NAME="${SLURM_JOB_NAME:-waiv-real}-${SLURM_JOB_ID:-local}"
OUT_DIR="$REPO/runs/$RUN_NAME"
mkdir -p "$OUT_DIR"
PY="$REPO/.venv/bin/python"

# Defaults reproduce job 369043 exactly; the env overrides exist so a sweep can vary one
# knob without forking this file (PLAN.md 3 risk 4 phase 8). Rank/alpha MUST come from
# here rather than through the trailing "$@", because embed_probe.py and the eval follower
# below are also given $LORA_RANK -- passing --lora-rank as an extra arg would train one
# rank and evaluate at another, and build_model hard-fails on that mismatch.
LORA_RANK="${WAIV_LORA_RANK:-32}"
LORA_ALPHA="${WAIV_LORA_ALPHA:-64}"
N_GROUPS="${WAIV_N_GROUPS:-2}"
GROUP_SIZE="${WAIV_GROUP_SIZE:-192}"
MAX_STEPS="${WAIV_MAX_STEPS:-4000}"
CKPT_EVERY="${WAIV_CKPT_EVERY:-500}"
LR="${WAIV_LR:-1e-4}"
TEMPERATURE="${WAIV_TEMPERATURE:-0.07}"

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} | run $RUN_NAME ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

# --- preflight: all 91, verified, and pin the probe's condition set --------------------
"$PY" - "$PACKED_DIR" "$OUT_DIR" <<'EOF'
import json, sys
from pathlib import Path
packed, out = Path(sys.argv[1]), Path(sys.argv[2])
man = json.loads((packed / "manifest.json").read_text())
ok = {k: v for k, v in man.items() if v.get("verified")}
# PLAN.md 3 phase 7 wants the full 13 x 7 grid; the smoke run only had part of it.
assert len(ok) == 91, f"expected 91 verified slides, manifest has {len(ok)} (repack incomplete)"
(out / "conditions_used.json").write_text(json.dumps(sorted(ok), indent=2))
stains = sorted({v["stain"] for v in ok.values()})
scanners = sorted({v["scanner"] for v in ok.values()})
print(f"[preflight] {len(ok)} verified slides | {len(stains)} stains {stains} "
      f"| {len(scanners)} scanners {scanners}")
assert len(stains) == 13 and len(scanners) == 7
EOF

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
  --format=csv,noheader -l 5 > "$OUT_DIR/gpu.csv" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

# --- embedding probe, BEFORE (base phikon-v2, pinned condition set) --------------------
CUDA_VISIBLE_DEVICES=1 "$PY" scripts/embed_probe.py \
  --packed-dir "$PACKED_DIR" \
  --conditions-file "$OUT_DIR/conditions_used.json" \
  --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" \
  --out "$OUT_DIR/probe_before.json"

# --- eval follower on GPU 1 -----------------------------------------------------------
# Starts now and idles on the poll until the first checkpoint lands. It exits once
# TRAIN_DONE exists and no unevaluated checkpoint remains.
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
trap 'kill $SAMPLER $FOLLOWER 2>/dev/null || true' EXIT
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
  --temperature "$TEMPERATURE" \
  --lr "$LR" \
  --max-steps "$MAX_STEPS" \
  --warmup-steps 200 \
  --n-groups "$N_GROUPS" \
  --group-size "$GROUP_SIZE" \
  --grad-checkpointing \
  --workers 20 \
  --amp bfloat16 \
  --log-every 20 \
  --ckpt-every "$CKPT_EVERY" \
  --eval-every "$CKPT_EVERY" \
  "$@"
TRAIN_RC=$?
set -e
echo "=== training exited rc=$TRAIN_RC ==="
touch "$OUT_DIR/TRAIN_DONE"

# --- let the follower drain the remaining checkpoints ---------------------------------
wait $FOLLOWER || true
kill $SAMPLER 2>/dev/null || true

echo "=== RI vs step ==="
"$PY" - "$OUT_DIR" <<'EOF'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "ri_curve.json"
print(p.read_text() if p.exists() else "no ri_curve.json written")
EOF
echo "=== done $(date) -> $OUT_DIR (train rc=$TRAIN_RC) ==="
exit $TRAIN_RC
