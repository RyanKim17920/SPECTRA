#!/bin/bash
#SBATCH --job-name=waiv-gridcmp3
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=350G
#SBATCH --time=10:00:00
# NOTE: there is deliberately no WAIV_TIME env passthrough -- sbatch parses #SBATCH
# directives from the file text at SUBMIT time, before any shell expansion, so an env var
# cannot reach this line. Override on the command line instead: `sbatch --time=14:00:00`.
# Arm f at C=12 T=800 runs ~36 s/step; 1500 steps needs ~15 h, NOT the 10 h above, which
# is what killed job 382243 at step 805.
#SBATCH --requeue
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# GRIDCMP-3: raise the BATCH BUDGET so "more negatives" stops being confounded with
# "fewer conditions".
#
# Every grid arm so far ran at B = C*T = 2400 images/step. At a fixed B, C and T trade
# against each other, so the three completed points moved negatives (T-1) and conditions
# (C) in OPPOSITE directions simultaneously:
#
#   GRID49  C=49 T=49    48 neg   115,248 rows   0.7973
#   GRID24  C=24 T=100   99 neg    55,200 rows   0.8043
#   GRID12  C=12 T=200   199 neg   26,400 rows   0.8021
#
# Nothing in that table separates "negatives helped" from "conditions hurt". The fix is
# to stop holding B fixed. Gradient checkpointing + --grid-forward-chunk make B=3600 fit
# on one 80 GiB H100, which buys a 1.5x larger tile budget -- enough to raise T while
# HOLDING C at the value its baseline used:
#
#   WAIV_ARM=a  --grid -C 12 -T 300   3600 img/step   299 neg   R=39,600
#   WAIV_ARM=b  --grid -C 24 -T 150   3600 img/step   149 neg   R=82,800
#   WAIV_ARM=c  --grid -C 6  -T 600   3600 img/step   599 neg   R=18,000
#   WAIV_ARM=d  --grid -C 2  -T 1800  3600 img/step  1799 neg   R=3,600
#   WAIV_ARM=e  --grid -C 2  -T 1975  3950 img/step  1974 neg   R=3,950
#   WAIV_ARM=f  --grid -C 12 -T 800   9600 img/step   799 neg   R=105,600 (offload)
#
# ARM A is GRID12's condition count (12) with 299 negatives instead of 199.
# ARM B is GRID24's condition count (24) with 149 negatives instead of 99.
# In both, C is UNCHANGED from the baseline and only negatives move -- that is the whole
# point, and it is why these two arms are the experiment and ARM C is the extrapolation.
# ARM C pushes negatives to 599 at the cost of only 6 conditions; it is the point that
# says whether the negatives axis keeps paying once conditions get genuinely scarce.
#
# NOT VARIED, deliberately: no --split-heads, no --pool-head. The pooling-head and
# split-head winners are a SEPARATE axis (headcmp / poolcmp) and combining winners is a
# later step. Every flag below except --grid-conditions / --grid-tiles /
# --grid-forward-chunk / --out-dir is identical to runs/gridcmp2-grid12-380862/config.json,
# which is what makes ARM A a clean +100-negatives delta on GRID12 rather than a new
# configuration that happens to also have more negatives.
#
# STEP SCHEDULE. 1500 steps with checkpoints every 250. Step 1000 at B=3600 is 3.6M
# images, which is exactly step 1500 at B=2400 -- so the ri_curve.json point at step 1000
# is the MATCHED-IMAGE comparison against the completed arms' final point, and step 1500
# here is the matched-STEP comparison. Both are wanted; do not trim the schedule.
#
#   WAIV_ARM=a sbatch --job-name=waiv-gridcmp3-a scripts/gridcmp3.sbatch

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
# At B=3600 the step keeps ~74 GiB live and then asks for one more large recompute
# buffer. The default caching allocator strands several GiB in reserved-but-unused
# segments and OOMs; expandable_segments hands that back. Load-bearing at this size.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ARM="${WAIV_ARM:?set WAIV_ARM to a | b | c | d | e | f}"

# --- CODE PINNING --------------------------------------------------------------------
# sbatch freezes the BATCH SCRIPT at submit time. It does NOT freeze the python source:
# a job that sits PENDING for hours reads scripts/ and src/ at START time, so any edit
# landing in the working tree in between silently enters the run. These arms are only
# meaningful as a controlled delta on GRID12/GRID24, so the source is pinned to a
# read-only snapshot taken at submit time.
if [ "$ARM" = "f" ]; then
  PIN="${WAIV_PIN:-/admin/home/ryan.kim/waiv-snapshots/gridcmp3-f-pinned}"
else
  PIN="${WAIV_PIN:-/admin/home/ryan.kim/waiv-snapshots/gridcmp3-pinned}"
fi
export PYTHONPATH="$PIN/src${PYTHONPATH:+:$PYTHONPATH}"
[ -d "$PIN/src/waivphaet" ]  || { echo "pin missing src/waivphaet: $PIN" >&2; exit 3; }
[ -d "$PIN/third_party/PathoROB" ] || { echo "pin missing third_party/PathoROB: $PIN" >&2; exit 3; }
[ -x "$PIN/.venv/bin/python" ]     || { echo "pin missing .venv/bin/python: $PIN" >&2; exit 3; }

PACKED_DIR="${WAIV_PACKED_DIR:-/data/plism/repacked}"
# SLURM_RESTART_COUNT keeps a requeued attempt out of the previous attempt's run dir
# (commit 42990ba). Preemption is routine here; without the suffix a requeue writes
# attempt-2 weights into attempt-1's directory and marries them to attempt-1's RI points.
RUN_NAME="gridcmp3-${ARM}-${SLURM_JOB_ID:-local}${SLURM_RESTART_COUNT:+.r$SLURM_RESTART_COUNT}"
OUT_DIR="$REPO/runs/$RUN_NAME"
mkdir -p "$OUT_DIR"
PY="$REPO/.venv/bin/python"

LORA_RANK=32
LORA_ALPHA=64
PROJ_OUT_DIM=512
PROBE_TILES=256

# How long the eval follower tolerates seeing NO new checkpoint before giving up.
# eval_checkpoints.py defaults this to 8 h and this script used to leave it unset, which
# truncated arm F: the follower quit while the trainer was still writing, so job 382243's
# step_0000750 landed on disk unscored and ri_curve.json stopped at [250, 500]. 20 h sits
# comfortably past the 10 h wall clock even after a --time override, so the follower now
# only ever exits via --stop-file (TRAIN_DONE) or a genuinely wedged trainer.
EVAL_MAX_WAIT_S="${WAIV_EVAL_MAX_WAIT_S:-$((20 * 3600))}"

# --- the ONLY things that differ between the arms ------------------------------------
CHUNK="${WAIV_CHUNK:-300}"
OFFLOAD="${WAIV_OFFLOAD:-0}"
case "$ARM" in
  a) COND=12; TILES=300 ;;
  b) COND=24; TILES=150 ;;
  c) COND=6;  TILES=600 ;;
  d) COND=2;  TILES=1800 ;;
  e) COND=2;  TILES=1975; CHUNK="${WAIV_CHUNK:-150}" ;;
  # ARM F / BOTH-HIGH: activation-offload arm, C=12 x T=800 -> B=9600, 799 negatives/row
  f) COND=12; TILES=800; CHUNK="${WAIV_CHUNK:-150}"; OFFLOAD=1 ;;
  *) echo "unknown WAIV_ARM=$ARM (want a | b | c | d | e | f)" >&2; exit 2 ;;
esac
SEED=0
SAMPLER_ARGS=(--grid --grid-conditions "$COND" --grid-tiles "$TILES" --grid-forward-chunk "$CHUNK")
if [ "$OFFLOAD" = "1" ]; then
  SAMPLER_ARGS+=(--activation-offload)
fi

# --- optional explicit resume ---------------------------------------------------------
# The auto-resume of commit af243c2 (--resume-from-prior-attempt) only finds SIBLING run
# dirs keyed on the CURRENT $SLURM_JOB_ID, so it cannot reach a run that died under a
# DIFFERENT job id (e.g. a --time timeout, which is not a requeue). WAIV_RESUME_FROM is
# the generic escape hatch: point it at any complete step_* dir and train_lora.py's
# --resume-from takes it, guarded by assert_resume_config_matches (so an arm/config
# mismatch fails loudly instead of silently mixing trajectories).
#
#   WAIV_ARM=f WAIV_RESUME_FROM=/abs/path/runs/<run>/step_0000750 \
#     sbatch --time=14:00:00 scripts/gridcmp3.sbatch
RESUME_ARGS=()
if [ -n "${WAIV_RESUME_FROM:-}" ]; then
  [ -d "$WAIV_RESUME_FROM" ] || { echo "WAIV_RESUME_FROM is not a directory: $WAIV_RESUME_FROM" >&2; exit 4; }
  RESUME_ARGS+=(--resume-from "$WAIV_RESUME_FROM")
  echo "=== resume: $WAIV_RESUME_FROM ==="
fi

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} | restart ${SLURM_RESTART_COUNT:-0} | run $RUN_NAME ==="
echo "=== arm $ARM | seed $SEED | B=$((COND * TILES)) | neg/row $((TILES - 1)) | rows $((COND * (COND - 1) * TILES)) ==="
echo "=== sampler ${SAMPLER_ARGS[*]} | pin $PIN ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

# Fail loudly at launch rather than at the first checkpoint: this is the exact check the
# two dead followers would have failed.
"$PY" - "$PIN" <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / "src"))
import waivphaet
from waivphaet.eval.pathorob_adapter import PathoRobPaths
assert waivphaet.__file__.startswith(sys.argv[1]), f"PYTHONPATH lost to .venv: {waivphaet.__file__}"
PathoRobPaths(root=Path(sys.argv[1]) / "third_party" / "PathoROB").check()
print(f"[pin] waivphaet={waivphaet.__file__} | PathoROB resolves: OK", flush=True)
EOF

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
# The completed arms all scored this exact condition set. If the manifest ever changes,
# the RI numbers stop being comparable -- so the digest is asserted, not just printed.
md5sum "$OUT_DIR/conditions_used.json"

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
  --format=csv,noheader -l 5 > "$OUT_DIR/gpu.csv" &
SAMPLER_PID=$!
trap 'kill $SAMPLER_PID 2>/dev/null || true' EXIT

# --- embedding probe, BEFORE (base phikon-v2, pinned condition set) --------------------
CUDA_VISIBLE_DEVICES=1 "$PY" "$PIN"/scripts/embed_probe.py \
  --packed-dir "$PACKED_DIR" \
  --conditions-file "$OUT_DIR/conditions_used.json" \
  --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" \
  --proj-out-dim "$PROJ_OUT_DIM" \
  --n-tiles "$PROBE_TILES" \
  --out "$OUT_DIR/probe_before.json"

# --- eval follower on GPU 1 -----------------------------------------------------------
# --run-dir is ABSOLUTE and --python is EXPLICIT. Both are load-bearing under a pinned
# source: eval_checkpoints.py derives its REPO from its own __file__, which under $PIN is
# the snapshot -- a relative run-dir would send embed_probe looking for <snapshot>/runs/,
# and the default --python would resolve to <snapshot>/.venv/bin/python.
#
# NO --model-prefix. The default ("waiv_" + run dir name, dashes to underscores) is what
# named every comparable arm's feature and results dirs; passing a hand-built prefix
# renames this arm out of the comparison.
rm -f "$OUT_DIR/TRAIN_DONE"
CUDA_VISIBLE_DEVICES=1 "$PY" "$PIN"/scripts/eval_checkpoints.py \
  --run-dir "$OUT_DIR" \
  --packed-dir "$PACKED_DIR" \
  --conditions-file "$OUT_DIR/conditions_used.json" \
  --python "$PY" \
  --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" \
  --proj-out-dim "$PROJ_OUT_DIM" \
  --probe-tiles "$PROBE_TILES" \
  --num-workers 8 \
  --max-wait-s "$EVAL_MAX_WAIT_S" \
  --follow --stop-file "$OUT_DIR/TRAIN_DONE" --poll-s 60 \
  > "$OUT_DIR/eval_follow.log" 2>&1 &
FOLLOWER=$!
trap 'kill $SAMPLER_PID $FOLLOWER 2>/dev/null || true' EXIT
echo "=== eval follower pid $FOLLOWER -> $OUT_DIR/eval_follow.log ==="

# --- training on GPU 0 ----------------------------------------------------------------
set +e
CUDA_VISIBLE_DEVICES=0 "$PY" "$PIN"/scripts/train_lora.py \
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
  ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
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
