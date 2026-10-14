#!/bin/bash
#SBATCH --job-name=waiv-armf-score
#SBATCH --partition=main
#SBATCH --account=max
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# Score whatever gridcmp3 ARM F checkpoints are on disk but missing from ri_curve.json.
#
# WHY THIS EXISTS. Arm F's in-job eval follower cannot be trusted to drain:
#
#   * job 382243 (the original arm F) TIMED OUT at --time=10:00:00. Its follower had
#     already exited before step_0000750 was flushed, so that checkpoint -- complete,
#     with adapter/ + optim.pt + projector.pt + metrics.json -- was never scored, and
#     ri_curve.json stops at [250, 500]. The RESUMED job (383342) restarts at global
#     step 750 with --ckpt-every 250, so its first checkpoint is step 1000: it will
#     NEVER re-emit a step-750 point. The orphan is the only source for RI@750.
#
#   * job 383342 itself inherits the follower's `--max-wait-s` default of 8h (see
#     eval_checkpoints.py). It runs ~38 s/step, so step 1500 lands ~8.2 h in -- AFTER
#     the follower has already printed "max wait exceeded" and exited. gridcmp3.sbatch
#     now passes an explicit WAIV_EVAL_MAX_WAIT_S (default 20 h), but 383342 was
#     submitted before that edit and runs from the read-only pin, so the fix does not
#     reach it. This script is the safety net for its tail.
#
# IDEMPOTENT. eval_checkpoints.py reads the existing ri_curve.json, skips every step
# already in it, reuses any probe_step_*.json already on disk, and skips PathoROB
# extraction for any features dir that already holds *.npz. Re-running it is always
# safe; it will simply find nothing to do.
#
# USAGE
#   # the step-750 orphan of the timed-out original (already done, job 383486):
#   sbatch scripts/score_armf_tail.sbatch runs/gridcmp3-f-382243
#
#   # the tail of the resumed run -- ONLY AFTER 383342 has finished:
#   sbatch scripts/score_armf_tail.sbatch
#
set -euo pipefail

REPO="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
cd "$REPO"

# Default target is the resumed arm F. Pass an explicit run dir to score another one.
RUN_DIR="${1:-runs/gridcmp3-f-383342}"
case "$RUN_DIR" in /*) ;; *) RUN_DIR="$REPO/$RUN_DIR" ;; esac
[ -d "$RUN_DIR" ] || { echo "no such run dir: $RUN_DIR" >&2; exit 2; }

# --- SAME PIN THE ARM TRAINED AND SCORED UNDER ----------------------------------------
# Load-bearing, not hygiene. eval_checkpoints.py derives REPO from its own __file__ and
# builds PathoRobPaths from it, so the PIN's copy writes features/results into
#   $PIN/third_party/PathoROB/{data/features,results/robustness_index}/waiv_<run>_s<step>/
# which is exactly where arm F's steps 250 and 500 already live. Running the working-tree
# copy instead would score into a DIFFERENT PathoROB tree against a DIFFERENT source
# snapshot, and the resulting number would not be comparable with the points it joins.
PIN="${WAIV_PIN:-/admin/home/ryan.kim/waiv-snapshots/gridcmp3-f-pinned}"
[ -d "$PIN/src/waivphaet" ]        || { echo "pin missing src/waivphaet: $PIN" >&2; exit 3; }
[ -d "$PIN/third_party/PathoROB" ] || { echo "pin missing third_party/PathoROB: $PIN" >&2; exit 3; }
export PYTHONPATH="$PIN/src${PYTHONPATH:+:$PYTHONPATH}"

export HF_HOME="${HF_HOME:-/data/huggingface}"
export HF_HUB_OFFLINE=0               # the PathoROB parquet is streamed
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1

PACKED_DIR="${WAIV_PACKED_DIR:-/data/plism/repacked}"
PY="$REPO/.venv/bin/python"

# Identical to gridcmp3.sbatch. These are the values arm F was TRAINED and SCORED with;
# build_model hard-fails on a rank/alpha mismatch rather than scoring a mis-shaped adapter.
LORA_RANK=32
LORA_ALPHA=64
PROJ_OUT_DIM=512
PROBE_TILES=256

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} ==="
echo "=== run-dir $RUN_DIR | pin $PIN ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

# What is on disk vs what the curve already has -- printed BEFORE the work so the log
# says up front which steps this pass intends to add.
"$PY" - "$RUN_DIR" <<'EOF'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
ck = sorted(int(d.name.split("_")[1]) for d in run.glob("step_*")
            if (d / "metrics.json").exists() and (d / "adapter").is_dir())
cur = run / "ri_curve.json"
have = sorted(p["step"] for p in json.loads(cur.read_text())["points"]) if cur.exists() else []
print(f"[plan] complete checkpoints on disk : {ck}")
print(f"[plan] steps already in ri_curve    : {have}")
print(f"[plan] this pass will score         : {[s for s in ck if s not in have]}")
EOF

# NO --model-prefix and NO --follow.
#   --model-prefix: the default is "waiv_" + run-dir name with dashes as underscores,
#     which is what named arm F's existing feature/result dirs. Passing a hand-built
#     prefix would rename this point out of the comparison.
#   --follow: this is a one-shot drain over what is already on disk. Without --follow the
#     loop makes exactly one pass and exits, so --max-wait-s is irrelevant here -- the
#     failure mode this script exists to repair cannot recur inside the script itself.
"$PY" "$PIN"/scripts/eval_checkpoints.py \
  --run-dir "$RUN_DIR" \
  --packed-dir "$PACKED_DIR" \
  --conditions-file "$RUN_DIR/conditions_used.json" \
  --python "$PY" \
  --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" \
  --proj-out-dim "$PROJ_OUT_DIM" \
  --probe-tiles "$PROBE_TILES" \
  --num-workers 12

echo "=== RI vs step ==="
"$PY" - "$RUN_DIR" <<'EOF'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "ri_curve.json"
d = json.loads(p.read_text())
for pt in d["points"]:
    print(f"  step {pt['step']:>5}  avgRI {pt['avg_robustness_index']:.4f}  "
          f"avgBA {pt['avg_balanced_accuracy']:.4f}")
EOF
echo "=== done $(date) -> $RUN_DIR ==="
