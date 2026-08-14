#!/bin/bash
#SBATCH --job-name=waiv-gridcmp2-follower
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# RESCUE follower for a gridcmp2 arm whose in-job follower died.
#
# Why this exists: `gridcmp2_pinned.sbatch` runs its arm against a read-only source
# snapshot at $PIN, but that snapshot contained only src/ scripts/ tests/ -- NOT
# third_party/. `eval_checkpoints.py` calls `PathoRobPaths.check()` at startup, which
# resolves PathoROB relative to the snapshot root, so the follower for jobs 380889
# (ctrlseed) and 380890 (grid2) died on its FIRST line with FileNotFoundError while
# training carried on happily. Those arms would have finished with a full set of
# checkpoints and an empty ri_curve.json -- silent, and only visible at collection time.
#
# $PIN/third_party is now a symlink to the real repo's third_party. This job re-attaches
# a follower to an already-running arm. Checkpoints persist on disk, so a follower
# started late still discovers and evaluates every step_* dir that already landed.
#
#   WAIV_RUN_DIR=runs/gridcmp2-ctrlseed-380889 sbatch scripts/gridcmp2_follower.sbatch
#
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

# Same pinned source the arm itself imports, so the eval path cannot drift from the
# training path while the working tree is still being edited by another session.
PIN="/admin/home/ryan.kim/waiv-snapshots/gridcmp2-pinned"
export PYTHONPATH="$PIN/src${PYTHONPATH:+:$PYTHONPATH}"

# NOTE: no apostrophes in this message -- bash applies quoting rules to the text inside
# ${var:?...}, so an unpaired ' opens a string and the script dies with a confusing
# "unexpected EOF" reported at some unrelated later line.
RUN_DIR="${WAIV_RUN_DIR:?set WAIV_RUN_DIR to the run directory of the arm}"
[ -d "$RUN_DIR" ] || { echo "no such run dir: $RUN_DIR" >&2; exit 2; }
PACKED_DIR="${WAIV_PACKED_DIR:-/data/plism/repacked}"
PY="$REPO/.venv/bin/python"

# Fail loudly here rather than 250 steps from now.
"$PY" - "$PIN" <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / "src"))
from waivphaet.eval.pathorob_adapter import PathoRobPaths
PathoRobPaths(root=Path(sys.argv[1]) / "third_party" / "PathoROB").check()
print("[follower] PathoROB resolves from the pinned snapshot: OK", flush=True)
EOF

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} | rescue follower for $RUN_DIR ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# --model-prefix must match what the in-job follower would have used, so feature dirs
# and results are named the same way as the other arms (derived from the run dir name).
PREFIX=$(basename "$RUN_DIR")
PREFIX=${PREFIX//-/_}

exec "$PY" "$PIN/scripts/eval_checkpoints.py" \
  --run-dir "$RUN_DIR" \
  --packed-dir "$PACKED_DIR" \
  --conditions-file "$RUN_DIR/conditions_used.json" \
  --model-prefix "$PREFIX" \
  --lora-rank 32 --lora-alpha 64 \
  --num-workers 8 \
  --follow --stop-file "$RUN_DIR/TRAIN_DONE" --poll-s 60 \
  --purge-features
