#!/bin/bash
#SBATCH --job-name=waiv-finalgem-score
#SBATCH --partition=main
#SBATCH --account=max
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# Score the three finalgem runs, whose in-job eval followers never scored ANYTHING.
#
# WHY THIS EXISTS. All three finalgem arms trained to completion (TRAIN_DONE present, all
# checkpoints on disk with adapter/ + projector*.pt + pool_head.pt + metrics.json) but every
# follower died on its FIRST line: the pin
#   /admin/home/ryan.kim/waiv-snapshots/finalgem-pinned
# was created WITHOUT the third_party/ and .venv/ symlinks, so PathoRobPaths.check() raised
# FileNotFoundError before a single checkpoint was read. That is failure mode #1 in
# scripts/make_pin.sh's own header. The pin has since been repaired by adding both symlinks;
# this script re-checks that repair (see PREFLIGHT below) rather than assuming it.
#
#   runs/finalgem-phikon-384585    8 ckpts (every 500, 4000 steps)  owkin/phikon-v2
#   runs/finalgem-midnight-384586  6 ckpts (every 250, 1500 steps)  kaiko-ai/midnight
#   runs/finalgem-virchow2-384587  6 ckpts (every 250, 1500 steps)  paige-ai/Virchow2
#
# IDEMPOTENT, same as scripts/score_armf_tail.sbatch, which this is modelled on:
# eval_checkpoints.py reads the existing ri_curve.json and skips every step already in it,
# reuses any probe_step_*.json on disk, and skips PathoROB extraction for any feature dir
# that already holds *.npz. Re-running is always safe.
#
# NO --pooling AND NO --pool-head -- THIS IS LOAD-BEARING, NOT AN OMISSION.
# eval_checkpoints.py defaults to clsmean, which is the protocol constant (PathoROB's
# reference row is phikonv2_clsmean), and it has no --infer-pool-head to pass: the trained
# GeM head in each checkpoint's pool_head.pt is deliberately NOT restored at eval. Quoting
# scripts/finalgem.sbatch's own header: every RI number in the seed study -- the entire basis
# for picking GEM -- was scored through the standard clsmean eval head, with the learned
# pooling reaching the readout only indirectly via the LoRA weights it shaped. Scoring these
# checkpoints WITH the learned pool at inference is a separate, additive experiment; doing it
# here would put these three runs on a different measurement axis from the decision that
# selected them and from the published reconstruction runs they exist to be compared against.
#
# Expect this line in the log on every checkpoint -- it is EXPECTED and benign on a
# split-head run (every eval path reads model.embed(); the projector is training-only):
#   [build_model] skipping projector: trained with a 1024-d input, evaluating at 2048-d
#
# USAGE
#   sbatch --job-name=waiv-score-phikon   scripts/score_finalgem.sbatch phikon
#   sbatch --job-name=waiv-score-midnight scripts/score_finalgem.sbatch midnight
#   sbatch --job-name=waiv-score-virchow2 scripts/score_finalgem.sbatch virchow2
set -euo pipefail

REPO="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
cd "$REPO"

ARM="${1:?usage: score_finalgem.sbatch <phikon|midnight|virchow2>}"
# Backbone args mirror finalgem.sbatch EXACTLY: phikon passes nothing (train_lora.py's and
# embed_probe.py's default is owkin/phikon-v2), the other two name the backbone. Both the
# probe and the PathoROB extractor cross-check this against the value recorded in the
# checkpoint and hard-fail on a contradiction, so a wrong value here cannot score silently.
case "$ARM" in
  phikon)   RUN_DIR="$REPO/runs/finalgem-phikon-384585";   BACKBONE_ARGS=() ;;
  midnight) RUN_DIR="$REPO/runs/finalgem-midnight-384586"; BACKBONE_ARGS=(--backbone kaiko-ai/midnight) ;;
  virchow2) RUN_DIR="$REPO/runs/finalgem-virchow2-384587"; BACKBONE_ARGS=(--backbone paige-ai/Virchow2) ;;
  *) echo "unknown arm '$ARM' (want phikon | midnight | virchow2)" >&2; exit 2 ;;
esac
[ -d "$RUN_DIR" ] || { echo "no such run dir: $RUN_DIR" >&2; exit 2; }

# --- THE SAME PIN THE ARMS TRAINED UNDER ----------------------------------------------
# eval_checkpoints.py derives REPO from its own __file__ and builds PathoRobPaths from it,
# and it shells out to scripts/embed_probe.py and scripts/extract_pathorob_features.py with
# cwd=REPO -- so running the PIN's copy keeps the whole eval chain on pinned source, which
# is what the three arms were trained against (commit in $PIN/PINNED_COMMIT).
PIN="${WAIV_PIN:-/admin/home/ryan.kim/waiv-snapshots/finalgem-pinned}"
[ -d "$PIN/src/waivphaet" ]        || { echo "pin missing src/waivphaet: $PIN" >&2; exit 3; }
[ -d "$PIN/third_party/PathoROB" ] || { echo "pin missing third_party/PathoROB: $PIN" >&2; exit 3; }
[ -x "$PIN/.venv/bin/python" ]     || { echo "pin missing .venv: $PIN" >&2; exit 3; }
export PYTHONPATH="$PIN/src${PYTHONPATH:+:$PYTHONPATH}"

export HF_HOME="${HF_HOME:-/data/huggingface}"
export HF_HUB_OFFLINE=0               # the PathoROB parquet is streamed
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PACKED_DIR="${WAIV_PACKED_DIR:-/data/plism/repacked}"
PY="$REPO/.venv/bin/python"

# Read back out of each run's config.json (encoder block), identical across all three arms.
# build_model hard-fails on a rank/alpha mismatch rather than scoring a mis-shaped adapter.
LORA_RANK=32
LORA_ALPHA=64
PROJ_OUT_DIM=512
PROBE_TILES=256

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} ==="
echo "=== arm $ARM | run-dir $RUN_DIR ==="
echo "=== pin $PIN ($(cat "$PIN/PINNED_COMMIT" 2>/dev/null || echo unknown)) ==="
echo "=== backbone ${BACKBONE_ARGS[*]:-<default owkin/phikon-v2>} ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

# --- PREFLIGHT: prove the repaired pin actually resolves ------------------------------
# This is the exact failure that killed all three followers, so it is checked FIRST and the
# job dies here rather than after burning an hour of extraction. Note PathoRobPaths takes the
# PATHOROB root (its default is the relative "third_party/PathoROB"), NOT the repo root --
# handing it the repo root reproduces the very FileNotFoundError we are recovering from.
"$PY" - "$PIN" <<'EOF'
import sys
from pathlib import Path
pin = Path(sys.argv[1])
sys.path.insert(0, str(pin / "src"))
import waivphaet
assert waivphaet.__file__.startswith(str(pin)), f"PYTHONPATH lost the pin: {waivphaet.__file__}"
from waivphaet.eval.pathorob_adapter import PathoRobPaths
paths = PathoRobPaths(pin / "third_party" / "PathoROB")
paths.check()
print(f"[pin] waivphaet   = {waivphaet.__file__}", flush=True)
print(f"[pin] pathorob    = {paths.root}", flush=True)
print(f"[pin] features -> {paths.features_dir}", flush=True)
print(f"[pin] results  -> {paths.results_dir}", flush=True)
EOF

# What is on disk vs what the curve already has, printed BEFORE the work so the log says up
# front which steps this pass intends to add. For these three runs the curve does not exist
# at all yet, so this should list every checkpoint.
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
#   --model-prefix: the default is "waiv_" + the run-dir name with dashes as underscores.
#     None of these three has any existing feature/result dir, so the default names them.
#   --follow: one-shot drain over what is already on disk. Without it the loop makes exactly
#     one pass and exits, so --max-wait-s is irrelevant -- the idle-timeout failure mode
#     cannot recur inside this script.
"$PY" "$PIN"/scripts/eval_checkpoints.py \
  --run-dir "$RUN_DIR" \
  --packed-dir "$PACKED_DIR" \
  --conditions-file "$RUN_DIR/conditions_used.json" \
  --python "$PY" \
  --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" \
  --proj-out-dim "$PROJ_OUT_DIM" \
  --probe-tiles "$PROBE_TILES" \
  "${BACKBONE_ARGS[@]}" \
  --num-workers 12

echo "=== RI vs step | $ARM ==="
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
