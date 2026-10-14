#!/bin/bash
#SBATCH --job-name=waiv-gem-s3
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# POOLING-HEAD comparison: ONE arm per job. Three separate 2-GPU jobs, not one 6-GPU
# job, so a node failure or a preemption on one arm does not take the other two with it.
# Structure lifted verbatim from headcmp.sbatch -- see the comparison note below.
#
#   WAIV_ARM=gem   --split-heads --cls-weight 0.5 --mean-weight 0.5 --pool-head gem
#   WAIV_ARM=attn  --split-heads --cls-weight 0.5 --mean-weight 0.5 --pool-head attn
#   WAIV_ARM=lse   --split-heads --cls-weight 0.5 --mean-weight 0.5 --pool-head lse
#
# THE BASELINE IS headcmp's SPLIT ARM, not the gridcmp CTRL. Every flag here is the SPLIT
# arm's (--split-heads, 0.5/0.5, 6 x 200 pair path, phikon-v2, rank 32 / alpha 64, lr 1e-4,
# tau 0.07, 1500 steps, 200 warmup, seed 0); the ONLY thing that moves is --pool-head,
# which replaces the pooling of the non-CLS head. The emitted config.json must therefore
# differ from SPLIT's in exactly pool_head, out_dir and the run name -- anything else
# moving confounds the comparison.
#
# WHY: plain mean pooling gives every token a PROVABLY IDENTICAL gradient (measured
# per-token spread 1.83e-07, i.e. float32 rounding). gem/attn/lse are token-dependent
# (3.94e-01 / 2.58e-01 / 4.35e-01). All three START within ~2% of plain mean in output
# space and must DEPART by learning -- so pool_gem_p / pool_attn_entropy / pool_lse_tau
# in history.json are the collapse check. Still sitting at init (p=3.000, entropy=0.996,
# tau=1.000) at step 1500 means the variant became the mean head again and any RI
# difference is noise, not mechanism.
#
# gem_clamp is deliberately NOT an arm: real phikon-v2 tokens are 50.11% negative by
# entry, so the textbook clamp zeroes half the input (measured 0.5011) and it is
# diagnostic-only.
#
# EVAL POOLING STAYS clsmean ON ALL THREE ARMS. It is a protocol constant (PathoROB's
# reference row is phikonv2_clsmean), not a model choice, and the training pooling must
# not leak into it -- exactly as on headcmp. What the pool head changes is the gradient
# the LoRA backbone receives during training; the eval reads model.embed() and the
# backbone weights are what carry the fine-tune.
#
# GPU 0 trains, GPU 1 runs eval_checkpoints.py --follow so each checkpoint gets its
# PathoROB RI live instead of doubling wall time inline. conditions_used.json is pinned
# from the preflight so the BEFORE probe and the follower score the SAME condition set the
# training run saw.
#
#   WAIV_ARM=gem sbatch --job-name=waiv-poolcmp-gem scripts/poolcmp.sbatch

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
ARM="${WAIV_ARM:?set WAIV_ARM to gem | attn | lse}"
# See gridcmp.sbatch: SLURM_RESTART_COUNT keeps a requeued attempt out of the previous
# attempt's run dir, so a preemption cannot silently mix two training trajectories.
RUN_NAME="gemseed3-${ARM}-${SLURM_JOB_ID:-local}${SLURM_RESTART_COUNT:+.r$SLURM_RESTART_COUNT}"
OUT_DIR="$REPO/runs/$RUN_NAME"
mkdir -p "$OUT_DIR"
PY="$REPO/.venv/bin/python"

LORA_RANK=32
LORA_ALPHA=64

# --- the ONE thing that differs between the three arms --------------------------------
# The loss-head block is FROZEN at headcmp's SPLIT setting on all three arms; only the
# trailing --pool-head moves.
SPLIT_ARGS=(--split-heads --cls-weight 0.5 --mean-weight 0.5)
case "$ARM" in
  gem)  HEAD_ARGS=("${SPLIT_ARGS[@]}" --pool-head gem) ;;
  attn) HEAD_ARGS=("${SPLIT_ARGS[@]}" --pool-head attn) ;;
  lse)  HEAD_ARGS=("${SPLIT_ARGS[@]}" --pool-head lse) ;;
  *) echo "unknown WAIV_ARM=$ARM (want gem | attn | lse)" >&2; exit 2 ;;
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
  --seed 3 \
  "$@"
TRAIN_RC=$?
set -e
echo "=== training exited rc=$TRAIN_RC ==="
touch "$OUT_DIR/TRAIN_DONE"

wait $FOLLOWER || true
kill $SAMPLER_PID 2>/dev/null || true

echo "=== per-head loss curve + learned pooling state ==="
# The pool_* columns are the COLLAPSE CHECK: if they are still at their init values
# (p=3.000, entropy=0.996, tau=1.000) at the last step, this arm pooled like the mean
# head all along and any RI delta vs SPLIT is noise, not the pooling.
"$PY" - "$OUT_DIR" <<'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
h = json.loads((d / "history.json").read_text()) if (d / "history.json").exists() else []
cfg = json.loads((d / "config.json").read_text()) if (d / "config.json").exists() else {}
heads = cfg.get("encoder", {}).get("split_heads", [])
POOL = ("pool_gem_p", "pool_attn_entropy", "pool_lse_tau", "pool_zero_fraction")
present = [k for k in POOL if h and any(k in r for r in h)]
for r in h[::10]:
    parts = " ".join(f"{k}={r.get('loss_' + k, float('nan')):.4f}" for k in heads)
    pool = " ".join(f"{k}={r[k]:.4f}" for k in present if k in r)
    print(f"step {r['step']:5d} loss={r['loss']:.4f} top1={r['top1']:.3f} {parts} {pool}")
if not present:
    print("!! NO pool_* key in history.json -- the collapse check is BLIND, do not "
          "interpret this arm's RI as a pooling effect")
else:
    first, last = h[0], h[-1]
    for k in present:
        print(f"[pool] {k}: {first.get(k)} (step {first['step']}) -> "
              f"{last.get(k)} (step {last['step']})")
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
