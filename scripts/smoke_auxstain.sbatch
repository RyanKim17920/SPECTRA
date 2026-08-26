#!/bin/bash
#SBATCH --job-name=waiv-smoke-auxstain
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# ======================================================================================
# SMOKE TEST for --aux-stain-weight / --aux-scanner-weight / --aux-detach-backbone
# (factored invariance: InfoNCE stays on the PROJECTION, aux CE runs on the BACKBONE)
#
# Pin under test: waiv-snapshots/auxstain-pinned (fork of falseneg-pinned)
# Baseline pin  : waiv-snapshots/falseneg-pinned  (the tree the mask arms actually run)
#
# Five tiny runs, one GPU, ~30 steps each. NOT a training slot -- C=2 x T=16.
#
#   A_old  falseneg-pinned, no aux flags        -- pre-change reference
#   A_new  auxstain-pinned, no aux flags        -- (a) must equal A_old
#   A_new2 auxstain-pinned, no aux flags        -- run-to-run noise floor
#   B      auxstain-pinned, aux 0.1 / 0.1       -- (b) CE finite and falling
#   C      auxstain-pinned, aux 0.1 / 0.1 + detach
#                                               -- (c) InfoNCE curve must equal A_new
#                                                      exactly (backbone grads unchanged)
#
# Every arm carries the real recipe's composition surface: --split-heads --pool-head gem
# --mask-same-core, so the new flag is proven to compose with them.
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
# No wandb, no eval follower -- this is a code-path smoke, not an experiment.
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export WAIV_NO_FOLLOWER=1

PY="$REPO/.venv/bin/python"
NEW_PIN="/admin/home/ryan.kim/waiv-snapshots/auxstain-pinned"
OLD_PIN="/admin/home/ryan.kim/waiv-snapshots/falseneg-pinned"
PACKED_DIR="${WAIV_PACKED_DIR:-/data/plism/repacked}"
BASE_OUT="${WAIV_SMOKE_OUT:-$REPO/runs/smoke-auxstain-${SLURM_JOB_ID:-local}}"

for P in "$NEW_PIN" "$OLD_PIN"; do
  [ -f "$P/scripts/train_lora.py" ] || { echo "pin missing train_lora.py: $P" >&2; exit 3; }
done
grep -q 'aux_stain_weight' "$NEW_PIN/src/waivphaet/train/contrastive.py" \
  || { echo "NEW PIN LACKS aux_stain_weight: $NEW_PIN" >&2; exit 3; }
if grep -q 'aux_stain_weight' "$OLD_PIN/src/waivphaet/train/contrastive.py"; then
  echo "OLD PIN ALREADY HAS aux_stain_weight -- not a clean baseline" >&2; exit 3
fi
[ -f "$REPO/runs/.plism_core_labels.npy" ] \
  || { echo "missing runs/.plism_core_labels.npy (needed by --mask-same-core)" >&2; exit 3; }

mkdir -p "$BASE_OUT"
echo "=== smoke start $(date) -> $BASE_OUT ==="

# Small on every axis, and warmup=2 with lr=1e-3 so 30 steps actually MOVE something --
# at the production warmup of 200 the LR is ~1e-6 here and no curve would be readable.
COMMON=(
  --packed-dir "$PACKED_DIR"
  --backbone owkin/phikon-v2
  --heldout-scanners GT450 S210
  --heldout-stains HRH KR MY
  --pooling clsmean
  --proj-out-dim 512
  --lora-rank 16 --lora-alpha 32
  --temperature 0.07
  --lr 1e-3
  --max-steps 30
  --warmup-steps 2
  --grid --grid-conditions 2 --grid-tiles 16
  --grid-forward-chunk 0
  --split-heads --cls-weight 0.5 --mean-weight 0.5 --pool-head gem
  --mask-same-core
  --grad-checkpointing
  --workers 4 --amp bfloat16 --log-every 1
  --ckpt-every 100000 --eval-every 100000
  --seed 0
)

run_arm () {
  local name="$1"; local pin="$2"; shift 2
  local out="$BASE_OUT/$name"
  local rc=0
  echo "--- arm $name (pin: $pin) ---"
  PYTHONPATH="$pin/src" CUDA_VISIBLE_DEVICES=0 "$PY" "$pin/scripts/train_lora.py" \
    --out-dir "$out" "${COMMON[@]}" "$@" > "$BASE_OUT/$name.log" 2>&1 || rc=$?
  echo "    rc=$rc -> $out"
  if [ "$rc" -ne 0 ]; then
    echo "ARM $name FAILED -- tail of $BASE_OUT/$name.log:" >&2
    tail -40 "$BASE_OUT/$name.log" >&2
    exit "$rc"
  fi
}

run_arm A_old  "$OLD_PIN"
run_arm A_new  "$NEW_PIN"
run_arm A_new2 "$NEW_PIN"
run_arm B      "$NEW_PIN" --aux-stain-weight 0.1 --aux-scanner-weight 0.1
run_arm C      "$NEW_PIN" --aux-stain-weight 0.1 --aux-scanner-weight 0.1 --aux-detach-backbone

echo "=== VERDICT ==="
"$PY" - "$BASE_OUT" <<'EOF'
import json, sys
from pathlib import Path

base = Path(sys.argv[1])

def hist(name):
    return json.loads((base / name / "history.json").read_text())

def col(name, key):
    return [r[key] for r in hist(name) if key in r]

A_old, A_new, A_new2 = col("A_old", "loss"), col("A_new", "loss"), col("A_new2", "loss")
B, C = col("B", "loss"), col("C", "loss")

def maxdiff(x, y):
    n = min(len(x), len(y))
    return max(abs(a - b) for a, b in zip(x[:n], y[:n])) if n else float("nan")

print(f"steps logged: A_old={len(A_old)} A_new={len(A_new)} B={len(B)} C={len(C)}")
print()
print("(a) DEFAULT PATH IS A NO-OP")
print(f"    A_new  vs A_old   max|dloss| = {maxdiff(A_new, A_old):.3e}")
print(f"    A_new2 vs A_new   max|dloss| = {maxdiff(A_new2, A_new):.3e}   <- run-to-run floor")
for arm in ("A_old", "A_new"):
    files = sorted(p.name for p in (base / arm).rglob("step_*/*"))
    print(f"    {arm} checkpoint files: {files}")
print()
print("(b) AUX HEADS LEARN (labels are real, not shuffled garbage)")
for arm in ("B", "C"):
    for axis in ("stain", "scanner"):
        ce = col(arm, f"loss_aux_{axis}")
        ac = col(arm, f"acc_aux_{axis}")
        if not ce:
            print(f"    {arm}/{axis}: ABSENT")
            continue
        finite = all(v == v and abs(v) != float("inf") for v in ce)
        print(f"    {arm}/{axis}: CE {ce[0]:.4f} -> {ce[-1]:.4f} "
              f"(first3 mean {sum(ce[:3])/3:.4f}, last3 mean {sum(ce[-3:])/3:.4f}) "
              f"acc {ac[0]:.3f} -> {ac[-1]:.3f} finite={finite}")
print()
print("(c) --aux-detach-backbone LEAVES THE BACKBONE UPDATE UNCHANGED")
print(f"    C vs A_new  max|d InfoNCE loss| = {maxdiff(C, A_new):.3e}  (expect ~= floor)")
print(f"    B vs A_new  max|d InfoNCE loss| = {maxdiff(B, A_new):.3e}  (expect >> floor)")
print()
print("    aux_heads.pt present?")
for arm in ("A_new", "B", "C"):
    ck = sorted((base / arm).glob("step_*"))
    have = [ (d.name, (d / "aux_heads.pt").exists()) for d in ck ]
    print(f"      {arm}: {have}")
EOF

echo "=== smoke done $(date) ==="
