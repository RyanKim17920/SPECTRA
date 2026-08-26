#!/bin/bash
#SBATCH --job-name=waiv-smoke-auxlearn
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
# LEARNABILITY smoke for the aux stain/scanner heads.
#
# WHY THIS EXISTS, SEPARATE FROM smoke_auxstain.sbatch. That script proves the three
# CONTRACT properties (no-op at 0, runs at >0, detach isolates the backbone) and it runs
# at the production geometry C=2. But C=2 means each step shows the heads only TWO
# conditions, i.e. at most 2 of 10 stain labels and 2 of 5 scanner labels. Over 30 steps
# that is far too few label draws to show a CE trend -- measured slope was -0.002/step
# against a chance CE of ln(10)=2.303, which proves nothing either way.
#
# Learnability is a different question from the contract, so it gets its own geometry:
# C=8 conditions per step (8 distinct (stain,scanner) pairs) and 150 steps. Still tiny --
# 8 x 16 = 128 images/step -- but now the head actually sees the label space.
#
# The claim under test is ONLY "the labels are real and linearly decodable from the
# backbone embedding", i.e. not shuffled garbage. Chance is CE = ln(10) = 2.3026 for
# stain and ln(5) = 1.6094 for scanner; accuracy chance is 0.10 and 0.20.
# ======================================================================================
set -euo pipefail

REPO="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
cd "$REPO"

export HF_HOME="${HF_HOME:-/data/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export WAIV_NO_FOLLOWER=1

PY="$REPO/.venv/bin/python"
PIN="/admin/home/ryan.kim/waiv-snapshots/auxstain-pinned"
PACKED_DIR="${WAIV_PACKED_DIR:-/data/plism/repacked}"
OUT="$REPO/runs/smoke-auxlearn-${SLURM_JOB_ID:-local}"

mkdir -p "$OUT"
echo "=== auxlearn start $(date) -> $OUT ==="

PYTHONPATH="$PIN/src" CUDA_VISIBLE_DEVICES=0 "$PY" "$PIN/scripts/train_lora.py" \
  --out-dir "$OUT/real" \
  --packed-dir "$PACKED_DIR" \
  --backbone owkin/phikon-v2 \
  --heldout-scanners GT450 S210 \
  --heldout-stains HRH KR MY \
  --pooling clsmean --proj-out-dim 512 \
  --lora-rank 16 --lora-alpha 32 \
  --temperature 0.07 --lr 1e-3 \
  --max-steps 150 --warmup-steps 2 \
  --grid --grid-conditions 8 --grid-tiles 16 --grid-forward-chunk 0 \
  --split-heads --cls-weight 0.5 --mean-weight 0.5 --pool-head gem \
  --mask-same-core \
  --grad-checkpointing \
  --workers 4 --amp bfloat16 --log-every 1 \
  --ckpt-every 100000 --eval-every 100000 --seed 0 \
  --aux-stain-weight 0.5 --aux-scanner-weight 0.5 \
  > "$OUT/real.log" 2>&1

echo "=== VERDICT ==="
"$PY" - "$OUT/real" <<'EOF'
import json, math, sys
from pathlib import Path

h = json.loads((Path(sys.argv[1]) / "history.json").read_text())

def ols(y):
    n = len(y); xm = (n - 1) / 2; ym = sum(y) / n
    num = sum((i - xm) * (v - ym) for i, v in enumerate(y))
    den = sum((i - xm) ** 2 for i in range(n))
    return num / den

print(f"steps: {len(h)}   C={h[0]['n_cond']:.0f} T={h[0]['n_tiles']:.0f}")
print()
for axis, n_cls in (("stain", 10), ("scanner", 5)):
    ce = [r[f"loss_aux_{axis}"] for r in h]
    ac = [r[f"acc_aux_{axis}"] for r in h]
    chance_ce, chance_acc = math.log(n_cls), 1.0 / n_cls
    q = len(ce) // 4
    print(f"{axis.upper()}  ({n_cls} classes; chance CE={chance_ce:.4f}, chance acc={chance_acc:.3f})")
    print(f"   CE   first-quarter {sum(ce[:q])/q:.4f}  ->  last-quarter {sum(ce[-q:])/q:.4f}"
          f"   (delta {sum(ce[-q:])/q - sum(ce[:q])/q:+.4f})")
    print(f"   CE   OLS slope/step {ols(ce):+.5f}   min {min(ce):.4f}   finite={all(math.isfinite(v) for v in ce)}")
    print(f"   ACC  first-quarter {sum(ac[:q])/q:.3f}  ->  last-quarter {sum(ac[-q:])/q:.3f}"
          f"   (chance {chance_acc:.3f})")
    below = sum(1 for v in ce[-q:] if v < chance_ce)
    print(f"   last-quarter steps with CE below chance: {below}/{q}")
    print()
print("InfoNCE term still training:",
      f"{h[0]['loss']:.4f} -> {h[-1]['loss']:.4f}")
EOF

echo "=== auxlearn done $(date) ==="
