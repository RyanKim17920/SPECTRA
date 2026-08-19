#!/bin/bash
#SBATCH --job-name=waiv-smoke-falseneg
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail

REPO="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
cd "$REPO"

export HF_HOME="${HF_HOME:-/data/huggingface}"
export HF_HUB_OFFLINE=0
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY="$REPO/.venv/bin/python"
PIN="/admin/home/ryan.kim/waiv-snapshots/falseneg-pinned"
export PYTHONPATH="$PIN/src${PYTHONPATH:+:$PYTHONPATH}"

[ -d "$PIN/src/waivphaet" ]         || { echo "pin missing src/waivphaet: $PIN" >&2; exit 3; }
[ -f "$PIN/scripts/train_lora.py" ] || { echo "pin missing train_lora.py: $PIN" >&2; exit 3; }
grep -q 'delegate rather than refuse' "$PIN/src/waivphaet/train/contrastive.py" \
  || { echo "PIN LACKS pool-head delegation" >&2; exit 3; }
grep -q 'mask_same_core' "$PIN/src/waivphaet/train/contrastive.py" \
  || { echo "PIN LACKS mask_same_core" >&2; exit 3; }
[ -f "$REPO/runs/.plism_core_labels.npy" ] \
  || { echo "MISSING core labels -- run derive_core_map.sbatch first" >&2; exit 3; }

PACKED_DIR="${WAIV_PACKED_DIR:-/data/plism/repacked}"
SMOKE_OUT="/tmp/smoke-falseneg-$$"
mkdir -p "$SMOKE_OUT"

echo "=== smoke: falseneg 20 steps ==="
CUDA_VISIBLE_DEVICES=0 "$PY" "$PIN/scripts/train_lora.py" \
  --packed-dir "$PACKED_DIR" \
  --out-dir "$SMOKE_OUT" \
  --heldout-scanners GT450 S210 \
  --heldout-stains HRH KR MY \
  --pooling clsmean \
  --proj-out-dim 512 \
  --lora-rank 32 --lora-alpha 64 \
  --temperature 0.07 \
  --lr 1e-4 \
  --max-steps 20 \
  --warmup-steps 5 \
  --grid --grid-conditions 2 --grid-tiles 900 \
  --grid-forward-chunk 0 \
  --split-heads --cls-weight 0.5 --mean-weight 0.5 --pool-head gem \
  --grad-checkpointing \
  --activation-offload \
  --workers 4 --amp bfloat16 --log-every 5 \
  --ckpt-every 20 --eval-every 20 \
  --seed 0 \
  --mask-same-core
RC=$?
echo "smoke rc=$RC"
echo "--- config.json mask_same_core ---"
python3 -c "import json; c=json.load(open('$SMOKE_OUT/config.json')); print('mask_same_core:', c.get('mask_same_core')); print('core_labels_path:', c.get('core_labels_path'))"
echo "--- history masked_per_row ---"
python3 -c "
import json
h = json.load(open('$SMOKE_OUT/history.json'))
rows = [e.get('masked_per_row', 'n/a') for e in h]
print('masked_per_row per step:', rows)
" 2>/dev/null || echo "(history.json not found)"
rm -rf "$SMOKE_OUT"
echo "=== done ==="
exit $RC
