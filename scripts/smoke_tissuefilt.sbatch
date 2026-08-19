#!/bin/bash
#SBATCH --job-name=waiv-smoke-tissuefilt
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
PIN="/admin/home/ryan.kim/waiv-snapshots/tissuefilt-pinned"
export PYTHONPATH="$PIN/src${PYTHONPATH:+:$PYTHONPATH}"

[ -d "$PIN/src/waivphaet" ]         || { echo "pin missing src/waivphaet: $PIN" >&2; exit 3; }
[ -f "$PIN/scripts/train_lora.py" ] || { echo "pin missing train_lora.py: $PIN" >&2; exit 3; }
grep -q 'delegate rather than refuse' "$PIN/src/waivphaet/train/contrastive.py" \
  || { echo "PIN LACKS pool-head delegation: $PIN" >&2; exit 3; }
grep -q 'min_tissue_frac' "$PIN/src/waivphaet/train/contrastive.py" \
  || { echo "PIN LACKS min_tissue_frac field: $PIN" >&2; exit 3; }

PACKED_DIR="${WAIV_PACKED_DIR:-/data/plism/repacked}"
SCRATCHPAD="/tmp/claude-1034600028/-admin-home-ryan-kim-waiv/b1353922-b3ae-4729-bb58-e9034ea942d2/scratchpad"

COMMON=(
  --packed-dir "$PACKED_DIR"
  --pooling clsmean --proj-out-dim 512
  --lora-rank 32 --lora-alpha 64
  --temperature 0.07 --lr 1e-4
  --max-steps 20 --warmup-steps 5
  --grid --grid-conditions 2 --grid-tiles 900 --grid-forward-chunk 0
  --split-heads --cls-weight 0.5 --mean-weight 0.5 --pool-head gem
  --grad-checkpointing --activation-offload
  --workers 4 --amp bfloat16 --log-every 5
  --ckpt-every 20 --eval-every 20
  --seed 0
)

echo "=== ARM 1: tissuefilt (tissue filter, normal heldout) ==="
SMOKE1="$SCRATCHPAD/smoke-arm1-tissuefilt"
mkdir -p "$SMOKE1"
CUDA_VISIBLE_DEVICES=0 "$PY" "$PIN/scripts/train_lora.py" \
  --out-dir "$SMOKE1" \
  --heldout-scanners GT450 S210 \
  --heldout-stains HRH KR MY \
  --min-tissue-frac 0.25 \
  "${COMMON[@]}"
echo "ARM1 rc=$?"
echo "--- ARM1 config.json ---"
python3 -c "import json; c=json.load(open('$SMOKE1/config.json')); print('min_tissue_frac:', c.get('min_tissue_frac')); print('grid_tiles:', c.get('grid_tiles'))"

echo ""
echo "=== ARM 2: all91 (all conditions, no tissue filter) ==="
SMOKE2="$SCRATCHPAD/smoke-arm2-all91"
mkdir -p "$SMOKE2"
CUDA_VISIBLE_DEVICES=0 "$PY" "$PIN/scripts/train_lora.py" \
  --out-dir "$SMOKE2" \
  --heldout-scanners \
  --heldout-stains \
  "${COMMON[@]}"
echo "ARM2 rc=$?"
echo "--- ARM2 config.json ---"
python3 -c "import json; c=json.load(open('$SMOKE2/config.json')); print('min_tissue_frac:', c.get('min_tissue_frac')); print('grid_tiles:', c.get('grid_tiles'))"
# Check training conditions count from conditions_used.json
echo "ARM2 training conditions count:"
python3 -c "import json; conds=json.load(open('$SMOKE2/conditions_used.json')); print(len(conds), 'conditions')"

echo ""
echo "=== ARM 3: fixed (tissue filter + all conditions) ==="
SMOKE3="$SCRATCHPAD/smoke-arm3-fixed"
mkdir -p "$SMOKE3"
CUDA_VISIBLE_DEVICES=0 "$PY" "$PIN/scripts/train_lora.py" \
  --out-dir "$SMOKE3" \
  --heldout-scanners \
  --heldout-stains \
  --min-tissue-frac 0.25 \
  "${COMMON[@]}"
echo "ARM3 rc=$?"
echo "--- ARM3 config.json ---"
python3 -c "import json; c=json.load(open('$SMOKE3/config.json')); print('min_tissue_frac:', c.get('min_tissue_frac')); print('grid_tiles:', c.get('grid_tiles'))"
python3 -c "import json; conds=json.load(open('$SMOKE3/conditions_used.json')); print(len(conds), 'conditions')"

echo "=== all smoke tests done ==="
