#!/bin/bash
#SBATCH --job-name=waiv-smoke
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#
# A *genuine* smoke test of the contrastive loop, on 1 H100 (PLAN.md 3 phase 7 dry run).
#
# Distinct from `scripts/smoke_test.py`, which was wiring-only: at the time only two
# slides were local and they shared the GIVH stain, so there was exactly ONE positive
# direction and the loss collapsed in a step. With the full stain x scanner grid
# repacked, this job answers the questions that one could not:
#
#   1. does the loss fall without collapsing to ~0 (collapse => trivial task or leakage)
#   2. tiles/s off the memmaps, and is the GPU data-starved (nvidia-smi sampled alongside)
#   3. memory headroom at this rank/batch, to size the real run
#   4. does the embedding move the RIGHT way -- matched cross-condition cosine up,
#      random-pair cosine flat (embed_probe.py before and after, same tiles, same GPU)
#
#   sbatch scripts/smoke_contrastive.sbatch
#   sbatch scripts/smoke_contrastive.sbatch --max-steps 600 --lora-rank 8

set -euo pipefail

REPO=/admin/home/ryan.kim/waiv
cd "$REPO"

export HF_HOME=/data/ryan.kim/hf_home
export HF_HUB_OFFLINE=1              # phikon-v2 is already cached; never hit the hub here
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1

PACKED_DIR=/data/ryan.kim/plism/repacked
RUN_NAME="${SLURM_JOB_NAME:-waiv-smoke}-${SLURM_JOB_ID:-local}"
OUT_DIR="$REPO/runs/$RUN_NAME"
mkdir -p "$OUT_DIR"
PY="$REPO/.venv/bin/python"

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

# The acquisition job may still be repacking. Use only slides the manifest marks
# verified=true, and snapshot that list so the numbers below are reproducible.
"$PY" - "$PACKED_DIR" "$OUT_DIR" <<'EOF'
import json, sys
from pathlib import Path
packed, out = Path(sys.argv[1]), Path(sys.argv[2])
man = json.loads((packed / "manifest.json").read_text())
ok = {k: v for k, v in man.items() if v.get("verified")}
(out / "conditions_used.json").write_text(json.dumps(sorted(ok), indent=2))
stains = sorted({v["stain"] for v in ok.values()})
scanners = sorted({v["scanner"] for v in ok.values()})
print(f"[preflight] {len(ok)} verified slides | {len(stains)} stains {stains} "
      f"| {len(scanners)} scanners {scanners}")
assert len(stains) >= 2 and len(scanners) >= 2, "need >1 stain AND >1 scanner"
EOF

# --- GPU sampler: the only honest answer to "is the GPU data-starved?" ----------------
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used \
  --format=csv,noheader -l 2 > "$OUT_DIR/gpu.csv" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

# --- 1. embedding probe, BEFORE -------------------------------------------------------
"$PY" scripts/embed_probe.py --packed-dir "$PACKED_DIR" --conditions-file "$OUT_DIR/conditions_used.json" --out "$OUT_DIR/probe_before.json"

# --- 2. short training run ------------------------------------------------------------
"$PY" scripts/train_lora.py \
  --packed-dir "$PACKED_DIR" \
  --out-dir "$OUT_DIR" \
  --heldout-scanners GT450 S210 \
  --heldout-stains HRH KR MY \
  --lora-rank 8 \
  --temperature 0.07 \
  --lr 1e-4 \
  --max-steps 300 \
  --warmup-steps 30 \
  --n-groups 2 \
  --group-size 32 \
  --workers 12 \
  --amp bfloat16 \
  --log-every 10 \
  --ckpt-every 150 \
  --eval-every 100 \
  "$@"

LAST_CKPT=$(ls -d "$OUT_DIR"/step_* | sort | tail -1)
echo "=== last checkpoint: $LAST_CKPT ==="

# --- 3. embedding probe, AFTER (same tiles, same seed, same GPU) ----------------------
"$PY" scripts/embed_probe.py --packed-dir "$PACKED_DIR" --lora-rank 8 \
  --conditions-file "$OUT_DIR/conditions_used.json" --adapter "$LAST_CKPT" --out "$OUT_DIR/probe_after.json"

kill $SAMPLER 2>/dev/null || true
echo "=== done $(date) -> $OUT_DIR ==="
