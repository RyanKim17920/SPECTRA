#!/bin/bash
#SBATCH --job-name=waiv-lora
#SBATCH --partition=main            # `main` = c-[1-6] + n-[1-8]; use `n` to pin the 8xH100 nodes
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64          # pair sampler is memmap-random-read heavy; give it workers
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#
# Registered-pair contrastive fine-tune of phikon-v2 (PLAN.md 3, phase 7).
#
#   sbatch scripts/train_lora.sbatch                     # defaults
#   sbatch scripts/train_lora.sbatch --lora-rank 32 --temperature 0.05
#
# PLAN.md 3 risk 4: there is no published recipe, so this is one point in a sweep, not
# "the" run. Vary --lr / --lora-rank / --temperature / --max-steps across jobs.

set -euo pipefail

REPO=/admin/home/ryan.kim/waiv
cd "$REPO"

# HF cache MUST live on /data -- /admin has only ~1.4 TB and must not fill (PLAN.md 5).
export HF_HOME=/data/ryan.kim/hf_home
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
# uint8 memmaps are read randomly across up to 91 slides; keep threads from thrashing
export MKL_NUM_THREADS=8

RUN_NAME="${SLURM_JOB_NAME:-waiv-lora}-${SLURM_JOB_ID:-local}"
OUT_DIR="$REPO/runs/$RUN_NAME"
mkdir -p "$OUT_DIR"

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true

# --- preflight ------------------------------------------------------------------------
# Repack must have run: 16,278 tiny HDF5 datasets per slide makes random pair sampling
# I/O bound (~112-125x slower than the contiguous memmap, measured).
PACKED_DIR=/data/ryan.kim/plism_packed
if [ ! -f "$PACKED_DIR/keys.json" ]; then
  echo "FATAL: $PACKED_DIR not repacked. Run:"
  echo "  ./.venv/bin/python -m waivphaet.data.repack --h5-dir /data/ryan.kim/plism \\"
  echo "      --out-dir $PACKED_DIR --verify"
  exit 1
fi
echo "repacked slides: $(ls -1 "$PACKED_DIR"/*.npy 2>/dev/null | wc -l) / 91"

# --- run ------------------------------------------------------------------------------
# Single-process for now: LoRA on ViT-L/16 @ 224 fits one H100 comfortably (PLAN.md 5).
# To scale out, wrap in `torchrun --nproc_per_node=8` and shard `batches_per_epoch`;
# the sampler is already seeded by (seed + epoch) so ranks stay in sync.
srun --unbuffered "$REPO/.venv/bin/python" scripts/train_lora.py \
  --packed-dir "$PACKED_DIR" \
  --out-dir "$OUT_DIR" \
  --heldout-scanners GT450 S210 \
  --heldout-stains HRH KR MY \
  --lora-rank 16 \
  --temperature 0.07 \
  --lr 1e-4 \
  --max-steps 5000 \
  --n-groups 4 \
  --group-size 64 \
  --workers 16 \
  --amp bfloat16 \
  --ckpt-every 500 \
  --eval-every 500 \
  "$@"

echo "=== done $(date) -> $OUT_DIR ==="
