#!/bin/bash
#SBATCH --job-name=waiv-lora
#SBATCH --partition=main            # `main` = c-[1-6] + n-[1-8]; use `n` to pin the 8xH100 nodes
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64          # pair sampler is memmap-random-read heavy; give it workers
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# Registered-pair contrastive fine-tune of phikon-v2 (PLAN.md 3, phase 7).
#
#   sbatch scripts/train_lora.sbatch                     # defaults
#   sbatch scripts/train_lora.sbatch --lora-rank 32 --temperature 0.05
#
# PLAN.md 3 risk 4: there is no published recipe, so this is one point in a sweep, not
# "the" run. Vary --lr / --lora-rank / --temperature / --max-steps across jobs.

set -euo pipefail

REPO="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
cd "$REPO"

# HF cache MUST live on /data -- /admin has only ~1.4 TB and must not fill (PLAN.md 5).
export HF_HOME="${HF_HOME:-/data/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
# uint8 memmaps are read randomly across up to 91 slides; keep threads from thrashing
export MKL_NUM_THREADS=8

RUN_NAME="${SLURM_JOB_NAME:-waiv-lora}-${SLURM_JOB_ID:-local}"
OUT_DIR="$REPO/runs/$RUN_NAME"
mkdir -p "$OUT_DIR"

# Backbone knob. UNSET => nothing is passed and train_lora.py's own default
# (owkin/phikon-v2) applies, i.e. byte-identical to every run submitted before this knob
# existed. Set WAIV_BACKBONE=kaiko-ai/midnight (or any HF id) to fine-tune something else.
#
# SWITCHING THE BACKBONE ALMOST CERTAINLY MEANS RE-SIZING THE BATCH. The --n-groups 4 /
# --group-size 64 below are sized for phikon-v2 (ViT-L/16, 24 blocks, 1024-d). midnight is
# ViT-g/14, 40 blocks, 1536-d -- roughly 3x the parameters -- and train_lora.py warns that
# a phikon-v2-sized batch will OOM on a wider backbone. Measure the new one first with
#     sbatch scripts/sizing_probe.sbatch kaiko-ai/midnight
# and pass the result through as `sbatch scripts/train_lora.sbatch --n-groups N
# --group-size M` (the trailing "$@" below). WAIV_BACKBONE deliberately does NOT change the
# batch defaults -- a silent re-size would make two runs incomparable with nothing in the
# log to say so.
BACKBONE="${WAIV_BACKBONE:-}"
BACKBONE_ARGS=()
if [ -n "$BACKBONE" ]; then BACKBONE_ARGS=(--backbone "$BACKBONE"); fi

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} ==="
echo "=== backbone ${BACKBONE:-<script default: owkin/phikon-v2>} ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true

# --- preflight ------------------------------------------------------------------------
# Repack must have run: 16,278 tiny HDF5 datasets per slide makes random pair sampling
# I/O bound (~112-125x slower than the contiguous memmap, measured).
PACKED_DIR="${WAIV_PACKED_DIR:-/data/plism/repacked}"
# Only used to print the fix-it command below; never read.
H5_DIR="${WAIV_H5_DIR:-/data/plism}"
if [ ! -f "$PACKED_DIR/keys.json" ]; then
  echo "FATAL: $PACKED_DIR not repacked. Run:"
  echo "  ./.venv/bin/python -m waivphaet.data.repack --h5-dir $H5_DIR \\"
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
  "${BACKBONE_ARGS[@]}" \
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
