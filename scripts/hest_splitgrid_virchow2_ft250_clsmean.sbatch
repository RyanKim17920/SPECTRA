#!/bin/bash
#SBATCH --job-name=waiv-hest-sg-virchow2-clsmean
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# HEST-Benchmark, Virchow2, FINAL split-head x grid checkpoint, 9-task leaderboard.
#
# CHECKPOINT: runs/splitgrid-386375/step_0000250 -- the final split-head x grid run
# (--split-heads --cls-weight 0.5 --mean-weight 0.5, --grid C=2 T=600), RI 0.9078 at
# step 250, its own RI-optimal step. This is NOT runs/waiv-virchow2-375367/step_0000250,
# which is the published-reconstruction checkpoint. BOTH ARE STEP 250 -- that coincidence
# already caused one agent to conclude this eval was "already done" because run names
# vft250_* exist. They exist for the OTHER model.
#
# POOLING = clsmean, and that is not the same choice as the other two backbones.
# Waiv never state a pooling protocol; it was determined empirically by scoring each
# base under both poolings and keeping whichever reproduces their published base:
#   phikon-v2 -> cls      (ours 0.37470 vs their 0.3747, exact)
#   midnight  -> cls      (ours 0.39521 vs their 0.3952, exact)
#   Virchow2  -> clsmean  (ours 0.40327 vs their 0.4034, within 0.00013)
# Virchow2 under cls gives 0.39791 -- off by 0.0055. Assuming a single protocol across
# backbones would manufacture a -0.0055 discrepancy here where none exists.
#
# CACHE KEY: run_hest.py keys the embedding cache on --exp-code ALONE (warning block at
# scripts/run_hest.py:114-155) -- pooling, backbone AND checkpoint are all absent from
# the key. Reusing an existing exp_code would silently rescore stale features and return
# a fabricated result. The exp_code below is fresh and was verified absent from both
# /data/ryan.kim/hest_work/embeddings/ and /data/ryan.kim/hest_work/results/.
#
# --num-workers 0 is REQUIRED, not a preference: the HEST dataloader crashes in shared
# memory teardown (c10::Error: could not unlink the shared memory file) at 8 workers and
# again at 2. Job 386354 died this way twice before dropping to 0.
#
#   sbatch --account=max --qos=high scripts/hest_splitgrid_virchow2_ft250_clsmean.sbatch

set -euo pipefail

REPO="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
cd "$REPO"

POOLING="clsmean"
EXP_CODE="${1:-splitgrid_virchow2_ft250_clsmean_v1}"
BACKBONE="paige-ai/Virchow2"
ADAPTER="$REPO/runs/splitgrid-386375/step_0000250"

export HF_HOME="${HF_HOME:-/data/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} ==="
echo "=== backbone=$BACKBONE pooling=$POOLING exp_code=$EXP_CODE ==="
echo "=== adapter=$ADAPTER ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true

[ -d "$ADAPTER/adapter" ] || { echo "FATAL: no adapter/ subdir under $ADAPTER" >&2; exit 3; }

# --tasks omitted => run_hest.py default = H.LEADERBOARD_TASKS, the 9 published tasks
# (IDC PRAD PAAD SKCM COAD READ CCRCC LUNG LYMPH_IDC), HCC excluded. --precision omitted
# => float32, matching every other row in the comparison.
ARGS=(--pooling "$POOLING" --exp-code "$EXP_CODE" --backbone "$BACKBONE" \
      --num-workers 0 --adapter "$ADAPTER" \
      --lora-rank 32 --lora-alpha 64 --proj-out-dim 512)

srun --unbuffered "$REPO/.venv-hest/bin/python" scripts/run_hest.py "${ARGS[@]}"

echo "=== done $(date) ==="
