#!/bin/bash
#SBATCH --job-name=waiv-thunder
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/data/ryan.kim/thunder/logs/slurm-%x-%j.out
#SBATCH --error=/data/ryan.kim/thunder/logs/slurm-%x-%j.err
#
# THUNDER retention eval -- our second retention detector (PLAN.md §3 risk 1, §4 phase 11).
#
#   sbatch scripts/run_thunder.sbatch mhist        "knn linear_probing"  cls base_cls
#   sbatch scripts/run_thunder.sbatch ocelot       "segmentation"        cls base_cls
#   sbatch scripts/run_thunder.sbatch break_his    "knn linear_probing"  cls step5000 "" runs/xxx/step5000
#
# One dataset per job so a slow dataset (esca/wilds/tcga_tils are 300k+ tiles) cannot
# stall the rest, and so a failure is isolated to one row of the table.
#
# DATA IS READ-ONLY AND NOT OURS. $THUNDER_BASE_DATA_FOLDER/datasets/<name> are symlinks
# into /data/thunder-data (owner: paul). Everything THUNDER writes -- data_splits/,
# embeddings/, outputs/ -- lands under /data/ryan.kim/thunder, which we own. Never point
# THUNDER_BASE_DATA_FOLDER at /data/thunder-data directly: it would try to write splits
# and embeddings into another user's tree.

set -euo pipefail

REPO=/admin/home/ryan.kim/waiv
cd "$REPO"

DATASET="${1:?usage: run_thunder.sbatch <dataset> <tasks> [pooling] [run_name] [ckpt] [adapter]}"
TASKS="${2:-knn linear_probing}"
POOLING="${3:-cls}"
RUN_NAME="${4:-base_${POOLING}}"
CKPT="${5:-}"
ADAPTER="${6:-}"

export THUNDER_BASE_DATA_FOLDER=/data/ryan.kim/thunder
export HF_HOME=/data/ryan.kim/hf_home
export TOKENIZERS_PARALLELISM=false
# wandb.login() is unconditional at thunder/benchmark.py:174 -- offline or the job blocks
# on an interactive prompt with no tty.
export WANDB_MODE=offline
export WANDB_SILENT=true
# A broken pandas in ~/.local shadowed a venv earlier on this box; isolate or the pins in
# .venv-thunder mean nothing.
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

# Read by src/waivphaet/eval/thunder_model.py. THUNDER instantiates the custom model with
# obj() and no arguments, so env vars are the only channel. WAIV_RUN_NAME becomes
# PretrainedModel.name, which is the <model> path component of
# outputs/res/<dataset>/<model>/<task>/<adaptation>/ -- get it wrong and one checkpoint's
# numbers silently overwrite another's.
export WAIV_POOLING="$POOLING"
export WAIV_RUN_NAME="$RUN_NAME"
[ -n "$CKPT" ]    && export WAIV_CHECKPOINT="$CKPT"
[ -n "$ADAPTER" ] && export WAIV_ADAPTER="$ADAPTER"

MODEL="custom:${REPO}/src/waivphaet/eval/thunder_model.py"

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} ==="
echo "dataset=$DATASET tasks='$TASKS' pooling=$POOLING run_name=$RUN_NAME"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true

for TASK in $TASKS; do
  echo "--- $(date) | $DATASET / $TASK ---"
  # image_pre_loading would cache decoded tiles in RAM; the big datasets here are 300k+
  # tiles so that OOMs a 128G box. online_loading is the default for a reason.
  srun --unbuffered "$REPO/.venv-thunder/bin/thunder" benchmark \
      "$MODEL" "$DATASET" "$TASK" --loading-mode online_loading
  echo "--- done $DATASET / $TASK rc=$? ---"
done

echo "=== done $(date) ==="
