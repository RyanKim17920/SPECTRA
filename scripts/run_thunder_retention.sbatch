#!/bin/bash
#SBATCH --job-name=spectra-thunder-ret
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=$SPECTRA_THUNDER/logs/slurm-%x-%j.out
#SBATCH --error=$SPECTRA_THUNDER/logs/slurm-%x-%j.err
#
# THUNDER retention at CHECKPOINT time -- the design spec §4 phase 8 ("evaluate retention at every
# checkpoint, not just at the end") and §6 (retention reported alongside every robustness
# claim, always as a pair).
#
#   sbatch scripts/run_thunder_retention.sbatch base_cls                       # reference row
#   sbatch scripts/run_thunder_retention.sbatch step5000 runs/waiv-real-369043/step5000
#
# Full FT checkpoint (backbone.safetensors format). Positionals are
# $1=RUN_NAME $2=ADAPTER $3=CKPT, so the adapter slot must be passed EMPTY -- a full-FT
# checkpoint in $2 would set SPECTRA_ADAPTER and be loaded as a LoRA adapter dir:
#   sbatch scripts/run_thunder_retention.sbatch full_ft_s1000 "" runs/full_ft/step_0000100
#
# Why a subset and not all 15: the full sweep is ~1.5M tiles of ViT-L inference and takes
# most of a day. Retention has to be cheap enough to run at every checkpoint or phase 8
# does not actually happen. These four are THUNDER's smallest classification sets --
# ~10k tiles total, a couple of minutes -- and they are diverse (breast microscopy, breast
# ROI, colorectal polyp, breast 4-class), so a broad forgetting event surfaces here.
#
# What it is NOT: a leaderboard number. Four datasets is not the benchmark. run_thunder.sbatch
# over the full 15 is what gets quoted; this is the tripwire between those runs. kNN only,
# because linear probing adds a 9-point lr x wd grid over 200 epochs for signal a kNN on
# the same embeddings already carries.
#
# Deliberately NOT a nested call to run_thunder.sbatch: that would nest srun inside srun and
# re-apply a second set of #SBATCH headers. The four lines of setup are cheaper than that.

set -euo pipefail

REPO="${SPECTRA_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
. "$REPO/scripts/_env.sh"
cd "$REPO"

RUN_NAME="${1:?usage: run_thunder_retention.sbatch <run-name> [adapter-dir] [checkpoint]}"
ADAPTER="${2:-}"
CKPT="${3:-}"
DATASETS="${THUNDER_RETENTION_DATASETS:-bach break_his mhist bracs}"

export THUNDER_BASE_DATA_FOLDER="${THUNDER_BASE_DATA_FOLDER:-$SPECTRA_THUNDER}"
export HF_HOME="${HF_HOME:-$SPECTRA_HF_HOME}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export WANDB_SILENT=true
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

export SPECTRA_POOLING="${SPECTRA_POOLING:-cls}"
export SPECTRA_RUN_NAME="$RUN_NAME"
if [ -n "$CKPT" ];    then export SPECTRA_CHECKPOINT="$CKPT"; fi
if [ -n "$ADAPTER" ]; then
  export SPECTRA_ADAPTER="$ADAPTER"
  export SPECTRA_LORA_RANK="${SPECTRA_LORA_RANK:-32}"
  export SPECTRA_LORA_ALPHA="${SPECTRA_LORA_ALPHA:-64}"
  export SPECTRA_PROJ_OUT_DIM="${SPECTRA_PROJ_OUT_DIM:-512}"
fi

MODEL="custom:${REPO}/src/waivphaet/eval/thunder_model.py"

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} ==="
echo "retention subset: run_name=$RUN_NAME adapter='${ADAPTER}' ckpt='${CKPT}'"
echo "datasets: $DATASETS"

for DS in $DATASETS; do
  echo "--- $(date) | $DS / knn ---"
  srun --unbuffered "$REPO/.venv-thunder/bin/thunder" benchmark \
      "$MODEL" "$DS" knn --loading-mode online_loading
done

echo "=== retention subset done: $RUN_NAME | $(date) ==="
"$REPO/.venv-thunder/bin/python" scripts/collect_thunder.py --model "$RUN_NAME"
