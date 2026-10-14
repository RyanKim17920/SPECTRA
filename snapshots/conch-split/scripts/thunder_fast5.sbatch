#!/bin/bash
#SBATCH --job-name=waiv-tfast5
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#
# Fast 5-dataset THUNDER readout: knn + linear_probing on bach, mhist, break_his, bracs, ccrcc.
# Used for seed-noise-floor calibration and quick arm comparisons.
#
# NOTE: --output/--error must be on shared storage. Logs go under
#   /admin/home/ryan.kim/waiv/logs/thunderfast5/
# which is set by the caller:
#   sbatch --output=.../logs/thunderfast5/tfast5-<tag>-%j.out \
#          --error=.../logs/thunderfast5/tfast5-<tag>-%j.err \
#          scripts/thunder_fast5.sbatch <run_tag> <adapter_dir>
#
# $1 = run_tag   : unique label; becomes the THUNDER run_name ("fast5_<tag>") so results
#                  land under outputs/res/<ds>/fast5_<tag>/... and never collide.
# $2 = adapter_dir: path to the step dir (must contain adapter/ subdirectory).
#
# Uses .venv-thunder and phikon-v2 backbone (all calibration arms are phikon-v2 LoRA).

set -euo pipefail

REPO="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
cd "$REPO"

RUN_TAG="${1:?usage: thunder_fast5.sbatch <run_tag> <adapter_dir>}"
ADAPTER="${2:?usage: thunder_fast5.sbatch <run_tag> <adapter_dir>}"

# Honor a caller-supplied WAIV_POOLING; default to cls (leaderboard-comparable path).
# Pass WAIV_POOLING=clsmean in the environment for the internal arm-ranking readout.
POOLING="${WAIV_POOLING:-cls}"

# Encode pooling in the run name so cls and clsmean results never overwrite each other.
# Results land under outputs/res/<ds>/fast5_<tag>_<pooling>/...
RUN_NAME="fast5_${RUN_TAG}_${POOLING}"

DATASETS="bach mhist break_his bracs ccrcc"
TASKS="knn linear_probing"

export THUNDER_BASE_DATA_FOLDER="${THUNDER_BASE_DATA_FOLDER:-/data/ryan.kim/thunder}"
export HF_HOME="${HF_HOME:-/data/huggingface}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export WANDB_SILENT=true
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

export WAIV_POOLING="$POOLING"
export WAIV_RUN_NAME="$RUN_NAME"
export WAIV_ADAPTER="$ADAPTER"
export WAIV_LORA_RANK="${WAIV_LORA_RANK:-32}"
export WAIV_LORA_ALPHA="${WAIV_LORA_ALPHA:-64}"
export WAIV_PROJ_OUT_DIM="${WAIV_PROJ_OUT_DIM:-512}"

MODEL="custom:${REPO}/src/waivphaet/eval/thunder_model.py"

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} ==="
echo "run_name=$RUN_NAME adapter=$ADAPTER"
echo "datasets: $DATASETS"
echo "tasks:    $TASKS"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true

if [ ! -d "$ADAPTER/adapter" ]; then
  echo "FATAL: no adapter/ subdir under $ADAPTER"; exit 2
fi

for DS in $DATASETS; do
  # embedding_pre_loading for classification tasks; cache integrity check mirrors run_thunder.sbatch.
  EMB_DIR="${THUNDER_BASE_DATA_FOLDER}/embeddings/${DS}/${RUN_NAME}"

  # Check for partial/corrupt cache (mirrors run_thunder.sbatch logic).
  if [ -d "$EMB_DIR" ]; then
    if ! "$REPO/.venv-thunder/bin/python" - "$EMB_DIR" <<'PYEOF'
import glob, os, sys
import h5py
emb_dir = sys.argv[1]
files = sorted(glob.glob(os.path.join(emb_dir, "*", "*.h5")))
if not files:
    sys.exit(1)
for f in files:
    try:
        with h5py.File(f, "r"):
            pass
    except Exception as exc:
        print(f"[cache-check] UNREADABLE {f}: {exc}")
        sys.exit(1)
print(f"[cache-check] ok, {len(files)} h5 file(s) readable under {emb_dir}")
PYEOF
    then
      echo "--- PARTIAL/CORRUPT embedding cache at $EMB_DIR -- removing and recomputing ---"
      rm -rf "$EMB_DIR"
    fi
  fi

  # Precompute embeddings if cache is cold.
  if [ ! -f "$EMB_DIR/train/labels.h5" ]; then
    echo "--- $(date) | $DS / pre_computing_embeddings ---"
    srun --unbuffered "$REPO/.venv-thunder/bin/thunder" benchmark \
        "$MODEL" "$DS" pre_computing_embeddings --loading-mode online_loading
  else
    echo "--- embedding cache warm at $EMB_DIR ---"
  fi

  for TASK in $TASKS; do
    echo "--- $(date) | $DS / $TASK ---"
    srun --unbuffered "$REPO/.venv-thunder/bin/thunder" benchmark \
        "$MODEL" "$DS" "$TASK" --loading-mode embedding_pre_loading
    echo "--- done $DS / $TASK ---"
  done
done

echo "=== fast5 done: $RUN_NAME | $(date) ==="
"$REPO/.venv-thunder/bin/python" scripts/collect_thunder.py --model "$RUN_NAME"
