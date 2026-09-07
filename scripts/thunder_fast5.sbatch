#!/bin/bash
#SBATCH --job-name=spectra-tfast5
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
#   $SPECTRA_REPO/logs/thunderfast5/
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

REPO="${SPECTRA_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
. "$REPO/scripts/_env.sh"
cd "$REPO"

RUN_TAG="${1:?usage: thunder_fast5.sbatch <run_tag> <adapter_dir>}"
ADAPTER="${2:?usage: thunder_fast5.sbatch <run_tag> <adapter_dir>}"

# Honor a caller-supplied SPECTRA_POOLING; default to cls (leaderboard-comparable path).
# Pass SPECTRA_POOLING=clsmean in the environment for the internal arm-ranking readout.
POOLING="${SPECTRA_POOLING:-cls}"

# Encode pooling in the run name so cls and clsmean results never overwrite each other.
# Results land under outputs/res/<ds>/fast5_<tag>_<pooling>/...
RUN_NAME="fast5_${RUN_TAG}_${POOLING}"

DATASETS="bach mhist break_his bracs ccrcc"
TASKS="knn linear_probing"

export THUNDER_BASE_DATA_FOLDER="${THUNDER_BASE_DATA_FOLDER:-$SPECTRA_THUNDER}"
export HF_HOME="${HF_HOME:-$SPECTRA_HF_HOME}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export WANDB_SILENT=true
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

export SPECTRA_POOLING="$POOLING"
export SPECTRA_RUN_NAME="$RUN_NAME"
export SPECTRA_ADAPTER="$ADAPTER"
export SPECTRA_LORA_RANK="${SPECTRA_LORA_RANK:-32}"
export SPECTRA_LORA_ALPHA="${SPECTRA_LORA_ALPHA:-64}"
export SPECTRA_PROJ_OUT_DIM="${SPECTRA_PROJ_OUT_DIM:-512}"

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
