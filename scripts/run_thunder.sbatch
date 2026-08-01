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
# Pooling is BACKBONE-DEPENDENT in THUNDER and it is not a free choice -- arXiv:2607.22861
# 3 line 106 uses CLS+mean only for Virchow2 / AquaViT / H0-mini / Midnight-12k, CLS
# elsewhere. Pass "auto" (or nothing, for the phikon-v2 default) to let
# thunder_model._default_pooling resolve it from WAIV_BACKBONE. An explicit 3rd argument
# still wins, so the in-flight phikon-v2 sweep (which passes `cls`) is unchanged.
POOLING="${3:-auto}"
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
if [ "$POOLING" != "auto" ]; then export WAIV_POOLING="$POOLING"; else unset WAIV_POOLING || true; fi
export WAIV_RUN_NAME="$RUN_NAME"
# WAIV_BACKBONE is inherited from the submitting environment if set; unset => phikon-v2.
# Written as if-blocks, not `[ -n "$X" ] && export ...`: under `set -e` a short-circuiting
# AND-OR list is a footgun, and the base run passes both of these empty.
if [ -n "$CKPT" ];    then export WAIV_CHECKPOINT="$CKPT"; fi
if [ -n "$ADAPTER" ]; then
  export WAIV_ADAPTER="$ADAPTER"
  # thunder_model.py defaults to r16/alpha32; the real run is r32/alpha64. Without these the
  # adapter load fails the config cross-check in build_model. Overridable from the caller.
  export WAIV_LORA_RANK="${WAIV_LORA_RANK:-32}"
  export WAIV_LORA_ALPHA="${WAIV_LORA_ALPHA:-64}"
  export WAIV_PROJ_OUT_DIM="${WAIV_PROJ_OUT_DIM:-512}"
fi

MODEL="custom:${REPO}/src/waivphaet/eval/thunder_model.py"

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} ==="
echo "dataset=$DATASET tasks='$TASKS' pooling=$POOLING run_name=$RUN_NAME"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true

# embedding_pre_loading, not the online_loading default. Under online_loading the linear
# probe re-runs the frozen ViT-L over every image on every one of its 200 epochs: measured
# 12 s/epoch on mhist, which is 1743 images -- ~40 min per hyper-parameter point, and the
# grid has nine. On wilds (302k train) that is not a slow run, it is an impossible one.
# The backbone is frozen, so those embeddings are identical every epoch; benchmark.py:220
# takes them straight from the cached h5 instead. Segmentation ignores the flag (same
# line) because its decoder needs the raw masks, so those jobs are unaffected.
# image_pre_loading is the wrong lever -- it caches decoded *tiles* in RAM and OOMs a 128G
# box on the 300k-tile sets.
MODE=embedding_pre_loading
case "$TASKS" in *segmentation*) MODE=online_loading ;; esac

# embedding_pre_loading READS a cache; it does not CREATE one. THUNDER exposes cache
# creation as its own task (`pre_computing_embeddings`, benchmark.py:25). Without it,
# every dataset whose cache is cold dies in ~37s on
#   FileNotFoundError: .../embeddings/<ds>/<model>/train/labels.h5
# which is exactly how 5 of the 6 base classification jobs failed (only mhist survived,
# because an earlier run had already populated its cache). Prepend the precompute step
# whenever the cache is cold, so a job is self-sufficient rather than depending on
# whatever happened to run before it.
EMB_DIR="${THUNDER_BASE_DATA_FOLDER}/embeddings/${DATASET}/${RUN_NAME}"
if [ "$MODE" = "embedding_pre_loading" ] && [ ! -f "$EMB_DIR/train/labels.h5" ]; then
  echo "--- cold embedding cache at $EMB_DIR -- precomputing first ---"
  TASKS="pre_computing_embeddings $TASKS"
else
  echo "--- embedding cache warm at $EMB_DIR (or segmentation) -- skipping precompute ---"
fi

for TASK in $TASKS; do
  echo "--- $(date) | $DATASET / $TASK ---"
  srun --unbuffered "$REPO/.venv-thunder/bin/thunder" benchmark \
      "$MODEL" "$DATASET" "$TASK" --loading-mode "$MODE"
  echo "--- done $DATASET / $TASK rc=$? ---"
done

echo "=== done $(date) ==="
