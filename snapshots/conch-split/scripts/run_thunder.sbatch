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

REPO="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
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

# Optional epochs override, read from the ENVIRONMENT (WAIV_EPOCHS), not a 7th positional.
# third_party/thunder/docs/guidelines.md:4 mandates overriding the generic frozen.yaml 200
# epochs for exactly two datasets -- `9` for segpath_epithelial and `21` for
# segpath_lymphocytes -- because of their size. Without it those runs are both infeasible
# (200 x 5170 s = 287 h for lymphocytes) and non-comparable to the published leaderboard.
#
# Env var, not a positional, deliberately: the positional contract ends in
# [ckpt] [adapter] and both existing callers (resubmit_thunder.sh,
# submit_midnight_thunder.sh) *conditionally append* the adapter, so a 7th slot could not
# be reached by a base job without also inventing a dummy adapter argument. An env var is
# strictly additive -- every existing call site is untouched and, with WAIV_EPOCHS unset,
# the thunder command line below is byte-identical to what produced the current results.
EPOCHS="${WAIV_EPOCHS:-}"
EPOCH_ARGS=()
if [ -n "$EPOCHS" ]; then
  # `thunder benchmark` collects unknown flags pairwise into **kwargs (main.py:83) and
  # get_config turns them into hydra `++` overrides (utils/config.py:32), so hydra parses
  # "9" as an int. Format must be two tokens: `--adaptation.epochs 9`.
  EPOCH_ARGS=(--adaptation.epochs "$EPOCHS")
fi

export THUNDER_BASE_DATA_FOLDER="${THUNDER_BASE_DATA_FOLDER:-/data/ryan.kim/thunder}"
export HF_HOME="${HF_HOME:-/data/huggingface}"
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
echo "dataset=$DATASET tasks='$TASKS' pooling=$POOLING run_name=$RUN_NAME epochs=${EPOCHS:-<thunder default>}"
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
# adversarial_attack joins segmentation here: it PGD-perturbs raw pixels and backprops
# through the frozen backbone (tasks/adversarial_attack.py builds its PatchDataset with
# embedding_pre_loading=False unconditionally), so a cached embedding is useless to it.
# Left on embedding_pre_loading the job still *works*, but the cold-cache branch below
# would prepend a multi-hour pre_computing_embeddings step that nothing then reads --
# and the phikon-v2/Midnight embedding caches were swept off /data, so every adversarial
# job would pay it.
case "$TASKS" in *segmentation*|*adversarial*) MODE=online_loading ;; esac

# embedding_pre_loading READS a cache; it does not CREATE one. THUNDER exposes cache
# creation as its own task (`pre_computing_embeddings`, benchmark.py:25). Without it,
# every dataset whose cache is cold dies in ~37s on
#   FileNotFoundError: .../embeddings/<ds>/<model>/train/labels.h5
# which is exactly how 5 of the 6 base classification jobs failed (only mhist survived,
# because an earlier run had already populated its cache). Prepend the precompute step
# whenever the cache is cold, so a job is self-sufficient rather than depending on
# whatever happened to run before it.
EMB_DIR="${THUNDER_BASE_DATA_FOLDER}/embeddings/${DATASET}/${RUN_NAME}"

# A PARTIAL cache is worse than a cold one. If a job is killed mid-precompute -- preemption,
# or ENOSPC on /data, both of which happened repeatedly during the Virchow2 sweep -- it
# leaves an embeddings.h5 that exists but is truncated, with HDF5's write-consistency flag
# still set. THUNDER's own check is presence-only, so the next job logs
#   "Pre-computed embeddings already found ... Not re-computing them"
# skips the precompute, and dies ~60s later on
#   OSError: Unable to synchronously open file (file is already open for write)
# Three separate jobs died this way on 2026-08-08/09, each time tripping the pilot's
# max-fast-failures guard and halting the whole sweep. A presence test cannot distinguish
# the two states, so test whether every h5 in the cache actually OPENS read-only, and treat
# a cache that fails as cold: wipe it and precompute. Recomputing costs an hour; reading a
# corrupt cache costs the sweep.
if [ "$MODE" = "embedding_pre_loading" ] && [ -d "$EMB_DIR" ]; then
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

if [ "$MODE" = "embedding_pre_loading" ] && [ ! -f "$EMB_DIR/train/labels.h5" ]; then
  echo "--- cold embedding cache at $EMB_DIR -- precomputing first ---"
  TASKS="pre_computing_embeddings $TASKS"
else
  echo "--- embedding cache warm at $EMB_DIR (or segmentation) -- skipping precompute ---"
fi

for TASK in $TASKS; do
  # Per-task mode. pre_computing_embeddings WRITES the cache, so it must not be told to
  # read it -- passing embedding_pre_loading to the precompute step makes it try to open
  # the very labels.h5 it is supposed to create, and it dies in ~45s (jobs 369135/6/8).
  # Everything downstream then finds nothing and fails identically, which is why the
  # first version of this fix looked like it engaged and still failed.
  TASK_MODE="$MODE"
  case "$TASK" in pre_computing_embeddings) TASK_MODE=online_loading ;; esac
  echo "--- $(date) | $DATASET / $TASK (loading-mode=$TASK_MODE) ---"
  # ${EPOCH_ARGS[@]+...} guard: an unset/empty array expanded as "${A[@]}" is an unbound
  # variable under `set -u` on some bashes. With WAIV_EPOCHS unset this expands to nothing
  # at all, so the command line is exactly what it was before this option existed.
  srun --unbuffered "$REPO/.venv-thunder/bin/thunder" benchmark \
      "$MODEL" "$DATASET" "$TASK" --loading-mode "$TASK_MODE" \
      ${EPOCH_ARGS[@]+"${EPOCH_ARGS[@]}"}
  rc=$?
  echo "--- done $DATASET / $TASK rc=$rc ---"
  if [ "$rc" -ne 0 ]; then
    echo "FATAL: $DATASET/$TASK failed rc=$rc -- aborting remaining tasks"
    exit "$rc"
  fi
done

echo "=== done $(date) ==="
