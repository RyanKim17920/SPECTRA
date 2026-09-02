#!/bin/bash
#SBATCH --job-name=waiv-hest-final5
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# HEST-Benchmark for one final5 checkpoint. Parameterized so all 15 runs go through ONE
# script instead of the per-checkpoint copies (hest_splitgrid_*.sbatch) that made the
# earlier round hard to audit.
#
# POOLING IS PER-BACKBONE AND IS NOT A FREE CHOICE. Waiv's published HEST numbers use a
# different pooling protocol per backbone and never say so; matching it is what turned the
# Midnight "null" into a real +0.0166 result (docs/FINAL_RESULTS.md sec.4.1). Scoring a
# backbone under the wrong pooling silently manufactures or erases the effect:
#   phikon-v2 -> cls       (base 0.37470 reproduces published 0.3747 exactly)
#   midnight  -> cls       (base 0.39521 reproduces published 0.3952 exactly)
#   Virchow2  -> clsmean   (base 0.40327 vs published 0.4034; under cls it is 0.39791, off by 0.0055)
# So POOLING is derived from ARM here, never passed in by hand.
#
# --num-workers 0 is REQUIRED, not a preference: the HEST dataloader crashes in shared
# memory teardown (c10::Error: could not unlink the shared memory file) at 8 workers and
# again at 2. Job 386354 died this way twice before dropping to 0.
#
# EXP_CODE must be unique per (run, step) or results collide silently on disk.
#
#   WAIV_RUN=final5-virchow2-s0-t900-XXXX WAIV_STEP=0000500 sbatch scripts/hest_final5.sbatch
set -euo pipefail

REPO="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
cd "$REPO"

RUN="${WAIV_RUN:?set WAIV_RUN (run dir name under runs/)}"
STEP="${WAIV_STEP:?set WAIV_STEP e.g. 0000500}"

# ARM is derived from the run name so pooling cannot drift from the checkpoint.
case "$RUN" in
  *phikon*)   ARM=phikon;   BACKBONE="owkin/phikon-v2";  POOLING="cls" ;;
  # openmidnight MUST precede midnight: `*midnight*` matches "openmidnight" too, and the
  # controller looped 30+ HEST jobs merging OpenMidnight adapters into kaiko-ai/midnight
  # (each died on "adapter keys not consumed") before this case existed.  cls like Midnight.
  *openmidnight*) ARM=openmidnight; BACKBONE="SophontAI/OpenMidnight"; POOLING="cls" ;;
  *midnight*) ARM=midnight; BACKBONE="kaiko-ai/midnight"; POOLING="cls" ;;
  # Virchow (v1) run names carry "-virchow-s<seed>"; `*virchow2*` never matches them.
  # clsmean: THUNDER and the HEST leaderboard both read Virchow as CLS ++ mean(patch).
  *-virchow-s*) ARM=virchow1; BACKBONE="paige-ai/Virchow"; POOLING="clsmean" ;;
  *virchow2*) ARM=virchow2; BACKBONE="paige-ai/Virchow2"; POOLING="clsmean" ;;
  # cls, not clsmean: arXiv:2607.22861 3 line 106 names the clsmean models (Virchow2,
  # AquaViT, H0-mini, Midnight-12k) and neither of these is in it.  H0-mini IS in it but
  # is a DISTILLATION of H-Optimus-0 and a separate row of their tables -- not this model.
  # Must agree with collect_final5.HEST_POOLING; tests/test_invariants.py asserts it does.
  # "hopt" is the abbreviated arm token used when the full "hoptimus" would push
  # RUN_NAME past the 64-char THUNDER key budget; scoreboard2._BACKBONES carries the
  # same alias.  "hopt" is a prefix of "hoptimus", so one glob would cover both, but
  # both are spelled out so a grep for either token finds this line.
  *hoptimus*|*hopt*) ARM=hoptimus; BACKBONE="bioptimus/H-optimus-0"; POOLING="cls" ;;
  *uni2*)     ARM=uni2;     BACKBONE="MahmoodLab/UNI2-h";     POOLING="cls" ;;
  *) echo "FATAL: cannot infer backbone from run name '$RUN'" >&2; exit 2 ;;
esac

ADAPTER="$REPO/runs/$RUN/step_$STEP"
EXP_CODE="${WAIV_EXP_CODE:-f5_${RUN}_s${STEP}_${POOLING}}"

export HF_HOME="${HF_HOME:-/data/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} ==="
echo "=== arm=$ARM backbone=$BACKBONE pooling=$POOLING ==="
echo "=== run=$RUN step=$STEP exp_code=$EXP_CODE ==="
echo "=== adapter=$ADAPTER ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

[ -d "$ADAPTER/adapter" ] || { echo "FATAL: no adapter/ subdir under $ADAPTER" >&2; exit 3; }

# --tasks omitted => run_hest.py default = H.LEADERBOARD_TASKS, the 9 published tasks
# (IDC PRAD PAAD SKCM COAD READ CCRCC LUNG LYMPH_IDC), HCC excluded. --precision omitted
# => float32, matching every other row in the comparison.
ARGS=(--pooling "$POOLING" --exp-code "$EXP_CODE" --backbone "$BACKBONE" \
      --num-workers 0 --adapter "$ADAPTER" \
      --lora-rank 32 --lora-alpha 64 --proj-out-dim 512)

srun --unbuffered "$REPO/.venv-hest/bin/python" scripts/run_hest.py "${ARGS[@]}"

echo "=== done $(date) | exp_code=$EXP_CODE ==="
