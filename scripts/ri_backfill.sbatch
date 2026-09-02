#!/bin/bash
#SBATCH --job-name=waiv-ribackfill
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# RI BACKFILL. The eval follower dies with its training job, so checkpoints written near
# the end of a cancelled run have no RI point even though the checkpoint itself is intact.
# This recomputes RI for every checkpoint already on disk -- no retraining required.
# eval_checkpoints.py without --follow processes existing steps and exits.
set -euo pipefail
REPO="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
cd "$REPO"
export HF_HOME="${HF_HOME:-/data/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
PY="$REPO/.venv/bin/python"
# Default to the gated snapshot: it resolves all five backbones, gemgrid-pinned only three.
PIN="${WAIV_PIN:-/admin/home/ryan.kim/waiv-snapshots/falseneg-gated}"
export PYTHONPATH="$PIN/src${PYTHONPATH:+:$PYTHONPATH}"
RUN="${WAIV_RUN:?set WAIV_RUN}"
OUT="$REPO/runs/$RUN"
# H-Optimus-0 and UNI2-h are gated: they resolve only under the falseneg-gated snapshot,
# which is the one carrying BACKBONE_LOCAL_DIRS.  Matching `*hopt*` as well as `*hoptimus*`
# because the short token is what the newer run names use.
case "$RUN" in
  *phikon*)   BB="" ;;
  *midnight*) BB="kaiko-ai/midnight" ;;
  *virchow2*) BB="paige-ai/Virchow2" ;;
  *hoptimus*|*hopt*) BB="bioptimus/H-optimus-0" ;;
  *uni2*)     BB="MahmoodLab/UNI2-h" ;;
  *) echo "bad run $RUN" >&2; exit 2 ;;
esac
BBA=(); [ -n "$BB" ] && BBA=(--backbone "$BB")
echo "=== $(date) | $(hostname) | RI backfill $RUN ==="
ls -d "$OUT"/step_* 2>/dev/null | sed 's/.*step_/  step /'
"$PY" "$PIN"/scripts/eval_checkpoints.py \
  --run-dir "$OUT" \
  --packed-dir "${WAIV_PACKED_DIR:-/data/plism/repacked}" \
  --conditions-file "$OUT/conditions_used.json" \
  --lora-rank 32 --lora-alpha 64 "${BBA[@]}" \
  --num-workers 8
echo "=== done $(date) ==="
