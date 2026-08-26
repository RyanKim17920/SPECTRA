#!/bin/bash
#SBATCH --job-name=waiv-rescoreri
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail
# Score RI for checkpoints that already exist but never got an eval follower run --
# e.g. an arm cancelled after its checkpoint was written but before the follower
# evaluated it. eval_checkpoints.py WITHOUT --follow scores whatever is on disk and exits.
REPO="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
cd "$REPO"
export HF_HOME="${HF_HOME:-/data/huggingface}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
RUN="${WAIV_RUN:?set WAIV_RUN (run dir name under runs/)}"
OUT="$REPO/runs/$RUN"
[ -d "$OUT" ] || { echo "FATAL: no run dir $OUT" >&2; exit 3; }
case "$RUN" in
  *phikon*)   BB="owkin/phikon-v2" ;;
  *midnight*) BB="kaiko-ai/midnight" ;;
  *virchow2*) BB="paige-ai/Virchow2" ;;
  *) echo "FATAL: cannot infer backbone from '$RUN'" >&2; exit 2 ;;
esac
echo "=== $(date) | rescore RI | run=$RUN backbone=$BB ==="
ls -d "$OUT"/step_* 2>/dev/null || { echo "FATAL: no checkpoints in $OUT" >&2; exit 3; }
"$REPO/.venv/bin/python" scripts/eval_checkpoints.py \
  --run-dir "$OUT" \
  --backbone "$BB" \
  --lora-rank 32 --lora-alpha 64 \
  --num-workers 8
echo "=== done rc=$? ==="
