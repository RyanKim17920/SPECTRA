#!/bin/bash
#SBATCH --job-name=main_1
#SBATCH --account=max
#SBATCH --qos=normal
#SBATCH --partition=n
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/data/ryan.kim/ashby/slurm/%x_%A_%a.out
# One (run, step) eval unit per array task; line $SLURM_ARRAY_TASK_ID of $1 = "run tag pathorob_readout step".
#   sbatch --array=1-N%4 scripts/conch_v2_task.sbatch tasks.txt
set -uo pipefail
REPO=/data/ryan.kim/ashby/third_party/SPECTRA-conch; RUNS=/data/ryan.kim/ashby/runs/spectra_conch_v2
RES=/data/ryan.kim/ashby/scratch/spectra_conch_pathorob/results
export HF_HOME=/data/huggingface HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 MPLBACKEND=Agg
export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID:-local} PYTHONPATH="$REPO/src"
PY=/data/ryan.kim/ashby/envs/conch/bin/python
export PATHOROB_WORKERS=${SLURM_CPUS_PER_TASK:-16}
read run tag ro s <<< "$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")"
d=$(printf "%s/%s/step_%07d" "$RUNS" "$run" "$s"); nm="${tag}_s${s}"
echo "task $SLURM_ARRAY_TASK_ID: $run $tag $ro $s"
chk="$nm"; [ "$ro" = both ] && chk="${nm}_clsmean"
[ "${SKIP_PATHOROB:-0}" = 1 ] || [ -f "$RES/$chk/tolkach_esca/-1_0/results_summary.json" ] || \
  $PY "$REPO/scripts/conch_pathorob.py" --name "$nm" --readout "$ro" --adapter "$d/adapter"
[ "${SKIP_EVAL:-0}" = 1 ] && { echo "TASK DONE (pathorob only) $(date)"; exit 0; }
$PY "$REPO/scripts/conch_eval.py" --run-dir "$RUNS/$run" --steps "$s" --readout export
[ "$ro" = both ] && $PY "$REPO/scripts/conch_eval.py" --run-dir "$RUNS/$run" --steps "$s" --readout clsmean
echo "TASK DONE $(date)"
