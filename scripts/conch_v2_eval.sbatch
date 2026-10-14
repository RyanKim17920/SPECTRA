#!/bin/bash
#SBATCH --job-name=main_1
#SBATCH --account=max
#SBATCH --qos=normal
#SBATCH --partition=n
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/data/ryan.kim/ashby/slurm/%x_%j.out
# v2 per-checkpoint eval of one run dir: PathoROB RI (export [+ clsmean]) on every step, then
# conch_eval (export [+ clsmean]).  READOUTS="export clsmean"  sbatch scripts/conch_v2_eval.sbatch <run_dir> <tag> [steps]
set -uo pipefail
REPO=/data/ryan.kim/ashby/third_party/SPECTRA-conch
export HF_HOME=/data/huggingface HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 MPLBACKEND=Agg
export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID:-local} PYTHONPATH="$REPO/src"
PY=/data/ryan.kim/ashby/envs/conch/bin/python
RUN="$1"; TAG="$2"; STEPS="${3:-50 100 150 200 250 300 350 400 450 500}"
PRO=export; [ "${READOUTS:-export}" = "export clsmean" ] && PRO=both
[ "${SKIP_PATHOROB:-0}" = 1 ] || for s in $STEPS; do
  d=$(printf "%s/step_%07d" "$RUN" "$s")
  [ -f "$d/metrics.json" ] || { echo "missing $d"; continue; }
  $PY "$REPO/scripts/conch_pathorob.py" --name "${TAG}_s${s}" --readout $PRO --adapter "$d/adapter"
done
[ "${SKIP_EVAL:-0}" = 1 ] && { echo "V2 EVAL DONE $(date)"; exit 0; }
for ro in ${READOUTS:-export}; do
  $PY "$REPO/scripts/conch_eval.py" --run-dir "$RUN" --readout $ro
done
echo "V2 EVAL DONE $(date)"
