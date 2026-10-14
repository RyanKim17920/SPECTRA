#!/bin/bash
#SBATCH --job-name=main_1
#SBATCH --account=max
#SBATCH --qos=normal
#SBATCH --partition=n
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=16:00:00
#SBATCH --output=/data/ryan.kim/ashby/slurm/%x_%j.out
# PathoROB RI follower over the v2 arms: scores each new checkpoint as it lands; exits when every
# listed run has TRAIN_DONE and all its checkpoints are scored.
#   sbatch scripts/conch_v2_follow.sbatch A_export-ninf_s0:A:export B_split-paper_s0:B:both ...
set -uo pipefail
REPO=/data/ryan.kim/ashby/third_party/SPECTRA-conch; RUNS=/data/ryan.kim/ashby/runs/spectra_conch_v2
RES=/data/ryan.kim/ashby/scratch/spectra_conch_pathorob/results
export HF_HOME=/data/huggingface HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 MPLBACKEND=Agg
export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID:-local} PYTHONPATH="$REPO/src"
PY=/data/ryan.kim/ashby/envs/conch/bin/python
while true; do
  did=0; alldone=1
  for spec in "$@"; do
    IFS=: read run tag ro <<< "$spec"
    [ -f "$RUNS/$run/TRAIN_DONE" ] || alldone=0
    for d in $(ls -d "$RUNS/$run"/step_* 2>/dev/null | sort); do
      [ -f "$d/metrics.json" ] || continue
      s=$((10#${d##*_})); nm="${tag}_s${s}"; chk="$nm"; [ "$ro" = clsmean ] && chk="${nm}_clsmean"
      [ "$ro" = both ] && chk="${nm}_clsmean"
      [ -f "$RES/$chk/tolkach_esca/-1_0/results_summary.json" ] && continue
      echo "=== $(date +%T) PathoROB $nm ($ro)"
      $PY "$REPO/scripts/conch_pathorob.py" --name "$nm" --readout "$ro" --adapter "$d/adapter" 2>&1 | grep -E "^extracted|^RI .* (tcga|camelyon|tolkach_esca) |Error|Traceback" | cut -c1-120
      did=1
    done
  done
  [ $did = 0 ] && [ $alldone = 1 ] && break
  [ $did = 0 ] && sleep 120
done
echo "FOLLOW DONE $(date)"
