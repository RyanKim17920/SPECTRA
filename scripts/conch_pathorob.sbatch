#!/bin/bash
#SBATCH --job-name=main_1
#SBATCH --account=max
#SBATCH --qos=normal
#SBATCH --partition=n
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=/data/ryan.kim/ashby/slurm/%x_%j.out
# sbatch scripts/conch_pathorob.sbatch name1[=adapter_dir] name2[=adapter_dir] ...
# A name ending in "_clsmean" is scored on the trunk clsmean readout, otherwise on the export.
set -uo pipefail
REPO=/data/ryan.kim/ashby/third_party/SPECTRA-conch
export HF_HOME=/data/huggingface HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 MPLBACKEND=Agg
export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID:-local}
export PYTHONPATH="$REPO/src"
for spec in "$@"; do
  name="${spec%%=*}"; ad=""; [ "$spec" != "$name" ] && ad="--adapter ${spec#*=}"
  ro=export; case "$name" in *_clsmean) ro=clsmean ;; esac
  /data/ryan.kim/ashby/envs/conch/bin/python "$REPO/scripts/conch_pathorob.py" --name "$name" --readout $ro $ad
done
