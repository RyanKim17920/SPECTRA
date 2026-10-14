#!/bin/bash
#SBATCH --job-name=main_1
#SBATCH --account=max
#SBATCH --qos=normal
#SBATCH --partition=n
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=/data/ryan.kim/ashby/slurm/%x_%j.out
# Generic SPECTRA-CONCH downstream runner:  sbatch scripts/conch_py.sbatch scripts/<script>.py [args...]
set -euo pipefail
REPO=/data/ryan.kim/ashby/third_party/SPECTRA-conch
source ~/.cache_env.sh 2>/dev/null || true
export HF_HOME=/data/huggingface HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 MPLBACKEND=Agg
export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID:-local}
export PYTHONPATH="$REPO/src"
cd "$REPO"
echo "=== $(date) $(hostname) job ${SLURM_JOB_ID:-} : $*"
/data/ryan.kim/ashby/envs/conch/bin/python "$@"
echo "=== done $(date)"
