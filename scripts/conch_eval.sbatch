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
# Per-checkpoint SPECTRA-CONCH eval:  sbatch scripts/conch_eval.sbatch <run_dir> [--base] [--steps 50,100]
set -euo pipefail
REPO=/data/ryan.kim/ashby/third_party/SPECTRA-conch
export HF_HOME=/data/huggingface HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16
export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID:-local}
export PYTHONPATH="$REPO/src"
/data/ryan.kim/ashby/envs/conch/bin/python "$REPO/scripts/conch_eval.py" --run-dir "$@"
