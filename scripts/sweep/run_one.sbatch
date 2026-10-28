#!/bin/bash
#SBATCH --job-name=q8
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#
# One Q8 run on ONE GPU. Submitted by scripts/sweep/submit.py, which writes the resolved
# config to $SWEEP_RUN_DIR/config.json and passes --output into the run dir. Cluster options
# (--account / --partition / --qos) go on the sbatch command line (submit.py --sbatch-args or
# $SPECTRA_SBATCH_ARGS).
#
# Environment:
#   SWEEP_RUN_DIR               run dir holding config.json (required)
#   SPECTRA_REPO                repository checkout (submit.py exports it)
#   SPECTRA_PY                  python interpreter (default $SPECTRA_REPO/.venv/bin/python)
#   SPECTRA_PLISM_DIR           PLISM repack directory (unless run.packed_dir is set)
#   SPECTRA_RUNS_DIR / SPECTRA_CACHE_DIR   run root / tile caches (see spectra.sweep.config)
#   SPECTRA_BACKBONE_LOCAL_DIRS "repo=/dir,..." local weights for gated backbones (optional)
#   HF_HOME                     Hugging Face cache (optional)
#
# Requeue / resume: the trainer rewrites $SWEEP_RUN_DIR/resume.pt every ckpt.resume_every
# steps and on SIGTERM (preemption) / SIGUSR1 (180 s before the time limit). Any rerun of this
# script -- SLURM requeue after preemption, the self-requeue below at the time limit, or
# status.py --resubmit -- resumes from it. A finished run (DONE) exits 0.
set -uo pipefail

RUN_DIR="${SWEEP_RUN_DIR:?set SWEEP_RUN_DIR}"
REPO="${SPECTRA_REPO:?set SPECTRA_REPO}"
PY="${SPECTRA_PY:-$REPO/.venv/bin/python}"

export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== $(date) | $(hostname) | job ${SLURM_JOB_ID:-local} restart ${SLURM_RESTART_COUNT:-0} | $RUN_DIR ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

if [ -f "$RUN_DIR/DONE" ]; then echo "already DONE"; exit 0; fi

"$PY" -m spectra.sweep.train --run-dir "$RUN_DIR" &
PID=$!
REQUEUE=0
trap 'echo "[sbatch] USR1 (time limit near): checkpointing"; REQUEUE=1; kill -USR1 $PID 2>/dev/null' USR1
trap 'echo "[sbatch] TERM: forwarding"; kill -TERM $PID 2>/dev/null' TERM
wait $PID; RC=$?
while kill -0 $PID 2>/dev/null; do wait $PID; RC=$?; done

echo "=== exit rc=$RC $(date) ==="
if [ "$REQUEUE" = 1 ] && [ ! -f "$RUN_DIR/DONE" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "[sbatch] self-requeue $SLURM_JOB_ID"
  scontrol requeue "$SLURM_JOB_ID"
fi
exit $RC
