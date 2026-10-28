#!/bin/bash
#SBATCH --job-name=ev-hest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/evals/hest-%j.out
#SBATCH --error=logs/evals/hest-%j.out
#
# HEST-Benchmark (9 tasks, PCA-256 + Ridge, Pearson, official patient splits) for ONE model
#   -> $SPECTRA_EVAL_OUT/<model_key>/hest.json
#
#   bash scripts/evals/run_hest.sbatch --backbone hoptimus0 --key hopt0-base
#   bash scripts/evals/run_hest.sbatch --backbone hoptimus0 --ckpt runs/sweeps/<run> --step 200
#   (bash mode resolves the spec on the login node, then self-submits with sbatch; the driver
#    is resumable per task, so the job is submitted with --requeue.)
#   Account / partition / qos are not set here: pass them through SPECTRA_SBATCH_ARGS, e.g.
#   SPECTRA_SBATCH_ARGS="--account A --partition P --qos Q" bash scripts/evals/run_hest.sbatch ...
#   Create ./logs/evals before submitting (slurm does not create the --output directory).
#
# ARGS: --backbone B [--ckpt RUN_DIR|CKPT.safetensors] [--step N] [--ema] [--key K]
#       [--precision fp32|fp16|bf16] [--retain] [--force] [--tasks T ...] [--pooling P] [--out-name N]
#
# ENVIRONMENT
#   SPECTRA_REPO       (required) checkout of this repository
#   SPECTRA_VENV_HEST  (required) venv built by scripts/evals/env/hest_build.sh
#                      (python 3.11, torch 2.11, timm 1.0.28; lock scripts/evals/env/hest_lock.txt)
#   HEST_DATA_DIR      (required) MahmoodLab/hest-bench download (<task>/{splits,patches})
#   SPECTRA_EVAL_OUT   results root (default runs/evals)
#   HF_HOME            Hugging Face cache holding the backbone weights (compute nodes run offline)
#   SPECTRA_SBATCH_ARGS  extra sbatch arguments for the self-submit (account, partition, qos)
#
# Default protocol: fp32, per-backbone pooling (spectra.evals.protocol.pooling_for),
# Resize(224,bilinear) transform, batch 256, --num-workers 0, HEST @ 3ddb5eaf + env/hest.patch
# (third_party/HEST).
#
# Concurrency: no shared writable state between jobs (each key has its own flock'd work dir;
# bench h5 files are read-only and HDF5 read locks are shared). Two jobs on the SAME key are
# refused by the driver.
set -euo pipefail

WT="${SPECTRA_REPO:?set SPECTRA_REPO}"
PY="${SPECTRA_VENV_HEST:?set SPECTRA_VENV_HEST}/bin/python"
: "${HEST_DATA_DIR:?set HEST_DATA_DIR}"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export PYTHONPATH="$WT/src${PYTHONPATH:+:$PYTHONPATH}"
EVALS="${SPECTRA_EVAL_OUT:-runs/evals}"
export SPECTRA_EVAL_OUT="$EVALS"

ORIG=("$@")
SPEC_ARGS=(); RUN_ARGS=(); FORCE=0; PRECISION=fp32
while [ $# -gt 0 ]; do
  case "$1" in
    --backbone|--ckpt|--step|--key) SPEC_ARGS+=("$1" "$2"); shift 2 ;;
    --ema) SPEC_ARGS+=("$1"); shift ;;
    --precision) PRECISION="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --retain) RUN_ARGS+=(--retain); shift ;;
    --tasks) shift; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do RUN_ARGS+=(--tasks-item "$1"); shift; done ;;
    --pooling|--out-name|--batch-size|--num-workers|--bench-data) RUN_ARGS+=("$1" "$2"); shift 2 ;;
    *) echo "run_hest: unknown arg $1" >&2; exit 2 ;;
  esac
done
# --tasks-item -> one --tasks list
TASKS=(); REST=()
for ((i=0; i<${#RUN_ARGS[@]}; i++)); do
  if [ "${RUN_ARGS[$i]}" = --tasks-item ]; then i=$((i+1)); TASKS+=("${RUN_ARGS[$i]}"); else REST+=("${RUN_ARGS[$i]}"); fi
done
[ ${#TASKS[@]} -gt 0 ] && REST+=(--tasks "${TASKS[@]}")
[ ${#SPEC_ARGS[@]} -gt 0 ] || { echo "run_hest: need --backbone and/or --ckpt" >&2; exit 2; }

cd "$WT"
SPEC_FORCE=(); [ "$FORCE" = 1 ] && SPEC_FORCE=(--force)
# Resolve the spec ONCE and file it under the key (refuses a different identity unless --force).
KEY=$("$PY" -m spectra.evals.spec "${SPEC_ARGS[@]}" --evals-dir "${SPEC_FORCE[@]}" --print-key)
SPEC="$EVALS/$KEY/spec.json"
echo "[run_hest] key=$KEY spec=$SPEC"

if [ -z "${SLURM_JOB_ID:-}" ]; then
  mkdir -p "$WT/logs/evals"
  # shellcheck disable=SC2086
  exec sbatch --requeue ${SPECTRA_SBATCH_ARGS:-} --job-name="ev-hest-$KEY" "$0" "${ORIG[@]}"
fi

echo "=== $(date) | node $(hostname) | job $SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} | key $KEY ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true
DRV=(--spec "$SPEC" --precision "$PRECISION" "${REST[@]}")
# --force recomputes on the first attempt only; a requeued attempt resumes from the done markers
if [ "$FORCE" = 1 ] && [ "${SLURM_RESTART_COUNT:-0}" = 0 ]; then DRV+=(--force); fi
srun --unbuffered "$PY" "$WT/scripts/evals/hest_run.py" "${DRV[@]}"
echo "=== done $(date) | key $KEY ==="
