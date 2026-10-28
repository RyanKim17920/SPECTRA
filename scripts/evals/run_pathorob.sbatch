#!/bin/bash
#SBATCH --job-name=pathorob-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=logs/evals/pathorob-%x-%j.out
#
# PathoROB robustness index (camelyon, tcga, tolkach_esca; clsmean) + CROMA for ONE model spec.
# Result: $SPECTRA_EVAL_OUT/<model_key>/pathorob.json  (+ croma.json, same features, no extra GPU work).
#
# USAGE
#   bash   scripts/evals/run_pathorob.sbatch --backbone hoptimus0 [--ckpt RUN_DIR|CKPT.safetensors] [--step N]
#              [--key K] [--ema] [--protocol resize224|resize256] [--precision fp32|fp16|bf16]
#              [--retain] [--force] [--no-bootstrap]
#        -> resolves the spec, writes/guards $SPECTRA_EVAL_OUT/<key>/spec.json, then self-submits
#           (1) GPU extraction job  ->  (2) CPU metrics+finalize job (--dependency=afterok). Idempotent/resumable.
#   sbatch scripts/evals/run_pathorob.sbatch ARGS
#        -> single-allocation mode: extraction (GPU) then metrics+finalize inline on the same node.
#
# Examples
#   bash scripts/evals/run_pathorob.sbatch --backbone hoptimus0 --key hopt0-base
#   bash scripts/evals/run_pathorob.sbatch --backbone hoptimus0 --ckpt runs/sweeps/<run> --step 200
#
# ENVIRONMENT (pins in scripts/evals/env/pathorob_*.txt)
#   SPECTRA_REPO                   (required) checkout of this repository
#   SPECTRA_VENV_PATHOROB_EXTRACT  (required) extraction venv (torch 2.8.0 timm 1.0.30 peft 0.20.0 pyarrow)
#   SPECTRA_VENV_PATHOROB_METRICS  (required) metrics venv (numpy 2.2.6 pandas 2.3.2)
#   PATHOROB_ROOT                  clone of PathoROB @6583cf0 + env/pathorob_bootstrap_ri.patch (default third_party/PathoROB)
#   PATHOROB_CROMA_ROOT            clone of croma @873612e (default third_party/croma)
#   SPECTRA_EVAL_OUT               results root (default runs/evals)
#   SPECTRA_PATHOROB_FEATURES      feature work dir root (default runs/pathorob_features; deleted after the
#                                  result JSON unless --retain)
#   HF_HOME                        Hugging Face cache with the backbone weights and the bifold-pathomics/PathoROB-*
#                                  datasets (jobs run with HF_HUB_OFFLINE=1)
#   SPECTRA_SBATCH_ARGS            extra sbatch arguments for the self-submit (e.g. "--account A --partition P --qos Q")
#   Create ./logs/evals before submitting (slurm does not create the --output directory).
# Protocol default `resize224` = fp32, Resize(224)->CenterCrop(224), cat(cls, mean patch).
# `resize256` = fp16, bicubic Resize 256 + CenterCrop 224.
# Expected cost per model: ~10 GPU-min extraction + ~10 CPU-min RI/CROMA (see pathorob.json gpu_hours/cpu_hours).

set -uo pipefail
EV=${SPECTRA_REPO:?set SPECTRA_REPO}
EXTRACT_PY=${SPECTRA_VENV_PATHOROB_EXTRACT:?set SPECTRA_VENV_PATHOROB_EXTRACT}/bin/python
METRICS_PY=${SPECTRA_VENV_PATHOROB_METRICS:?set SPECTRA_VENV_PATHOROB_METRICS}/bin/python
export SPECTRA_VENV_PATHOROB_METRICS
export SPECTRA_EVAL_OUT=${SPECTRA_EVAL_OUT:-runs/evals}
RUN=$EV/scripts/evals/pathorob_run.py
export PYTHONPATH=$EV/src${PYTHONPATH:+:$PYTHONPATH}
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
mkdir -p logs/evals

ARGS=("$@")
PASS=("${ARGS[@]}")
# flags every subcommand understands (prepare additionally takes the spec flags)
COMMON=(); SPECF=(); i=0
while [ $i -lt ${#PASS[@]} ]; do
  case "${PASS[$i]}" in
    --protocol|--precision|--num-workers|--batch-size) COMMON+=("${PASS[$i]}" "${PASS[$((i+1))]}"); i=$((i+2));;
    --retain|--force|--no-bootstrap) COMMON+=("${PASS[$i]}"); i=$((i+1));;
    --backbone|--ckpt|--step|--key) SPECF+=("${PASS[$i]}" "${PASS[$((i+1))]}"); i=$((i+2));;
    --ema) SPECF+=("--ema"); i=$((i+1));;
    *) echo "unknown arg ${PASS[$i]}" >&2; exit 2;;
  esac
done

STAGE=${PATHOROB_STAGE:-}
if [ -z "$STAGE" ]; then
  eval "$("$EXTRACT_PY" "$RUN" prepare "${COMMON[@]}" "${SPECF[@]}")" || exit 1
  echo "[run_pathorob] key=$KEY result_done=$RESULT_DONE extract_done=$EXTRACT_DONE"
  if [ "$RESULT_DONE" = 1 ] && [[ ! " ${COMMON[*]} " =~ " --force " ]]; then
    echo "[run_pathorob] $SPECTRA_EVAL_OUT/$KEY/pathorob.json already complete; nothing to do (use --force)"; exit 0
  fi
  KEYARG=(--key "$KEY")
  if [ -z "${SLURM_JOB_ID:-}" ]; then
    # ---- bash mode: self-submit ----
    EXPORT="ALL,PATHOROB_COMMON=${COMMON[*]}"
    DEP=()
    if [ "$EXTRACT_DONE" != 1 ]; then
      J1=$(sbatch --parsable --requeue ${SPECTRA_SBATCH_ARGS:-} --job-name "pathorob-x-$KEY" \
           --export="$EXPORT,PATHOROB_STAGE=extract,PATHOROB_KEY=$KEY" "$EV/scripts/evals/run_pathorob.sbatch")
      echo "[run_pathorob] extract job $J1"; DEP=(--dependency=afterok:$J1 --kill-on-invalid-dep=yes)
    fi
    J2=$(sbatch --parsable --requeue ${SPECTRA_SBATCH_ARGS:-} --gres=gpu:0 --cpus-per-task=16 --mem=64G --time=02:00:00 \
         --job-name "pathorob-m-$KEY" "${DEP[@]}" \
         --export="$EXPORT,PATHOROB_STAGE=metrics,PATHOROB_KEY=$KEY" "$EV/scripts/evals/run_pathorob.sbatch")
    echo "[run_pathorob] metrics+finalize job $J2 (afterok ${J1:-none}) -> $SPECTRA_EVAL_OUT/$KEY/pathorob.json"
    exit 0
  fi
  # ---- single-allocation sbatch mode ----
  PATHOROB_STAGE=all
  STAGE=all
else
  KEY=$PATHOROB_KEY
  read -r -a COMMON <<< "${PATHOROB_COMMON:-}"
fi
# (stage runs below; COMMON/KEY are set in all paths)
KEYARG=(--key "$KEY")
echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} | stage $STAGE | key $KEY ==="
rc=0
if [ "$STAGE" = extract ] || [ "$STAGE" = all ]; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  "$EXTRACT_PY" "$RUN" extract "${COMMON[@]}" "${KEYARG[@]}" --num-workers 16 || rc=$?
  [ $rc -ne 0 ] && { echo "extract failed rc=$rc"; exit $rc; }
fi
if [ "$STAGE" = metrics ] || [ "$STAGE" = all ]; then
  "$METRICS_PY" "$RUN" metrics "${COMMON[@]}" "${KEYARG[@]}" || rc=$?
  [ $rc -ne 0 ] && { echo "metrics failed rc=$rc"; exit $rc; }
  "$EXTRACT_PY" "$RUN" finalize "${COMMON[@]}" "${KEYARG[@]}" || rc=$?
  [ $rc -ne 0 ] && { echo "finalize failed rc=$rc"; exit $rc; }
fi
echo "=== done $(date) ==="
