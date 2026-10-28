#!/usr/bin/env bash
#SBATCH --job-name=thunder
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=logs/evals/thunder_%x_%A_%a.out
#
# THUNDER (current website leaderboard): 16 classification datasets x {kNN, linear probing,
# calibration ECE, 16-shot SimpleShot, PGD attack} + 4 segmentation datasets, for one model
# -> $SPECTRA_EVAL_OUT/<model_key>/thunder.json
#
# USAGE
#   bash scripts/evals/run_thunder.sbatch --backbone hoptimus0                        # base model
#   bash scripts/evals/run_thunder.sbatch --backbone hoptimus0 --ckpt runs/sweeps/<run> --step 200
#   options: --key K (<=52 chars; THUNDER appends _optimized) --tasks cached|online|all (default all)
#            --precision fp16|bf16|fp32 (encoder autocast, default fp16)
#            --adv-precision fp16|fp32 (PGD forward/backward; default fp16)
#            --pooling cls|mean|clsmean (override spectra.evals.protocol.THUNDER_POOLING)
#            --gpus N (max concurrent GPUs per stage array, default 2)
#            --overlap (start segmentation right after preflight, concurrently with the cached
#              chain: up to 2N GPUs; shortest wall time)
#            --retain (keep classification embeddings)  --force (re-bind key: wipes thunder work)
#            --dry-run (print the sbatch commands only)
#   `bash` self-submits: preflight -> precompute[2] -> cached[2] -> cleanup -> online[6] -> summary
#   (afterok chain; every stage is requeueable and resumes per dataset).
#   `sbatch scripts/evals/run_thunder.sbatch ARGS` instead runs every stage serially inside
#   that one 1-GPU allocation (slow; ~1 day for a ViT-g).
#   Re-running the same command resumes: finished datasets/tasks are skipped.
#
# ENVIRONMENT
#   SPECTRA_REPO         (required) checkout of this repository
#   SPECTRA_VENV_THUNDER (required) venv built from scripts/evals/env/thunder_requirements.lock.txt
#                        (python 3.13, torch 2.11, timm 1.0.20; full freeze: thunder_venv_freeze.txt)
#   THUNDER_DATA         (required) THUNDER data folder (official data_splits/ + the datasets)
#   THUNDER_UPSTREAM     clone of MICS-Lab/thunder at 3d1cc95 with env/thunder*.patch applied
#                        (default $SPECTRA_REPO/third_party/thunder)
#   SPECTRA_EVAL_OUT     results root (default runs/evals)
#   HF_HOME              Hugging Face cache holding the backbone weights
#   SPECTRA_SBATCH_ARGS  extra sbatch arguments for every stage job (e.g. "--account A --partition P --qos Q")
#   Create ./logs/evals before submitting (slurm does not create the --output directory).
#
# PROTOCOL: upstream MICS-Lab/thunder 3d1cc95 + scripts/evals/env/thunder.patch (vectorized
#   KNN / SimpleShot / linear heads / seg metrics, int8 seg token cache, wandb disabled) +
#   env/thunder_local.patch (user+job-scoped /tmp staging only). Encoder fp16 autocast, features
#   fp32, timm pretrained_cfg transform (H-optimus-0: bicubic Resize(256) + CenterCrop(224)),
#   extraction batch 1024, attack batch 128 (32 for fp32 PGD), adaptation batch 64, SegPath
#   epithelial/lymphocytes 9/21 epochs.
#   Outputs: $SPECTRA_EVAL_OUT/<key>/thunder_work/outputs/res/{results.csv,<ds>/<key>_optimized/...};
#   thunder.json carries the six roll-ups, per-dataset scores, gpu_hours (+ by stage), jobs.
#   GPU-hours per H-optimus-0 model ~13.2 (preflight 0.02, precompute 1.4-1.6, cached 3.4-3.5,
#   segmentation 7.4, PGD 0.8).
set -euo pipefail
export PYTHONNOUSERSITE=1

WT=${SPECTRA_REPO:?set SPECTRA_REPO}
PY=${SPECTRA_VENV_THUNDER:?set SPECTRA_VENV_THUNDER}/bin/python
: "${THUNDER_DATA:?set THUNDER_DATA}"
DRIVER=$WT/scripts/evals/thunder_eval.py
SELF=$WT/scripts/evals/run_thunder.sbatch
EVALS=${SPECTRA_EVAL_OUT:-runs/evals}
LOGS=${SPECTRA_LOGS:-logs/evals}
export SPECTRA_EVAL_OUT=$EVALS
export PYTHONPATH="$WT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$LOGS"

# ------------------------------------------------------------------------------------------
# worker mode: one stage (array element) of a self-submitted chain
# ------------------------------------------------------------------------------------------
if [[ -n "${THUNDER_STAGE:-}" ]]; then
  : "${THUNDER_SPEC:?}"
  export THUNDER_SPEC
  i=${SLURM_ARRAY_TASK_ID:-0}
  if [[ "$THUNDER_STAGE" == cleanup || "$THUNDER_STAGE" == summary ]]; then
    export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
  else
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  fi
  echo "START stage=$THUNDER_STAGE index=$i host=$(hostname) job=${SLURM_JOB_ID:-} spec=$THUNDER_SPEC"
  case "$THUNDER_STAGE" in
    preflight) exec "$PY" "$DRIVER" preflight ;;
    precompute) exec "$PY" "$DRIVER" precompute_group "$i" ;;
    cached) exec "$PY" "$DRIVER" cached_group "$i" ;;
    cleanup) exec "$PY" "$DRIVER" cleanup ;;
    online)
      items=("segmentation 1" "segmentation 2" "segmentation 3" "segmentation 0"
             "adversarial_group 0" "adversarial_group 1")
      read -r -a t <<< "${items[$i]}"
      exec "$PY" "$DRIVER" "${t[@]}" ;;
    seg)
      items=(1 2 3 0)
      exec "$PY" "$DRIVER" segmentation "${items[$i]}" ;;
    adv) exec "$PY" "$DRIVER" adversarial_group "$i" ;;
    summary) exec "$PY" "$DRIVER" summary ;;
    *) echo "unknown THUNDER_STAGE=$THUNDER_STAGE" >&2; exit 2 ;;
  esac
fi

# ------------------------------------------------------------------------------------------
# argument parsing (submitter / monolithic)
# ------------------------------------------------------------------------------------------
backbone="" ckpt="" step="" key="" precision=fp16 adv_precision=fp16 pooling=""
retain=0 force=0 tasks=all gpus=2 overlap=0 dry=0 attack_batch=""
extraction_batch=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backbone) backbone=$2; shift 2 ;;
    --ckpt) ckpt=$2; shift 2 ;;
    --step) step=$2; shift 2 ;;
    --key) key=$2; shift 2 ;;
    --precision) precision=$2; shift 2 ;;
    --adv-precision) adv_precision=$2; shift 2 ;;
    --pooling) pooling=$2; shift 2 ;;
    --tasks) tasks=$2; shift 2 ;;
    --gpus) gpus=$2; shift 2 ;;
    --attack-batch) attack_batch=$2; shift 2 ;;
    --extraction-batch) extraction_batch=$2; shift 2 ;;
    --overlap) overlap=1; shift ;;
    --retain) retain=1; shift ;;
    --force) force=1; shift ;;
    --dry-run) dry=1; shift ;;
    -h|--help) sed -n '2,50p' "$SELF"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ "$tasks" =~ ^(cached|online|all|summary)$ ]] || { echo "--tasks cached|online|all|summary" >&2; exit 2; }
[[ "$precision" =~ ^(fp16|bf16|fp32)$ && "$adv_precision" =~ ^(fp16|bf16|fp32)$ ]] || { echo "bad precision" >&2; exit 2; }
[[ -n "$backbone$ckpt" ]] || { echo "need --backbone and/or --ckpt" >&2; exit 2; }

spec_args=()
[[ -n "$backbone" ]] && spec_args+=(--backbone "$backbone")
[[ -n "$ckpt" ]] && spec_args+=(--ckpt "$ckpt")
[[ -n "$step" ]] && spec_args+=(--step "$step")
[[ -n "$key" ]] && spec_args+=(--key "$key")
[[ "$force" == 1 ]] && spec_args+=(--force)
key=$("$PY" -m spectra.evals.spec "${spec_args[@]}" --evals-dir --print-key)
[[ ${#key} -le 52 ]] || { echo "model key '$key' is ${#key} > 52 chars; pass --key" >&2; exit 2; }
spec=$EVALS/$key/spec.json
work=$EVALS/$key/thunder_work
if [[ "$force" == 1 && -d "$work" ]]; then
  echo "--force: removing $work and $EVALS/$key/thunder.json"
  rm -rf "$work" "$EVALS/$key/thunder.json"
fi
mkdir -p "$work"

export THUNDER_SPEC=$spec THUNDER_PRECISION=$precision THUNDER_ADV_PRECISION=$adv_precision
export RETAIN_EMBEDDINGS=$retain
[[ -n "$pooling" ]] && export THUNDER_POOLING=$pooling
[[ -n "$attack_batch" ]] && export ATTACK_BATCH=$attack_batch
[[ -n "$extraction_batch" ]] && export EXTRACTION_BATCH=$extraction_batch

if [[ -f "$EVALS/$key/thunder.json" && "$tasks" != summary ]] && "$PY" - "$EVALS/$key/thunder.json" "$spec" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1])); s = json.load(open(sys.argv[2]))
sys.exit(0 if r.get("status") == "complete" and (r.get("spec") or {}).get("identity_sha256") == s["identity_sha256"] else 1)
EOF
then
  echo "$key: thunder.json already complete (use --force to recompute)"; exit 0
fi

# ------------------------------------------------------------------------------------------
# monolithic mode: `sbatch run_thunder.sbatch ARGS` -> run everything in this allocation
# ------------------------------------------------------------------------------------------
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "{\"job_id\": \"$SLURM_JOB_ID\", \"stage\": \"monolithic\"}" >> "$work/slurm_jobs.jsonl"
  export OMP_NUM_THREADS=1
  "$PY" "$DRIVER" preflight
  if [[ "$tasks" == cached || "$tasks" == all ]]; then
    for g in 0 1; do "$PY" "$DRIVER" precompute_group $g; done
    for g in 0 1; do "$PY" "$DRIVER" cached_group $g; done
    "$PY" "$DRIVER" cleanup
  fi
  if [[ "$tasks" == online || "$tasks" == all ]]; then
    for s in 1 2 3 0; do "$PY" "$DRIVER" segmentation $s; done
    for g in 0 1; do "$PY" "$DRIVER" adversarial_group $g; done
  fi
  exec "$PY" "$DRIVER" summary
fi

# ------------------------------------------------------------------------------------------
# submitter mode: self-submit the stage chain
# ------------------------------------------------------------------------------------------
st=$("$PY" "$DRIVER" status)
cached_done=$(echo "$st" | "$PY" -c 'import json,sys; print(int(json.load(sys.stdin)["cached_complete"]))')
echo "$key status: $st"
exp="ALL,THUNDER_SPEC=$spec,THUNDER_PRECISION=$precision,THUNDER_ADV_PRECISION=$adv_precision,RETAIN_EMBEDDINGS=$retain"
[[ -n "$pooling" ]] && exp+=",THUNDER_POOLING=$pooling"
[[ -n "$attack_batch" ]] && exp+=",ATTACK_BATCH=$attack_batch"
[[ -n "$extraction_batch" ]] && exp+=",EXTRACTION_BATCH=$extraction_batch"
# shellcheck disable=SC2206
A=(--parsable --requeue ${SPECTRA_SBATCH_ARGS:-})
CPU=(--gres=none --cpus-per-task=4 --mem=16G --time=02:00:00)
submit() {  # submit STAGE NAME [sbatch args...]
  local stage=$1 name=$2; shift 2
  if [[ "$dry" == 1 ]]; then
    echo "DRY sbatch ${A[*]} --job-name=$name $* --export=$exp,THUNDER_STAGE=$stage $SELF" >&2
    echo "DRY_$stage"; return
  fi
  local id
  id=$(sbatch "${A[@]}" --job-name="$name" --output="$LOGS/thunder_${key}_${stage}_%A_%a.out" "$@" \
       --export="$exp,THUNDER_STAGE=$stage" "$SELF")
  id=${id%%;*}
  echo "{\"job_id\": \"$id\", \"stage\": \"$stage\", \"submitted\": \"$(date -Is)\"}" >> "$work/slurm_jobs.jsonl"
  echo "$id"
}
dep() { [[ -n "$1" ]] && echo "--dependency=afterok:$1" || true; }
short=${key:0:20}

if [[ "$tasks" == summary ]]; then
  sm=$(submit summary "th-sum-$short" --array=0-0 "${CPU[@]}")
  echo "THUNDER $key summary=$sm"; exit 0
fi
pre=$(submit preflight "th-pre-$short" --array=0-0 --mem=32G --time=01:00:00)
last_cached=$pre
jobs="preflight=$pre"
if [[ "$tasks" == cached || "$tasks" == all ]] && [[ "$cached_done" == 0 ]]; then
  p=$(submit precompute "th-emb-$short" --array=0-1%"$gpus" $(dep "$pre"))
  c=$(submit cached "th-cache-$short" --array=0-1%"$gpus" --mem=32G $(dep "$p"))
  cl=$(submit cleanup "th-clean-$short" --array=0-0 "${CPU[@]}" $(dep "$c"))
  last_cached=$cl
  jobs+=" precompute=$p cached=$c cleanup=$cl"
fi
last=$last_cached
if [[ "$tasks" == online || "$tasks" == all ]]; then
  if [[ "$overlap" == 1 ]]; then
    sg=$(submit seg "th-seg-$short" --array=0-3%"$gpus" $(dep "$pre"))
    ad=$(submit adv "th-adv-$short" --array=0-1%"$gpus" $(dep "$last_cached"))
    last="$sg:$ad"
    jobs+=" seg=$sg adv=$ad"
  else
    on=$(submit online "th-online-$short" --array=0-5%"$gpus" $(dep "$last_cached"))
    last=$on
    jobs+=" online=$on"
  fi
fi
sm=$(submit summary "th-sum-$short" --array=0-0 "${CPU[@]}" $(dep "$last"))
jobs+=" summary=$sm"
echo "THUNDER $key (gpus/stage=$gpus overlap=$overlap tasks=$tasks): $jobs"
echo "  logs: $LOGS/thunder_${key}_<stage>_<job>_<idx>.out   result: $EVALS/$key/thunder.json"
