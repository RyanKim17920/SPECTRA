#!/usr/bin/env bash
#SBATCH --job-name=cptac-driver
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/evals/cptac-driver-%j.out
#
# CPTAC (Patho-Bench subset) eval: 38 classification tasks (macro-OvR AUC, logistic probe,
# official splits, 100 test bootstraps) + 12 CoxNet survival fits.
#
# Usage (driver; self-submits the stage jobs with afterok dependencies, resumable/idempotent):
#   bash scripts/evals/run_cptac.sbatch --backbone hoptimus0 [--ckpt RUN_DIR|CKPT.safetensors] [--step N]
#        [--key K] [--precision fp16] [--concurrency 2] [--shards 8] [--batch 1024]
#        [--pooling cls|clsmean] [--retain] [--force]
#   (or `sbatch scripts/evals/run_cptac.sbatch ARGS`: same, from a small CPU job)
#
# ENVIRONMENT
#   SPECTRA_REPO        (required) checkout of this repository
#   SPECTRA_VENV_CPTAC  (required) venv built from scripts/evals/env/cptac_lock.txt
#   PATHOBENCH_DIR      (required) Patho-Bench data folder (splits/ + job_dir/ with the 20x 224px tiles)
#   SPECTRA_EVAL_OUT    results root (default runs/evals)
#   HF_HOME             Hugging Face cache holding the backbone weights (stages run offline)
#   SPECTRA_SBATCH_ARGS      extra sbatch arguments for every stage job (e.g. "--account A --qos Q")
#   SPECTRA_SBATCH_ARGS_GPU  extra sbatch arguments for the GPU extract stage (e.g. "--partition gpu")
#   Create ./logs/evals before submitting (slurm does not create the --output directory).
#
# Stages (each re-run skips completed work; rerunning the same command resumes):
#   extract  GPU array, one element per slide shard (greedy patch-balanced), %concurrency,
#            --requeue; slides are flushed to h5 as they complete -> per-slide resume.
#   pool     CPU, slide means -> case means.
#   probe    CPU array 0-49 (38 linprobe + 4 survival x 3 alphas).
#   finalize CPU, writes $SPECTRA_EVAL_OUT/<key>/cptac.json (headline
#            mean_classification_macro_ovr_auc, 25-task matched mean, per-task, gpu_hours);
#            deletes embeddings unless --retain.
# Work dir: $SPECTRA_EVAL_OUT/<key>/cptac_work. Logs: logs/evals/cptac-<stage>-*.out.
# Upstream: third_party/Patho-Bench @660e770 + env/cptac_patho_bench.patch (pins: env/cptac*.txt).
set -euo pipefail

REPO=${SPECTRA_REPO:?set SPECTRA_REPO}
PY=${SPECTRA_VENV_CPTAC:?set SPECTRA_VENV_CPTAC}/bin/python
: "${PATHOBENCH_DIR:?set PATHOBENCH_DIR}"
SELF=$REPO/scripts/evals/run_cptac.sbatch
LOGS=${SPECTRA_LOGS:-logs/evals}
export SPECTRA_EVAL_OUT=${SPECTRA_EVAL_OUT:-runs/evals}
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 HDF5_USE_FILE_LOCKING=FALSE HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# ---------------------------------------------------------------- stage worker
if [[ -n "${CPTAC_STAGE:-}" ]]; then
  cd "$REPO"
  key=$CPTAC_KEY
  echo "[cptac:$CPTAC_STAGE] key=$key job=${SLURM_JOB_ID:-} array=${SLURM_ARRAY_TASK_ID:-} host=$(hostname) $(date -Is)"
  case "$CPTAC_STAGE" in
    extract)
      export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
      extra=()
      [[ -n "${CPTAC_POOLING:-}" ]] && extra+=(--pooling "$CPTAC_POOLING")
      exec "$PY" scripts/evals/cptac_extract.py --key "$key" --shard "$SLURM_ARRAY_TASK_ID" \
        --shards "$CPTAC_SHARDS" --precision "$CPTAC_PRECISION" --batch "$CPTAC_BATCH" \
        --workers "$(( ${SLURM_CPUS_PER_TASK:-16} ))" "${extra[@]}" ;;
    pool)
      exec "$PY" scripts/evals/cptac_probe.py pool --key "$key" --shards "$CPTAC_SHARDS" ;;
    probe)
      export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} \
        OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
      exec "$PY" scripts/evals/cptac_probe.py probe "$SLURM_ARRAY_TASK_ID" --key "$key" ;;
    finalize)
      extra=()
      [[ "${CPTAC_RETAIN:-0}" == 1 ]] && extra+=(--retain)
      exec "$PY" scripts/evals/cptac_probe.py finalize --key "$key" --jobs "$CPTAC_WORK/jobs.json" "${extra[@]}" ;;
    *) echo "unknown CPTAC_STAGE=$CPTAC_STAGE" >&2; exit 2 ;;
  esac
fi

# ---------------------------------------------------------------- driver
backbone="" ckpt="" step="" key="" precision=fp16
concurrency=2 shards=8 batch=1024 pooling="" retain=0 force=0 gpu_time=12:00:00
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backbone) backbone=$2; shift 2 ;;
    --ckpt) ckpt=$2; shift 2 ;;
    --step) step=$2; shift 2 ;;
    --key) key=$2; shift 2 ;;
    --precision) precision=$2; shift 2 ;;
    --concurrency) concurrency=$2; shift 2 ;;
    --shards) shards=$2; shift 2 ;;
    --batch) batch=$2; shift 2 ;;
    --pooling) pooling=$2; shift 2 ;;
    --gpu-time) gpu_time=$2; shift 2 ;;
    --retain) retain=1; shift ;;
    --force) force=1; shift ;;
    -h|--help) sed -n 2,40p "$SELF"; exit 0 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done
[[ -n "$backbone$ckpt" ]] || { echo "need --backbone and/or --ckpt" >&2; exit 2; }
mkdir -p "$LOGS"
# preflight: pinned upstream + patch, env imports (cheap; fails before anything is queued)
pb=$REPO/third_party/Patho-Bench
[[ "$(git -c safe.directory='*' -C "$pb" rev-parse HEAD)" == 660e77044640e3d7d2f1150cc6721e97454993bf ]] || { echo "Patho-Bench not at pinned commit" >&2; exit 4; }
cmp -s "$REPO/scripts/evals/env/cptac_patho_bench.patch" <(git -c safe.directory='*' -C "$pb" diff --binary HEAD) || { echo "Patho-Bench working tree != cptac_patho_bench.patch" >&2; exit 4; }
"$PY" -c "import h5py, yaml, sksurv, patho_bench.ExperimentFactory, spectra.evals.spec" || { echo "env import failure" >&2; exit 4; }

spec_args=()
[[ -n "$backbone" ]] && spec_args+=(--backbone "$backbone")
[[ -n "$ckpt" ]] && spec_args+=(--ckpt "$ckpt")
[[ -n "$step" ]] && spec_args+=(--step "$step")
[[ -n "$key" ]] && spec_args+=(--key "$key")
[[ "$force" == 1 ]] && spec_args+=(--force)
key=$(cd "$REPO" && "$PY" -m spectra.evals.spec "${spec_args[@]}" --evals-dir --print-key)
out=$SPECTRA_EVAL_OUT/$key
work=$out/cptac_work
echo "[cptac] model_key=$key spec=$out/spec.json work=$work"

if [[ -f "$out/cptac.json" && "$force" == 0 ]] && \
   "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('status')=='complete' else 1)" "$out/cptac.json"; then
  echo "[cptac] already complete: $out/cptac.json (pass --force to redo)"; exit 0
fi
if [[ "$force" == 1 && -d "$work" ]]; then
  echo "[cptac] --force: clearing $work"; rm -rf "$work"; rm -f "$out/cptac.json"
fi
mkdir -p "$work"

# Protocol knobs that change the embeddings are pinned per work dir.
knobs="precision=$precision pooling=${pooling:-default} shards=$shards"
if [[ -f "$work/knobs.txt" && "$(cat "$work/knobs.txt")" != "$knobs" ]]; then
  echo "[cptac] $work was started with '$(cat "$work/knobs.txt")' != '$knobs'; pass --force to restart" >&2; exit 3
fi
echo "$knobs" > "$work/knobs.txt"

# shellcheck disable=SC2206
common=(--parsable --requeue --kill-on-invalid-dep=yes ${SPECTRA_SBATCH_ARGS:-})
# shellcheck disable=SC2206
gpu_args=(${SPECTRA_SBATCH_ARGS_GPU:-})
exp="ALL,CPTAC_KEY=$key,CPTAC_WORK=$work,CPTAC_SHARDS=$shards,CPTAC_PRECISION=$precision,CPTAC_BATCH=$batch,CPTAC_POOLING=$pooling,CPTAC_RETAIN=$retain"
jobs_json=$work/jobs.json
record() {  # stage job_id
  "$PY" - "$jobs_json" "$1" "$2" <<'EOF'
import json, sys, os
p, stage, jid = sys.argv[1:]
d = json.load(open(p)) if os.path.exists(p) else []
d.append({"stage": stage, "job_id": jid})
json.dump(d, open(p, "w"), indent=1)
EOF
}

pending=()
for ((s = 0; s < shards; s++)); do
  [[ -f "$work/extract_${s}_of_${shards}.done" ]] || pending+=("$s")
done
dep=""
if [[ ${#pending[@]} -gt 0 && ! -f "$work/.pooled" ]]; then
  list=$(IFS=,; echo "${pending[*]}")
  jid=$(sbatch "${common[@]}" --job-name="cptac-ext-$key" "${gpu_args[@]}" \
        --gres=gpu:1 --cpus-per-task=16 --mem=128G --time="$gpu_time" --array="$list%$concurrency" \
        --output="$LOGS/cptac-extract-${key}-%A_%a.out" --export="$exp,CPTAC_STAGE=extract" "$SELF")
  record extract "$jid"; dep="--dependency=afterok:$jid"
  echo "[cptac] extract shards [$list] -> job $jid"
fi
if [[ ! -f "$work/.pooled" ]]; then
  jid=$(sbatch "${common[@]}" $dep --job-name="cptac-pool-$key" --cpus-per-task=4 --mem=16G \
        --time=02:00:00 --output="$LOGS/cptac-pool-${key}-%j.out" \
        --export="$exp,CPTAC_STAGE=pool" "$SELF")
  record pool "$jid"; dep="--dependency=afterok:$jid"
  echo "[cptac] pool -> job $jid"
fi
jid=$(sbatch "${common[@]}" $dep --job-name="cptac-probe-$key" --cpus-per-task=8 --mem=24G \
      --time=06:00:00 --array=0-49%16 --output="$LOGS/cptac-probe-${key}-%A_%a.out" \
      --export="$exp,CPTAC_STAGE=probe" "$SELF")
record probe "$jid"; dep="--dependency=afterok:$jid"
echo "[cptac] probes -> job $jid"
jid=$(sbatch "${common[@]}" $dep --job-name="cptac-final-$key" --cpus-per-task=2 --mem=8G \
      --time=01:00:00 --output="$LOGS/cptac-finalize-${key}-%j.out" \
      --export="$exp,CPTAC_STAGE=finalize" "$SELF")
record finalize "$jid"
echo "[cptac] finalize -> job $jid ; result will be $out/cptac.json"
