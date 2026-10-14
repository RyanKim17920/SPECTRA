#!/bin/bash
#SBATCH --job-name=main_1
#SBATCH --account=max
#SBATCH --qos=normal
#SBATCH --partition=n
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=/data/ryan.kim/ashby/slurm/%x_%j.out
#
# SPECTRA-CONCH: the gentle.sbatch production recipe (genMASK-c50: grid C=2 T=900,
# same-core false-negative mask, 500 steps, ckpt every 50, lr 1e-4, r=32/alpha=64,
# warmup 200, wd 0.05, tau 0.07) applied to CONCH v1's visual tower, plus an optional
# embedding anchor  lambda * (1 - cos(f_lora(x), f_base(x)))  on the exported 512-d vector.
#
# Differences from gentle.sbatch, and why:
#   * PIN = snapshots/conch-anchor (= falseneg-gated + CONCH loader + anchor term).
#   * --pooling cls, single projector, NO --split-heads/--pool-head gem: CONCH exports ONE
#     vector (encode_image(proj_contrast=False, normalize=False), the attentional-pooler
#     output SlideChat and the SAE read); see src/waivphaet/models/conch_visual.py.
#     The production bias for the CLS head (+3.0) is applied as the single head's
#     --same-core-logit-bias, since this head plays the CLS head's role.
#   * --grid-forward-chunk 600 (memory only, exact: one graph, loss sees all C*T; allowed
#     because the single head uses no learned pool head). Unchunked peaked at 78/80 GB.
#   * 1 GPU, --mem=256G (not 700G): 700G pays for pinned offloaded activations of ViT-L/g
#     at T=900. ViT-B/16@448 checkpointed block inputs at B=1800 are ~52 GB, so offload is
#     kept (exact, avoids an 80 GB edge) and 256G covers it with margin.
#   * No RI eval follower (eval_checkpoints.py has no CONCH path); evals are run per
#     checkpoint by scripts/conch_eval.py afterwards.
#   * Job name main_<ngpus>, account max/normal, partition n (cluster convention).
#
#   CONCH_ARM=anc0 CONCH_ANCHOR=0 sbatch scripts/conch_gentle.sbatch
#   CONCH_ARM=anc1 CONCH_ANCHOR=1 sbatch scripts/conch_gentle.sbatch
set -euo pipefail

REPO="${SPECTRA_REPO:-/data/ryan.kim/ashby/third_party/SPECTRA-conch}"
PIN="${SPECTRA_PIN:-$REPO/snapshots/conch-anchor}"
PY="${CONCH_PY:-/data/ryan.kim/ashby/envs/conch/bin/python}"
PACKED_DIR="${SPECTRA_PACKED_DIR:-/data/plism/repacked}"
CORE_LABELS="${SPECTRA_CORE_LABELS:-/data/SPECTRA-runs/runs/.plism_core_labels.npy}"
ARM="${CONCH_ARM:?set CONCH_ARM (e.g. anc0 / anc1)}"
ANCHOR="${CONCH_ANCHOR:?set CONCH_ANCHOR (lambda, e.g. 0 or 1)}"
SEED="${SPECTRA_SEED:-0}"
T="${SPECTRA_T:-900}"
MAX_STEPS="${SPECTRA_MAX_STEPS:-500}"
CKPT_EVERY="${SPECTRA_CKPT_EVERY:-50}"
BCLS="${SPECTRA_BCLS:-3.0}"
OUT_ROOT="${CONCH_OUT_ROOT:-/data/ryan.kim/ashby/runs/spectra_conch}"
OUT_DIR="$OUT_ROOT/$ARM"

source ~/.cache_env.sh 2>/dev/null || true
export HF_HOME=/data/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID:-local}
export PYTHONPATH="$PIN/src"
export WAIV_RUNS="${WAIV_RUNS:-/data/SPECTRA-runs/runs}"

mkdir -p "$OUT_DIR"
echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} restart ${SLURM_RESTART_COUNT:-0} ==="
echo "arm $ARM anchor_lambda=$ANCHOR space=${CONCH_ANCHOR_SPACE:-export} sched=${SPECTRA_CKPT_SCHEDULE:-} seed=$SEED T=$T steps=$MAX_STEPS ckpt=$CKPT_EVERY bcls=$BCLS"
echo "pin $PIN  out $OUT_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

"$PY" - "$PIN" <<'EOF'
import importlib, sys
from pathlib import Path
pin = Path(sys.argv[1]); pkg = importlib.import_module("waivphaet")
assert Path(pkg.__file__).is_relative_to(pin), f"PIN LEAK: {pkg.__file__}"
print(f"[pin] waivphaet={pkg.__file__}", flush=True)
EOF

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
  --format=csv,noheader -l 10 > "$OUT_DIR/gpu.csv" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

RESUME=()
# SLURM requeue: continue from the newest complete checkpoint in this arm's dir.
LAST=$(ls -d "$OUT_DIR"/step_* 2>/dev/null | while read d; do [ -f "$d/metrics.json" ] && echo "$d"; done | sort | tail -1 || true)
[ -n "$LAST" ] && RESUME=(--resume-from "$LAST") && echo "resume from $LAST"

set +e
"$PY" "$PIN"/scripts/train_lora.py \
  --packed-dir "$PACKED_DIR" \
  --out-dir "$OUT_DIR" \
  --heldout-scanners GT450 S210 \
  --heldout-stains HRH KR MY \
  --backbone MahmoodLab/conch \
  --pooling cls \
  --proj-out-dim 512 \
  --lora-rank 32 --lora-alpha 64 \
  --temperature 0.07 --lr 1e-4 --weight-decay 0.05 \
  --mask-same-core --core-labels-path "$CORE_LABELS" \
  --same-core-logit-bias="$BCLS" \
  --anchor-cos-weight "$ANCHOR" --anchor-space "${CONCH_ANCHOR_SPACE:-export}" \
  --max-steps "$MAX_STEPS" --warmup-steps "${SPECTRA_WARMUP:-200}" \
  --grid --grid-conditions 2 --grid-tiles "$T" --grid-forward-chunk "${SPECTRA_CHUNK:-600}" \
  --grad-checkpointing --activation-offload \
  --workers "${SPECTRA_WORKERS:-10}" --amp bfloat16 --log-every 10 \
  --ckpt-every "$CKPT_EVERY" --eval-every "$CKPT_EVERY" ${SPECTRA_CKPT_SCHEDULE:+--ckpt-schedule "$SPECTRA_CKPT_SCHEDULE"} \
  --seed "$SEED" "${RESUME[@]}"
RC=$?
set -e
echo "rc=$RC" > "$OUT_DIR/TRAIN_DONE"
echo "=== done $(date) rc=$RC ==="
exit $RC
