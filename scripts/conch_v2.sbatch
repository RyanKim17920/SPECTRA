#!/bin/bash
#SBATCH --job-name=main_1
#SBATCH --account=max
#SBATCH --qos=normal
#SBATCH --partition=n
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=10:00:00
#SBATCH --output=/data/ryan.kim/ashby/slurm/%x_%j.out
#
# SPECTRA-CONCH v2: paper-faithful arms (no anchor loss). Canonical gentle.sbatch recipe:
# grid C=2 T=900, mask_same_core, lr 1e-4, warmup 200, wd 0.05, r32/a64, tau 0.07, 500 steps,
# ckpt/eval every 50, seed $SPECTRA_SEED.
#
#   CONCH_V2_ARM=export-ninf  : conch-anchor pin, single head on the 512-d export,
#                               --same-core-logit-bias=-inf (RI-readout canonical), anchor 0.
#   CONCH_V2_ARM=split-paper  : conch-split pin, backbone MahmoodLab/conch-trunk (trunk tokens),
#                               --pooling clsmean --split-heads cls+mean 0.5/0.5 --pool-head gem,
#                               bias cls=+3 / mean=-inf  == gentle.sbatch HEAD_ARGS.
# Only deviation from gentle.sbatch on split-paper: --grid-forward-chunk 600 (canonical 0). The
# conch-split pin pools GeM per chunk, which is exact (per-image op; projectors still run once
# over the full C*T batch) -- verified max|dz|=0.0 vs unchunked. 1 GPU (no in-job RI follower;
# PathoROB RI runs per checkpoint via conch_pathorob.py).
#
#   CONCH_V2_ARM=export-ninf OUT_NAME=A_export-ninf_s0 sbatch scripts/conch_v2.sbatch
set -euo pipefail
REPO="${SPECTRA_REPO:-/data/ryan.kim/ashby/third_party/SPECTRA-conch}"
PY="${CONCH_PY:-/data/ryan.kim/ashby/envs/conch/bin/python}"
PACKED_DIR="${SPECTRA_PACKED_DIR:-/data/plism/repacked}"
CORE_LABELS="${SPECTRA_CORE_LABELS:-/data/SPECTRA-runs/runs/.plism_core_labels.npy}"
ARM="${CONCH_V2_ARM:?export-ninf|split-paper}"
SEED="${SPECTRA_SEED:-0}"
OUT_DIR="${CONCH_OUT_ROOT:-/data/ryan.kim/ashby/runs/spectra_conch_v2}/${OUT_NAME:?set OUT_NAME}"
case "$ARM" in
  export-ninf)
    PIN="$REPO/snapshots/conch-anchor"
    HEAD=(--backbone MahmoodLab/conch --pooling cls --same-core-logit-bias=-inf --anchor-cos-weight 0) ;;
  split-paper)
    PIN="$REPO/snapshots/conch-split"
    HEAD=(--backbone MahmoodLab/conch-trunk --pooling clsmean
          --split-heads --cls-weight 0.5 --mean-weight 0.5 --pool-head gem --retention-kl-weight 0
          --same-core-logit-bias-cls=3.0 --same-core-logit-bias-mean=-inf --anchor-cos-weight 0) ;;
  *) echo "bad arm $ARM" >&2; exit 2 ;;
esac
source ~/.cache_env.sh 2>/dev/null || true
export HF_HOME=/data/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_${SLURM_JOB_ID:-local}
export PYTHONPATH="$PIN/src"
export WAIV_RUNS="${WAIV_RUNS:-/data/SPECTRA-runs/runs}"
mkdir -p "$OUT_DIR"
echo "=== $(date) | $(hostname) | job ${SLURM_JOB_ID:-none} | arm $ARM seed $SEED | pin $PIN | out $OUT_DIR"
echo "head ${HEAD[*]}"
"$PY" - "$PIN" <<'PYEOF'
import importlib, sys
from pathlib import Path
pin = Path(sys.argv[1]); pkg = importlib.import_module("waivphaet")
assert Path(pkg.__file__).is_relative_to(pin), f"PIN LEAK: {pkg.__file__}"
print(f"[pin] waivphaet={pkg.__file__}", flush=True)
PYEOF
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -l 10 > "$OUT_DIR/gpu.csv" &
SAMPLER=$!; trap 'kill $SAMPLER 2>/dev/null || true' EXIT
RESUME=()
LAST=$(ls -d "$OUT_DIR"/step_* 2>/dev/null | while read d; do [ -f "$d/metrics.json" ] && echo "$d"; done | sort | tail -1 || true)
[ -n "$LAST" ] && RESUME=(--resume-from "$LAST") && echo "resume from $LAST"
set +e
"$PY" "$PIN"/scripts/train_lora.py \
  --packed-dir "$PACKED_DIR" --out-dir "$OUT_DIR" \
  --heldout-scanners GT450 S210 --heldout-stains HRH KR MY \
  "${HEAD[@]}" \
  --proj-out-dim 512 --lora-rank 32 --lora-alpha 64 \
  --temperature 0.07 --lr 1e-4 --weight-decay 0.05 \
  --mask-same-core --core-labels-path "$CORE_LABELS" \
  --max-steps 500 --warmup-steps 200 \
  --grid --grid-conditions 2 --grid-tiles 900 --grid-forward-chunk "${SPECTRA_CHUNK:-600}" \
  --grad-checkpointing --activation-offload \
  --workers 10 --amp bfloat16 --log-every 10 \
  --ckpt-every 50 --eval-every 50 --seed "$SEED" "${RESUME[@]}"
RC=$?
set -e
echo "rc=$RC" > "$OUT_DIR/TRAIN_DONE"; echo "=== done $(date) rc=$RC"; exit $RC
