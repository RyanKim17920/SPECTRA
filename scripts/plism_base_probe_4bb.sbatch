#!/bin/bash
#SBATCH --job-name=waiv-plism-base-4bb
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
#
# BASE (untrained) PLISM held-out retrieval probe for the four backbones that have NO
# probe_before.json anywhere under $SPECTRA_RUNS -- hoptimus0, uni2h,
# virchow1, openmidnightsq were bound locally after the finalgem/gridcmp era that
# produced probe_before.json for phikon2/midnight/virchow2, so their c50 training runs
# never generated one.
#
# probe_before.json is model-agnostic given (backbone, conditions, n_tiles, seed): no
# adapter is loaded (--adapter omitted), so this is exactly the same computation
# embed_probe.py's own preflight ran for the other three backbones. Verified byte-for-
# byte reproducible: genMASK-c50 runs' own probe_step_*.json (n_tiles=256, seed=1234,
# heldout n_pairs 73/171) match the finalgem probe_before.json numbers exactly, and all
# runs share the identical conditions_used.json (md5 777fb7ccdef5ff42051c1ed02dd7eb73,
# 91 verified slides). Reusing that same pinned conditions file here.
#
#   sbatch --account=max --qos=high scripts/plism_base_probe_4bb.sbatch

set -euo pipefail

REPO="${SPECTRA_REPO:-${WAIV_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}}"
. "$REPO/scripts/_env.sh"
cd "$REPO"

PY="$REPO/.venv-hest/bin/python"
CONDITIONS="$SPECTRA_RUNS/finalgem-phikon-384585/conditions_used.json"
OUT_DIR="$SPECTRA_RUNS/plism_base_probes"
mkdir -p "$OUT_DIR"

export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$SPECTRA_HF_HOME}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true
test -f "$CONDITIONS" || { echo "FATAL: conditions file missing: $CONDITIONS"; exit 2; }

declare -A BACKBONES=(
  [hoptimus0]="bioptimus/H-optimus-0"
  [uni2h]="MahmoodLab/UNI2-h"
  [virchow1]="paige-ai/Virchow"
  [openmidnightsq]="SophontAI/OpenMidnight"
)

for key in "${!BACKBONES[@]}"; do
  bb="${BACKBONES[$key]}"
  out="$OUT_DIR/${key}_probe_before.json"
  echo "--- $key ($bb) -> $out ---"
  "$PY" scripts/embed_probe.py \
    --conditions-file "$CONDITIONS" \
    --backbone "$bb" \
    --lora-rank 32 --lora-alpha 64 --proj-out-dim 512 \
    --n-tiles 256 --seed 1234 \
    --out "$out"
done

echo "=== done $(date) ==="
