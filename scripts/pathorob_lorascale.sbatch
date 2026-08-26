#!/bin/bash
#SBATCH --job-name=waiv-robls
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --error=logs/slurm-%x-%j.err
set -euo pipefail
# PathoROB RI for a WiSE-FT-interpolated LoRA checkpoint: scaling the LoRA delta by s at
# eval IS theta = (1-s)*theta_base + s*theta_ft, exactly. HEST is flat across s for the
# Virchow2 mask arm; this asks whether RI is flat too (Pareto point) or collapses toward
# the base backbone (s just "mostly base model").
#
# NOTE: Virchow2 adapters have base_model_name_or_path=null, so --backbone is MANDATORY
# or phikon-v2 is silently built and the adapter keys fail to load.
REPO="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
cd "$REPO"
export HF_HOME="${HF_HOME:-/data/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

ADAPTER="${WAIV_ADAPTER:?set WAIV_ADAPTER (checkpoint dir)}"
S="${WAIV_S:?set WAIV_S (lora scale)}"
BACKBONE="${WAIV_BACKBONE:?set WAIV_BACKBONE}"
POOL="${WAIV_POOL:-clsmean}"
BASETAG="${WAIV_TAG:?set WAIV_TAG (features model-name stem)}"
DATASETS="${WAIV_DATASETS:-camelyon tcga tolkach_esca}"

# Derived from S, never hand-written: the features dir keys on --model-name alone, so a
# stale token would score another scale's embeddings and read as "interpolation flat".
TAG=$("$REPO/.venv/bin/python" -c "import sys; sys.path.insert(0,'src'); from waivphaet.models.encoder import lora_scale_tag; print(lora_scale_tag($S))")
MODEL="${BASETAG}_${TAG}"

echo "=== $(date) | node $(hostname) | job ${SLURM_JOB_ID:-none} ==="
echo "=== adapter=$ADAPTER s=$S pool=$POOL model=$MODEL ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

for DS in $DATASETS; do
  echo "--- extract $DS ---"
  srun --unbuffered "$REPO/.venv/bin/python" scripts/extract_pathorob_features.py \
    --dataset "$DS" --model-name "$MODEL" \
    --backbone "$BACKBONE" --pooling "$POOL" \
    --adapter "$ADAPTER" --lora-rank 32 --lora-alpha 64 --proj-out-dim 512 \
    --lora-scale "$S" \
    --batch-size "${BATCH_SIZE:-256}" --num-workers 8
done

echo "--- score RI ---"
"$REPO/.venv/bin/python" - "$MODEL" "$S" $DATASETS <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, "src")
from waivphaet.eval.pathorob_adapter import PathoRobPaths, read_results, run_robustness_index

model, s, datasets = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
paths = PathoRobPaths(root=Path("third_party/PathoROB"))
run_robustness_index(model, datasets, paths=paths)
ri = {d: float(read_results(model, d, paths=paths)["robustness_index"]) for d in datasets}
ri["avg"] = sum(ri.values()) / len(datasets)
print("[RI] " + json.dumps({"model": model, "lora_scale": s, "ri": ri}, indent=2), flush=True)
out = f"docs/lorascale_ri_{model}.json"
with open(out, "w") as f:
    json.dump({"model": model, "lora_scale": s, "ri": ri}, f, indent=2)
print(f"[RI] wrote {out}", flush=True)
PY
echo "=== done rc=$? model=$MODEL ==="
