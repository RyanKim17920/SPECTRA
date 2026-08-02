#!/usr/bin/env bash
# THUNDER sweep for Midnight-12k: base + fine-tuned (step 500, its best PathoROB point).
#
# Midnight had PathoROB only; this closes that gap so the second backbone gets the same
# retention treatment phikon-v2 got.
#
# Pooling: pass "auto" so thunder_model._default_pooling resolves it from WAIV_BACKBONE.
# arXiv:2607.22861 §3 line 106 uses CLS+mean for Midnight-12k in THUNDER (unlike
# phikon-v2, which is CLS-only there) -- getting this wrong invalidates the comparison.
#
# Submitted HELD; scripts/thunder_pilot.py drains them at bounded concurrency.
#
#   bash scripts/submit_midnight_thunder.sh          # dry run
#   bash scripts/submit_midnight_thunder.sh --go
set -uo pipefail
cd "$(dirname "$0")/.."

GO=0
[ "${1:-}" = "--go" ] && GO=1

export WAIV_BACKBONE=kaiko-ai/midnight
CLS_TASKS="knn linear_probing simple_shot"
SEG_TASKS="segmentation"
CLASSIFICATION="bach bracs break_his ccrcc crc esca mhist patch_camelyon tcga_crc_msi tcga_tils tcga_uniform wilds"
SEGMENTATION="ocelot pannuke"   # segpath_lymphocytes infeasible, segpath_epithelial absent
FT_ADAPTER="runs/waiv-midnight-369159/step_0000500"

ROOT="${THUNDER_BASE_DATA_FOLDER:-/data/ryan.kim/thunder}"
ACTIVE=$(squeue -u ryan.kim -h -o "%j" | sort -u)

submit() {  # <jobname> <dataset> <tasks> <run_name> [adapter]
  local name="$1" ds="$2" tasks="$3" run="$4" adapter="${5:-}"
  if grep -qx "$name" <<<"$ACTIVE"; then echo "SKIP $name -- in queue"; return; fi
  local complete=1
  for t in $tasks; do
    [ -f "$ROOT/outputs/res/$ds/$run/$t/frozen/outputs.json" ] || complete=0
  done
  if [ "$complete" -eq 1 ]; then echo "SKIP $name -- results complete"; return; fi
  # "auto" pooling -> resolved from WAIV_BACKBONE (clsmean for midnight).
  local args=(--hold --job-name="$name" --export=ALL,WAIV_BACKBONE=kaiko-ai/midnight
              scripts/run_thunder.sbatch "$ds" "$tasks" auto "$run" "")
  [ -n "$adapter" ] && args+=("$adapter")
  if [ $GO -eq 1 ]; then sbatch "${args[@]}" | tail -1; else echo "DRYRUN sbatch ${args[*]}"; fi
}

for ds in $CLASSIFICATION; do
  submit "mthd-$ds"     "$ds" "$CLS_TASKS" mbase_clsmean
  submit "mthdft-$ds"   "$ds" "$CLS_TASKS" mft500_clsmean "$FT_ADAPTER"
done
for ds in $SEGMENTATION; do
  submit "mthd-$ds"     "$ds" "$SEG_TASKS" mbase_clsmean
  submit "mthdft-$ds"   "$ds" "$SEG_TASKS" mft500_clsmean "$FT_ADAPTER"
done

echo
echo "Submitted HELD. Drain with:"
echo "  nohup .venv/bin/python scripts/thunder_pilot.py --cap 4 --interval 120 \\"
echo "      --max-fast-failures 3 >> /data/ryan.kim/thunder_pilot.log 2>&1 &"
