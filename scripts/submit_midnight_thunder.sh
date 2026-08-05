#!/usr/bin/env bash
# THUNDER sweep for Midnight-12k: base + fine-tuned (step 500, its best PathoROB point).
#
# Midnight had PathoROB only; this closes that gap so the second backbone gets the same
# retention treatment phikon-v2 got.
#
# Pooling: CLASSIFICATION passes "auto" so thunder_model._default_pooling resolves it from
# WAIV_BACKBONE. arXiv:2607.22861 §3 line 106 uses CLS+mean for Midnight-12k in THUNDER
# (unlike phikon-v2, which is CLS-only there) -- getting this wrong invalidates the
# comparison.
#
# SEGMENTATION passes "cls" EXPLICITLY, and must not be switched back to "auto"/clsmean.
# clsmean advertises emb_dim = 2 * hidden (3072 for Midnight's ViT-g), and THUNDER sizes
# its segmentation decoder from emb_dim -- but get_segmentation_embeddings returns raw
# per-patch tokens, which are hidden-d (1536). The two disagree and the job dies at
# thunder/models/task_specific_models.py:121 (`x = self.proj_dec(x)`):
#     RuntimeError: mat1 and mat2 shapes cannot be multiplied (16384x1536 and 3072x768)
# phikon-v2 never hit this because its CLS dim equals its patch dim, so cls/clsmean is a
# no-op there. Four Midnight seg jobs died this way on 2026-08-02 before this fix.
# cls is also the honest setting for segmentation: pooling is not applied to patch tokens
# at all, so nothing about the published protocol is lost by using it here.
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
# segpath_lymphocytes and segpath_epithelial are deliberately NOT here: they need the
# guidelines.md:4 epochs override (21 / 9) and a much longer wall, so they have their own
# submitter -- scripts/submit_segpath_thunder.sh, which covers both backbones at once.
SEGMENTATION="ocelot pannuke"
FT_ADAPTER="runs/waiv-midnight-369159/step_0000500"

ROOT="${THUNDER_BASE_DATA_FOLDER:-/data/ryan.kim/thunder}"
ACTIVE=$(squeue -u ryan.kim -h -o "%j" | sort -u)

submit() {  # <jobname> <dataset> <tasks> <pooling> <run_name> [adapter]
  local name="$1" ds="$2" tasks="$3" pooling="$4" run="$5" adapter="${6:-}"
  if grep -qx "$name" <<<"$ACTIVE"; then echo "SKIP $name -- in queue"; return; fi
  local complete=1
  for t in $tasks; do
    [ -f "$ROOT/outputs/res/$ds/$run/$t/frozen/outputs.json" ] || complete=0
  done
  if [ "$complete" -eq 1 ]; then echo "SKIP $name -- results complete"; return; fi
  # "auto" -> resolved from WAIV_BACKBONE (clsmean for midnight); "cls" wins explicitly.
  local args=(--hold --job-name="$name" --export=ALL,WAIV_BACKBONE=kaiko-ai/midnight
              scripts/run_thunder.sbatch "$ds" "$tasks" "$pooling" "$run" "")
  [ -n "$adapter" ] && args+=("$adapter")
  if [ $GO -eq 1 ]; then sbatch "${args[@]}" | tail -1; else echo "DRYRUN sbatch ${args[*]}"; fi
}

for ds in $CLASSIFICATION; do
  submit "mthd-$ds"     "$ds" "$CLS_TASKS" auto mbase_clsmean
  submit "mthdft-$ds"   "$ds" "$CLS_TASKS" auto mft500_clsmean "$FT_ADAPTER"
done
# cls, not auto -- see the emb_dim/patch-dim mismatch note in the header. The run_name
# carries the pooling too, so these land in outputs/res/<ds>/m{base,ft500}_cls/ and cannot
# collide with the classification rows.
for ds in $SEGMENTATION; do
  submit "mthd-$ds"     "$ds" "$SEG_TASKS" cls mbase_cls
  submit "mthdft-$ds"   "$ds" "$SEG_TASKS" cls mft500_cls "$FT_ADAPTER"
done

echo
echo "Submitted HELD. Drain with:"
echo "  nohup .venv/bin/python scripts/thunder_pilot.py --cap 4 --interval 120 \\"
echo "      --max-fast-failures 3 >> /data/ryan.kim/thunder_pilot.log 2>&1 &"
