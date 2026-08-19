#!/bin/bash
# Submit HEST + THUNDER for one final5 checkpoint.
#
#   scripts/submit_final5_evals.sh <run_name> <step> [--hest-only|--thunder-only]
#
# RUN NAME UNIQUENESS IS LOAD-BEARING. WAIV_RUN_NAME becomes PretrainedModel.name, the
# <model> path component of outputs/res/<dataset>/<model>/<task>/<adaptation>/. Reusing a
# name silently overwrites another checkpoint's numbers AND reuses its cached embeddings --
# which is how a previous round concluded a Virchow2 eval was "already done" when it was
# actually reading features dated 2026-08-10 from a different checkpoint. The name below
# embeds run+step so two final5 checkpoints can never collide.
set -euo pipefail
REPO="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
cd "$REPO"

RUN="${1:?usage: submit_final5_evals.sh <run_name> <step> [--hest-only|--thunder-only]}"
STEP="${2:?step, e.g. 0000500}"
MODE="${3:-all}"

case "$RUN" in
  *phikon*)   BACKBONE="owkin/phikon-v2" ;;
  *midnight*) BACKBONE="kaiko-ai/midnight" ;;
  *virchow2*) BACKBONE="paige-ai/Virchow2" ;;
  *) echo "FATAL: cannot infer backbone from '$RUN'" >&2; exit 2 ;;
esac

ADAPTER="$REPO/runs/$RUN/step_$STEP"
[ -d "$ADAPTER/adapter" ] || { echo "FATAL: no adapter/ under $ADAPTER" >&2; exit 3; }
TAG="f5_${RUN}_s${STEP}"

# Classification datasets get knn + linear_probing + simple_shot; the two segmentation
# datasets get segmentation. This is the full 4-task protocol, matching what Waiv's
# leaderboard mean is computed over -- NOT the fast-5 subset, which is fine for arm
# comparison but cannot feed the published-comparison chart.
CLS_DS="bach bracs break_his ccrcc crc esca mhist patch_camelyon tcga_crc_msi tcga_tils tcga_uniform wilds"
SEG_DS="ocelot pannuke"

if [ "$MODE" != "--thunder-only" ]; then
  jid=$(WAIV_RUN="$RUN" WAIV_STEP="$STEP" sbatch --parsable --account=max --qos=high \
        --job-name="hest-${TAG}" \
        --export=ALL,WAIV_RUN="$RUN",WAIV_STEP="$STEP" \
        scripts/hest_final5.sbatch)
  echo "HEST    $TAG -> $jid"
fi

if [ "$MODE" != "--hest-only" ]; then
  # pooling=auto so thunder_model._default_pooling resolves the per-backbone protocol
  # from WAIV_BACKBONE (CLS+mean for Virchow2/Midnight, CLS elsewhere) rather than us
  # hardcoding one protocol across three backbones.
  for ds in $CLS_DS; do
    jid=$(WAIV_BACKBONE="$BACKBONE" sbatch --parsable --account=max --qos=high \
          --job-name="thd-${TAG}-${ds}" \
          --export=ALL,WAIV_BACKBONE="$BACKBONE" \
          scripts/run_thunder.sbatch "$ds" "knn linear_probing simple_shot" auto "$TAG" "" "$ADAPTER")
    echo "THUNDER $TAG $ds -> $jid"
  done
  for ds in $SEG_DS; do
    # guidelines.md mandates non-default epochs for the two big segpath sets; ocelot and
    # pannuke are not those, so no WAIV_EPOCHS override here.
    jid=$(WAIV_BACKBONE="$BACKBONE" sbatch --parsable --account=max --qos=high \
          --job-name="thd-${TAG}-${ds}" \
          --export=ALL,WAIV_BACKBONE="$BACKBONE" \
          scripts/run_thunder.sbatch "$ds" "segmentation" auto "$TAG" "" "$ADAPTER")
    echo "THUNDER $TAG $ds -> $jid"
  done
fi
