#!/bin/bash
# THUNDER base fan-out for the two GATED backbones (H-Optimus-0, UNI2-h).
#
# Why this exists instead of scripts/submit_thunder.sh: that script's
# backbone_spec() only knows phikon-v2|midnight|virchow2 and exits 2 on
# anything else. This emits the same command lines it would.
#
# POOLING IS PASSED EXPLICITLY AS `cls`, never `auto`. thunder_model.py:89
# carries a STALE duplicate THUNDER_CLS_BACKBONES = frozenset({"owkin/phikon-v2"})
# and _default_pooling() reads that local copy, not the corrected roster in
# thunder_protocol.py:44 -- so `auto` hard-RuntimeErrors on both gated keys.
#
# Roster is PAPER_CLS_WAIV16 (collect_final5.py): the 12 THUNDER classification
# datasets Waiv published plus the 4 SPIDER sets that postdate the paper. The
# 4-dataset subset is indefensible (its error exceeds Waiv's whole gain).
#
# Usage:  scripts/thunder_gated_fanout.sh hoptimus|uni2 [--seg-only|--cls-only]

set -euo pipefail
REPO=/admin/home/ryan.kim/waiv
ARM="${1:?usage: $0 hoptimus|uni2 [--cls-only|--seg-only]}"
MODE="${2:-all}"

case "$ARM" in
  hoptimus) BACKBONE="bioptimus/H-optimus-0" ;;
  uni2)     BACKBONE="MahmoodLab/UNI2-h" ;;
  *) echo "unknown arm: $ARM (expected hoptimus|uni2)" >&2; exit 2 ;;
esac

RUN_NAME="${ARM}_base_cls"          # 17 chars; THUNDER run_tags hard-caps at 64
CLS_TASKS="knn linear_probing simple_shot"
SEG_TASKS="segmentation"

CLASSIFICATION="bach bracs break_his ccrcc crc esca mhist patch_camelyon tcga_crc_msi tcga_tils tcga_uniform wilds"
SPIDER="spider_breast spider_colorectal spider_skin spider_thorax"
SEGMENTATION="ocelot pannuke"

submit () {  # $1=dataset $2=tasks $3=run_name $4=jobtag
  local ds=$1 tasks=$2 rn=$3 tag=$4
  if [ ${#rn} -gt 64 ]; then echo "run_name >64 chars, THUNDER will reject: $rn" >&2; exit 3; fi
  sbatch --account=max --qos=high -J "$tag" \
    --export="ALL,WAIV_BACKBONE=$BACKBONE" \
    "$REPO/scripts/run_thunder.sbatch" "$ds" "$tasks" cls "$rn"
}

if [ "$MODE" != "--seg-only" ]; then
  for ds in $CLASSIFICATION $SPIDER; do
    [ "$ARM/$ds" = "hoptimus/bach" ] && { echo "skip hoptimus/bach (canary 394088)"; continue; }
    submit "$ds" "$CLS_TASKS" "$RUN_NAME" "thu-${ARM}-${ds}"
  done
fi

if [ "$MODE" != "--cls-only" ]; then
  for ds in $SEGMENTATION; do
    submit "$ds" "$SEG_TASKS" "${ARM}_base_cls_seg" "thuseg-${ARM}-${ds}"
  done
fi
