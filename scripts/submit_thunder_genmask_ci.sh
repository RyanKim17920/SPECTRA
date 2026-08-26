#!/bin/bash
# THUNDER roster for the 2026-08-25 genMASK (c3s) checkpoint-interval arms, 6 checkpoints
# x 12 classification datasets x {knn, linear_probing, simple_shot}.
#
# Dataset list, tasks, pooling=auto and the per-dataset time limits are transcribed
# VERBATIM from scripts/submit_thunder_seed1_virchow2.sh so these rows are comparable to
# the existing seed-floor and mask rosters. Adapter path is the CHECKPOINT dir (the 6th
# positional of run_thunder.sbatch), not the adapter/ subdir.
#
# break_his/phikon-s0 is intentionally skipped: it is canary job 392918, already running.
set -euo pipefail

SB=/admin/home/ryan.kim/waiv/scripts/run_thunder.sbatch
RUNS=/admin/home/ryan.kim/waiv/runs
TASKS="knn linear_probing simple_shot"
OUT=$RUNS/.thunder_genmask_ci_jobs
P=genMASK-c3s-lr1e-4-r32-kl0-t0.07-wd0.05-ms500-pd512

ROSTER="
bach=1-00:00:00
bracs=1-00:00:00
break_his=1-00:00:00
ccrcc=1-00:00:00
crc=2-00:00:00
esca=5-00:00:00
mhist=1-00:00:00
patch_camelyon=3-00:00:00
tcga_crc_msi=2-00:00:00
tcga_tils=3-00:00:00
tcga_uniform=4-00:00:00
wilds=4-00:00:00
"

# tag|backbone|run_dir|step
ARMS="
phikon-s0|owkin/phikon-v2|$P-phikon-s0-t900-392669|0000250
phikon-s1|owkin/phikon-v2|$P-phikon-s1-t900-392672|0000250
midnight-s0|kaiko-ai/midnight|$P-midnight-s0-t900-392670|0000125
midnight-s1|kaiko-ai/midnight|$P-midnight-s1-t900-392673|0000125
virchow2-s0|paige-ai/Virchow2|$P-virchow2-s0-t900-392671|0000125
virchow2-s1|paige-ai/Virchow2|$P-virchow2-s1-t900-392674|0000125
"

: > "$OUT"
for arm in $ARMS; do
  TAG="$(echo "$arm" | cut -d'|' -f1)"
  BB="$(echo "$arm" | cut -d'|' -f2)"
  RD="$(echo "$arm" | cut -d'|' -f3)"
  ST="$(echo "$arm" | cut -d'|' -f4)"
  JOBID="${RD##*-}"
  RUN="f5_ci-${TAG%-*}-${TAG##*-}-${JOBID}_s${ST}"
  ADAPTER="$RUNS/$RD/step_$ST"
  [ -f "$ADAPTER/adapter/adapter_model.safetensors" ] || { echo "MISSING $ADAPTER" >&2; exit 1; }
  for entry in $ROSTER; do
    DS="${entry%%=*}"; TL="${entry#*=}"
    if [ "$TAG" = "phikon-s0" ] && [ "$DS" = "break_his" ]; then
      echo "skip  break_his/$TAG (canary 392918)"; continue
    fi
    jid=$(sbatch --parsable \
        --job-name="ci-${TAG}-${DS}" \
        --time="$TL" \
        --export=ALL,WAIV_BACKBONE="$BB" \
        "$SB" "$DS" "$TASKS" auto "$RUN" "" "$ADAPTER")
    echo "$jid  ci-${TAG}-${DS}  $RUN  $TL"
    echo -n "$jid " >> "$OUT"
  done
done
echo; echo "job ids -> $OUT"
