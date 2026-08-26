#!/bin/bash
# THUNDER SEED-FLOOR measurement, Virchow2, ret0.01 recipe, seed 1.
#
# Pairs with the already-submitted seed-0 roster (jobs 392027-392038, run name
# f5_ret0.01-virchow2-s0-t900-391059_s0000250). The ONLY difference between the two
# training runs is `seed` (0 vs 1) and `out_dir` -- verified by diffing the two
# config.json files; every other key, including encoder/backbone/lora/pooling, is
# byte-identical. So the seed-1 minus seed-0 difference on each THUNDER cell IS the
# seed noise, which is the quantity scripts/scoreboard.py currently hardcodes as
# THUNDER_TASK_2SE = 0.0025 with no replicate behind it.
#
# Submit lines below are transcribed from `sacct --format=SubmitLine -j 392027-392038`
# with two substitutions ONLY: the run name and the adapter path. Time limits are copied
# verbatim from seed-0 so the two arms are also matched in scheduling treatment; SLURM
# lets you DECREASE a limit later (`scontrol update JobId=<id> TimeLimit=...`) but never
# increase one, so they are deliberately generous.
#
# All jobs are submitted HELD. Release with:
#   scontrol release $(cat /admin/home/ryan.kim/waiv/runs/.thunder_seed1_virchow2_jobs)
set -euo pipefail

RUN=f5_ret0.01-virchow2-s1-t900-392045_s0000250
ADAPTER=/admin/home/ryan.kim/waiv/runs/ret0.01-virchow2-s1-t900-392045/step_0000250
TASKS="knn linear_probing simple_shot"
SB=/admin/home/ryan.kim/waiv/scripts/run_thunder.sbatch
OUT=/admin/home/ryan.kim/waiv/runs/.thunder_seed1_virchow2_jobs

# dataset=timelimit -- identical to the seed-0 roster.
# SEPARATOR IS '=', NOT ':'. A ':' separator here is a real bug: SLURM time limits
# themselves contain colons, so `${entry##*:}` strips to the last one and yields "00",
# which sbatch silently ignores, leaving every job at the partition default (UNLIMITED).
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

: > "$OUT"
for entry in $ROSTER; do
  DS="${entry%%=*}"
  TL="${entry#*=}"
  jid=$(sbatch --hold --parsable \
      --job-name="tsdv-${DS}" \
      --time="$TL" \
      --export=ALL,WAIV_BACKBONE=paige-ai/Virchow2 \
      "$SB" "$DS" "$TASKS" auto "$RUN" "" "$ADAPTER")
  echo "$jid  tsdv-${DS}  $TL"
  echo -n "$jid " >> "$OUT"
done
echo
echo "held job ids written to $OUT:"
cat "$OUT"; echo
