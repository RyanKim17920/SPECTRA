#!/bin/bash
# Watch mask+lr3e-5+ms1500 runs (392188 phikon / 392189 midnight / 392190 virchow2).
#  - auto-submit HEST at step 250 AND step 500 as each checkpoint appears
#  - log each new RI point once
#  - log each HEST job's completion once
# All events are logged AT MOST ONCE, state seeded from the log so restarts are idempotent.
REPO=/admin/home/ryan.kim/waiv
cd "$REPO" || exit 1
LOG="$REPO/logs/mask1500_watch.log"
PFX="runs/genMASK-lr3e-5-r32-kl0-t0.07-wd0.05-ms1500-pd512"
ARMS="phikon midnight virchow2"
declare -A JOBS=( [phikon]=392188 [midnight]=392189 [virchow2]=392190 )
echo "$(date) watcher v2 start (steps 250+500, HEST completion)" >> "$LOG"
seen() { grep -q "^SEEN $1$" "$LOG" 2>/dev/null; }
mark() { echo "SEEN $1" >> "$LOG"; }
for i in $(seq 1 2880); do
  for ARM in $ARMS; do
    JID=${JOBS[$ARM]}
    D=$(ls -d ${PFX}-${ARM}-s0-t900-${JID}* 2>/dev/null | head -1)
    [ -z "$D" ] && continue
    RUN=$(basename "$D")
    for STEP in 0000250 0000500; do
      if [ -d "$D/step_$STEP/adapter" ] && ! seen "hest-$ARM-$STEP"; then
        HJ=$(WAIV_RUN="$RUN" WAIV_STEP=$STEP sbatch --parsable scripts/hest_final5.sbatch 2>>"$LOG")
        mark "hest-$ARM-$STEP"
        echo "$(date) HEST_SUBMITTED $ARM step=$STEP job=$HJ" >> "$LOG"
      fi
    done
    if [ -f "$D/ri_curve.json" ]; then
      for S in $("$REPO/.venv/bin/python" -c "
import json,sys
try:
  d=json.load(open('$D/ri_curve.json'))
  print(' '.join('%d:%.4f'%(p['step'],p['avg_robustness_index']) for p in d['points']))
except Exception: pass" 2>/dev/null); do
        if ! seen "ri-$ARM-$S"; then mark "ri-$ARM-$S"; echo "$(date) RI_POINT $ARM $S" >> "$LOG"; fi
      done
    fi
  done
  for F in logs/slurm-waiv-hest-final5-*.out; do
    [ -f "$F" ] || continue
    J=$(basename "$F" .out); J=${J##*-}
    if grep -q "=== done" "$F" 2>/dev/null && ! seen "hestdone-$J"; then
      mark "hestdone-$J"
      echo "$(date) HEST_DONE job=$J $(grep -m1 'exp_code=' "$F" | tr -d '\n')" >> "$LOG"
    fi
  done
  sleep 60
done
echo "$(date) watcher v2 exit" >> "$LOG"
