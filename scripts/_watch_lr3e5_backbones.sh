#!/bin/bash
# Waits for step_0000250 adapters on the lr3e-5 ms1500 backbone-coverage runs
# (392183 virchow2 / 392184 phikon), then auto-submits HEST with DEFAULT exp_code.
# Filesystem signals only. Glob-tolerant: a requeue appends .rN to the run dir.
REPO=/admin/home/ryan.kim/waiv
cd "$REPO"
PATS="gen-lr3e-5-r32-kl0-t0.07-wd0.05-ms1500-pd512-virchow2-s0-t900-392183* gen-lr3e-5-r32-kl0-t0.07-wd0.05-ms1500-pd512-phikon-s0-t900-392184*"
declare -A hdone
for i in $(seq 1 5760); do
  pending=0
  for P in $PATS; do
    hit=0
    for D in runs/$P; do
      [ -d "$D" ] || continue
      R=$(basename "$D")
      if [ -n "${hdone[$R]:-}" ]; then hit=1; continue; fi
      if [ -d "$D/step_0000250/adapter" ]; then
        sleep 20
        out=$(WAIV_RUN="$R" WAIV_STEP=0000250 sbatch "$REPO/scripts/hest_final5.sbatch" 2>&1)
        echo "$(date -Is) HEST_SUBMITTED run=$R -> $out"
        hdone[$R]=1; hit=1
      elif [ -f "$D/TRAIN_DONE" ] && ! grep -q 'rc=0' "$D/TRAIN_DONE"; then
        echo "$(date -Is) WATCH_TRAIN_FAILED run=$R $(cat "$D/TRAIN_DONE")"
        hdone[$R]=1; hit=1
      fi
    done
    [ "$hit" = "1" ] || pending=1
  done
  if [ "$pending" = "0" ]; then echo "$(date -Is) WATCH_ALL_RESOLVED"; break; fi
  sleep 30
done
echo "$(date -Is) WATCH_EXIT"
