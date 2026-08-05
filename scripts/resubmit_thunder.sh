#!/usr/bin/env bash
# Cancel the stale held THUNDER jobs and resubmit them against the CURRENT sbatch.
#
# Why cancel rather than `scontrol release`: SLURM copies the batch script into its spool
# at *submission* time. The 17 held jobs were submitted before the cold-cache fix
# (pre_computing_embeddings), so releasing them would re-run the old script and each would
# die in ~37s on FileNotFoundError .../embeddings/<ds>/<model>/train/labels.h5.
# Python changes DO reach queued jobs (read at runtime); .sbatch changes do NOT.
#
# Resubmits in HELD state so scripts/thunder_pilot.py governs concurrency.
#
#   bash scripts/resubmit_thunder.sh          # dry run, prints what it would do
#   bash scripts/resubmit_thunder.sh --go     # actually do it
set -uo pipefail
cd "$(dirname "$0")/.."

GO=0
[ "${1:-}" = "--go" ] && GO=1

CLS_TASKS="knn linear_probing simple_shot"
SEG_TASKS="segmentation"
CLASSIFICATION="bach bracs break_his ccrcc crc esca mhist patch_camelyon tcga_crc_msi tcga_tils tcga_uniform wilds"
# segpath_lymphocytes was removed here because 5048 s/epoch x frozen.yaml's generic 200
# epochs = ~281 h vs a 12 h wall (job 369061). That was the WRONG protocol, not an
# impossible dataset: guidelines.md:4 mandates --adaptation.epochs 21 for this set (and 9
# for segpath_epithelial), which is ~30 h. Both segpath sets are now submitted by
# scripts/submit_segpath_thunder.sh, which passes that override via WAIV_EPOCHS and a
# longer --time. Keep them out of THIS script -- it passes no override.
SEGMENTATION="ocelot pannuke"

FT_ADAPTER="runs/waiv-real-369043/step_0001000"

# Cancel every held THUNDER job; they carry the stale script.
held=$(squeue -u ryan.kim -h -t PD -o "%i|%j|%r" | awk -F'|' '$3=="JobHeldUser" && ($2 ~ /^thd-/ || $2 ~ /^thdft1k-/) {print $1}' | paste -sd, -)
if [ -n "$held" ]; then
  echo "cancel stale held jobs: $held"
  [ $GO -eq 1 ] && scancel "$held"
else
  echo "no held THUNDER jobs to cancel"
fi

ROOT="${THUNDER_BASE_DATA_FOLDER:-/data/ryan.kim/thunder}"
# Names still in the queue after the cancel above: the 3 base segmentation jobs are
# RUNNING right now and must not be duplicated.
# Exclude the held jobs cancelled above -- in --go they are already gone, and in a dry run
# they are still listed, which would otherwise make every dataset look "already queued"
# and the dry run would show nothing to do.
ACTIVE=$(squeue -u ryan.kim -h -o "%j|%r" \
         | awk -F'|' '$2!="JobHeldUser" {print $1}' | sort -u)

submit() {  # submit <jobname> <dataset> <tasks> <pooling> <run_name> [adapter]
  local name="$1" ds="$2" tasks="$3" pool="$4" run="$5" adapter="${6:-}"

  # Guard 1: already in the queue (running or pending). Resubmitting would run the same
  # dataset twice on separate GPUs for nothing.
  if grep -qx "$name" <<<"$ACTIVE"; then
    echo "SKIP $name -- already in queue"; return
  fi
  # Guard 2: results already on disk for every task we would run. mhist/base_cls is the
  # one dataset that completed before the cold-cache bug was found; the 3 base
  # segmentation runs will land here too.
  local complete=1
  for t in $tasks; do
    [ -f "$ROOT/outputs/res/$ds/$run/$t/frozen/outputs.json" ] || complete=0
  done
  if [ "$complete" -eq 1 ]; then
    echo "SKIP $name -- results already complete for [$tasks]"; return
  fi

  local args=(--job-name="$name" scripts/run_thunder.sbatch "$ds" "$tasks" "$pool" "$run" "")
  [ -n "$adapter" ] && args+=("$adapter")
  if [ $GO -eq 1 ]; then
    sbatch --hold "${args[@]}" | tail -1
  else
    echo "DRYRUN sbatch --hold ${args[*]}"
  fi
}

for ds in $CLASSIFICATION; do
  submit "thd-$ds"     "$ds" "$CLS_TASKS" cls base_cls
  submit "thdft1k-$ds" "$ds" "$CLS_TASKS" cls ft1000_cls "$FT_ADAPTER"
done
for ds in $SEGMENTATION; do
  submit "thd-$ds"     "$ds" "$SEG_TASKS" cls base_cls
  submit "thdft1k-$ds" "$ds" "$SEG_TASKS" cls ft1000_cls "$FT_ADAPTER"
done

echo
echo "All resubmitted HELD. Start the pilot to drain them:"
echo "  nohup .venv/bin/python scripts/thunder_pilot.py --cap 4 --interval 120 \\"
echo "      > /data/ryan.kim/thunder_pilot.log 2>&1 &"
