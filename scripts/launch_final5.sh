#!/bin/bash
# Launch the full 3-backbone x 5-seed final-config matrix.
#
#   scripts/launch_final5.sh <T> [seeds...]      # default seeds: 0 1 2 3 4
#
# T is REQUIRED and has no default on purpose. A single shared T across all three
# backbones is the entire point of this study: the previous three-backbone table
# (docs/FINAL_RESULTS.md sec.2) ran phikon at T=1800, Virchow2 at T=600 and midnight at
# T=450, which makes "the method helps phikon but not the others" inseparable from
# "phikon got 4x the negatives". Defaulting T here would invite that confound straight
# back in, so the caller must state it and it is recorded in every run name.
set -euo pipefail
REPO="${WAIV_REPO:-/admin/home/ryan.kim/waiv}"
cd "$REPO"

T="${1:?usage: launch_final5.sh <T> [seeds...]   e.g. launch_final5.sh 900 0 1 2 3 4}"
shift || true
SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2 3 4)

echo "=== launching final5: T=$T seeds=${SEEDS[*]} ==="
for ARM in phikon midnight virchow2; do
  for S in "${SEEDS[@]}"; do
    jid=$(sbatch --parsable --account=max --qos=high \
          --job-name="f5-${ARM}-s${S}" \
          --export=ALL,WAIV_ARM=$ARM,WAIV_SEED=$S,WAIV_T=$T \
          scripts/final5.sbatch)
    echo "  $ARM seed=$S T=$T -> $jid"
  done
done
echo "=== 15 jobs submitted; runs land in runs/final5-<arm>-s<seed>-t${T}-<jobid> ==="
