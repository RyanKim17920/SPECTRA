#!/usr/bin/env bash
# THUNDER sweep for the two remaining SEGMENTATION datasets: segpath_epithelial and
# segpath_lymphocytes, x {base, fine-tuned} x {phikon-v2, Midnight-12k} = 8 jobs.
#
# Closing these two rows takes our coverage to Waiv's full 16 datasets
# (12 classification + 4 segmentation: ocelot, pannuke, segpath_epithelial,
# segpath_lymphocytes).
#
# ---------------------------------------------------------------------------------------
# THE EPOCHS OVERRIDE IS THE WHOLE POINT OF THIS SCRIPT
# ---------------------------------------------------------------------------------------
# third_party/thunder/docs/guidelines.md:4 mandates, for these two datasets *only*:
#     segpath_epithelial   --adaptation.epochs 9
#     segpath_lymphocytes  --adaptation.epochs 21
# because of the size of the datasets. frozen.yaml's generic `epochs: 200` is what
# run_thunder.sbatch would otherwise inherit, and at the measured 5170 s/epoch that is
# 287 h for lymphocytes -- infeasible, AND a different protocol from the published
# leaderboard, so the number would not be comparable even if it finished. This is passed
# through run_thunder.sbatch's WAIV_EPOCHS env var (added 2026-08-03), which is a no-op
# for every other dataset.
#
# ---------------------------------------------------------------------------------------
# WALL TIME
# ---------------------------------------------------------------------------------------
# run_thunder.sbatch's #SBATCH --time=12:00:00 is right for classification and for
# ocelot/pannuke and is deliberately NOT edited -- an sbatch command-line --time overrides
# the in-script directive, so the walls below apply to these 8 jobs and to nothing else.
# All partitions are MaxTime=UNLIMITED, so over-requesting costs only backfill priority.
#
# Derivation from measured runs (1 tqdm `it` = 1 epoch, incl. that epoch's validation):
#   phikon-v2  ocelot   146.7 s/ep (135 steps) -> 1.09 s/step | pannuke  42.6 (82) -> 0.52
#   Midnight   ocelot   171.7 s/ep (135 steps) -> 1.27 s/step | pannuke 101.4 (82) -> 1.24
# Midnight's per-step cost is ~1.25 s on BOTH, 4x apart in image size: images are resized
# to a fixed model input, so that ~1.25 s is its GPU-bound ceiling, not a scaling factor.
# phikon-v2 on segpath_lymphocytes measured 5047 s (epoch 1) and 5169 s (epoch 2) over
# 1471+153 steps = 3.18 s/step -- already 2.5x above Midnight's GPU ceiling, i.e. segpath
# is I/O-bound (984x984 tiles, num_workers=1) and the backbone swap adds little. Note
# epoch 2 was SLOWER than epoch 1, so there is no page-cache speedup to bank on; 5170 is
# used throughout. Epithelial scales by train samples: 197208/94077 = 2.09x.
#
#   job                            epochs  s/epoch   estimate   wall
#   thd-segpath_lymphocytes          21      5170     30.2 h    72 h
#   thdft1k-segpath_lymphocytes      21      5185     30.2 h    72 h
#   thd-segpath_epithelial            9     10805     27.0 h    72 h
#   thdft1k-segpath_epithelial        9     10836     27.1 h    72 h
#   mthd-segpath_lymphocytes         21     ~5400     31.5 h    96 h
#   mthdft-segpath_lymphocytes       21     ~5500     32.1 h    96 h
#   mthd-segpath_epithelial           9    ~11300     28.2 h    96 h
#   mthdft-segpath_epithelial         9    ~11500     28.7 h    96 h
#
# The Midnight numbers are the I/O-bound central estimate. 48 h was the original plan and
# is NOT safe for them: if segpath's 3.18 s/step turned out to be compute-bound after all,
# the pannuke ratio (2.43x) would apply and mthdft-segpath_lymphocytes would need 73 h.
# 96 h covers that worst case; 48 h does not. Hence 96 h for the four Midnight jobs.
#
# ---------------------------------------------------------------------------------------
# POOLING: cls, EXPLICITLY, for all 8 -- never "auto"
# ---------------------------------------------------------------------------------------
# "auto" resolves to clsmean for Midnight, which advertises emb_dim = 2*hidden (3072 on
# ViT-g) while get_segmentation_embeddings returns hidden-d (1536) patch tokens. THUNDER
# sizes the seg decoder from emb_dim and dies at task_specific_models.py:121 with
#   RuntimeError: mat1 and mat2 shapes cannot be multiplied (16384x1536 and 3072x768)
# Four Midnight seg jobs died this way on 2026-08-02. Pooling is not applied to patch
# tokens at all in segmentation, so cls loses nothing from the published protocol.
#
# ---------------------------------------------------------------------------------------
# --only: SUBMIT ONE DATASET AT A TIME
# ---------------------------------------------------------------------------------------
# The two datasets do not become ready together -- segpath_lymphocytes (23 GB) was already
# mirrored under /data/thunder-data, while segpath_epithelial is a separate 47.5 GB Zenodo
# pull. Guard 0 below is deliberately all-or-nothing over whatever set is SELECTED, so
# without --only a single absent dataset blocks the four jobs that are perfectly ready.
#
#   bash scripts/submit_segpath_thunder.sh                                 # dry run, both
#   bash scripts/submit_segpath_thunder.sh --only segpath_lymphocytes      # dry run, 4 jobs
#   bash scripts/submit_segpath_thunder.sh --only segpath_lymphocytes --go # submit those 4
#   bash scripts/submit_segpath_thunder.sh --go                            # submit all 8
set -uo pipefail
cd "$(dirname "$0")/.."

# epochs per guidelines.md:4 -- 9 for epithelial, 21 for lymphocytes. Do not "harmonise".
ALL_SPECS=("segpath_epithelial 9" "segpath_lymphocytes 21")

GO=0
ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --go)   GO=1; shift ;;
    --only) ONLY="${2:-}"; shift 2 ;;
    --only=*) ONLY="${1#--only=}"; shift ;;
    *) echo "unknown argument: $1"; echo "usage: $0 [--only segpath_epithelial|segpath_lymphocytes] [--go]"; exit 2 ;;
  esac
done

SPECS=()
if [ -z "$ONLY" ]; then
  SPECS=("${ALL_SPECS[@]}")
else
  for spec in "${ALL_SPECS[@]}"; do
    [ "${spec%% *}" = "$ONLY" ] && SPECS+=("$spec")
  done
  if [ "${#SPECS[@]}" -eq 0 ]; then
    echo "unknown dataset for --only: $ONLY"
    echo "  valid: segpath_epithelial segpath_lymphocytes"
    exit 2
  fi
fi

SEG_TASKS="segmentation"
PHIKON_ADAPTER="runs/waiv-real-369043/step_0001000"
MIDNIGHT_ADAPTER="runs/waiv-midnight-369159/step_0000500"
# Third backbone. Unlike the two above, its adapter is not a constant: the checkpoint is
# picked by the blind "best PathoROB checkpoint" rule after training, so both the path and
# the step encoded in the run name arrive from the environment. When the adapter is absent
# the vthdft- row is SKIPPED and the vthd- base row still goes -- these jobs run 96h, so
# the base half must not wait on the fine-tune. Same property as submit_thunder.sh
# --base-only: nothing lands under vft*_ without a real adapter.
VIRCHOW2_ADAPTER="${WAIV_VIRCHOW2_ADAPTER:-}"
VIRCHOW2_FT_RUN="${WAIV_VIRCHOW2_FT_RUN:-vft500_cls}"

ROOT="${THUNDER_BASE_DATA_FOLDER:-/data/ryan.kim/thunder}"
ACTIVE=$(squeue -u ryan.kim -h -o "%j" | sort -u)

# Guard 0: the data has to actually be there. segpath_epithelial is downloaded separately
# (Zenodo record 7412731, 47.5 GB tarball) and its splits json is written by
# `thunder download-datasets segpath_epithelial --make-splits`, which is also what
# verifies the md5sum in config/dataset/segpath_epithelial.yaml. Submitting before that
# lands gives jobs that each burn a GPU slot and die on a missing split file.
# Only the SELECTED datasets are checked -- an unselected one is not being submitted, so
# its absence is irrelevant. Note the datasets/<ds> dir can exist mid-download, so the
# splits json (written last, after the md5 check) is the real readiness signal.
missing=0
for spec in "${SPECS[@]}"; do
  ds="${spec%% *}"
  [ -e "$ROOT/datasets/$ds" ]            || { echo "MISSING $ROOT/datasets/$ds"; missing=1; }
  [ -f "$ROOT/data_splits/$ds.json" ] || [ -f "$ROOT/datasets/data_splits/$ds.json" ] \
                                        || { echo "MISSING data_splits/$ds.json"; missing=1; }
done
if [ "$missing" -eq 1 ]; then
  echo "REFUSING to submit -- see missing paths above."
  echo "  fix: THUNDER_BASE_DATA_FOLDER=$ROOT thunder download-datasets <ds> --make-splits"
  exit 1
fi

submit() {  # <jobname> <dataset> <run_name> <epochs> <wall> <backbone|""> <adapter|"">
  local name="$1" ds="$2" run="$3" epochs="$4" wall="$5" backbone="$6" adapter="${7:-}"

  if grep -qx "$name" <<<"$ACTIVE"; then echo "SKIP $name -- in queue"; return; fi
  if [ -f "$ROOT/outputs/res/$ds/$run/segmentation/frozen/outputs.json" ]; then
    echo "SKIP $name -- results complete"; return
  fi

  # --export: ALL keeps the submitting environment (that is how the existing sweeps work);
  # WAIV_EPOCHS is the guidelines-mandated override and is set ONLY here. WAIV_BACKBONE is
  # appended only for Midnight -- unset means phikon-v2 (thunder_model.py's default).
  local exports="ALL,WAIV_EPOCHS=$epochs"
  [ -n "$backbone" ] && exports="$exports,WAIV_BACKBONE=$backbone"

  # Positional contract of run_thunder.sbatch is unchanged:
  #   <dataset> <tasks> <pooling> <run_name> <ckpt> [adapter]
  local args=(--hold --job-name="$name" --time="$wall" --export="$exports"
              scripts/run_thunder.sbatch "$ds" "$SEG_TASKS" cls "$run" "")
  [ -n "$adapter" ] && args+=("$adapter")

  if [ $GO -eq 1 ]; then sbatch "${args[@]}" | tail -1; else echo "DRYRUN sbatch ${args[*]}"; fi
}

# Job-name prefixes reuse the four the pilot already knows
# (scripts/thunder_pilot.py PREFIXES = thd-/thdft1k-/mthd-/mthdft-); a prefix it does not
# recognise makes it see an empty queue and declare DONE while these sit held forever.
# run_names reuse the existing per-backbone seg row names so scripts/collect_thunder.py
# merges them into the same table: base_cls/ft1000_cls for phikon-v2, mbase_cls/mft500_cls
# for Midnight (the _cls seg rows, NOT the _clsmean classification ones).
for spec in "${SPECS[@]}"; do
  set -- $spec; ds="$1"; ep="$2"
  submit "thd-$ds"     "$ds" base_cls    "$ep" 72:00:00 ""                 ""
  submit "thdft1k-$ds" "$ds" ft1000_cls  "$ep" 72:00:00 ""                 "$PHIKON_ADAPTER"
  submit "mthd-$ds"    "$ds" mbase_cls   "$ep" 96:00:00 kaiko-ai/midnight  ""
  submit "mthdft-$ds"  "$ds" mft500_cls  "$ep" 96:00:00 kaiko-ai/midnight  "$MIDNIGHT_ADAPTER"
  submit "vthd-$ds"    "$ds" vbase_cls   "$ep" 96:00:00 paige-ai/Virchow2  ""
  if [ -n "$VIRCHOW2_ADAPTER" ] && [ -d "$VIRCHOW2_ADAPTER" ]; then
    submit "vthdft-$ds" "$ds" "$VIRCHOW2_FT_RUN" "$ep" 96:00:00 paige-ai/Virchow2 "$VIRCHOW2_ADAPTER"
  else
    echo "SKIP vthdft-$ds -- no Virchow2 adapter (set WAIV_VIRCHOW2_ADAPTER + WAIV_VIRCHOW2_FT_RUN)"
  fi
done

echo
echo "Submitted HELD. Drain with:"
echo "  nohup .venv/bin/python scripts/thunder_pilot.py --cap 4 --interval 120 \\"
echo "      --max-fast-failures 3 >> /data/ryan.kim/thunder_pilot.log 2>&1 &"
