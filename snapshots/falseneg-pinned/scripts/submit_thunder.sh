#!/usr/bin/env bash
# THUNDER sweep submitter for the 12 classification + 2 short segmentation datasets,
# parameterised by BACKBONE. Replaces the two byte-duplicated submitters:
#
#   scripts/resubmit_thunder.sh        == this script with --backbone phikon-v2 --cancel-held
#   scripts/submit_midnight_thunder.sh == this script with --backbone midnight
#
# Both of those are now thin wrappers around this file, kept because
# scripts/run_thunder.sbatch:51-52 and several run logs reference them by name.
#
# Modelled on scripts/submit_segpath_thunder.sh, which already had the right shape: ONE
# submit() that takes the backbone explicitly and appends WAIV_BACKBONE to --export only
# when it is non-empty. Empty backbone == phikon-v2, which is thunder_model.py's default
# (src/waivphaet/eval/thunder_model.py:140, `os.environ.get("WAIV_BACKBONE") or None`), so
# the phikon-v2 jobs must carry NO --export at all -- that is what the in-flight sweep did
# and changing it would change nothing but is gratuitous churn on a live queue.
#
# ---------------------------------------------------------------------------------------
# WHY POOLING IS PER-BACKBONE AND NOT A FLAG
# ---------------------------------------------------------------------------------------
# CLASSIFICATION pooling is backbone-dependent in THUNDER, not a free choice:
# arXiv:2607.22861 3 line 106 uses CLS+mean for Midnight-12k and CLS elsewhere. Midnight
# therefore passes "auto" (thunder_model._default_pooling resolves it from WAIV_BACKBONE);
# phikon-v2 passes "cls" explicitly, matching the sweep already on disk.
#
# SEGMENTATION pooling is "cls" for EVERY backbone and must never become "auto": clsmean
# advertises emb_dim = 2*hidden (3072 on Midnight's ViT-g) while get_segmentation_
# embeddings returns hidden-d (1536) patch tokens, and THUNDER sizes its seg decoder from
# emb_dim -- the job dies at thunder/models/task_specific_models.py:121 with
#   RuntimeError: mat1 and mat2 shapes cannot be multiplied (16384x1536 and 3072x768)
# Four Midnight seg jobs died this way on 2026-08-02. phikon-v2 never hit it because its
# CLS dim equals its patch dim. cls is also the honest setting: pooling is not applied to
# patch tokens at all, so nothing about the published protocol is lost.
#
# ---------------------------------------------------------------------------------------
# ADDING A BACKBONE  (phikon-v2, midnight and virchow2 are already here)
# ---------------------------------------------------------------------------------------
# Add one case to backbone_spec() below AND add its two job-name prefixes to
# scripts/thunder_pilot.py's PREFIXES (or pass THUNDER_PREFIXES). A prefix the pilot does
# not recognise makes it see an empty queue, print DONE, and exit while the jobs sit held
# forever -- exactly what happened on the first Midnight submission.
#
# segpath_epithelial / segpath_lymphocytes are deliberately NOT here: they need the
# guidelines.md:4 epochs override (9 / 21) and a much longer wall, so they keep their own
# submitter, scripts/submit_segpath_thunder.sh.
#
# Submitted HELD; scripts/thunder_pilot.py drains them at bounded concurrency.
#
#   bash scripts/submit_thunder.sh                                  # dry run, phikon-v2
#   bash scripts/submit_thunder.sh --backbone midnight              # dry run, Midnight
#   bash scripts/submit_thunder.sh --backbone virchow2              # dry run, Virchow2
#   bash scripts/submit_thunder.sh --backbone midnight --go         # submit
#   bash scripts/submit_thunder.sh --cancel-held --go               # resubmit flow
#   bash scripts/submit_thunder.sh --backbone virchow2 --base-only --go   # base rows only
#
# --base-only submits the *_BASE jobs and nothing else. It exists so a backbone's base
# sweep can drain while its fine-tune is still training -- the base rows depend on no
# adapter, and THUNDER is the long pole. It therefore also lifts the adapter-exists
# refusal below, which is safe precisely because no *_FT job is submitted: the property
# that guard protects is "nothing lands under *ft*_ without an adapter", and base-only
# writes only under *base_.
set -uo pipefail
cd "$(dirname "$0")/.."

BACKBONE_KEY="phikon-v2"
GO=0
CANCEL_HELD=0
BASE_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --go) GO=1; shift ;;
    --backbone) BACKBONE_KEY="${2:-}"; shift 2 ;;
    --backbone=*) BACKBONE_KEY="${1#--backbone=}"; shift ;;
    --cancel-held) CANCEL_HELD=1; shift ;;
    --base-only) BASE_ONLY=1; shift ;;
    *) echo "unknown argument: $1"
       echo "usage: $0 [--backbone phikon-v2|midnight|virchow2] [--base-only] [--cancel-held] [--go]"; exit 2 ;;
  esac
done

# Everything that differs between backbones lives here and nowhere else.
#   BACKBONE   value for WAIV_BACKBONE; "" means thunder_model.py's default (phikon-v2)
#              and suppresses --export entirely
#   P_BASE/P_FT  job-name prefixes -- MUST be known to thunder_pilot.py
#   CLS_POOL     classification pooling ("cls" or "auto"); segmentation is always "cls"
#   *_RUN        THUNDER run_name, i.e. the row name scripts/collect_thunder.py merges on
#   ADAPTER      fine-tuned LoRA adapter for the *_FT jobs
backbone_spec() {
  case "$1" in
    phikon-v2|phikon|"")
      BACKBONE=""
      P_BASE="thd-";  P_FT="thdft1k-"
      CLS_POOL="cls"
      CLS_BASE_RUN="base_cls";      CLS_FT_RUN="ft1000_cls"
      SEG_BASE_RUN="base_cls";      SEG_FT_RUN="ft1000_cls"
      ADAPTER="runs/waiv-real-369043/step_0001000"
      ;;
    midnight|kaiko-ai/midnight)
      BACKBONE="kaiko-ai/midnight"
      P_BASE="mthd-"; P_FT="mthdft-"
      CLS_POOL="auto"
      CLS_BASE_RUN="mbase_clsmean"; CLS_FT_RUN="mft500_clsmean"
      SEG_BASE_RUN="mbase_cls";     SEG_FT_RUN="mft500_cls"
      ADAPTER="runs/waiv-midnight-369159/step_0000500"
      ;;
    # Third backbone. Naming follows Midnight exactly, with v- for Virchow2 where
    # Midnight uses m-: job prefixes vthd-/vthdft- (thunder_pilot.py DEFAULT_PREFIXES),
    # run names vbase_*/vft500_* (collect_thunder.py BACKBONE_RUN_PREFIXES). Virchow2 is
    # clsmean in THUNDER (arXiv:2607.22861 3 line 106 names it first), so CLS_POOL is
    # "auto" -- thunder_model._default_pooling resolves paige-ai/Virchow2 to clsmean --
    # and the classification rows carry _clsmean while the two segmentation rows carry
    # _cls, for the emb_dim reason in the header (2560 advertised vs 1280 patch tokens).
    virchow2|virchow|paige-ai/Virchow2)
      BACKBONE="paige-ai/Virchow2"
      P_BASE="vthd-"; P_FT="vthdft-"
      CLS_POOL="auto"
      # The FT step is not known until training picks a checkpoint by the blind
      # "best PathoROB checkpoint" rule, so it comes from the environment rather than
      # being frozen at 500 the way Midnight's is. collect_thunder.py matches on the
      # bare "vft" prefix (BACKBONE_RUN_PREFIXES), so any step merges into the table.
      V_FT_STEP="${WAIV_VIRCHOW2_FT_STEP:-500}"
      CLS_BASE_RUN="vbase_clsmean"; CLS_FT_RUN="vft${V_FT_STEP}_clsmean"
      SEG_BASE_RUN="vbase_cls";     SEG_FT_RUN="vft${V_FT_STEP}_cls"
      # PLACEHOLDER. No Virchow2 fine-tune has been trained yet, so there is no adapter
      # to point at. The path below does not exist on purpose: a --go run refuses rather
      # than quietly submitting the *_FT jobs with no adapter, which would produce a full
      # sweep of base numbers filed under vft500_* and silently corrupt the comparison.
      # Override with WAIV_VIRCHOW2_ADAPTER=runs/<run>/step_XXXXXXX once one exists.
      ADAPTER="${WAIV_VIRCHOW2_ADAPTER:-runs/PLACEHOLDER-virchow2-adapter}"
      # --base-only submits no *_FT job, so a missing adapter cannot mislabel anything.
      if [ $GO -eq 1 ] && [ $BASE_ONLY -eq 0 ] && [ ! -d "$ADAPTER" ]; then
        echo "refusing to submit: Virchow2 adapter '$ADAPTER' does not exist."
        echo "  set WAIV_VIRCHOW2_ADAPTER=runs/<run>/step_XXXXXXX, pass --base-only for the"
        echo "  base rows alone, or drop --go for a dry run."
        exit 2
      fi
      ;;
    *)
      echo "unknown backbone: $1"
      echo "  valid: phikon-v2 midnight virchow2"
      exit 2 ;;
  esac
}
backbone_spec "$BACKBONE_KEY"

CLS_TASKS="knn linear_probing simple_shot"
SEG_TASKS="segmentation"
CLASSIFICATION="bach bracs break_his ccrcc crc esca mhist patch_camelyon tcga_crc_msi tcga_tils tcga_uniform wilds"
SEGMENTATION="ocelot pannuke"

ROOT="${THUNDER_BASE_DATA_FOLDER:-/data/ryan.kim/thunder}"
USER_NAME="${USER:-ryan.kim}"

# --cancel-held: SLURM copies the batch script into its spool at *submission* time, so
# `scontrol release` on a job submitted before an .sbatch fix would re-run the OLD script.
# (Python changes DO reach queued jobs; .sbatch changes do NOT.) Cancelling and
# resubmitting is the only way to pick up an .sbatch fix. Scoped to THIS backbone's two
# prefixes so a phikon-v2 resubmit cannot nuke a held Midnight sweep.
if [ $CANCEL_HELD -eq 1 ]; then
  held=$(squeue -u "$USER_NAME" -h -t PD -o "%i|%j|%r" \
         | awk -F'|' -v a="$P_BASE" -v b="$P_FT" \
             '$3=="JobHeldUser" && (index($2,a)==1 || index($2,b)==1) {print $1}' \
         | paste -sd, -)
  if [ -n "$held" ]; then
    echo "cancel stale held jobs: $held"
    [ $GO -eq 1 ] && scancel "$held"
  else
    echo "no held THUNDER jobs to cancel"
  fi
fi

# Guard 1's input. With --cancel-held the held jobs are gone (in --go) or about to be, so
# they must NOT count as "already queued" -- otherwise every dataset looks busy and a dry
# run shows nothing to do. Without --cancel-held a held job is a real, live submission and
# resubmitting it would run the same dataset twice on two GPUs for nothing.
if [ $CANCEL_HELD -eq 1 ]; then
  ACTIVE=$(squeue -u "$USER_NAME" -h -o "%j|%r" \
           | awk -F'|' '$2!="JobHeldUser" {print $1}' | sort -u)
else
  ACTIVE=$(squeue -u "$USER_NAME" -h -o "%j" | sort -u)
fi

submit() {  # submit <jobname> <dataset> <tasks> <pooling> <run_name> [adapter]
  local name="$1" ds="$2" tasks="$3" pool="$4" run="$5" adapter="${6:-}"

  # Guard 1: already running or pending.
  if grep -qx "$name" <<<"$ACTIVE"; then
    echo "SKIP $name -- in queue"; return
  fi
  # Guard 2: results already on disk for every task we would run.
  local complete=1
  for t in $tasks; do
    [ -f "$ROOT/outputs/res/$ds/$run/$t/frozen/outputs.json" ] || complete=0
  done
  if [ "$complete" -eq 1 ]; then
    echo "SKIP $name -- results already complete for [$tasks]"; return
  fi

  # --export is appended ONLY when a backbone is named. ALL keeps the submitting
  # environment, which is how the existing sweeps work.
  local args=(--hold --job-name="$name")
  [ -n "$BACKBONE" ] && args+=(--export="ALL,WAIV_BACKBONE=$BACKBONE")
  # Positional contract of run_thunder.sbatch:
  #   <dataset> <tasks> <pooling> <run_name> <ckpt> [adapter]
  args+=(scripts/run_thunder.sbatch "$ds" "$tasks" "$pool" "$run" "")
  [ -n "$adapter" ] && args+=("$adapter")

  if [ $GO -eq 1 ]; then sbatch "${args[@]}" | tail -1; else echo "DRYRUN sbatch ${args[*]}"; fi
}

for ds in $CLASSIFICATION; do
  submit "$P_BASE$ds" "$ds" "$CLS_TASKS" "$CLS_POOL" "$CLS_BASE_RUN"
  [ $BASE_ONLY -eq 1 ] || submit "$P_FT$ds" "$ds" "$CLS_TASKS" "$CLS_POOL" "$CLS_FT_RUN" "$ADAPTER"
done
# cls, not $CLS_POOL -- see the emb_dim/patch-dim note in the header. The run_name carries
# the pooling too, so on Midnight these land in outputs/res/<ds>/m{base,ft500}_cls/ and
# cannot collide with the _clsmean classification rows.
for ds in $SEGMENTATION; do
  submit "$P_BASE$ds" "$ds" "$SEG_TASKS" cls "$SEG_BASE_RUN"
  [ $BASE_ONLY -eq 1 ] || submit "$P_FT$ds" "$ds" "$SEG_TASKS" cls "$SEG_FT_RUN" "$ADAPTER"
done

echo
echo "Submitted HELD. Drain with:"
echo "  nohup .venv/bin/python scripts/thunder_pilot.py --cap 4 --interval 120 \\"
echo "      --max-fast-failures 3 >> /data/ryan.kim/thunder_pilot.log 2>&1 &"
