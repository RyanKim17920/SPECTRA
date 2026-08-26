#!/bin/bash
# Fan out SPIDER THUNDER evals. Canary (spider_thorax / base_cls) is submitted separately.
set -euo pipefail
cd /admin/home/ryan.kim/waiv
TASKS="knn linear_probing simple_shot"
RUNS=runs/genMASK-c3s-lr1e-4-r32-kl0-t0.07-wd0.05-ms500-pd512
DS_ALL="spider_thorax spider_breast spider_colorectal spider_skin"

# model spec: backbone|run_name|adapter(empty for base)|short
SPECS=(
  "owkin/phikon-v2|base_cls||pbase"
  "kaiko-ai/midnight|mbase_clsmean||mbase"
  "paige-ai/Virchow2|vbase_clsmean||vbase"
  "owkin/phikon-v2|f5_ci-phikon-s0-392669_s0000250|$RUNS-phikon-s0-t900-392669/step_0000250|ph-s0"
  "owkin/phikon-v2|f5_ci-phikon-s1-392672_s0000250|$RUNS-phikon-s1-t900-392672/step_0000250|ph-s1"
  "kaiko-ai/midnight|f5_ci-midnight-s0-392670_s0000125|$RUNS-midnight-s0-t900-392670/step_0000125|mn-s0"
  "kaiko-ai/midnight|f5_ci-midnight-s1-392673_s0000125|$RUNS-midnight-s1-t900-392673/step_0000125|mn-s1"
  "paige-ai/Virchow2|f5_ci-virchow2-s0-392671_s0000125|$RUNS-virchow2-s0-t900-392671/step_0000125|v2-s0"
  "paige-ai/Virchow2|f5_ci-virchow2-s1-392674_s0000125|$RUNS-virchow2-s1-t900-392674/step_0000125|v2-s1"
)

i=0
for spec in "${SPECS[@]}"; do
  IFS='|' read -r BB RN AD SH <<<"$spec"
  [ -n "$AD" ] && [ ! -d "$AD" ] && { echo "MISSING ADAPTER: $AD"; exit 1; }
  for DS in $DS_ALL; do
    # skip the canary cell (already submitted)
    if [ "$DS" = "spider_thorax" ] && [ "$RN" = "base_cls" ]; then echo "skip canary cell"; continue; fi
    ORGAN=${DS#spider_}
    JN="sp-${ORGAN}-${SH}"
    # alternate accounts to spread load
    if [ $((i % 2)) -eq 0 ]; then ACCT="--account=max --qos=high"; else ACCT="--account=training"; fi
    i=$((i+1))
    export WAIV_BACKBONE="$BB"
    if [ -n "$AD" ]; then
      sbatch $ACCT -J "$JN" scripts/run_thunder.sbatch "$DS" "$TASKS" auto "$RN" "" "$AD"
    else
      sbatch $ACCT -J "$JN" scripts/run_thunder.sbatch "$DS" "$TASKS" auto "$RN"
    fi
  done
done
echo "submitted $i jobs"
