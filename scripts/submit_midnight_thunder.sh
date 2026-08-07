#!/usr/bin/env bash
# THUNDER sweep for Midnight-12k: base + fine-tuned (step 500, its best PathoROB point).
# Thin wrapper -- all logic now lives in scripts/submit_thunder.sh, which is the same
# script resubmit_thunder.sh calls with --backbone phikon-v2.
#
# Kept as a separate entry point (rather than deleted) because the name is referenced from
# scripts/run_thunder.sbatch:51-52 and from the notes/logs of the Midnight sweep.
#
# Midnight had PathoROB only; this closes that gap so the second backbone gets the same
# retention treatment phikon-v2 got. The backbone-specific choices it used to hardcode
# (WAIV_BACKBONE=kaiko-ai/midnight, "auto" classification pooling per arXiv:2607.22861 3
# line 106, explicit "cls" segmentation pooling to dodge the emb_dim/patch-dim crash at
# task_specific_models.py:121, the mbase_*/mft500_* run names, the mthd-/mthdft- job
# prefixes and the step-500 adapter) now live in submit_thunder.sh's backbone_spec()
# midnight case, with the reasoning kept in that file's header.
#
# Submitted HELD; scripts/thunder_pilot.py drains them at bounded concurrency.
#
#   bash scripts/submit_midnight_thunder.sh          # dry run
#   bash scripts/submit_midnight_thunder.sh --go
set -uo pipefail
exec bash "$(dirname "$0")/submit_thunder.sh" --backbone midnight "$@"
