#!/usr/bin/env bash
# Cancel the stale held phikon-v2 THUNDER jobs and resubmit them against the CURRENT
# sbatch. Thin wrapper -- all logic now lives in scripts/submit_thunder.sh, which is the
# same script submit_midnight_thunder.sh calls with a different --backbone.
#
# Kept as a separate entry point (rather than deleted) because the name is referenced from
# scripts/run_thunder.sbatch:51-52 and from the run logs/notes of the in-flight sweep, so
# the reproduction instructions there stay valid.
#
# Why cancel rather than `scontrol release`: SLURM copies the batch script into its spool
# at *submission* time. The 17 held jobs were submitted before the cold-cache fix
# (pre_computing_embeddings), so releasing them would re-run the old script and each would
# die in ~37s on FileNotFoundError .../embeddings/<ds>/<model>/train/labels.h5.
# Python changes DO reach queued jobs (read at runtime); .sbatch changes do NOT.
# That is what --cancel-held below does; it is scoped to thd-/thdft1k- only.
#
# segpath_lymphocytes/_epithelial are NOT covered here: guidelines.md:4 mandates
# --adaptation.epochs 21/9 for them versus frozen.yaml's generic 200 (~281 h vs a 12 h
# wall, job 369061), so they have their own submitter with the WAIV_EPOCHS override and a
# longer --time -- scripts/submit_segpath_thunder.sh.
#
# Resubmits in HELD state so scripts/thunder_pilot.py governs concurrency.
#
#   bash scripts/resubmit_thunder.sh          # dry run, prints what it would do
#   bash scripts/resubmit_thunder.sh --go     # actually do it
set -uo pipefail
exec bash "$(dirname "$0")/submit_thunder.sh" --backbone phikon-v2 --cancel-held "$@"
