#!/usr/bin/env bash
# ======================================================================================
# Regenerate every LaTeX table the paper \input's, into $SPECTRA_PAPER/tables.
# ======================================================================================
#
#     paper/make_tables.sh
#
# Environment (all optional; unset ones fall back to .env, then to the repo-relative
# defaults in src/spectra/paths.py -- print the resolved values with
# `python3 scripts/_config.py`):
#
#   SPECTRA_PAPER      output root; tables land in $SPECTRA_PAPER/tables
#                      (default <repo>/paper -> <repo>/paper/tables)
#   SPECTRA_DATA       root the corpora / caches default under
#   SPECTRA_EVALS      harness outputs (THUNDER / PathoROB / CPTAC eval results)
#   SPECTRA_CELLS      per-cell model directories (seed cells, base controls)
#   SPECTRA_RUNS       training runs (ri_curve.json, per-benchmark summaries)
#   SPECTRA_HEST_WORK  HEST eval outputs
#   PYTHON             interpreter to use (default: python3)
#
# Inputs that are read from the checkout: data/*_leaderboard*.tsv (published fields)
# and docs/*.md (seed_stats.md, hest_per_task.md, plism_retrieval.md, ...).
#
# Order matters:
#   (plism_retrieval.py is deliberately NOT run: plism_retrieval.tex is not a paper input,
#    and pathorob_ranks reads the committed docs/plism_retrieval.md.  Rerunning it today
#    rewrites that doc with '--' cells because the finalgem-* runs' probe_before.json and
#    most tuned probe_step_*.json files have been deleted from $SPECTRA_RUNS, which in
#    turn blanks the PLISM column of pathorob_ranks.tex.)
#   1. thunder_ranks     -> thunder_ranks{,_full}.tex
#   2. hest_ranks        -> hest_ranks.tex        (make_hest_per_task reads it)
#   3. pathorob_ranks    -> pathorob_ranks.tex
#   4. overall_ranks     -> overall_ranks.tex     (imports the three rank modules)
#   5. pathorob_submetrics -> pathorob_submetrics.tex
#   6. cptac_table       -> cptac.tex; --per-task -> cptac_per_task.tex
#   7. seed_spread_tables -> seed_hest.tex, seed_thunder_{knn,lp,fewshot,seg,ece,adv}.tex
#   8. seed_raw_table --tex-dir -> perseed_{pathorob,hest,cptac,thunder}.tex
#   9. paper/scripts/combine_seed_thunder.py -> thunder_per_ds_{cls,other}.tex
#  10. paper/scripts/make_hest_per_task.py   -> hest_per_task.tex   (needs hest_ranks.tex)
#  11. paper/scripts/make_hest_cptac.py      -> hest_cptac.tex      (needs hest_per_task + cptac)
#
# Not generated: tables/ablations.tex is hand-written; it is copied from paper/data/tables/.
#
# Side effect: the in-repo generators also rewrite their docs/*.md summaries; on
# unchanged inputs those rewrites are byte-identical (check with `git diff docs/`).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
PY="${PYTHON:-python3}"

TABLES="$("$PY" -c 'import sys; sys.path.insert(0, "scripts"); from _config import PAPER_TABLES; print(PAPER_TABLES)')"
mkdir -p "$TABLES"
echo "writing tables to $TABLES"

SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT

run() { echo "+ $PY $*"; "$PY" "$@"; }

run scripts/thunder_ranks.py
run scripts/hest_ranks.py
run scripts/pathorob_ranks.py
run scripts/overall_ranks.py
run scripts/pathorob_submetrics.py
run scripts/cptac_table.py
run scripts/cptac_table.py --per-task
run scripts/seed_spread_tables.py
run scripts/seed_raw_table.py --out-dir "$SCRATCH/seed_raw" --tex-dir "$TABLES"

run paper/scripts/combine_seed_thunder.py
run paper/scripts/make_hest_per_task.py
run paper/scripts/make_hest_cptac.py

# ablations.tex is hand-written (values transcribed from the appendix prose); it is
# tracked as paper/data/tables/ablations.tex and copied, not generated.
cp paper/data/tables/ablations.tex "$TABLES/ablations.tex"

echo "done: $(ls "$TABLES"/*.tex | wc -l) tables in $TABLES"
