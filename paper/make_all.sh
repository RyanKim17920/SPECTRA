#!/usr/bin/env bash
# Regenerate every paper figure (and, with --tables, every paper table) into $SPECTRA_PAPER.
#
#   paper/make_all.sh            figures only; needs nothing outside the repo
#   paper/make_all.sh --tables   also run paper/make_tables.sh (needs the eval outputs:
#                                SPECTRA_EVALS, SPECTRA_CELLS, SPECTRA_RUNS, SPECTRA_HEST_WORK)
#
# Output: $SPECTRA_PAPER/figures/*.pdf (+ .png previews), $SPECTRA_PAPER/tables/*.tex.
# SPECTRA_PAPER defaults to <repo>/paper (paper/figures and paper/tables are gitignored).
# Python: ${PYTHON:-python3} with numpy, matplotlib and Pillow (WebP support).
#
# Every figure renders from the frozen extracts in paper/data/. Each extract has its own
# regeneration command against the raw data (see "Reproducing the paper" in README.md).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
run() { echo "+ $PY $*"; "$PY" "$@"; }

# Main paper
run scripts/paper_figures.py                       # Fig 1 left panel (grid_batch_left)
run paper/scripts/annotate_grid_batch.py           # Fig 1: grid_batch
run paper/scripts/make_base_to_tuned.py            # Fig 2: base_to_tuned
# Appendix
run paper/scripts/ri_vs_step.py                    # ri_vs_step
run scripts/thunder_per_ds.py --from-extract       # thunder_per_ds
# Figures drafted for the paper but not in the current build (figures/unused/ or not \included)
run scripts/embedding_shift.py --from-extract      # embedding_shift
run scripts/retrieval_qualitative.py --from-extract # retrieval_examples
run paper/scripts/make_pipeline.py                 # pipeline
run paper/scripts/make_split.py                    # split
run paper/scripts/plism_traj.py                    # plism_traj

if [[ "${1:-}" == "--tables" ]]; then
    bash paper/make_tables.sh
fi
echo "done: figures in $("$PY" -c 'import sys; sys.path.insert(0,"scripts"); from _config import PAPER_FIGURES; print(PAPER_FIGURES)')"
