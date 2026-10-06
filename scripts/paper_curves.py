#!/usr/bin/env python3
"""RI-vs-step figure for the paper: shows the checkpoint-selection rule in action.

Reads runs/<RUN>/ri_curve.json for exactly the cells the paper's tables use (resolved via
pathorob_submetrics.cells_for) and plots RI against optimisation step, marking each run's
selected checkpoint. All 3 adapter seeds per backbone are plotted, matching the table n.

    ./.venv/bin/python scripts/paper_curves.py
      -> $SPECTRA_PAPER_FIGURES/ri_vs_step.pdf (+ .png preview)

Plotting is shared with paper/scripts/ri_vs_step.py, which renders the same figure from the
frozen paper/data/ri_vs_step.json (written by paper/scripts/extract_ri_vs_step.py).

The point of the figure: every selected checkpoint lands in 100--200 steps, well before
the 500-step schedule ends, and RI is flat-to-decaying afterwards. That claim is made in
prose in the paper (Sec. checkpoint selection) with no supporting exhibit.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import CELLS, PAPER_FIGURES, RUNS  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RUNS_ROOT = RUNS
SEED_STATS = REPO / "docs/seed_stats.md"
OUT = PAPER_FIGURES

# seed_stats.md section name -> scoreboard backbone label
SLUG = {"phikon2": "Phikon-v2", "midnight": "Midnight-12k", "virchow2": "Virchow2",
        "hoptimus0": "H-Optimus-0", "uni2h": "UNI2-h", "virchow1": "Virchow",
        "openmidnightsq": "OpenMidnight"}

def parse_curves():
    """-> list of (backbone, seed, selected_step, [(step, avg RI), ...])

    Reads runs/<RUN>/ri_curve.json for EXACTLY the cells the paper's tables use, resolved
    through pathorob_submetrics.cells_for.  An earlier version parsed the curve strings out
    of docs/final_scoreboard.md, which is a stale snapshot: it records "no curve" for runs
    whose ri_curve.json has since landed, so it under-reported the seed count per panel
    (Midnight-12k showed 1 curve when all 3 seeds have curves on disk).
    """
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pathorob_submetrics as pm

    rows = []
    for slug, label in SLUG.items():
        _base, seed_cells = pm.cells_for(slug)
        for cell in seed_cells:
            text = (CELLS / cell / "model.py").read_text()
            run = re.search(r'^RUN = "(.*)"$', text, re.M)
            step = re.search(r'^STEP = "(.*)"$', text, re.M)
            if not (run and step):
                continue
            sel = int(step.group(1).replace("step_", ""))
            cur = RUNS_ROOT / run.group(1) / "ri_curve.json"
            if not cur.exists():
                print(f"  no ri_curve.json for {cell} ({run.group(1)})")
                continue
            pts = sorted((q["step"], q["avg_robustness_index"])
                         for q in json.loads(cur.read_text())["points"]
                         if q.get("avg_robustness_index") is not None)
            if len(pts) < 2:
                print(f"  {cell}: only {len(pts)} scored checkpoint(s), skipped")
                continue
            seed = re.search(r"-s(\d+)-", cell)
            rows.append((label, int(seed.group(1)) if seed else -1, sel, pts))
    return rows


def parse_base_ri():
    """-> {backbone label: base PathoROB RI}, read from the same doc the tables use."""
    base, cur = {}, None
    for line in SEED_STATS.read_text().splitlines():
        if line.startswith("## "):
            cur = SLUG.get(line[3:].strip())
        elif cur and line.startswith("| PathoROB RI |"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            try:
                base.setdefault(cur, float(cells[1]))
            except ValueError:
                pass
    return base


def figure():
    """Plot straight from the run directories (needs $SPECTRA_RUNS / $SPECTRA_CELLS).

    The plotting code lives in paper/scripts/ri_vs_step.py so the paper copy, which renders
    from the frozen paper/data/ri_vs_step.json, and this raw-data path cannot drift apart.
    Several of the paper's run dirs have since been deleted; for the figure as published use
    paper/scripts/extract_ri_vs_step.py + paper/scripts/ri_vs_step.py instead.
    """
    sys.path.insert(0, str(REPO / "paper" / "scripts"))
    from ri_vs_step import render
    render(parse_curves(), parse_base_ri(), OUT)


if __name__ == "__main__":
    figure()
