#!/usr/bin/env python3
"""Freeze the data behind the appendix RI-vs-step figure into paper/data/ri_vs_step.json.

Data dependency
---------------
* Cells: exactly the seed cells the paper's tables use, resolved through
  ``scripts/pathorob_submetrics.cells_for`` (which reads data/seed_stats.txt) and each cell's
  ``$SPECTRA_CELLS/<cell>/model.py`` (``RUN = ...`` / ``STEP = ...``).
* Curves: ``$SPECTRA_RUNS/<RUN>/ri_curve.json`` (``points[].step`` /
  ``points[].avg_robustness_index``) -- what scripts/paper_curves.py plotted on 2026-09-06.
* Base RI per backbone: the ``| PathoROB RI |`` row of data/seed_stats.txt.

For runs whose directory is gone, the curve is recovered from the published figure's own
vector paths (``--recover-from-pdf <ri_vs_step.pdf>``): per-curve marker offsets are read
from the PDF content stream and mapped back to data with a per-panel transform calibrated on
the gridlines/tick labels (see _pdfvec.py). Validated on the panels whose ri_curve.json
still exists; recovered values are kept to 10 dp (measured error <= 1.2e-9). Otherwise the
existing entry in paper/data/ri_vs_step.json is kept unchanged. Each row records its source.

    python3 paper/scripts/extract_ri_vs_step.py [--recover-from-pdf PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import CELLS, RUNS  # noqa: E402

DATA = REPO / "paper/data/ri_vs_step.json"
SEED_STATS = REPO / "data/seed_stats.txt"

# keep in sync with scripts/paper_curves.py
SLUG = {"phikon2": "Phikon-v2", "midnight": "Midnight-12k", "virchow2": "Virchow2",
        "hoptimus0": "H-Optimus-0", "uni2h": "UNI2-h", "virchow1": "Virchow",
        "openmidnightsq": "OpenMidnight"}
ORDER = ["Phikon-v2", "Midnight-12k", "Virchow2", "H-Optimus-0", "UNI2-h",
         "Virchow", "OpenMidnight"]
DISPLAY = {"H-Optimus-0": "H-optimus-0"}
ACCENT, GRID = "#2563eb", "#e5e7eb"

SRC_DISK = "ri_curve.json"
SRC_PDF = "recovered from published ri_vs_step.pdf vector paths"
AGREE_TOL = 1e-5


def cells():
    """-> [(label, seed, cell, run, selected_step)] in the order paper_curves.py plotted."""
    import pathorob_submetrics as pm
    out = []
    for slug, label in SLUG.items():
        _base, seed_cells = pm.cells_for(slug)
        for cell in seed_cells:
            text = (CELLS / cell / "model.py").read_text()
            run = re.search(r'^RUN = "(.*)"$', text, re.M)
            step = re.search(r'^STEP = "(.*)"$', text, re.M)
            if not (run and step):
                continue
            seed = re.search(r"-s(\d+)-", cell)
            out.append((label, int(seed.group(1)) if seed else -1, cell, run.group(1),
                        int(step.group(1).replace("step_", ""))))
    return out


def disk_curve(run):
    cur = RUNS / run / "ri_curve.json"
    if not cur.exists():
        return None
    return sorted((q["step"], q["avg_robustness_index"])
                  for q in json.loads(cur.read_text())["points"]
                  if q.get("avg_robustness_index") is not None)


def base_ri():
    base, cur = {}, None
    for line in SEED_STATS.read_text().splitlines():
        if line.startswith("## "):
            cur = SLUG.get(line[3:].strip())
        elif cur and line.startswith("| PathoROB RI |"):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            try:
                base.setdefault(cur, float(c[1]))
            except ValueError:
                pass
    return base


def recover_pdf(pdf):
    """-> {panel title: [[(step, ri), ...] per curve in drawing order]} from the figure."""
    import _pdfvec as pv
    pg = pv.read(pdf)
    acc, grid = pv.hex_rgb(ACCENT), pv.hex_rgb(GRID)
    frames = pv.axes_frames(pg, grid)
    panels = {}
    for clip, fr in frames.items():
        fx, fy, resid = pv.calibrate(pg, clip, fr, xticks=[0, 250, 500])
        assert resid < 1e-3, (clip, resid)
        curves = []
        for it in pg.items:
            if getattr(it, "clip", None) != clip:
                continue
            if isinstance(it, pv.Path_) and it.op == "S" and pv.close(it.stroke, acc) \
                    and abs(it.width - 1.0) < 1e-6 and len(it.pts) >= 2:
                curves.append({"line": it.pts, "dots": []})
            elif isinstance(it, pv.Mark) and pv.close(it.fill, acc) and curves:
                # a Line2D's '.' markers are one q..Q group right after its line
                curves[-1]["dots"].append((it.x, it.y))
        rows = []
        for c in curves:
            dots = c["dots"] or c["line"]
            assert len(dots) == len(c["line"]), "marker/vertex count mismatch"
            # markers (10 dp) must sit on the line vertices (6 dp)
            assert max(max(abs(a[0] - b[0]), abs(a[1] - b[1]))
                       for a, b in zip(dots, c["line"])) < 1e-4
            pts = [(fx(x), fy(y)) for x, y in dots]
            for s, _ in pts:
                assert abs(s - round(s)) < 1e-3, s
            rows.append([(int(round(s)), v) for s, v in pts])
        panels[pv.title_above(pg, clip)] = rows
    return panels


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recover-from-pdf", type=Path, default=None,
                    help="published ri_vs_step.pdf to recover deleted runs' curves from")
    ap.add_argument("--out", type=Path, default=DATA)
    args = ap.parse_args()

    prev = {}
    if args.out.exists():
        prev = {r["cell"]: r for r in json.loads(args.out.read_text())["rows"]}
    rec = recover_pdf(args.recover_from_pdf) if args.recover_from_pdf else None
    if rec is not None:
        by_label = {}
        for label, seed, cell, run, sel in cells():
            by_label.setdefault(label, []).append(cell)
        rec_by_cell = {}
        for label, cl in by_label.items():
            curves = rec.get(DISPLAY.get(label, label))
            assert curves is not None and len(curves) == len(cl), (label, len(curves or []), cl)
            rec_by_cell.update(zip(cl, curves))

    rows, worst = [], 0.0
    for label, seed, cell, run, sel in cells():
        pts, src = disk_curve(run), SRC_DISK
        r = rec_by_cell.get(cell) if rec is not None else None
        if pts is not None:
            if r is not None:
                err = max(abs(a[1] - b[1]) for a, b in zip(pts, r)) if [a[0] for a in pts] == [b[0] for b in r] else float("inf")
                worst = max(worst, err)
                print(f"  validate {cell:32s} n={len(pts):2d} max|recovered - ri_curve.json| = {err:.2e}")
        elif r is not None:
            pts, src = [(s, round(y, 10)) for s, y in r], SRC_PDF
        elif cell in prev:
            rows.append(prev[cell])
            print(f"  {cell}: kept existing entry ({prev[cell]['source']})")
            continue
        else:
            print(f"  {cell}: no curve available ({run}), skipped")
            continue
        if len(pts) < 2:
            print(f"  {cell}: only {len(pts)} scored checkpoint(s), skipped")
            continue
        rows.append({"backbone": label, "seed": seed, "cell": cell, "run": run,
                     "selected_step": sel, "points": [[s, y] for s, y in pts], "source": src})
        print(f"  {cell:32s} n={len(pts):2d} sel={sel} <- {src}")
    if rec is not None:
        print(f"recovery validation: worst max-abs error vs ri_curve.json = {worst:.2e}")

    out = {"description": "Appendix figure ri_vs_step: PathoROB avg robustness index vs "
                          "optimisation step for the paper's seed cells, plus each backbone's "
                          "base (untuned) RI. Rendered by paper/scripts/ri_vs_step.py; "
                          "regenerated by paper/scripts/extract_ri_vs_step.py.",
           "base_ri": base_ri(), "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
