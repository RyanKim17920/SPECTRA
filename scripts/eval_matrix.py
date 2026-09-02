#!/usr/bin/env python3
"""The eval plan as a coverage matrix: every cell x every suite, grouped by the role the
cell plays in the paper.

`docs/final_scoreboard.md` answers "what do the numbers say"; it lists what is missing only
where a graded cell needed it.  This answers the other question -- "what is still running,
and what is it for" -- over the whole `pathfm-cells` tree, so the queue can be read against
the paper's argument rather than against directory names.

The five completion probes below are the same ones the saturation controller uses to decide
what to submit (waiv_status/watch/saturate.py:has_results).  They are duplicated rather than
imported because that controller lives outside this repo; if one moves, move both.

    ./.venv/bin/python scripts/eval_matrix.py    ->  docs/eval_matrix.md
"""
import glob
import os
import re
from pathlib import Path

CELLS = Path("/admin/home/ryan.kim/pathfm-cells")
OUT = Path("/data/ryan.kim/pathfm-full-evals")
HEST_RES = "/data/ryan.kim/hest_work/results"
REPO = Path(__file__).resolve().parent.parent

# The 1-SE-selected operating point per backbone (docs/final_scoreboard.md section 1a).
# A cell at this step is the one the headline table reads; a cell at any other step is a
# sensitivity point, and is labelled as such rather than being silently equivalent.
OPERATING_STEP = {"phikon2": 200, "midnight": 150, "virchow2": 100,
                  "hoptimus0": 100, "uni2h": 100}
# UNI2-h's two seeds plateau at different steps, so both are operating points for it.
EXTRA_OPERATING = {"uni2h": {150}}

ROLES = [
    ("c50", "FINAL RECIPE (WAIV_BCLS=3.0 / WAIV_BMEAN=-inf)"),
    ("b00", "ABLATION: bias 0/0 -- arithmetically identical to no same-core masking"),
    ("bm3", "ABLATION: symmetric bias +3/+3 on both heads"),
    ("base-control", "BASE CONTROL: published backbone, no adapter"),
    ("c3s", "SUPERSEDED generation (kept for the three-generation comparison)"),
    ("c3m", "SUPERSEDED generation (kept for the three-generation comparison)"),
]


def role_of(cell):
    for token, _ in ROLES:
        if f"-{token}-" in cell or cell.endswith(f"-{token}"):
            return token
    return "other"


def backbone_of(cell):
    return cell.split("-")[0]


def step_of(cell):
    tail = cell.rsplit("-", 1)[-1]
    return int(tail[4:]) if tail.startswith("step") else None


def run_step_of(cell):
    """The (run directory, zero-padded step) this cell's model.py merges, read off the
    literals rather than parsed out of the cell name -- a cell may point at a `.rN` requeue
    sibling whose name the cell name does not carry."""
    text = (CELLS / cell / "model.py").read_text()
    run = re.search(r'^RUN = "(.*)"$', text, re.M).group(1)
    step = re.search(r'^STEP = "(.*)"$', text, re.M).group(1)
    return run, step.replace("step_", "")


def has_results(cell, suite):
    if suite == "hest":
        # HEST is NOT a cell suite.  It is measured out of band, per (run, step), by
        # scripts/hest_final5.sbatch, which writes
        # `<HEST_RES>/f5_<base run>_s<step>_<pool>_summary.json` -- so probing the cell
        # tree for it reports every cell missing forever.  Pooling is cls for
        # phikon/hoptimus/uni2 and clsmean for midnight/virchow2; the glob covers both.
        run, step = run_step_of(cell)
        if not run:
            return bool(glob.glob(f"{HEST_RES}/base_*_summary.json"))
        base = re.sub(r"\.r\d+$", "", run)
        return bool(glob.glob(f"{HEST_RES}/f5_{base}_s{step}_*_summary.json"))
    if suite == "cptac":
        return (OUT / "cptac" / cell / "aggregate.json").exists()
    if suite == "pathorob":
        return all(glob.glob(f"{OUT}/pathorob/results/{m}/{cell}*")
                   for m in ("robustness_index", "apd", "clustering_score"))
    if suite == "thunder":
        # the cached-probe stage: knn / linear / few-shot / calibration
        return (OUT / f"thunder/outputs/res/mhist/{cell}_optimized/knn/frozen/outputs.json").exists()
    # "online": segmentation over all four datasets (+PGD).  Probing only ocelot marks a
    # cell complete while the SegPath tasks are still failing.
    return all((OUT / f"thunder/outputs/res/{ds}/{cell}_optimized/segmentation/frozen/outputs.json").exists()
               for ds in ("ocelot", "pannuke", "segpath_epithelial", "segpath_lymphocytes"))


SUITES = ["thunder", "online", "pathorob", "cptac", "hest"]


def main():
    cells = sorted(d.name for d in CELLS.iterdir()
                   if d.is_dir() and (d / "model.py").exists())
    lines = [
        "# Eval coverage matrix",
        "",
        "**Generated file -- do not hand-edit.**  Regenerate with:",
        "",
        "```",
        "./.venv/bin/python scripts/eval_matrix.py",
        "```",
        "",
        "Every cell under `/admin/home/ryan.kim/pathfm-cells`, by the role it plays in the",
        "paper.  `done` means the suite's terminal artifact is on disk; it is read from disk",
        "at generation time and is never inferred from a queue state.",
        "",
        "`op` marks a cell sitting at its backbone's 1-SE-selected operating point -- the",
        "checkpoint the headline table in `docs/final_scoreboard.md` reads.  Cells at other",
        "steps are sensitivity points, not substitutes for it.",
        "",
        "`online` is THUNDER segmentation + PGD; it is a separate row from `thunder` because",
        "`submit_partial.sh` deliberately skips it (4-4.5 of a ~7 GPU-hour path).",
        "",
    ]
    totals = {s: [0, 0] for s in SUITES}
    for token, blurb in ROLES + [("other", "unclassified")]:
        group = [c for c in cells if role_of(c) == token]
        if not group:
            continue
        lines += [f"## {token} -- {blurb}", "",
                  "| cell | backbone | step | " + " | ".join(SUITES) + " | complete |",
                  "|---|---|---|" + "---|" * (len(SUITES) + 1)]
        for c in sorted(group):
            bb, st = backbone_of(c), step_of(c)
            ops = {OPERATING_STEP.get(bb)} | EXTRA_OPERATING.get(bb, set())
            step_cell = "base" if st is None else (f"**{st} (op)**" if st in ops else str(st))
            marks, ndone = [], 0
            for s in SUITES:
                ok = has_results(c, s)
                ndone += ok
                totals[s][0] += ok
                totals[s][1] += 1
                marks.append("done" if ok else "--")
            lines.append(f"| `{c}` | {bb} | {step_cell} | " + " | ".join(marks)
                         + f" | {ndone}/{len(SUITES)} |")
        lines.append("")

    lines += ["## Totals", "",
              "| suite | done | cells | remaining |", "|---|---|---|---|"]
    for s in SUITES:
        done, total = totals[s]
        lines.append(f"| {s} | {done} | {total} | {total - done} |")
    grand_done = sum(v[0] for v in totals.values())
    grand_total = sum(v[1] for v in totals.values())
    lines += ["",
              f"**{grand_done} of {grand_total} cell x suite pairs complete "
              f"({grand_total - grand_done} remaining).**", ""]

    dest = REPO / "docs/eval_matrix.md"
    dest.write_text("\n".join(lines))
    print(f"wrote {dest}")
    print(f"{grand_done}/{grand_total} cell x suite pairs complete")


if __name__ == "__main__":
    main()
