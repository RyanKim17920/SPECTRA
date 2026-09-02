#!/usr/bin/env python3
"""Across-seed mean and SD for the shipped recipe, per backbone and metric.

This replaces the "seed floor" framing.  A floor borrowed from another arm, another panel
or another corpus is always arguing by analogy; once the recipe itself has been run at
n>=2 seeds, the honest scale for "is this delta real" is the SD of THIS recipe's own
results.  So: report base, per-seed values, mean, SD, and the gain expressed in SDs.

Seeds are matched by each run's OWN 1-SE-selected checkpoint, not by a shared step. That
means the SD includes checkpoint-selection variance, which is correct -- the selection rule
is part of the procedure being reported, so its variability belongs in the error bar.

n=2 gives a spread, not an SD worth quoting; the table prints n and marks n<3 explicitly.

    ./.venv/bin/python scripts/seed_stats.py   ->  docs/seed_stats.md
"""
import csv
import glob
import json
import os
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parent.parent
CELLS = Path("/admin/home/ryan.kim/pathfm-cells")
OUT = Path("/data/ryan.kim/pathfm-full-evals")
HEST_RES = "/data/ryan.kim/hest_work/results"
RI_DATASETS = ("tcga", "camelyon", "tolkach_esca")

BACKBONES = ["phikon2", "midnight", "virchow2", "hoptimus0", "uni2h", "openmidnightsq", "virchow1", "virchow2f"]
# cell prefix -> the key the scoreboard's verdict JSON uses for the same backbone
VERDICT_KEY = {"phikon2": "phikon", "midnight": "midnight", "virchow2": "virchow2",
               "hoptimus0": "hoptimus", "uni2h": "uni2",
               "openmidnightsq": "openmidnight", "virchow1": "virchow1",
               "virchow2f": "virchow2"}
# cells whose base-control lives under a different name (fp32-attack Virchow2 base)
BASE_CELL = {"virchow2f": "virchow2-basectrl-fp32adv"}


def base_from_verdict(bb, metric):
    """Base-control value as the scoreboard computed it.

    NOT re-derived here: the only base HEST summaries on disk are the generic
    `base_cls` / `base_clsmean` pair, which are not per-backbone, so globbing them
    returns whichever sorts first and silently attributes one backbone's base to
    another. The generated verdict JSON already carries the correct per-backbone base.
    """
    f = REPO / "docs/final_recipe_verdict.json"
    if not f.exists():
        return None
    cells = json.loads(f.read_text()).get("cells", {})
    v = cells.get(VERDICT_KEY[bb], {}).get(metric, {}).get("base")
    if v is None and metric == "HEST":
        # Backbones added after the verdict JSON was generated: read the per-backbone base
        # summary that collect_final5 registers (raises at import if the file is absent).
        import collect_final5 as _c5
        v = _c5.HEST_BASE.get(VERDICT_KEY[bb])
    return v
THUNDER_TASKS = [("benchmark_knn", "knn", "f1", ""),
                 ("benchmark_linear_probing", "linear_probing", "f1", ""),
                 ("benchmark_simple_shot", "simple_shot", "f1", "16"),
                 ("benchmark_segmentation", "segmentation", "f1", ""),
                 ("benchmark_calibration", "linear_probing", "ECE", ""),
                 ("benchmark_adversarial_attack", "adversarial_attack", "f1", "drop")]


def rule_selected(cell):
    """True when this cell sits at ITS OWN run's 1-SE-selected checkpoint.

    Cells are created ahead of the RI curve being fully scored, so several exist at steps
    the rule does not end up choosing once the curve fills in (e.g. midnight seed 1 has a
    cell at 150 from before its backfill, but the completed curve selects 100). Averaging
    those in alongside the selected ones would mix checkpoints across seeds and inflate the
    SD with step choice rather than seed noise.
    """
    text = (CELLS / cell / "model.py").read_text()
    run = re.search(r'^RUN = "(.*)"$', text, re.M).group(1)
    step = int(re.search(r'^STEP = "(.*)"$', text, re.M).group(1).replace("step_", ""))
    run_dir = REPO / "runs" / run
    if not (run_dir / "ri_curve.json").exists():
        return False
    import final_recipe_report as frr
    return frr.select_step_1se(run_dir, 0.007)[0] == step


def ri_of(cell):
    vals = []
    for ds in RI_DATASETS:
        p = OUT / "pathorob/results/robustness_index" / f"{cell}_clsmean" / ds / "-1_0/results_summary.json"
        if not p.exists():
            return None
        vals.append(json.loads(p.read_text())["robustness_index"])
    return sum(vals) / len(vals)


def hest_of(cell):
    text = (CELLS / cell / "model.py").read_text()
    run = re.search(r'^RUN = "(.*)"$', text, re.M).group(1)
    step = re.search(r'^STEP = "(.*)"$', text, re.M).group(1).replace("step_", "")
    if not run:
        hits = glob.glob(f"{HEST_RES}/base_*_summary.json")
    else:
        hits = glob.glob(f"{HEST_RES}/f5_{re.sub(r'.r[0-9]+$', '', run)}_s{step}_*_summary.json")
    if not hits:
        return None
    d = json.loads(Path(sorted(hits)[0]).read_text())
    return d.get("hest_perf_per_encoder", {}).get("custom_encoder")


def thunder_of(rows, cell):
    return {bench: rows.get(cell, {}).get((bench, task, metric, setting))
            for bench, task, metric, setting in THUNDER_TASKS}


def load_thunder():
    rows = {}
    csv_path = OUT / "thunder/outputs/res/results.csv"
    if not csv_path.exists():
        return rows
    with csv_path.open() as h:
        for r in csv.DictReader(h):
            try:
                v = float(r["metric_score"])
            except (TypeError, ValueError):
                continue
            rows.setdefault(r["model"].removesuffix("_optimized"), {})[
                (r["dataset"], r["task"], r["metric"], r["setting"])] = v
    return rows


def fmt(vals, base, higher=True):
    """per-seed list -> 'mean +/- sd (n)  gain  = k SD'."""
    have = [v for v in vals if v is not None]
    if not have:
        return "--", "--"
    m = statistics.mean(have)
    sd = statistics.stdev(have) if len(have) > 1 else None
    cell = f"{m:.4f}" if sd is None else f"{m:.4f} +/- {sd:.4f}"
    cell += f" (n={len(have)})"
    if base is None:
        return cell, "--"
    gain = m - base
    if sd is None:
        return cell, f"{gain:+.4f} (n=1, no SD)"
    if sd == 0:
        # Every seed landed on the same value. That is a real observation (the metric is
        # coarse enough that seeds do not separate), not a missing one -- but the ratio is
        # undefined, so say which it is rather than reusing the n=1 label.
        return cell, f"{gain:+.4f} (SD=0 across seeds)"
    k = abs(gain) / sd
    # Past ~10 SD the ratio stops being informative -- it says the seeds agree closely,
    # not that the effect is 40x more certain than a 20-SD one. Report it as a bound.
    return cell, (f"{gain:+.4f} = {k:.1f} SD" if k < 10 else f"{gain:+.4f} = >10 SD")


def main():
    rows = load_thunder()
    cells = sorted(d.name for d in CELLS.iterdir()
                   if d.is_dir() and (d / "model.py").exists())
    lines = ["# Across-seed mean +/- SD, shipped recipe", "",
             "**Generated file -- do not hand-edit.**  Regenerate with:", "",
             "```", "./.venv/bin/python scripts/seed_stats.py", "```", "",
             "Significance is judged against the SD of this recipe's own seeds, not against a",
             "floor imported from another arm or panel.  Each seed contributes its own 1-SE",
             "selected checkpoint, so the SD includes selection variance -- part of the",
             "procedure, therefore part of the error bar.", "",
             "`n=2` gives a spread rather than an SD; cells are marked with their n and",
             "anything below n=3 should be read as provisional.", ""]

    for bb in BACKBONES:
        base = BASE_CELL.get(bb, f"{bb}-base-control")
        seeds = sorted(c for c in cells
                       if c.startswith(f"{bb}-c50-s") and c != base and rule_selected(c))
        lines += [f"## {bb}", "",
                  f"seed cells: {', '.join(s.replace(bb + '-', '') for s in seeds) or 'none'}", "",
                  "| metric | base | tuned (mean +/- SD) | gain |", "|---|---|---|---|"]
        rb = ri_of(base) or base_from_verdict(bb, "RI")
        c, g = fmt([ri_of(s) for s in seeds], rb)
        lines.append(f"| PathoROB RI | {'--' if rb is None else f'{rb:.4f}'} | {c} | {g} |")
        hb = base_from_verdict(bb, "HEST")
        c, g = fmt([hest_of(s) for s in seeds], hb)
        lines.append(f"| HEST | {'--' if hb is None else f'{hb:.4f}'} | {c} | {g} |")
        for bench, task, metric, setting in THUNDER_TASKS:
            k = (bench, task, metric, setting)
            b = rows.get(base, {}).get(k)
            c, g = fmt([rows.get(s, {}).get(k) for s in seeds], b)
            name = bench.replace("benchmark_", "")
            lines.append(f"| THUNDER {name} | {'--' if b is None else f'{b:.1f}'} | {c} | {g} |")
        lines.append("")

    dest = REPO / "docs/seed_stats.md"
    dest.write_text("\n".join(lines))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
