#!/usr/bin/env python3
"""CPTAC (Patho-Bench subset) macro-OvR AUC, base -> fine-tuned, per backbone.

Unlike PathoROB / HEST / THUNDER there is no published per-model Patho-Bench leaderboard
to rank against, so this table reports the base -> fine-tuned movement only.

Cell selection is imported from seed_stats rather than re-implemented: the same
`rule_selected` 1-SE checkpoint filter and the same `{bb}-base-control` convention, so
this table cannot silently drift from the other generated tables.

    ./.venv/bin/python scripts/cptac_table.py
      -> docs/cptac.md, waiv-asci/tables/cptac.tex

    ./.venv/bin/python scripts/cptac_table.py --per-task
      -> waiv-asci/tables/cptac_per_task.tex   (does not touch the default outputs)
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import EVALS, PAPER_TABLES  # noqa: E402

import seed_stats as ss

REPO = Path(__file__).resolve().parent.parent
OUT = EVALS
TEX_OUT = PAPER_TABLES / "cptac.tex"
PER_TASK_TEX_OUT = PAPER_TABLES / "cptac_per_task.tex"
METRIC = "mean_classification_macro_ovr_auc"
PER_TASK_METRIC = "classification_macro_ovr_auc"
# base -> fine-tuned aggregate values this table's "Mean (38)" row must reproduce
# (from waiv-asci/tables/hest_cptac.tex, CPTAC AUC column).
HEST_CPTAC_MEAN = {"phikon2": (0.6491, 0.6870), "midnight": (0.6643, 0.6898),
                    "virchow2": (0.6789, 0.6879), "hoptimus0": (0.6728, 0.6895),
                    "uni2h": (0.6750, 0.6980), "virchow1": (0.6608, 0.6839),
                    "openmidnightsq": (0.6561, 0.6844)}

NAME = {"phikon2": "Phikon-v2", "midnight": "Midnight-12k", "virchow2": "Virchow2",
        "hoptimus0": "H-optimus-0", "uni2h": "UNI2-h",
        "virchow1": "Virchow", "openmidnightsq": "OpenMidnight"}
ORDER = ["phikon2", "midnight", "virchow2", "hoptimus0", "uni2h", "virchow1", "openmidnightsq"]


def cptac_of(cell):
    p = OUT / "cptac" / cell / "aggregate.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get(METRIC)


def n_tasks(cell):
    p = OUT / "cptac" / cell / "aggregate.json"
    if not p.exists():
        return None
    return len(json.loads(p.read_text()).get("classification_macro_ovr_auc", {}))


def per_task_of(cell):
    """Per-task macro-OvR AUC dict (task key -> value), or None if missing."""
    p = OUT / "cptac" / cell / "aggregate.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get(PER_TASK_METRIC)


def _task_label(key):
    cohort, task = key.split("/", 1)
    cohort = cohort.replace("cptac_", "").upper()
    return cohort, f"{cohort} / {task.replace('_', ' ')}"


def gen_per_task():
    cells = sorted(d.name for d in ss.CELLS.iterdir()
                   if d.is_dir() and (d / "model.py").exists())

    per_bb = {}
    task_order = None
    for bb in ORDER:
        base_cell = ss.BASE_CELL.get(bb, f"{bb}-base-control")
        seeds = sorted(c for c in cells
                       if c.startswith(f"{bb}-c50-s") and c != base_cell
                       and ss.rule_selected(c))
        base_pt = per_task_of(base_cell)
        seed_pts = [d for d in (per_task_of(s) for s in seeds) if d is not None]
        if base_pt is None:
            raise SystemExit(f"missing base aggregate.json for {bb} ({base_cell})")
        if task_order is None:
            task_order = list(base_pt.keys())
        elif list(base_pt.keys()) != task_order:
            raise SystemExit(f"{bb}: task key set/order differs from {ORDER[0]}")
        per_bb[bb] = (base_pt, seed_pts)

    n_cohorts = {}
    for k in task_order:
        c, _ = _task_label(k)
        n_cohorts[c] = n_cohorts.get(c, 0) + 1

    n_improve = 0
    n_regress = 0
    per_bb_improve = {bb: 0 for bb in ORDER}

    lines = [r"\begin{tabular}{l" + "c" * len(ORDER) + "}", r"\toprule",
             "Task & " + " & ".join(NAME[bb] for bb in ORDER) + r" \\", r"\midrule"]

    prev_cohort = None
    bb_sums = {bb: [0.0, 0.0] for bb in ORDER}  # base_sum, tuned_sum (over 38 tasks)
    for key in task_order:
        cohort, label = _task_label(key)
        if prev_cohort is not None and cohort != prev_cohort:
            lines.append(r"\midrule")
        prev_cohort = cohort

        cells_tex = []
        for bb in ORDER:
            base_pt, seed_pts = per_bb[bb]
            b = base_pt[key]
            vals = [d[key] for d in seed_pts if key in d]
            m = statistics.mean(vals) if vals else None
            bb_sums[bb][0] += b
            if m is not None:
                bb_sums[bb][1] += m
                delta = m - b
                if delta > 1e-12:
                    n_improve += 1
                    per_bb_improve[bb] += 1
                elif delta < -1e-12:
                    n_regress += 1
                tuned = f"{m:.3f}"
                tuned = f"\\gain{{{tuned}}}" if delta > 0 else (
                    f"\\loss{{{tuned}}}" if delta < 0 else tuned)
                cells_tex.append(f"{b:.3f}$\\to${tuned}")
            else:
                cells_tex.append(f"{b:.3f}$\\to$--")
        lines.append(f"{label} & " + " & ".join(cells_tex) + r" \\")

    lines.append(r"\midrule")
    mean_cells = []
    n_tasks_total = len(task_order)
    for bb in ORDER:
        b_mean = bb_sums[bb][0] / n_tasks_total
        t_mean = bb_sums[bb][1] / n_tasks_total
        want_b, want_t = HEST_CPTAC_MEAN[bb]
        assert abs(b_mean - want_b) < 5e-5, (bb, "base", b_mean, want_b)
        assert abs(t_mean - want_t) < 5e-5, (bb, "tuned", t_mean, want_t)
        delta = t_mean - b_mean
        tuned = f"{t_mean:.3f}"
        tuned = f"\\gain{{{tuned}}}" if delta > 0 else (f"\\loss{{{tuned}}}" if delta < 0 else tuned)
        mean_cells.append(f"{b_mean:.3f}$\\to${tuned}")
    lines.append(f"Mean ({n_tasks_total}) & " + " & ".join(mean_cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    header = ["% Generated by scripts/cptac_table.py --per-task -- do not hand-edit."]
    PER_TASK_TEX_OUT.write_text("\n".join(header + lines) + "\n")
    print("wrote", PER_TASK_TEX_OUT)
    print("tasks per cohort:", n_cohorts)
    print(f"cells improving/regressing (of {len(ORDER)}x{n_tasks_total}"
          f"={len(ORDER) * n_tasks_total}): {n_improve} improve, {n_regress} regress, "
          f"{len(ORDER) * n_tasks_total - n_improve - n_regress} flat")
    print("per-backbone improving task count:", per_bb_improve)


def main():
    cells = sorted(d.name for d in ss.CELLS.iterdir()
                   if d.is_dir() and (d / "model.py").exists())

    md = ["# CPTAC (Patho-Bench subset): base -> fine-tuned", "",
          "**Generated file -- do not hand-edit.**  Regenerate with:", "",
          "```", "./.venv/bin/python scripts/cptac_table.py", "```", "",
          f"Metric: `{METRIC}` from each cell's `aggregate.json`.",
          "Seed cells are the 1-SE-selected checkpoints, matching `docs/seed_stats.md`.",
          "There is no published per-model Patho-Bench leaderboard, so no rank column.", "",
          "| backbone | base | fine-tuned (mean +/- SD) | gain | n | tasks |",
          "|---|---|---|---|---|---|"]
    tex = [r"\begin{tabular}{lcccc}", r"\toprule",
           r"Backbone & Base AUC & Fine-tuned AUC & $\Delta$ & $n$ \\", r"\midrule"]

    n_task_seen = set()
    for bb in ORDER:
        base_cell = ss.BASE_CELL.get(bb, f"{bb}-base-control")
        seeds = sorted(c for c in cells
                       if c.startswith(f"{bb}-c50-s") and c != base_cell
                       and ss.rule_selected(c))
        b = cptac_of(base_cell)
        vals = [v for v in (cptac_of(s) for s in seeds) if v is not None]
        nt = n_tasks(base_cell)
        if nt is not None:
            n_task_seen.add(nt)
        if b is None or not vals:
            md.append(f"| {NAME[bb]} | {'--' if b is None else f'{b:.4f}'} | -- | -- | 0 | "
                      f"{nt if nt is not None else '--'} |")
            tex.append(f"{NAME[bb]} & {'--' if b is None else f'{b:.4f}'} & -- & -- & 0 \\\\")
            continue
        m = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else None
        tuned_md = f"{m:.4f}" if sd is None else f"{m:.4f} +/- {sd:.4f}"
        tuned_tex = f"${m:.4f}$" if sd is None else f"${m:.4f}\\pm{sd:.4f}$"
        md.append(f"| {NAME[bb]} | {b:.4f} | {tuned_md} | {m - b:+.4f} | {len(vals)} | "
                  f"{nt if nt is not None else '--'} |")
        d = m - b
        dtex = f"${d:+.4f}$"
        dtex = f"\\gain{{{dtex}}}" if d > 0 else (f"\\loss{{{dtex}}}" if d < 0 else dtex)
        tex.append(f"{NAME[bb]} & {b:.4f} & {tuned_tex} & {dtex} & {len(vals)} \\\\")

    tex += [r"\bottomrule", r"\end{tabular}"]
    md.append("")

    (REPO / "docs/cptac.md").write_text("\n".join(md) + "\n")
    TEX_OUT.write_text("\n".join(tex) + "\n")
    print("wrote docs/cptac.md and", TEX_OUT)
    print("task counts seen:", sorted(n_task_seen))
    for line in md[md.index("|---|---|---|---|---|---|") + 1:]:
        if line.strip():
            print(line)


if __name__ == "__main__":
    if "--per-task" in sys.argv[1:]:
        gen_per_task()
    else:
        main()
