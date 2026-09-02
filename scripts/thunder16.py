#!/usr/bin/env python3
"""Full THUNDER, our base vs our fine-tuned, from a single corpus.

This replaces the percent-of-Waiv framing.  The claim is an absolute delta measured in
one harness, not a ratio to a number we did not compute, so there is no external target
column here and no pass/fail grade derived from one.

Two things this fixes over the older collectors:

  * ALL 16 classification datasets.  `collect_final5.PAPER_CLS` defaults to the 12
    THUNDER-paper sets and treats the 4 SPIDER sets as an extra; that split only ever
    existed to line our average up with an external 16-set average.  THUNDER's own
    `benchmark_*` roll-up rows already average all 16, so we read those directly.
  * ALL SIX tasks, calibration and the PGD attack included.  The old harness computed
    neither, which is why the earlier report carried four.

Source: /data/ryan.kim/pathfm-full-evals/thunder/outputs/res/results.csv -- the second
corpus, which uses the leaderboard's Resize(256, bicubic) transform.  The older
/data/ryan.kim/thunder corpus is NOT read: it scored SPIDER for only 10 of our runs and
preprocesses with Resize(224, bilinear), so it cannot support a 16-set average.

    ./.venv/bin/python scripts/thunder16.py   ->  docs/thunder16.md
"""
import csv
from collections import defaultdict
from pathlib import Path

RESULTS = Path("/data/ryan.kim/pathfm-full-evals/thunder/outputs/res/results.csv")
REPO = Path(__file__).resolve().parent.parent

CLS16 = ["bach", "bracs", "break_his", "ccrcc", "crc", "esca", "mhist", "patch_camelyon",
         "spider_breast", "spider_colorectal", "spider_skin", "spider_thorax",
         "tcga_crc_msi", "tcga_tils", "tcga_uniform", "wilds"]
SEG4 = ["ocelot", "pannuke", "segpath_epithelial", "segpath_lymphocytes"]

# (roll-up row, task, metric, higher-is-better).  Calibration is an ECE emitted by the
# linear-probing stage, which is why its task column reads linear_probing.
TASKS = [
    ("benchmark_knn", "knn", "f1", True),
    ("benchmark_linear_probing", "linear_probing", "f1", True),
    ("benchmark_simple_shot", "simple_shot", "f1", True),
    ("benchmark_segmentation", "segmentation", "f1", True),
    ("benchmark_calibration", "linear_probing", "ECE", False),
    ("benchmark_adversarial_attack", "adversarial_attack", "f1", False),
]
BACKBONES = ["phikon2", "midnight", "virchow2", "hoptimus0", "uni2h", "openmidnightsq", "virchow1", "virchow2f"]

# The final recipe's own arm at its 1-SE-selected operating step.  The delta reported is
# THIS arm's, not the best arm's: a max taken over seven arms -- which include the
# superseded c3s generation and the bm3 arm the ablation shows losing -- is a selection
# effect, not a result the recipe can claim.
FINAL_ARM = {"phikon2": "c50-s0-step200", "midnight": "c50-s0-step150",
             "virchow2": "c50-s0-step100", "hoptimus0": "c50-s0-step100",
             "uni2h": "c50-s0-step100",
             "openmidnightsq": "c50-s0-step150", "virchow1": "c50-s0-step150",
             "virchow2f": "c50-s0-step100"}
BASE_CELL = {"virchow2f": "virchow2-basectrl-fp32adv"}

# Virchow2's PGD column is excluded, and the reason is internal rather than external: all
# three Virchow2 models lose 0.1-0.3pp of f1 under the attack where every other backbone
# loses 19-32pp.  A near-zero drop measures an attack that did not land, not a robust
# model, so averaging it in would read as robustness we did not demonstrate.
ADV_DEGENERATE = {"virchow2"}


def load():
    rows = defaultdict(dict)
    per_ds = defaultdict(dict)
    with RESULTS.open() as handle:
        for r in csv.DictReader(handle):
            model = r["model"].removesuffix("_optimized")
            try:
                score = float(r["metric_score"])
            except (TypeError, ValueError):
                continue
            key = (r["dataset"], r["task"], r["metric"], r["setting"])
            rows[model][key] = score
            if r["dataset"] in CLS16 + SEG4 and r["metric"] == "f1":
                per_ds[(model, r["task"], r["setting"])][r["dataset"]] = score
    return rows, per_ds


# The `setting` column is not empty for every task: the attack rows split clean /
# adversarial / drop and we want the drop, and simple_shot carries its shot count there.
# Matching on an empty setting for all six tasks silently reports simple_shot MISSING.
SETTING = {"adversarial_attack": "drop", "simple_shot": "16"}


RES_DIR = RESULTS.parent


def probes_on_disk(model, task):
    """True when this cell has per-dataset outputs for `task` but the roll-up has not been
    written yet.  Distinguishes "measured, awaiting the summary stage" from "not run"."""
    panel = SEG4 if task == "segmentation" else CLS16
    n = sum((RES_DIR / d / f"{model}_optimized" / task / "frozen" / "outputs.json").exists()
            for d in panel)
    return n == len(panel)


def get(rows, model, bench, task, metric):
    """Harness roll-up only.

    The per-dataset outputs.json files are NOT averaged here as a fallback.  knn and
    simple_shot store a single key and would be safe, but linear_probing stores 14 and
    segmentation 6 -- the harness applies its own selection across those, and guessing it
    would put a number in the paper that no longer matches what THUNDER reports.  A cell
    whose probes are done but whose summary stage has not run yet is marked PENDING
    instead, so a blank is never mistaken for a measurement that failed.
    """
    return rows.get(model, {}).get((bench, task, metric, SETTING.get(task, "")))


def main():
    rows, per_ds = load()
    models = sorted(rows)
    lines = [
        "# Full THUNDER -- our base vs our fine-tuned",
        "",
        "**Generated file -- do not hand-edit.**  Regenerate with:",
        "",
        "```",
        "./.venv/bin/python scripts/thunder16.py",
        "```",
        "",
        f"Source: `{RESULTS}`.  All six THUNDER tasks over all **16** classification",
        "datasets and 4 segmentation datasets, read from the harness's own `benchmark_*`",
        "roll-up rows.  No external target: every column below is ours, measured in one",
        "harness, so the quantity of interest is the base-to-tuned delta.",
        "",
        "Calibration is ECE and the adversarial column is the PGD f1 *drop*; for both,",
        "**lower is better**, so a negative delta is an improvement.",
        "",
    ]

    for bb in BACKBONES:
        base = BASE_CELL.get(bb, f"{bb}-base-control")
        arms = [m for m in models if m.startswith(f"{bb}-") and m != base]
        if base not in rows:
            lines += [f"## {bb}", "", "TODO: no base-control on disk -- no delta can be "
                      "expressed for this backbone.", ""]
            continue
        lines += [f"## {bb}", "",
                  "| task | base | " + " | ".join(a.replace(f"{bb}-", "") for a in arms)
                  + f" | delta ({FINAL_ARM[bb]}) |",
                  "|---|---|" + "---|" * (len(arms) + 1)]
        for bench, task, metric, higher in TASKS:
            if task == "adversarial_attack" and bb in ADV_DEGENERATE:
                lines.append(f"| {bench.replace('benchmark_', '')} | EXCLUDED | "
                             + " | ".join("EXCLUDED" for _ in arms)
                             + " | attack degenerate on this backbone (see note) |")
                continue
            b = get(rows, base, bench, task, metric)
            final = get(rows, f"{bb}-{FINAL_ARM[bb]}", bench, task, metric)
            cells = []
            for a in arms:
                v = get(rows, a, bench, task, metric)
                if v is None:
                    cells.append("PENDING" if probes_on_disk(a, task) else "--")
                else:
                    cells.append(f"{v:.1f}")
            if b is None:
                lines.append(f"| {bench.replace('benchmark_', '')} | MISSING | "
                             + " | ".join(cells) + " | MISSING |")
                continue
            if final is None:
                tail = " | PENDING |" if probes_on_disk(f"{bb}-{FINAL_ARM[bb]}", task) else " | -- |"
            else:
                d = final - b
                # ECE and the PGD drop are lower-is-better, so mark whether the recipe's
                # delta is an improvement rather than leaving the sign to be misread.
                # A delta that rounds to zero is flat, not an improvement: labelling
                # +0.0 as "better" reads as a win the measurement does not support.
                verdict = "flat" if abs(d) < 0.05 else ("better" if (d > 0) == higher
                                                        else "worse")
                tail = f" | {d:+.1f} ({verdict}) |"
            lines.append(f"| {bench.replace('benchmark_', '')} | {b:.1f} | "
                         + " | ".join(cells) + tail)
        lines.append("")
        got = {d for a in arms for d in per_ds.get((a, "knn", ""), {})}
        lines += [f"Classification datasets scored for this backbone: "
                  f"{len(got & set(CLS16))}/16.", ""]

    lines += [
        "## Notes",
        "",
        "* **Virchow2 adversarial is excluded.** Its three models lose 0.1-0.3pp of f1",
        "  under PGD where the other four backbones lose 19-32pp. That is an attack that",
        "  failed to land, not a robust representation; reporting it would claim",
        "  robustness we did not measure.",
        "* **Seed floors do not carry over from the 12-set panel.** The floors in",
        "  `docs/thunder_seed_floor_12ds.md` were measured over 12 datasets; a 16-set",
        "  average has a different variance. TODO: re-measure the per-(backbone, task)",
        "  floor on the 16-set panel before calling any delta above significant.",
        "* `PENDING` means every per-dataset probe for that cell is on disk but THUNDER's",
        "  summary stage has not written the roll-up yet; `--` means the task has not run.",
        "* Single-seed cells are not marked here; see `docs/eval_matrix.md` for which",
        "  (backbone, arm) pairs have a second seed on disk.",
        "",
    ]

    dest = REPO / "docs/thunder16.md"
    dest.write_text("\n".join(lines))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
