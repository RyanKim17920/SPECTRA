#!/usr/bin/env python3
"""A seed-noise floor for the 16-dataset THUNDER panel.

`docs/thunder_seed_floor_12ds.md` measures the floor over the 12-set panel. A 16-set mean
averages more datasets and therefore has a different variance, so those numbers cannot be
carried across -- and without a floor the small deltas in `docs/thunder16.md` (most of the
H-Optimus-0 and UNI2-h rows sit inside +/-0.5) cannot be told apart from noise.

The estimate here is the paired difference between two arms that differ ONLY by training
seed: `c3s-s0` and `c3s-s1` at the same step, which exist for all five backbones. It is a
single paired difference per (backbone, task), so it is an order-of-magnitude scale, not a
standard error with a confidence interval -- reported as such. It also comes from the c3s
generation rather than the final c50 recipe; the c50 seed replicates are still on the queue,
and when they land this script reads them instead by extending SEED_PAIRS.

    ./.venv/bin/python scripts/thunder16_floor.py   ->  docs/thunder16_floor.md
"""
from pathlib import Path

from thunder16 import BACKBONES, RESULTS, SETTING, TASKS, load

REPO = Path(__file__).resolve().parent.parent

# (backbone, arm_seed0, arm_seed1) -- identical except the training seed.
SEED_PAIRS = [
    ("phikon2", "c3s-s0-step250", "c3s-s1-step250"),
    ("midnight", "c3s-s0-step125", "c3s-s1-step125"),
    ("virchow2", "c3s-s0-step125", "c3s-s1-step125"),
    ("hoptimus0", "c3s-s0-step125", "c3s-s1-step125"),
    ("uni2h", "c3s-s0-step125", "c3s-s1-step125"),
]


def main():
    rows, _ = load()
    lines = [
        "# THUNDER 16-set panel -- seed-noise floor",
        "",
        "**Generated file -- do not hand-edit.**  Regenerate with:",
        "",
        "```",
        "./.venv/bin/python scripts/thunder16_floor.py",
        "```",
        "",
        "One paired difference between two arms identical except for the training seed, per",
        "(backbone, task), over the 16-dataset panel.  Read it as the scale below which a",
        "delta in `docs/thunder16.md` is not interpretable -- **not** as a standard error:",
        "n=2 supports a magnitude, not an interval.",
        "",
        "Measured on the c3s generation, the only arms with a seed replicate on disk today.",
        "The c50 replicates are queued; when they land, extend `SEED_PAIRS` and re-run.",
        "",
        "| backbone | pair | " + " | ".join(b.replace("benchmark_", "") for b, _, _, _ in TASKS) + " |",
        "|---|---|" + "---|" * len(TASKS),
    ]
    per_task = {b: [] for b, _, _, _ in TASKS}
    for bb, a0, a1 in SEED_PAIRS:
        cells = []
        for bench, task, metric, _ in TASKS:
            k = (bench, task, metric, SETTING.get(task, ""))
            v0 = rows.get(f"{bb}-{a0}", {}).get(k)
            v1 = rows.get(f"{bb}-{a1}", {}).get(k)
            if v0 is None or v1 is None:
                cells.append("--")
                continue
            d = abs(v0 - v1)
            per_task[bench].append(d)
            cells.append(f"{d:.2f}")
        lines.append(f"| {bb} | {a0} vs {a1} | " + " | ".join(cells) + " |")

    lines += ["", "| task | max seed gap across backbones | n backbones |", "|---|---|---|"]
    for bench, _, _, _ in TASKS:
        vals = per_task[bench]
        lines.append(f"| {bench.replace('benchmark_', '')} | "
                     + (f"{max(vals):.2f}" if vals else "MISSING")
                     + f" | {len(vals)} |")
    lines += [
        "",
        "## How to use this",
        "",
        "A final-recipe delta in `docs/thunder16.md` smaller than its own (backbone, task)",
        "gap above is inside seed noise and must not be reported as an effect.  The floor",
        "varies by backbone -- it is not one constant -- so compare each cell against its own",
        "row, never against the cross-backbone maximum.",
        "",
    ]
    dest = REPO / "docs/thunder16_floor.md"
    dest.write_text("\n".join(lines))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
