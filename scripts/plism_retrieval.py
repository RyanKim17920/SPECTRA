#!/usr/bin/env python3
"""Held-out PLISM cross-scanner / cross-stain retrieval, base -> fine-tuned (c50).

This is embed_probe.py's own instrument, NOT PathoROB's robustness_index: scripts/
eval_common.py already retired a false claim that probe_before.json feeds RI (F-F fix,
2026-08-26) -- that file has no `robustness_index` field. What it DOES carry is
groups.heldout.{cross_scanner,cross_stain}.embedding.top1: among the held-out PLISM
scanner/stain conditions (GT450/S210 scanners, HRH/KR/MY stains -- never seen during
training), can the model retrieve the SAME tile's embedding as its nearest neighbour
under a different acquisition condition, out of a pool of `n_pairs` sampled tile pairs.
top1=1.0 is perfect retrieval; the "random" field in the raw JSON is the null (different
tile, same condition pair) and separation = matched - random, but this table follows
the paper and reports top1 alone since that is the headline number quoted for the
shipped arm.

BASE (untuned backbone) values come from a `probe_before.json` (adapter=None,
checkpoint=None) -- deterministic given (backbone, conditions_used.json, n_tiles=256,
seed=1234), so ANY run of that backbone reproduces the identical numbers; verified
exactly for phikon2/midnight/virchow2 (their genMASK-c50 runs' OWN probe_step_*.json
heldout n_pairs [73, 171] and top1 values agree with the finalgem probe_before.json to
full float precision). hoptimus0/uni2h/virchow1/openmidnightsq were bound locally AFTER
the finalgem/gridcmp era, so no run of theirs ever wrote a probe_before.json; those four
were computed directly with `scripts/plism_base_probe_4bb.sbatch` (same embed_probe.py,
same pinned conditions_used.json, no adapter) into
$SPECTRA_RUNS/plism_base_probes/<key>_probe_before.json.

TUNED (c50, shipped recipe) values come from each seed cell's OWN probe_step_<STEP>.json
at that cell's 1-SE-selected checkpoint (`RUN`/`STEP` read from
pathfm-cells/<cell>/model.py, same convention as seed_stats.py / cptac_table.py) --
never a fixed shared step across seeds.

    ./.venv-hest/bin/python scripts/plism_retrieval.py
      -> docs/plism_retrieval.md, waiv-asci/tables/plism_retrieval.tex
"""
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import PAPER_TABLES, RUNS  # noqa: E402

import seed_stats as ss

REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "runs"
TEX_OUT = PAPER_TABLES / "plism_retrieval.tex"
BASE_PROBE_DIR = RUNS / "plism_base_probes"

NAME = {"phikon2": "Phikon-v2", "midnight": "Midnight-12k", "virchow2": "Virchow2",
        "hoptimus0": "H-optimus-0", "uni2h": "UNI2-h",
        "virchow1": "Virchow", "openmidnightsq": "OpenMidnight"}
ORDER = ["phikon2", "midnight", "virchow2", "hoptimus0", "uni2h", "virchow1", "openmidnightsq"]

# probe_before.json elsewhere on disk (finalgem era) for the three backbones that had
# one; the other four are read from BASE_PROBE_DIR below. Any run of a given backbone
# reproduces identical numbers (see module docstring), so a single representative run
# is sufficient and is not "cherry-picked" -- it is the only free variable that does
# NOT vary the result.
BASE_PROBE_BEFORE = {
    "phikon2": str(RUNS / "finalgem-phikon-384585/probe_before.json"),
    "midnight": str(RUNS / "finalgem-midnight-384586/probe_before.json"),
    "virchow2": str(RUNS / "finalgem-virchow2-384587/probe_before.json"),
}
BASE_PROBE_LOCAL = {
    "hoptimus0": BASE_PROBE_DIR / "hoptimus0_probe_before.json",
    "uni2h": BASE_PROBE_DIR / "uni2h_probe_before.json",
    "virchow1": BASE_PROBE_DIR / "virchow1_probe_before.json",
    "openmidnightsq": BASE_PROBE_DIR / "openmidnightsq_probe_before.json",
}


def heldout_top1(probe_json):
    """(cross_scanner top1, cross_stain top1) from a probe_*.json, or (None, None)."""
    p = Path(probe_json)
    if not p.exists():
        return None, None
    h = json.loads(p.read_text()).get("groups", {}).get("heldout", {})
    scanner = h.get("cross_scanner.embedding", {}).get("top1")
    stain = h.get("cross_stain.embedding", {}).get("top1")
    return scanner, stain


def base_of(bb):
    if bb in BASE_PROBE_BEFORE:
        return heldout_top1(BASE_PROBE_BEFORE[bb])
    return heldout_top1(BASE_PROBE_LOCAL[bb])


def cell_run_step(cell):
    text = (ss.CELLS / cell / "model.py").read_text()
    run = re.search(r'^RUN = "(.*)"$', text, re.M).group(1)
    step = re.search(r'^STEP = "(.*)"$', text, re.M).group(1)
    return run, step


def tuned_of(cell):
    run, step = cell_run_step(cell)
    if not run or not step:
        return None, None
    return heldout_top1(RUNS / run / f"probe_step_{step.replace('step_', '')}.json")


def fmt_pair(vals):
    have = [v for v in vals if v is not None]
    if not have:
        return None, None, 0
    m = statistics.mean(have)
    # TWO sample SDs, matching the convention already used by the base->tuned figure
    # (waiv-asci/scripts/make_base_to_tuned.py) and its table inputs.
    sd = 2 * statistics.stdev(have) if len(have) > 1 else None
    return m, sd, len(have)


def main():
    cells = sorted(d.name for d in ss.CELLS.iterdir()
                   if d.is_dir() and (d / "model.py").exists())

    md = ["# Held-out PLISM retrieval (top-1): base -> fine-tuned (c50)", "",
          "**Generated file -- do not hand-edit.**  Regenerate with:", "",
          "```", "./.venv-hest/bin/python scripts/plism_retrieval.py", "```", "",
          "Top-1 retrieval of the SAME tile across a held-out scanner or stain switch",
          "(GT450/S210 scanners, HRH/KR/MY stains -- never seen during training), CLS+",
          "mean-patch (`clsmean`) embedding space, `n_tiles=256`, seed 1234. Base is the",
          "untrained backbone's `probe_before.json` (adapter=None); tuned is the mean +/-",
          "2 SD over the shipped c50 recipe's seeds at EACH seed's own 1-SE-selected",
          "checkpoint (matching `docs/seed_stats.md`), not a fixed shared step.",
          "See the script docstring for the base-value provenance per backbone.", "",
          "| backbone | axis | base | tuned (mean +/- 2 SD) | gain | n |",
          "|---|---|---|---|---|---|"]
    # Compact form: ONE ROW PER BACKBONE, both axes side by side. n is constant (3) and the
    # SD range is small, so both live in the caption instead of costing two columns.
    tex = [r"\begin{tabular}{lcc}", r"\toprule",
           r" & \multicolumn{2}{c}{top-1 retrieval, base $\to$ fine-tuned} \\",
           r"\cmidrule(lr){2-3}",
           r"Backbone & across scanners & across stains \\", r"\midrule"]

    for bb in ORDER:
        base_cell = ss.BASE_CELL.get(bb, f"{bb}-base-control")
        bscan, bstain = base_of(bb)
        seeds = sorted(c for c in cells
                       if c.startswith(f"{bb}-c50-s") and c != base_cell
                       and ss.rule_selected(c))
        scan_vals = [tuned_of(s)[0] for s in seeds]
        stain_vals = [tuned_of(s)[1] for s in seeds]

        row = []
        for axis, base, vals in (("cross-scanner", bscan, scan_vals),
                                  ("cross-stain", bstain, stain_vals)):
            m, sd, n = fmt_pair(vals)
            bstr = "--" if base is None else f"{base:.4f}"
            if m is None:
                md.append(f"| {NAME[bb]} | {axis} | {bstr} | -- | -- | 0 |")
                row.append("--")
                continue
            tuned_md = f"{m:.4f}" if sd is None else f"{m:.4f} +/- {sd:.4f}"
            gain_md = "--" if base is None else f"{m - base:+.4f}"
            md.append(f"| {NAME[bb]} | {axis} | {bstr} | {tuned_md} | {gain_md} | {n} |")
            # three decimals is enough at this precision and keeps the cell narrow
            tuned_tex = f"{m:.3f}"
            if base is None:
                row.append(tuned_tex)
            else:
                d = m - base
                mark = "\\gain" if d > 0 else ("\\loss" if d < 0 else "")
                cell = f"{base:.3f}$\\to${tuned_tex}"
                row.append(f"{cell[:cell.index('$')]}$\\to${mark}{{{tuned_tex}}}"
                           if mark else cell)
        tex.append(f"{NAME[bb]} & " + " & ".join(row) + " \\\\")

    tex += [r"\bottomrule", r"\end{tabular}"]
    md.append("")

    (REPO / "docs/plism_retrieval.md").write_text("\n".join(md) + "\n")
    TEX_OUT.parent.mkdir(parents=True, exist_ok=True)
    TEX_OUT.write_text("\n".join(tex) + "\n")
    print("wrote docs/plism_retrieval.md and", TEX_OUT)
    for line in md[md.index("|---|---|---|---|---|---|") + 1:]:
        if line.strip():
            print(line)


if __name__ == "__main__":
    main()
