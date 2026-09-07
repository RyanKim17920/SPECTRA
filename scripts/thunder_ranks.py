#!/usr/bin/env python3
"""THUNDER leaderboard rank of each backbone, base vs fine-tuned.

The leaderboard prints "79.9 (8)": value and rank among all 32 models in the 16-dataset
panel -- histopathology AND natural-image models, not histopathology alone.  Ranking is
DENSE (ties share a rank, the next distinct value takes the next integer), not competition
ranking.  Both facts are load-bearing: getting either wrong shifts most ranks by 1-4 places.
`validate()` asserts that our rule reproduces all 32 published per-task ranks and all 32
published rank sums before any of our own numbers are ranked, so the table cannot silently
drift from the leaderboard's own methodology.

To rank one of our models we replace that backbone's own published row with our value --
our model IS that backbone, fine-tuned -- and re-rank the 32-model field.

    ./.venv/bin/python scripts/thunder_ranks.py
      -> docs/thunder_ranks.md, waiv-asci/tables/thunder_ranks.tex
"""
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import EVALS, PAPER_TABLES  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
LB_TSV = REPO / "data/thunder_leaderboard_16ds.tsv"
TEX_OUT = PAPER_TABLES / "thunder_ranks.tex"
# Appendix twin: the SAME rows in the uncut format (per-task ranks and +/-2SD kept).
# The main-body table drops both from the six metric cells so it fits \textwidth at
# \footnotesize without \resizebox; nothing is lost, it just moves to the appendix.
TEX_FULL_OUT = PAPER_TABLES / "thunder_ranks_full.tex"

COLS = ["knn", "lin", "few", "seg", "ece", "adv"]
LOWER = {"ece", "adv"}                     # lower is better
# arrows state the direction of "better" for each column, so a reader never has to
# remember that ECE and adversarial drop are lower-is-better.
HDR = ["$k$-NN\\,$\\uparrow$", "Lin.\\ probe\\,$\\uparrow$", "Few-shot\\,$\\uparrow$",
       "Seg.\\,$\\uparrow$", "ECE\\,$\\downarrow$", "Adv.\\ drop\\,$\\downarrow$"]
LBNAME = {"phikon2": "PHIKON2", "midnight": "MIDNIGHT", "virchow2": "VIRCHOW2",
          "hoptimus0": "H-OPTIMUS-0", "uni2h": "UNI2-H",
          "virchow1": "VIRCHOW", "openmidnightsq": "OpenMidnight"}
NAME = {"phikon2": "Phikon-v2", "midnight": "Midnight-12k", "virchow2": "Virchow2",
        "hoptimus0": "H-optimus-0", "uni2h": "UNI2-h",
        "virchow1": "Virchow", "openmidnightsq": "OpenMidnight"}
ORDER = ["phikon2", "midnight", "virchow2", "hoptimus0", "uni2h", "virchow1", "openmidnightsq"]
KEY = {"THUNDER knn": 0, "THUNDER linear_probing": 1, "THUNDER simple_shot": 2,
       "THUNDER segmentation": 3, "THUNDER calibration": 4, "THUNDER adversarial_attack": 5}

# fp32-attack cells (base, tuned mean), READ FROM DISK -- the fp16 adversarial numbers in
# seed_stats.md are not usable (an autocast in our wrapper weakened PGD on every backbone).
# Hardcoding these went stale the moment more cells landed, so they are derived instead.
FP32_CELLS = {
    "phikon2": ("phikon2f-base-control",
                ["phikon2f-c50-s0-step200", "phikon2f-c50-s1-step200", "phikon2f-c50-s2-step200"]),
    "midnight": ("midnightf-base-control",
                 ["midnightf-c50-s0-step150", "midnightf-c50-s1-step100", "midnightf-c50-s3-step100"]),
    "virchow2": ("virchow2-basectrl-fp32adv",
                 ["virchow2f-c50-s0-step100", "virchow2f-c50-s1-step150", "virchow2f-c50-s3-step100"]),
    "hoptimus0": ("hoptimus0f-base-control",
                  ["hoptimus0f-c50-s0-step100", "hoptimus0f-c50-s1-step100", "hoptimus0f-c50-s3-step100"]),
    "uni2h": ("uni2hf-base-control",
              ["uni2hf-c50-s0-step100", "uni2hf-c50-s1-step150", "uni2hf-c50-s2-step100"]),
    # Virchow v1 and OpenMidnight cells were patched IN PLACE with the fp32 attack, so their
    # own cells already carry fp32 adversarial -- no separate `*f-` family exists for them.
    "virchow1": ("virchow1-base-control",
                 ["virchow1-c50-s0-step150", "virchow1-c50-s1-step150", "virchow1-c50-s2-step150"]),
    "openmidnightsq": ("openmidnightsq-base-control",
                       ["openmidnightsq-c50-s0-step150", "openmidnightsq-c50-s1-step150",
                        "openmidnightsq-c50-s2-step150"]),
}


def _fp32_adv():
    """(base, tuned mean) per backbone from the harness rollup; absent until a cell lands."""
    import statistics
    path = EVALS / "thunder/outputs/res/results.csv"
    got = {}
    if path.exists():
        with path.open() as h:
            for r in csv.DictReader(h):
                if r["dataset"] == "benchmark_adversarial_attack" and r["metric"] == "f1":
                    got[r["model"].removesuffix("_optimized")] = float(r["metric_score"])
    out = {}
    for bb, (base, seeds) in FP32_CELLS.items():
        vals = [got[s] for s in seeds if s in got]
        if base in got and vals:
            sd = statistics.stdev(vals) if len(vals) > 1 else None
            out[bb] = (got[base], statistics.mean(vals), sd)
    return out


FP32_ADV = _fp32_adv()

# Published per-task ranks, transcribed alongside the values, used only to validate the rule.
PUB = {
    "UNI2-H": [2, 1, 1, 4, 9, 2], "VIRCHOW2": [3, 4, 12, 1, 9, 1],
    "GenBio-PFM": [1, 2, 2, 8, 12, 5], "H0-mini": [9, 6, 11, 3, 8, 4],
    "MIDNIGHT": [8, 5, 18, 5, 2, 6], "UNI": [7, 8, 3, 10, 8, 10],
    "H-OPTIMUS-1": [4, 3, 4, 15, 6, 20], "KEEP": [5, 9, 5, 9, 10, 15],
    "H-OPTIMUS-0": [6, 6, 8, 13, 10, 14], "GIGAPATH": [10, 10, 9, 16, 5, 11],
    "HIBOU-B": [12, 16, 7, 10, 4, 17], "HIBOU-L": [15, 7, 13, 6, 16, 9],
    "VIRCHOW": [18, 11, 17, 2, 14, 7], "KAIKO-B/16": [14, 15, 6, 12, 18, 8],
    "OpenMidnight": [11, 5, 30, 3, 19, 7], "CONCH": [13, 13, 15, 7, 11, 19],
    "KAIKO-S/16": [16, 14, 10, 12, 15, 12], "CONCH 1.5": [8, 12, 11, 5, 15, 30],
    "PHIKON": [19, 18, 14, 9, 20, 3], "PHIKON2": [20, 19, 17, 11, 9, 13],
    "MUSK": [17, 17, 16, 14, 10, 29], "DINOv3-S": [21, 22, 19, 20, 1, 25],
    "DINOv3-L": [22, 20, 23, 19, 3, 26], "DINOv3-B": [23, 21, 21, 17, 7, 27],
    "ViT-L/16": [26, 25, 28, 18, 13, 14], "ViT-B/16": [28, 26, 27, 21, 10, 16],
    "PLIP": [25, 27, 22, 26, 14, 21], "QUILTNET": [24, 28, 20, 25, 20, 18],
    "DINOv2-L": [23, 23, 25, 24, 17, 23], "DINOv2-B": [27, 24, 24, 23, 18, 24],
    "CLIP-L/14": [29, 29, 26, 22, 8, 28], "CLIP-B/32": [30, 30, 29, 27, 17, 22],
}


def leaderboard():
    rows = list(csv.DictReader(
        (l for l in LB_TSV.read_text().splitlines() if not l.startswith("#")), delimiter="\t"))
    vals = {r["model"]: [float(r[c]) for c in COLS] for r in rows}
    ranksum = {r["model"]: int(r["ranksum"]) for r in rows}
    return vals, ranksum


def dense_rank(value, i, field):
    """Dense rank of `value` on task i within `field` (ties share a rank)."""
    distinct = sorted({v[i] for v in field.values()}, reverse=(COLS[i] not in LOWER))
    return distinct.index(value) + 1


def validate(vals, ranksum):
    for m, vs in vals.items():
        got = [dense_rank(v, i, vals) for i, v in enumerate(vs)]
        assert got == PUB[m], f"rank rule wrong for {m}: {got} != {PUB[m]}"
        assert sum(got) == ranksum[m], f"rank sum wrong for {m}: {sum(got)} != {ranksum[m]}"
    return len(vals)


def substituted(vals, bb, replacement):
    """The 32-model field with this backbone's row replaced by our values."""
    field = dict(vals)
    field[LBNAME[bb]] = replacement
    return field


def tuned_means():
    return {bb: {i: c[1] for i, c in d.items()} for bb, d in our_cells().items()}


def our_cells():
    """{backbone: {task index: (base, tuned mean, sd, n)}} from docs/seed_stats.md.

    Both terms are OUR measurement in OUR harness, so the pair is comparable. They differ
    from the published leaderboard row by at most 0.1 on classification (5 of 15 cells),
    which is the rounding of a one-decimal published value; see the caption note."""
    text = (REPO / "docs/seed_stats.md").read_text()
    out, bb = {}, None
    for line in text.splitlines():
        m = re.match(r"## (\w+)", line)
        if m:
            bb = m.group(1)
            continue
        m = re.match(r"\| (.+?) \| ([\d.]+|--) \| ([\d.]+) \+/- ([\d.]+) \(n=(\d+)\)", line)
        if m and m.group(1) in KEY and bb in NAME:
            base = None if m.group(2) == "--" else float(m.group(2))
            out.setdefault(bb, {})[KEY[m.group(1)]] = (
                base, float(m.group(3)), float(m.group(4)), int(m.group(5)))
    return out


def mark(delta, text):
    r"""Wrap `text` in \gain (improved) or \loss (worsened); leave unchanged alone.

    `delta` is oriented so positive = better. The macros are defined in the paper preamble."""
    if delta > 0:
        return f"\\gain{{{text}}}"
    if delta < 0:
        return f"\\loss{{{text}}}"
    return text


def dense_pos(value, values):
    """Dense position of a rank sum among all rank sums (lower rank sum is better)."""
    return sorted(set(values)).index(value) + 1


def ranksum_and_position(bb, vals, our):
    """(rank sum base, rank sum tuned, position base, position tuned) for one backbone.

    THUNDER orders models by rank sum, so position is what the rank sum buys. Every model's
    rank sum is RECOMPUTED in the substituted field, because inserting our value displaces
    other models on that task and therefore moves their rank sums too. scripts/overall_ranks.py
    computes the THUNDER column this way; the two must agree."""
    # `our` is substituted into the field for BOTH terms. Previously rs_base was computed on
    # the unsubstituted `vals`, i.e. it returned the PUBLISHED rank sum while the per-task base
    # ranks printed beside it came from our measured base -- so the six visible ranks did not
    # add up to the printed base total on any row.
    f = substituted(vals, bb, our)
    rs = {m: sum(dense_rank(v[i], i, f) for i in range(6)) for m, v in f.items()}
    v = rs[LBNAME[bb]]
    return v, v, dense_pos(v, rs.values()), dense_pos(v, rs.values())


def main():
    vals, ranksum = leaderboard()
    n = validate(vals, ranksum)
    tuned = tuned_means()

    md = ["# THUNDER leaderboard rank, base vs fine-tuned", "",
          "**Generated file -- do not hand-edit.**  Regenerate with",
          "`./.venv/bin/python scripts/thunder_ranks.py`.", "",
          f"Rank is position in the {n}-model 16-dataset panel (histopathology AND",
          "natural-image models), using DENSE ranking, the leaderboard's own convention.",
          f"The rule is validated against all {n} published per-task ranks and rank sums",
          "before use. To rank one of our models we replace that backbone's published row.",
          "ECE and adversarial drop are lower-is-better; adversarial uses fp32-attack cells.",
          "", "| backbone | task | base | rank | fine-tuned | rank | move |",
          "|---|---|---|---|---|---|---|"]
    # MERGED TABLE. Replaces the old value-only table and the old value+rank table with one
    # exhibit: our own paired base -> tuned +/- SD per task, then the rank sum those values
    # earn in the published field and the leaderboard position that rank sum implies.
    cells_ours = our_cells()
    header = ["% Generated by scripts/thunder_ranks.py -- do not hand-edit.",
              "\\begin{tabular}{l" + "c" * len(HDR) + "c}", "\\toprule",
              "Backbone & " + " & ".join(HDR) + " & $\\Sigma$rank\\,(position)\\,$\\downarrow$ \\\\", "\\midrule"]
    tex = list(header)        # main body: values only in the six metric cells
    tex_full = list(header)   # appendix: same rows with per-task ranks and +/-2SD

    for bb in ORDER:
        base_vals = list(vals[LBNAME[bb]])       # published row, used only for the base rank
        our_base = list(base_vals)
        our_vals = list(base_vals)
        cells = []
        cells_full = []
        for i in range(6):
            cell = cells_ours.get(bb, {}).get(i)
            b = cell[0] if cell else None
            t = cell[1] if cell else None
            sd, n = (cell[2], cell[3]) if cell else (None, None)
            if i == 5:                            # adversarial: fp32 cells, not seed_stats
                if bb not in FP32_ADV:
                    cells.append("--")
                    cells_full.append("--")
                    md.append(f"| {NAME[bb]} | {COLS[i]} | -- | -- | -- | -- | pending |")
                    continue
                b, t, sd = FP32_ADV[bb]
            if t is None:
                cells.append("--")
                cells_full.append("--")
                continue
            if b is not None:
                our_base[i] = b
            our_vals[i] = t
            # Rank OUR base, not the published row. The table displays our measured base,
            # so ranking the published value there pairs a number with a rank that belongs
            # to a different number (Virchow2 seg displayed 69.0 with the rank of the
            # published 69.3, showing a 2-place "move" for an unchanged value).
            rb = dense_rank(our_base[i], i, substituted(vals, bb, our_base))
            rt = dense_rank(t, i, substituted(vals, bb, our_vals))
            # Uncertainty is reported as TWO sample SDs over adapter seeds, matching the
            # figures and the 2-SD resolution rule used in the per-dataset panel.
            sdtxt = f"$\\pm${2 * sd:.1f}" if sd is not None else ""
            # superscript is the rank MOVE on this task: keeps the per-task rank story the
            # text tells without spending a second table on rank pairs.
            # value carries its OWN rank in parentheses, before and after; the fine-tuned
            # rank is marked when it moved.
            rt_txt = mark(rb - rt, f"({rt})")   # appendix twin: rank move (ranks are shown)
            head = f"{b:.1f}~({rb})$\\to$" if b is not None else ""
            cells_full.append(f"{head}{t:.1f}{sdtxt}~{rt_txt}")
            # Main-body cell: value pair only. Per-task ranks and the +/-2SD term live in
            # the appendix twin -- editorial policy keeps the SD term on aggregate columns
            # only. Because the ranks are NOT shown here, the colour must encode the change
            # in the VALUE the reader can see, not the rank move it would otherwise track
            # (that made a 5.9-point adversarial improvement print uncoloured next to a
            # coloured +0.1 k-NN cell). Direction comes from COLS/LOWER, the same source the
            # header arrows come from, so the two can never disagree. The comparison is on
            # the ROUNDED printed values: a cell that prints the same number twice stays
            # uncoloured. The appendix twin keeps rank-move colouring, where ranks ARE shown.
            head_c = f"{b:.1f}$\\to$" if b is not None else ""
            if b is None:
                vdelta = 0.0
            else:
                db, dt = round(b, 1), round(t, 1)
                vdelta = (db - dt) if COLS[i] in LOWER else (dt - db)
            cells.append(f"{head_c}{mark(vdelta, f'{t:.1f}')}")
            md.append(f"| {NAME[bb]} | {COLS[i]} | {base_vals[i]:.1f} | {rb} | {t:.1f} | "
                      f"{rt} | {rb-rt:+d} |".replace("+0", "0"))
        rs_b, _, pb, _ = ranksum_and_position(bb, vals, our_base)
        _, rs_t, _, pt = ranksum_and_position(bb, vals, our_vals)
        # rank sum carries the field position it implies, in parentheses. Both are
        # lower-is-better, so a DECREASE is the improvement.
        rs_txt = mark(rs_b - rs_t, f"{rs_t}~({pt})")
        tail = f" & {rs_b}~({pb})$\\to${rs_txt} \\\\"
        tex.append(f"{NAME[bb]} & " + " & ".join(cells) + tail)
        tex_full.append(f"{NAME[bb]} & " + " & ".join(cells_full) + tail)

    # rank sums, with and without ECE, over the tasks we can fill
    md += ["", "## Rank sum (lower is better)", "",
           "`4 tasks` excludes ECE and adversarial, the subset with complete $n=3$ coverage on",
           "every backbone. `5 tasks` adds adversarial where the fp32 cells have landed.", "",
           "| backbone | 4 tasks base | tuned | move | +ECE base | tuned | move |",
           "|---|---|---|---|---|---|---|"]
    for bb in ORDER:
        base_vals = list(vals[LBNAME[bb]])
        our = list(base_vals)
        for i in range(5):
            t = tuned.get(bb, {}).get(i)
            if t is not None:
                our[i] = t
        f4 = [0, 1, 2, 3]
        f5 = [0, 1, 2, 3, 4]
        rb4 = sum(dense_rank(base_vals[i], i, vals) for i in f4)
        rt4 = sum(dense_rank(our[i], i, substituted(vals, bb, our)) for i in f4)
        rb5 = sum(dense_rank(base_vals[i], i, vals) for i in f5)
        rt5 = sum(dense_rank(our[i], i, substituted(vals, bb, our)) for i in f5)
        md.append(f"| {NAME[bb]} | {rb4} | {rt4} | {rb4-rt4:+d} | {rb5} | {rt5} | "
                  f"{rb5-rt5:+d} |".replace("+0", "0"))

    tex += ["\\bottomrule", "\\end{tabular}"]
    tex_full += ["\\bottomrule", "\\end{tabular}"]
    (REPO / "docs/thunder_ranks.md").write_text("\n".join(md) + "\n")
    TEX_OUT.parent.mkdir(parents=True, exist_ok=True)
    TEX_OUT.write_text("\n".join(tex) + "\n")
    TEX_FULL_OUT.write_text("\n".join(tex_full) + "\n")
    print(f"validated rank rule against all {n} published rows; wrote docs/thunder_ranks.md")


if __name__ == "__main__":
    main()
