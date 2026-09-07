#!/usr/bin/env python3
"""PathoROB rank and rank sum, base vs fine-tuned.

PathoROB publishes no rank column, so ranks are computed with the same dense-ranking
convention used for THUNDER (validated there against 32 published rows).  The model field is
the leaderboard maintained in the benchmark authors' own repo, github.com/bifold-pathomics/
PathoROB, transcribed to data/pathorob_leaderboard.tsv.

VALIDATION.  Unlike THUNDER and HEST, this table ships no quantity we can recompute from
its own columns, so the check runs the other way: our independently measured base RI is
compared against their published row for the four backbones we share.  Virchow2 and UNI2-h
reproduce to 0.000 and Phikon-v2 to 0.004, which is strong evidence our PathoROB harness
matches theirs.  H-optimus-0 does NOT (average 0.7997 vs 0.812, driven by Camelyon
0.6785 vs 0.705 and TCGA 0.8025 vs 0.812, while Tolkach-ESCA agrees exactly); the cause is
unresolved, so its row is reported with that discrepancy attached rather than silently
ranked.  Base and fine-tuned are both OUR measurements, so the delta is within-harness and
unaffected either way.

Midnight-12k is absent from the PathoROB field, so it has no rank or position.  We ran it
through our own PathoROB harness all the same, so its RI is reported like any other row and
only the rank columns are blank.  Its Camelyon RI moves 0.478 -> 0.884, which the previous
"not in the PathoROB field" banner hid entirely.

    ./.venv/bin/python scripts/pathorob_ranks.py
      -> docs/pathorob_ranks.md, waiv-asci/tables/pathorob_ranks.tex
"""
import csv
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from _config import PAPER_TABLES  # noqa: E402
LB_TSV = REPO / "data/pathorob_leaderboard.tsv"
TEX_OUT = PAPER_TABLES / "pathorob_ranks.tex"

DS = [("tcga", "ri_tcga2x2"), ("camelyon", "ri_camelyon"), ("tolkach_esca", "ri_tolkach_esca")]
DSLABEL = ["TCGA $2\\times2$", "Camelyon", "Tolkach-ESCA"]
# The backbones that appear in the published PathoROB field. Only Midnight-12k and
# OpenMidnight are genuinely absent from it; they carry measured RI but no rank.
# Virchow (v1) IS in the field -- our base reproduces its published row to 0.0004, the
# closest agreement of any backbone -- so it must be ranked, not blanked.
LBNAME = {"phikon2": "Phikon-v2", "virchow2": "Virchow2",
          "hoptimus0": "H-optimus-0", "uni2h": "UNI2-h", "virchow1": "Virchow"}
ORDER = ["phikon2", "midnight", "virchow2", "hoptimus0", "uni2h", "virchow1",
         "openmidnightsq"]
NAME = {"phikon2": "Phikon-v2", "midnight": "Midnight-12k", "virchow2": "Virchow2",
        "hoptimus0": "H-optimus-0", "uni2h": "UNI2-h",
        "virchow1": "Virchow", "openmidnightsq": "OpenMidnight"}
TOL = 0.005          # agreement threshold for the base-reproduction check


def leaderboard():
    rows = list(csv.DictReader(
        (l for l in LB_TSV.read_text().splitlines() if not l.startswith("#")), delimiter="\t"))
    return {r["model"]: [float(r[c]) for _, c in DS] for r in rows}, {
        r["model"]: r.get("validated_by_authors", "") for r in rows}


def dense_rank(value, i, field):
    distinct = sorted({v[i] for v in field.values()}, reverse=True)   # RI: higher is better
    return distinct.index(value) + 1


def mark(delta, text):
    r"""\gain when better, \loss when worse (delta oriented so positive = better)."""
    return f"\\gain{{{text}}}" if delta > 0 else (f"\\loss{{{text}}}" if delta < 0 else text)


def mark_2sd(delta, sd2, text, nd=3):
    r"""Single paper-wide VALUE-cell rule: \gain / \loss only when |delta| clears
    two sample SDs (compared at the printed precision), plain otherwise and plain
    when there is no spread.  `delta` is oriented so positive = better; `sd2` is
    already two sample SDs.  Rank cells keep the ungated `mark`."""
    if sd2 is None or not round(abs(delta), nd) > round(sd2, nd):
        return text
    return mark(delta, text)


# The main-body table reports AGGREGATES only (mean RI, Sigma-rank, PLISM retrieval). The
# three per-dataset RI columns were dropped so the table fits \textwidth unscaled at \small;
# per-dataset RI is still reported, in the appendix submetrics table (tables/pathorob_submetrics).


def ranksum(vals, field):
    """Sum of per-dataset dense ranks for one model's 3 RI values within `field`."""
    return sum(dense_rank(vals[i], i, field) for i in range(len(DS)))


def position(vals, field):
    """Field position by MEAN RI over the three datasets, which is how PathoROB orders models
    and how scripts/overall_ranks.py reports it. Deliberately NOT derived from the rank sum:
    the rank sum is reported alongside as a per-dataset consistency summary, and the two are
    different aggregations that need not agree."""
    m = sum(vals) / len(vals)
    distinct = sorted({sum(v) / len(v) for v in field.values()}, reverse=True)
    return distinct.index(m) + 1


def ours():
    import pathorob_submetrics as pm
    out = {}
    for bb in ORDER:
        base_cell, seeds = pm.cells_for(bb)
        b, t, sd = [], [], []
        for ds, _ in DS:
            s = pm.summary(base_cell, ds)
            b.append(s["robustness_index"] if s else None)
            vals = [pm.summary(c, ds) for c in seeds]
            vals = [v["robustness_index"] for v in vals if v]
            t.append(statistics.mean(vals) if vals else None)
            sd.append(2 * statistics.stdev(vals) if len(vals) > 1 else None)
        out[bb] = (b, t, len(seeds), sd)
    return out



def seed_stats_cell(metric):
    """{backbone: (base, tuned mean, sd, n)} for 'PathoROB RI'.

    Computed directly from the same raw results_summary.json files as
    pathorob_submetrics.py / seed_stats.py (via seed_stats.ri_of), NOT by re-parsing the
    4-decimal text in docs/seed_stats.md. Parsing that rounded text and rounding again to
    3 dp here double-rounds and can land on a different final digit than rounding the exact
    mean once (e.g. hoptimus0's exact mean 0.90552... rounds to 0.906 directly, but its
    docs/seed_stats.md text '0.9055' re-parses to a float just under 0.9055 and rounds down
    to 0.905) -- this was the cause of pathorob_ranks.tex disagreeing with
    pathorob_submetrics.tex on H-optimus-0's mean RI.
    """
    assert metric == "PathoROB RI", f"only PathoROB RI is supported here, got {metric!r}"
    import seed_stats as _ss
    import pathorob_submetrics as _pm
    out = {}
    for bb in ORDER:
        base_cell, seeds = _pm.cells_for(bb)
        rb = _ss.ri_of(base_cell)
        vals = [v for v in (_ss.ri_of(s) for s in seeds) if v is not None]
        if not vals:
            continue
        mt = statistics.mean(vals)
        msd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out[bb] = (rb, mt, msd, len(vals))
    return out


def _retr_cell(retr, bb):
    """'0.858->0.995 / 0.696->0.897' for one backbone, or '--'."""
    e = retr.get(bb)
    if not e:
        return "--"
    parts = []
    for axis in ("cross-scanner", "cross-stain"):
        if axis not in e:
            parts.append("--")
            continue
        b, t, s2 = e[axis]
        tail = "" if s2 is None else f"$\\pm${s2:.3f}"
        parts.append(f"{b:.3f}$\\to$" + mark_2sd(t - b, s2, f"{t:.3f}{tail}"))
    return " / ".join(parts)


def rank_cell(bb, mine, lb):
    """'58~(21)$\\to$22~(6)' (daggered when the backbone is inserted into the field) for the
    merged appendix table. Same arithmetic as main()."""
    b, t, _, _ = mine[bb]
    key = LBNAME.get(bb, NAME[bb])
    fb = dict(lb); fb[key] = b
    ft = dict(lb); ft[key] = t
    rs_b, rs_t = ranksum(b, fb), ranksum(t, ft)
    pb, pt = position(b, fb), position(t, ft)
    return f"{rs_b}~({pb})$\\to${rs_t}~({pt})" + ("" if bb in LBNAME else "$^\\dagger$")


def retr_cells(retr, bb):
    """['0.858$\\to$0.995$\\pm$0.001', '0.696$\\to$0.897$\\pm$0.003'] (scanner, stain), unmarked."""
    e = retr.get(bb, {})
    out = []
    for axis in ("cross-scanner", "cross-stain"):
        if axis not in e:
            out.append("--"); continue
        b, t, s2 = e[axis]
        out.append(f"{b:.3f}$\\to${t:.3f}" + ("" if s2 is None else f"$\\pm${s2:.3f}"))
    return out


def plism_retrieval():
    """{backbone slug: {axis: (base, tuned)}} from docs/plism_retrieval.md.

    Held-out PLISM top-1 retrieval is the cross-ACQUISITION half of the robustness story;
    PathoROB RI is the cross-CENTRE half. They share this table rather than costing a
    second float. Generated by scripts/plism_retrieval.py.
    """
    inv = {"Phikon-v2": "phikon2", "Midnight-12k": "midnight", "Virchow2": "virchow2",
           "H-optimus-0": "hoptimus0", "UNI2-h": "uni2h", "Virchow": "virchow1",
           "OpenMidnight": "openmidnightsq"}
    f = REPO / "docs/plism_retrieval.md"
    out = {}
    if not f.exists():
        return out
    for line in f.read_text().splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if len(c) >= 4 and c[0] in inv and c[1] in ("cross-scanner", "cross-stain"):
            try:
                b = float(c[2])
                parts = c[3].split()
                t = float(parts[0])
                s2 = float(parts[-1]) if "+/-" in c[3] else None
            except ValueError:
                continue
            out.setdefault(inv[c[0]], {})[c[1]] = (b, t, s2)
    return out


def main():
    lb, validated = leaderboard()
    mine = ours()

    md = ["# PathoROB rank and rank sum, base vs fine-tuned", "",
          "**Generated file -- do not hand-edit.**  Regenerate with",
          "`./.venv/bin/python scripts/pathorob_ranks.py`.", "",
          f"Field is the {len(lb)}-model leaderboard from the PathoROB authors' repo",
          "(github.com/bifold-pathomics/PathoROB); three of its rows are cited from external",
          "publications rather than computed by them. PathoROB publishes no ranks, so dense",
          "ranking is applied as for THUNDER. Midnight-12k is not in the field and cannot be",
          "ranked.", "",
          "## Check: our measured base RI vs their published row", "",
          "| backbone | dataset | ours | theirs | diff |", "|---|---|---|---|---|"]
    for bb in ORDER:
        if bb not in LBNAME:
            continue
        b = mine[bb][0]
        for i, (ds, _) in enumerate(DS):
            th = lb[LBNAME[bb]][i]
            flag = " **MISMATCH**" if abs(b[i] - th) > TOL else ""
            md.append(f"| {NAME[bb]} | {ds} | {b[i]:.4f} | {th:.3f} | {b[i]-th:+.3f}{flag} |")

    md += ["", "## Rank and rank sum", "",
           "| backbone | dataset | base | rank | tuned | rank | move |", "|---|---|---|---|---|---|---|"]
    tex = ["% Generated by scripts/pathorob_ranks.py -- do not hand-edit.",
           "\\begin{tabular}{lcccc}", "\\toprule",
           " & \\multicolumn{1}{c}{PathoROB (cross-centre)} & & "
           "\\multicolumn{1}{c}{PLISM (cross-acquisition)} \\\\",
           "\\cmidrule(lr){2-2}\\cmidrule(lr){4-4}",
           "Backbone & "
           "Mean RI\\,$\\uparrow$ & $\\Sigma$rank\\,(pos.)\\,$\\downarrow$ & "
           "top-1 scanner / stain\\,$\\uparrow$ \\\\", "\\midrule"]
    retr = plism_retrieval()
    ss = seed_stats_cell("PathoROB RI")
    for bb in ORDER:
        b, t, n, sd2 = mine[bb]
        # Midnight-12k is absent from the published field, so it has no rank or position.
        # Its RI is our own measurement all the same, and is reported rather than blanked.
        ranked = bb in LBNAME
        if not ranked:
            # Absent from the published field, so there is no row to substitute. We INSERT the
            # model instead, which makes the field one larger (24, not 23); the resulting rank
            # is an estimate of where it would have placed and is daggered in the table.
            fb = dict(lb); fb[NAME[bb]] = b
            ft = dict(lb); ft[NAME[bb]] = t
            rs_b, rs_t = ranksum(b, fb), ranksum(t, ft)
            pb, pt = position(b, fb), position(t, ft)
            mb, mt, msd, mn = ss[bb]
            # 3 dp throughout this table: the per-dataset RI and retrieval columns are 3 dp, so
            # a 4 dp mean column was the only inconsistent precision in the table.
            meancol = f"{mb:.3f}$\\to$" + mark_2sd(mt - mb, 2 * msd, f"{mt:.3f}$\\pm${2 * msd:.3f}")
            tex.append(f"{NAME[bb]} & {meancol} & "
                       f"{rs_b}~({pb})$\\to$" + mark(rs_b - rs_t, f"{rs_t}~({pt})")
                       + "$^\\dagger$" + f" & {_retr_cell(retr, bb)} \\\\")
            for i, (ds, _) in enumerate(DS):
                rb = dense_rank(b[i], i, fb); rt = dense_rank(t[i], i, ft)
                md.append(f"| {NAME[bb]} | {ds} | {b[i]:.4f} | {rb}* | {t[i]:.4f} | {rt}* | "
                          f"{rb-rt:+d} (inserted into field) |".replace("+0", "0"))
            md.append(f"| {NAME[bb]} | **rank sum** | | {rs_b}* | | {rs_t}* | "
                      f"**{rs_b-rs_t:+d}** (estimated) |".replace("+0", "0"))
            continue
        fb = dict(lb); fb[LBNAME[bb]] = b
        ft = dict(lb); ft[LBNAME[bb]] = t
        rs_b, rs_t = ranksum(b, fb), ranksum(t, ft)
        pb, pt = position(b, fb), position(t, ft)
        for i, (ds, _) in enumerate(DS):
            rb = dense_rank(b[i], i, fb)
            rt = dense_rank(t[i], i, ft)
            md.append(f"| {NAME[bb]} | {ds} | {b[i]:.4f} | {rb} | {t[i]:.4f} | {rt} | "
                      f"{rb-rt:+d} |".replace("+0", "0"))
        mb, mt, msd, mn = ss[bb]
        # 3 dp throughout this table: the per-dataset RI and retrieval columns are 3 dp, so
        # a 4 dp mean column was the only inconsistent precision in the table.
        meancol = f"{mb:.3f}$\\to$" + mark_2sd(mt - mb, 2 * msd, f"{mt:.3f}$\\pm${2 * msd:.3f}") 
        tex.append(f"{NAME[bb]} & {meancol} & {rs_b}~({pb})$\\to$"
                   + mark(rs_b - rs_t, f"{rs_t}~({pt})")
                   + f" & {_retr_cell(retr, bb)} \\\\")
        md.append(f"| {NAME[bb]} | **rank sum** | | {rs_b} | | {rs_t} | "
                  f"**{rs_b-rs_t:+d}** |".replace("+0", "0"))
    tex += ["\\bottomrule", "\\end{tabular}"]
    (REPO / "docs/pathorob_ranks.md").write_text("\n".join(md) + "\n")
    TEX_OUT.parent.mkdir(parents=True, exist_ok=True)
    TEX_OUT.write_text("\n".join(tex) + "\n")
    print(f"wrote docs/pathorob_ranks.md ({len(lb)}-model field)")


if __name__ == "__main__":
    main()
