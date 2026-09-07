#!/usr/bin/env python3
"""Overall leaderboard position, base -> fine-tuned, on all three benchmarks.

Per-task rank movements are in the per-benchmark tables; this is the one-line summary a
reader actually wants: where does the model sit in each published field before and after.

Position is defined the way each leaderboard itself orders models:
  THUNDER  -- by RANK SUM over its six tasks (its own headline column), 32-model field.
  HEST     -- by mean Pearson r over nine tasks, 26-model field.
  PathoROB -- by mean RI over three datasets, 23-model field.
Ties share a position (dense), matching THUNDER's validated convention.

Every underlying table is validated by its own rank script before use; see
scripts/{thunder,hest,pathorob}_ranks.py.

    ./.venv/bin/python scripts/overall_ranks.py
      -> docs/overall_ranks.md, waiv-asci/tables/overall_ranks.tex
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from _config import PAPER_TABLES  # noqa: E402
TEX_OUT = PAPER_TABLES / "overall_ranks.tex"

import thunder_ranks as T          # noqa: E402
import hest_ranks as H             # noqa: E402
import pathorob_ranks as P         # noqa: E402

ORDER = ["phikon2", "midnight", "virchow2", "hoptimus0", "uni2h", "virchow1", "openmidnightsq"]
NAME = {"phikon2": "Phikon-v2", "midnight": "Midnight-12k", "virchow2": "Virchow2",
        "hoptimus0": "H-optimus-0", "uni2h": "UNI2-h",
        "virchow1": "Virchow", "openmidnightsq": "OpenMidnight"}


def dense_pos(value, values, higher_better=True):
    return sorted(set(values), reverse=higher_better).index(value) + 1


def thunder_positions():
    lb, rs = T.leaderboard()
    T.validate(lb, rs)
    tuned = T.tuned_means()
    out = {}
    for bb in ORDER:
        base = list(lb[T.LBNAME[bb]])
        our = list(base)
        for i in range(5):
            v = tuned.get(bb, {}).get(i)
            if v is not None:
                our[i] = v
        provisional = bb not in T.FP32_ADV
        if not provisional:
            our[5] = T.FP32_ADV[bb][1]
        # OUR measured base must be substituted in too, exactly as for the tuned value.
        # Using the unsubstituted `lb` here returns the PUBLISHED rank sum, which disagreed
        # with scripts/thunder_ranks.py (Midnight-12k base position printed 4 vs 5).
        cells = T.our_cells()
        our_base = list(base)
        for i in range(5):
            c = cells.get(bb, {}).get(i)
            if c and c[0] is not None:
                our_base[i] = c[0]
        if not provisional:
            our_base[5] = T.FP32_ADV[bb][0]
        fb = T.substituted(lb, bb, our_base)
        rs_base = {m: sum(T.dense_rank(v[i], i, fb) for i in range(6)) for m, v in fb.items()}
        f = T.substituted(lb, bb, our)
        rs_tuned = {m: sum(T.dense_rank(v[i], i, f) for i in range(6)) for m, v in f.items()}
        pb = dense_pos(rs_base[T.LBNAME[bb]], rs_base.values(), higher_better=False)
        pt = dense_pos(rs_tuned[T.LBNAME[bb]], rs_tuned.values(), higher_better=False)
        out[bb] = (pb, pt, len(lb), provisional,
                   rs_base[T.LBNAME[bb]], rs_tuned[T.LBNAME[bb]])
    return out


def hest_positions():
    lb, _ = H.leaderboard()
    ours = H.our_values()
    out = {}
    for bb in ORDER:
        b, t = ours[bb][0], ours[bb][1]
        if not b or not t:
            continue
        fb = dict(lb); fb[H.LBNAME[bb]] = b
        ft = dict(lb); ft[H.LBNAME[bb]] = t
        pb = dense_pos(sum(b) / 9, [sum(v) / 9 for v in fb.values()])
        pt = dense_pos(sum(t) / 9, [sum(v) / 9 for v in ft.values()])
        out[bb] = (pb, pt, len(lb), False, sum(b) / 9, sum(t) / 9)
    return out


def pathorob_positions():
    lb, _ = P.leaderboard()
    mine = P.ours()
    out = {}
    for bb in ORDER:
        b, t, *_ = mine[bb]
        # In the field: substitute our value for the published row (field stays 23).
        # Absent from it: insert the model, so the field becomes 24 and the position is an
        # ESTIMATE of where it would have placed. `est` flags that for the caption.
        key = P.LBNAME.get(bb, P.NAME[bb])
        est = bb not in P.LBNAME
        fb = dict(lb); fb[key] = b
        ft = dict(lb); ft[key] = t
        pb = dense_pos(sum(b) / 3, [sum(v) / 3 for v in fb.values()])
        pt = dense_pos(sum(t) / 3, [sum(v) / 3 for v in ft.values()])
        out[bb] = (pb, pt, len(fb), est, sum(b) / 3, sum(t) / 3)
    return out


def main():
    th, he, pr = thunder_positions(), hest_positions(), pathorob_positions()
    md = ["# Overall leaderboard position, base -> fine-tuned", "",
          "**Generated file -- do not hand-edit.**  Regenerate with",
          "`./.venv/bin/python scripts/overall_ranks.py`.", "",
          "Position is how each leaderboard orders models: THUNDER by rank sum over six tasks",
          "(32 models), HEST by mean $r$ over nine tasks (26 models), PathoROB by mean RI over",
          "three datasets (23 models). `*` marks a THUNDER row whose adversarial column is still",
          "the fp16 value and is therefore provisional.", "",
          "| backbone | PathoROB | HEST | THUNDER |", "|---|---|---|---|"]
    tex = ["% Generated by scripts/overall_ranks.py -- do not hand-edit.",
           "\\begin{tabular}{lccc}", "\\toprule",
           "Backbone & PathoROB\\,$\\downarrow$ & HEST\\,$\\downarrow$ & THUNDER\\,$\\downarrow$ \\\\", "\\midrule"]
    for bb in ORDER:
        cells_md, cells_tex = [], []
        for src in (pr, he, th):
            if bb not in src:
                cells_md.append("not in field"); cells_tex.append("--"); continue
            pb, pt, n, prov, vb, vt = src[bb]
            star = "$^\\dagger$" if prov else ""
            mv = pb - pt
            cells_md.append(f"{pb} -> {pt} ({mv:+d})".replace("+0", "0")
                            + (" (estimated)" if prov else ""))
            pos = f"{pt}{star}"
            pos = f"\\gain{{{pos}}}" if mv > 0 else (f"\\loss{{{pos}}}" if mv < 0 else pos)
            cells_tex.append(f"{pb}$\\to${pos}")
        md.append(f"| {NAME[bb]} | " + " | ".join(cells_md) + " |")
        tex.append(f"{NAME[bb]} & " + " & ".join(cells_tex) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (REPO / "docs/overall_ranks.md").write_text("\n".join(md) + "\n")
    TEX_OUT.parent.mkdir(parents=True, exist_ok=True)
    TEX_OUT.write_text("\n".join(tex) + "\n")
    print("\n".join(md[md.index("| backbone | PathoROB | HEST | THUNDER |"):]))


if __name__ == "__main__":
    main()
