#!/usr/bin/env python3
"""HEST-Benchmark rank and rank sum, base vs fine-tuned.

HEST publishes per-task Pearson r for 26 models but NO rank column and no rank sum, so both
are computed here.  Two guards against silently inventing a methodology:

  * the transcribed table is validated by recomputing each model's published Average from its
    nine task columns (26/26 must match), so the parse cannot be wrong; and
  * ranking uses DENSE ranking, the convention THUNDER's leaderboard uses and which
    scripts/thunder_ranks.py validates against 32 published rows -- reused here for
    consistency rather than invented.  HEST itself sanctions no convention; this is ours.

Base and fine-tuned both come from OUR harness, so the delta is within-harness and free of
protocol offsets.  Our base reproduces the published row closely (UNI2-h average 0.4138 vs
published 0.4141), which is reported as a check rather than assumed.

    ./.venv-hest/bin/python scripts/hest_ranks.py     (any venv; stdlib only)
      -> docs/hest_ranks.md, waiv-asci/tables/hest_ranks.tex
"""
import csv
import glob
import json
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from _config import PAPER_TABLES  # noqa: E402
LB_TSV = REPO / "data/hest_leaderboard_9task.tsv"
TEX_OUT = PAPER_TABLES / "hest_ranks.tex"

TASKS = ["IDC", "PRAD", "PAAD", "SKCM", "COAD", "READ", "CCRCC", "LUNG", "LYMPH_IDC"]
LBNAME = {"phikon2": "Phikon-v2", "midnight": "Midnight-12k", "virchow2": "Virchow2",
          "hoptimus0": "H-Optimus-0", "uni2h": "UNI2-h",
          "virchow1": "Virchow", "openmidnightsq": "OpenMidnight"}
ORDER = ["phikon2", "midnight", "virchow2", "hoptimus0", "uni2h", "virchow1", "openmidnightsq"]
BASE_SUMMARY = {"phikon2": "base_cls_summary.json",
                "midnight": "midnight_base_cls_9task_v1_summary.json",
                "virchow2": "vbase_clsmean_summary.json",
                "hoptimus0": "hoptimus_base_cls_9task_v1_summary.json",
                "uni2h": "uni2_base_cls_9task_v1_summary.json",
                "virchow1": "virchow1_base_clsmean_9task_v1_summary.json",
                "openmidnightsq": "openmidnight_base_cls_9task_v1_summary.json"}


def leaderboard():
    rows = list(csv.DictReader(
        (l for l in LB_TSV.read_text().splitlines() if not l.startswith("#")), delimiter="\t"))
    vals = {r["model"]: [float(r[t]) for t in TASKS] for r in rows}
    pub_avg = {r["model"]: float(r["average"]) for r in rows}
    bad = [m for m in vals if abs(sum(vals[m]) / 9 - pub_avg[m]) > 6e-5]
    assert not bad, f"parse does not reproduce published Average for {bad}"
    return vals, pub_avg


def dense_rank(value, i, field):
    distinct = sorted({v[i] for v in field.values()}, reverse=True)   # all tasks higher-better
    return distinct.index(value) + 1


def our_values():
    """Per-task base and fine-tuned means from our own HEST summaries."""
    import seed_stats as ss
    res = {}
    cells = sorted(d.name for d in ss.CELLS.iterdir() if (d / "model.py").exists())
    for bb in ORDER:
        base = None
        for cand in (Path(ss.HEST_RES) / BASE_SUMMARY[bb],
                     REPO / "results_backup/hest_work_results" / BASE_SUMMARY[bb]):
            if cand.exists():
                d = json.loads(cand.read_text())
                if "results" in d:
                    base = [d["results"][t] for t in TASKS]
                break
        per = []
        for c in [x for x in cells if x.startswith(f"{bb}-c50-s") and ss.rule_selected(x)]:
            txt = (ss.CELLS / c / "model.py").read_text()
            run = re.search(r'^RUN = "(.*)"$', txt, re.M).group(1)
            step = re.search(r'^STEP = "(.*)"$', txt, re.M).group(1).replace("step_", "")
            hits = glob.glob(f"{ss.HEST_RES}/f5_{re.sub(r'.r[0-9]+$', '', run)}_s{step}_*_summary.json")
            if hits:
                d = json.loads(Path(sorted(hits)[0]).read_text())
                if "results" in d:
                    per.append([d["results"][t] for t in TASKS])
        tuned = [statistics.mean(x[i] for x in per) for i in range(9)] if per else None
        res[bb] = (base, tuned, len(per), per)
    return res



def seed_stats_cell(metric):
    """{backbone: (base, tuned mean, sd, n)} for 'PathoROB RI' or 'HEST' from seed_stats.md.
    Same source the standalone value tables used before they were merged in here."""
    import re as _re
    text = (REPO / "docs/seed_stats.md").read_text()
    out, bb = {}, None
    for line in text.splitlines():
        m = _re.match(r"## (\w+)", line)
        if m:
            bb = m.group(1); continue
        m = _re.match(r"\| (PathoROB RI|HEST) \| ([\d.]+) \| ([\d.]+) \+/- ([\d.]+) \(n=(\d+)\)", line)
        if m and m.group(1) == metric:
            out[bb] = (float(m.group(2)), float(m.group(3)), float(m.group(4)), int(m.group(5)))
    return out


def g(d, txt):
    return f"\\gain{{{txt}}}" if d > 0 else (f"\\loss{{{txt}}}" if d < 0 else txt)


def main():
    lb, pub_avg = leaderboard()
    ours = our_values()
    md = ["# HEST-Benchmark rank and rank sum, base vs fine-tuned", "",
          "**Generated file -- do not hand-edit.**  Regenerate with",
          "`./.venv/bin/python scripts/hest_ranks.py`.", "",
          f"HEST publishes per-task Pearson $r$ for {len(lb)} models but no rank column and no",
          "rank sum, so both are computed here with dense ranking (the convention THUNDER uses",
          "and which scripts/thunder_ranks.py validates against 32 published rows). The",
          "transcribed table is validated by reproducing every published Average from its nine",
          "task columns. Base and tuned are both from our harness, so the delta is",
          "within-harness; our base reproduces the published row closely.", "",
          "| backbone | our base avg | published avg | base rank sum | tuned rank sum | move | base avg-rank | tuned avg-rank |",
          "|---|---|---|---|---|---|---|---|"]
    tex = ["% Generated by scripts/hest_ranks.py -- do not hand-edit.",
           "\\begin{tabular}{l" + "c" * len(TASKS) + "cc}", "\\toprule",
           " & \\multicolumn{%d}{c}{per-task rank, base $\\to$ fine-tuned\\,$\\downarrow$}"
           " & & \\\\" % len(TASKS),
           "\\cmidrule(lr){2-%d}" % (1 + len(TASKS)),
           "Backbone & " + " & ".join(t.replace("_", "\\_") for t in TASKS)
           + " & Mean $r$\\,$\\uparrow$ & $\\Sigma$rank\\,(position)\\,$\\downarrow$ \\\\",
           "\\midrule"]
    ss = seed_stats_cell("HEST")
    for bb in ORDER:
        base, tuned, n, per = ours[bb]
        if base is None or tuned is None:
            md.append(f"| {LBNAME[bb]} | -- | -- | -- | -- | -- | -- | -- |")
            continue
        f_base = dict(lb); f_base[LBNAME[bb]] = base
        f_tuned = dict(lb); f_tuned[LBNAME[bb]] = tuned
        rs_b = sum(dense_rank(base[i], i, f_base) for i in range(9))
        rs_t = sum(dense_rank(tuned[i], i, f_tuned) for i in range(9))
        ab, at = sum(base) / 9, sum(tuned) / 9
        avg_field_b = {m: [sum(v) / 9] for m, v in f_base.items()}
        avg_field_t = {m: [sum(v) / 9] for m, v in f_tuned.items()}
        ra = dense_rank(ab, 0, avg_field_b)
        rt = dense_rank(at, 0, avg_field_t)
        md.append(f"| {LBNAME[bb]} | {ab:.4f} | {pub_avg[LBNAME[bb]]:.4f} | {rs_b} | {rs_t} | "
                  f"{rs_b-rs_t:+d} | {ra} | {rt} |".replace("+0", "0"))
        _g = lambda d, txt: f"\\gain{{{txt}}}" if d > 0 else (f"\\loss{{{txt}}}" if d < 0 else txt)
        # per-task rank pairs: these NINE numbers are exactly what the rank sum adds up,
        # so the reader can verify the sum instead of taking it on trust.
        taskcells = []
        for i in range(9):
            rb_i = dense_rank(base[i], i, f_base)
            rt_i = dense_rank(tuned[i], i, f_tuned)
            taskcells.append(f"{rb_i}$\\to$" + g(rb_i - rt_i, str(rt_i)))
        sd, n = (ss[bb][2], ss[bb][3]) if bb in ss else (None, None)
        tail = f"$\\pm${2 * sd:.4f}" if sd is not None else ""  # two sample SDs
        ncol = ""  # seed count is stated once in the text, not repeated per table
        tex.append(f"{LBNAME[bb]} & " + " & ".join(taskcells)
                   + f" & {ab:.4f}$\\to$" + g(at - ab, f"{at:.4f}{tail}") + ncol
                   + f" & {rs_b}~({ra})$\\to$" + g(rs_b - rs_t, f"{rs_t}~({rt})") + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    raw = ["# HEST per-task raw values, base vs fine-tuned", "",
           "**Generated file -- do not hand-edit.**  Regenerate with",
           "`./.venv-hest/bin/python scripts/hest_ranks.py`.", "",
           "Mean Pearson $r$ per task, our harness. Fine-tuned is mean +/- sample SD over the",
           "n adapter seeds at each seed's own selected checkpoint. These are the values the",
           "per-task ranks in docs/hest_ranks.md are computed from.", "",
           "| backbone | " + " | ".join(TASKS) + " | average | n |",
           "|" + "---|" * (len(TASKS) + 3)]
    for bb in ORDER:
        base, tuned, n, per = ours[bb]
        if base is None or tuned is None:
            continue
        raw.append(f"| {LBNAME[bb]} base | " + " | ".join(f"{v:.4f}" for v in base)
                   + f" | {sum(base)/9:.4f} | 1 |")
        cells = []
        for i in range(9):
            sd = statistics.stdev([x[i] for x in per]) if len(per) > 1 else 0.0
            cells.append(f"{tuned[i]:.4f} +/- {sd:.4f}")
        raw.append(f"| {LBNAME[bb]} tuned | " + " | ".join(cells)
                   + f" | {sum(tuned)/9:.4f} | {n} |")
        raw.append(f"| {LBNAME[bb]} delta | "
                   + " | ".join(f"{tuned[i]-base[i]:+.4f}" for i in range(9))
                   + f" | {(sum(tuned)-sum(base))/9:+.4f} | |")
    (REPO / "docs/hest_per_task.md").write_text("\n".join(raw) + "\n")
    print("wrote docs/hest_per_task.md")
    (REPO / "docs/hest_ranks.md").write_text("\n".join(md) + "\n")
    TEX_OUT.parent.mkdir(parents=True, exist_ok=True)
    TEX_OUT.write_text("\n".join(tex) + "\n")
    print(f"validated {len(lb)}/{len(lb)} published averages; wrote docs/hest_ranks.md")


if __name__ == "__main__":
    main()
