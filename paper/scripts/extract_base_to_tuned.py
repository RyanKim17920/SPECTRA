#!/usr/bin/env python3
"""Extract the numbers Figure 2 (base_to_tuned) plots into paper/data/base_to_tuned.json.

Figure 2 is drawn from four generated LaTeX tables, which are themselves produced by the
repo's table generators from the eval outputs (so they need /data):

    tables/pathorob_ranks.tex      <- scripts/pathorob_ranks.py   (mean RI, cells[0])
    tables/hest_ranks.tex          <- scripts/hest_ranks.py       (mean Pearson r, cells[9])
    tables/thunder_ranks_full.tex  <- scripts/thunder_ranks.py    (6 task means, cells[0..5])
    tables/cptac.tex               <- scripts/cptac_table.py      (base / tuned +- 1 SD)

This script parses those tables exactly as the figure script always did and freezes the
result as JSON, so the figure renders from the repo alone.  Every SD stored is TWO sample
SDs (the tables carry 2 SD except cptac.tex, which carries 1 and is doubled here).

    python3 paper/scripts/extract_base_to_tuned.py [--tables DIR]
      (default DIR: $SPECTRA_PAPER/tables) -> paper/data/base_to_tuned.json
"""
import argparse, json, re, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from _config import PAPER_TABLES  # noqa: E402

OUT = REPO / "paper/data/base_to_tuned.json"
ORDER = ["Phikon-v2", "OpenMidnight", "UNI2-h", "Midnight-12k", "H-optimus-0", "Virchow", "Virchow2"]


def norm(n): return {"H-Optimus-0": "H-optimus-0"}.get(n, n)


def rows(path):
    out = {}
    for line in open(path):
        if ("$\\to$" not in line and "\\pm" not in line) or line.startswith("%") or "&" not in line: continue
        cells = [c.strip() for c in line.rstrip("\\\n").split("&")]
        out[norm(cells[0].replace("\\", "").strip())] = cells[1:]
    return out


num = r"([-+]?\d*\.\d+|\d+)"


def pair(cell):
    """'0.47$\\to$\\gain{0.8356$\\pm$0.0069}~(3)' or '73.9~(20)$\\to$77.4$\\pm$0.2~\\gain{(18)}' -> base,tuned,sd"""
    b, t = cell.split("$\\to$")
    base = float(re.search(num, b).group(1))
    m = re.search(num + r"\$\\pm\$" + num, t)
    if m: return base, float(m.group(1)), float(m.group(2))
    t = re.sub(r"\\(gain|loss)\{[^}]*\}", "", t)
    return base, float(re.search(num, t).group(1)), np.nan


def cptac_pair(cells):
    """cptac.tex carries ONE sample SD; doubled to the figure's 2-SD convention."""
    base = float(re.search(num, cells[0]).group(1))
    m = re.search(num + r"\\pm" + num, cells[1])
    return base, float(m.group(1)), 2 * float(m.group(2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", type=Path, default=PAPER_TABLES)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    P = rows(a.tables / "pathorob_ranks.tex"); H = rows(a.tables / "hest_ranks.tex")
    T = rows(a.tables / "thunder_ranks_full.tex"); C = rows(a.tables / "cptac.tex")
    getters = {"pathorob_ri": lambda b: pair(P[b][0]), "hest_r": lambda b: pair(H[b][9]),
               "cptac_auc": lambda b: cptac_pair(C[b])}
    for j, k in enumerate(["thunder_knn", "thunder_lp", "thunder_fewshot", "thunder_seg",
                           "thunder_ece", "thunder_adv"]):
        getters[k] = (lambda j: lambda b: pair(T[b][j]))(j)
    data = {k: {b: [None if np.isnan(v) else v for v in g(b)] for b in ORDER} for k, g in getters.items()}
    doc = {"_about": "Figure 2 inputs: [base, fine-tuned mean, 2 sample SDs (null = no spread)] per "
                     "backbone. Extracted by paper/scripts/extract_base_to_tuned.py from the generated "
                     "tables pathorob_ranks.tex, hest_ranks.tex, thunder_ranks_full.tex, cptac.tex.",
           "order": ORDER, "panels": data}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
