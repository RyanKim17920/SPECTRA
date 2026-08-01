#!/usr/bin/env python
"""Collect THUNDER results into a per-task absolute-score table.

Deliberately does NOT compute the rank sum. THUNDER's rank sum is a rank over whatever
roster of models happens to be on the leaderboard, so it moves when the roster moves and
is not reproducible from our own numbers alone: Waiv quote 97 for phikon-v2, the THUNDER
paper says 77, and the live leaderboard says 89 -- three different numbers for one model.
Absolute per-dataset F1 is the only figure that means the same thing on every run, and it
is what a retention regression has to be measured in.

Reads   $THUNDER_BASE_DATA_FOLDER/outputs/res/<dataset>/<model>/<task>/<adaptation>/outputs.json
Writes  a markdown/CSV table to stdout.

    python scripts/collect_thunder.py --model base_cls
    python scripts/collect_thunder.py --model base_cls --csv > runs/thunder_base.csv
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# The paper's 16. thunder/utils/results.py ships a *different* 16 (it swapped the two
# segpath sets for the four SPIDER sets, which postdate the paper), so aggregating with
# their helper does not reproduce the published table. Kept explicit here for that reason.
PAPER_CLS = [
    "bach", "bracs", "break_his", "ccrcc", "crc", "esca", "mhist",
    "patch_camelyon", "tcga_crc_msi", "tcga_tils", "tcga_uniform", "wilds",
]
PAPER_SEG = ["ocelot", "pannuke", "segpath_epithelial", "segpath_lymphocytes"]

TASKS = ["knn", "linear_probing", "simple_shot", "segmentation"]


def _score(blob: dict, task: str) -> tuple[float | None, float | None]:
    """Pull (f1, ece) out of one outputs.json.

    knn nests under the k value and simple_shot under the shot count, while
    linear_probing and segmentation are flat -- so the shape has to be sniffed rather than
    assumed.

    knn searches k over [1,3,5,10,20,30,40,50] on val and persists ONLY the selected k
    (mhist came back as a lone "3"), so there is exactly one key to take and picking a
    fixed k=20 would just miss the file. simple_shot is the opposite -- every shot count
    is kept -- so there we do want a specific one, and 16 is the shot count in THUNDER's
    published simple_shot row.
    """
    def _flat(d: dict) -> tuple[float | None, float | None]:
        f1 = d.get("f1", {}).get("metric_score") if isinstance(d.get("f1"), dict) else d.get("f1")
        ece = d.get("ECE", {}).get("metric_score") if isinstance(d.get("ECE"), dict) else d.get("ECE")
        return f1, ece

    if task in ("linear_probing", "segmentation"):
        return _flat(blob)

    # nested: {"<selected k>": {...}} for knn, {"1".."16": {...}} for simple_shot
    keys = [k for k in blob if k.isdigit()]
    if not keys:
        return _flat(blob)
    if task == "knn":
        key = keys[0] if len(keys) == 1 else max(keys, key=int)
    else:
        key = "16" if "16" in keys else max(keys, key=int)
    return _flat(blob[key])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("THUNDER_BASE_DATA_FOLDER", "/data/ryan.kim/thunder"))
    ap.add_argument("--model", required=True, help="PretrainedModel.name, i.e. WAIV_RUN_NAME")
    ap.add_argument("--adaptation", default="frozen")
    ap.add_argument("--csv", action="store_true")
    args = ap.parse_args()

    res = Path(args.root) / "outputs" / "res"
    table: dict[str, dict[str, tuple]] = {}
    for ds in PAPER_CLS + PAPER_SEG:
        for task in TASKS:
            p = res / ds / args.model / task / args.adaptation / "outputs.json"
            if not p.is_file():
                continue
            try:
                f1, ece = _score(json.loads(p.read_text()), task)
            except Exception as e:  # a truncated file from a killed job must not hide the rest
                print(f"# WARN unreadable {p}: {e}")
                continue
            table.setdefault(ds, {})[task] = (f1, ece)

    cols = [t for t in TASKS if any(t in v for v in table.values())]
    if args.csv:
        print("dataset,split," + ",".join(cols))
        for ds in PAPER_CLS + PAPER_SEG:
            if ds not in table:
                continue
            grp = "classification" if ds in PAPER_CLS else "segmentation"
            print(f"{ds},{grp}," + ",".join(
                "" if t not in table[ds] or table[ds][t][0] is None else f"{table[ds][t][0]:.4f}"
                for t in cols))
        return

    print(f"| dataset | {' | '.join(c + ' F1' for c in cols)} | LP ECE |")
    print("|" + "---|" * (len(cols) + 2))
    for ds in PAPER_CLS + PAPER_SEG:
        if ds not in table:
            status = "MISSING (no data on this cluster)" if ds == "segpath_epithelial" else "not run"
            print(f"| {ds} | " + " | ".join("--" for _ in cols) + f" | -- |  <!-- {status} -->")
            continue
        cells = []
        for t in cols:
            f1 = table[ds].get(t, (None, None))[0]
            cells.append("--" if f1 is None else f"{f1:.4f}")
        ece = table[ds].get("linear_probing", (None, None))[1]
        print(f"| {ds} | {' | '.join(cells)} | {'--' if ece is None else f'{ece:.4f}'} |")

    # A mean over a partial roster is not the paper's mean; label it so nobody quotes it.
    for grp, names in (("classification", PAPER_CLS), ("segmentation", PAPER_SEG)):
        for t in cols:
            vals = [table[d][t][0] for d in names if d in table and table[d].get(t, (None,))[0] is not None]
            if vals:
                print(f"# mean {grp} {t} F1 over {len(vals)}/{len(names)} present = {sum(vals)/len(vals):.4f}")


if __name__ == "__main__":
    main()
