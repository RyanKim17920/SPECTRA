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

--model takes one or more run names, searched in order, first hit per dataset wins. The
Midnight-12k sweep needs this: clsmean pooling (3072-d) crashes the segmentation decoder on
ViT-g, so its 12 classification sets land under <run>_clsmean and its 2 segmentation sets
under <run>_cls. That is a real methodological difference, not a bookkeeping one, so as soon
as more than one run name is given the table grows a `run_name` column (and the CSV a
`run_name` field) naming the run that supplied each row, plus a provenance footnote.

    python scripts/collect_thunder.py --model mbase_clsmean mbase_cls

The published columns and the `# cross-check` line are BACKBONE-GATED: they appear only for
a backbone this file actually holds published numbers for (today: phikon-v2). See PUBLISHED
and BACKBONE_RUN_PREFIXES below for how the backbone is determined and how to override it
with --backbone / WAIV_BACKBONE.
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

PHIKONV2 = "owkin/phikon-v2"
MIDNIGHT = "kaiko-ai/midnight"
VIRCHOW2 = "paige-ai/Virchow2"

# Published absolutes, KEYED BY BACKBONE, for cross-checking our base run. The key is the
# whole point: a published row belongs to one encoder, and a column headed "pub" next to a
# different encoder's F1 is not a cross-check, it is a backbone comparison wearing a
# reproduction's clothes. Only a backbone that has an entry here gets pub/delta columns.
#
# phikon-v2 only, today. No one on this cluster has ever run phikon-v2 through THUNDER --
# /data/vbelagali, /data/eva-data and /data/anis hold only in-house DINOv2/I-JEPA runs and
# OpenMidnight -- so the external reference is the paper appendix, not a neighbouring
# results tree.
#   knn            arXiv:2507.07860v3 Table S37   (12-dataset mean 70.1 = Table 4 KNN col)
#   linear_probing arXiv:2507.07860v3 Table S39   (also on the live leaderboard, identical)
#   segmentation   arXiv:2507.07860v3 Table S50, reported as Dice; binary Dice == F1
# Tolerance: Waiv's own mixed-precision re-run of phikon-v2 moves lin-probe ~-0.4 and
# segmentation ~-0.9 against these, so treat ~1 point as agreement, not a regression.
#
# THUNDER's leaderboard DOES carry a Midnight-12k row, but nobody has transcribed it into
# this file, so there is deliberately no MIDNIGHT key below. Adding one (same task ->
# dataset -> percent shape, plus a PUBLISHED_SOURCE entry) is all that is needed to turn
# the Midnight cross-check back on; until then Midnight runs print an explicit
# "NO published counterpart" note rather than a comparison against phikon-v2's numbers.
# The same holds for paige-ai/Virchow2: no transcribed row, so no key, so no columns.
PUBLISHED = {
    PHIKONV2: {
        "knn": {
            "bach": 53.1, "bracs": 45.9, "break_his": 51.6, "ccrcc": 77.2, "crc": 92.1,
            "esca": 75.8, "mhist": 66.1, "patch_camelyon": 82.2, "tcga_crc_msi": 56.8,
            "tcga_tils": 80.8, "tcga_uniform": 69.1, "wilds": 91.0,
        },
        "linear_probing": {
            "bach": 64.7, "bracs": 58.2, "break_his": 53.0, "ccrcc": 76.7, "crc": 92.0,
            "esca": 77.3, "mhist": 79.2, "patch_camelyon": 90.8, "tcga_crc_msi": 61.1,
            "tcga_tils": 91.0, "tcga_uniform": 77.7, "wilds": 95.8,
        },
        "segmentation": {
            "ocelot": 78.7, "pannuke": 61.0, "segpath_epithelial": 69.1,
            "segpath_lymphocytes": 60.9,
        },
    },
}

# Citation printed on the `# cross-check` line, per backbone.
PUBLISHED_SOURCE = {PHIKONV2: "arXiv:2507.07860v3"}

# HOW THE BACKBONE IS DETERMINED, and why it is not read from metadata.
#
# There is no metadata to read. THUNDER writes nothing about the encoder into its results
# tree: outputs/res/<ds>/<run>/<task>/frozen/config.json carries exactly
#   adaptation, ckpt_saving, dataset, data_loading, task, wandb, embedding_recomputing,
#   model_retraining
# and not one of those mentions the model -- THUNDER only ever sees `custom:<path>.py`, and
# thunder_model.py resolves WAIV_BACKBONE internally at import time. run_thunder.sbatch's
# banner echoes dataset/tasks/pooling/run_name/epochs but NOT WAIV_BACKBONE, so the slurm
# logs do not carry it either (checked: no hit for backbone/midnight/phikon in the
# mbase_clsmean job logs). The only physical trace is the embedding cache's width
# (phikon-v2 ViT-L: 1024 cls / 2048 clsmean; Midnight ViT-g: 1536 / 3072), which lives
# inside embeddings.h5 -- unreadable without h5py, which is not in the interpreter this
# script is normally run with, and absent entirely for segmentation-only runs.
#
# So the backbone is an INPUT: --backbone is authoritative, and the prefix table below is
# the documented default for the run names this repo actually produces
# (scripts/submit_midnight_thunder.sh and scripts/submit_segpath_thunder.sh name Midnight
# runs mbase_*/mft*_* and phikon-v2 runs base_*/ft*_*). A run name matching NO prefix
# resolves to None -- unknown, no published columns -- which is the safe answer for a third
# backbone, rather than defaulting it into phikon-v2's table.
#
# Virchow2 runs are named vbase_*/vft*_* (scripts/submit_thunder.sh --backbone virchow2).
# Order matters only in that the longer, more specific prefixes must be tried first --
# "vbase_clsmean" does not start with "base", so there is no actual collision here, but
# keeping the per-backbone letters grouped above the bare phikon-v2 pair preserves that
# property if a future run name ever drops the leading letter.
#
# There is deliberately no VIRCHOW2 key in PUBLISHED: this repo holds no transcribed
# THUNDER leaderboard row for Virchow2, so it takes the same "NO published counterpart"
# path Midnight takes, and its F1s are never diffed against phikon-v2's appendix.
BACKBONE_RUN_PREFIXES = (
    ("mbase", MIDNIGHT),
    ("mft", MIDNIGHT),
    ("vbase", VIRCHOW2),
    ("vft", VIRCHOW2),
    ("base", PHIKONV2),
    ("ft", PHIKONV2),
)


def infer_backbone(run_name: str) -> str | None:
    """Map a WAIV_RUN_NAME to its backbone, or None when the convention does not cover it."""
    for prefix, backbone in BACKBONE_RUN_PREFIXES:
        if run_name.startswith(prefix):
            return backbone
    return None


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
    ap.add_argument("--model", required=True, nargs="+",
                    help="one or more PretrainedModel.name (WAIV_RUN_NAME), searched in order; "
                         "the first with results on disk supplies each dataset row")
    ap.add_argument("--adaptation", default="frozen")
    ap.add_argument("--backbone", default=os.environ.get("WAIV_BACKBONE"),
                    help="encoder these runs were produced with, e.g. owkin/phikon-v2 or "
                         "kaiko-ai/midnight. Decides whether published columns are shown. "
                         "Defaults to WAIV_BACKBONE, else inferred from the run name "
                         "prefix (see BACKBONE_RUN_PREFIXES)")
    ap.add_argument("--csv", action="store_true")
    args = ap.parse_args()

    # Resolve once, up front: every --model name is merged into ONE table, so they had
    # better be one encoder. Disagreement is a bookkeeping error, not something to average.
    if args.backbone:
        backbone = args.backbone
    else:
        guesses = {m: infer_backbone(m) for m in args.model}
        distinct = {b for b in guesses.values() if b is not None}
        if len(distinct) > 1:
            ap.error("--model names map to more than one backbone "
                     + ", ".join(f"{m}->{b}" for m, b in guesses.items())
                     + "; they cannot share one table. Pass --backbone if this is wrong.")
        backbone = next(iter(distinct)) if distinct else None
    pub_tbl: dict[str, dict[str, float]] = PUBLISHED.get(backbone or "", {})

    res = Path(args.root) / "outputs" / "res"
    table: dict[str, dict[str, tuple]] = {}
    # Which run_name actually supplied each row. With one --model this is constant and the
    # column is suppressed so single-name output is unchanged; with several it is the whole
    # point of the merge and must be visible.
    source: dict[str, str] = {}
    for ds in PAPER_CLS + PAPER_SEG:
        found: dict[str, dict[str, tuple]] = {}
        for model in args.model:
            got: dict[str, tuple] = {}
            for task in TASKS:
                p = res / ds / model / task / args.adaptation / "outputs.json"
                if not p.is_file():
                    continue
                try:
                    f1, ece = _score(json.loads(p.read_text()), task)
                except Exception as e:  # a truncated file from a killed job must not hide the rest
                    print(f"# WARN unreadable {p}: {e}")
                    continue
                got[task] = (f1, ece)
            if got:
                found[model] = got
        if not found:
            continue
        names = list(found)
        if len(names) > 1:
            # Two pooling variants of the same dataset are not interchangeable -- say so.
            print(f"# WARN {ds} has results under {' and '.join(names)}; using {names[0]}")
        table[ds] = found[names[0]]
        source[ds] = names[0]

    show_src = len(args.model) > 1
    cols = [t for t in TASKS if any(t in v for v in table.values())]
    if args.csv:
        print("dataset,split," + ("run_name," if show_src else "") + ",".join(cols))
        for ds in PAPER_CLS + PAPER_SEG:
            if ds not in table:
                continue
            grp = "classification" if ds in PAPER_CLS else "segmentation"
            print(f"{ds},{grp}," + (f"{source[ds]}," if show_src else "") + ",".join(
                "" if t not in table[ds] or table[ds][t][0] is None else f"{table[ds][t][0]:.4f}"
                for t in cols))
        return

    hdr = []
    for c in cols:
        hdr.append(f"{c} F1")
        if c in pub_tbl:
            hdr += [f"{c} pub", f"{c} Δ"]
    print(f"| dataset |{' run_name |' if show_src else ''} {' | '.join(hdr)} | LP ECE |")
    print("|" + "---|" * (len(hdr) + 2 + (1 if show_src else 0)))
    deltas: dict[str, list[float]] = {}
    for ds in PAPER_CLS + PAPER_SEG:
        src = f" {source.get(ds, '--')} |" if show_src else ""
        if ds not in table:
            # segpath_epithelial was absent from this cluster until 2026-08-03, when it was
            # downloaded from Zenodo record 7412731 into /data/ryan.kim/thunder/datasets.
            # Both segpath sets are now submittable via scripts/submit_segpath_thunder.sh.
            status = "not run"
            print(f"| {ds} |{src} " + " | ".join("--" for _ in hdr) + f" | -- |  <!-- {status} -->")
            continue
        cells = []
        for t in cols:
            f1 = table[ds].get(t, (None, None))[0]
            # THUNDER stores fractions, the paper prints percentages.
            pct = None if f1 is None else f1 * 100
            cells.append("--" if pct is None else f"{pct:.1f}")
            if t in pub_tbl:
                pub = pub_tbl[t].get(ds)
                cells.append("--" if pub is None else f"{pub:.1f}")
                if pub is None or pct is None:
                    cells.append("--")
                else:
                    d = pct - pub
                    deltas.setdefault(t, []).append(d)
                    cells.append(f"{d:+.1f}")
        ece = table[ds].get("linear_probing", (None, None))[1]
        print(f"| {ds} |{src} {' | '.join(cells)} | {'--' if ece is None else f'{ece:.4f}'} |")
    if show_src:
        by: dict[str, list[str]] = {}
        for ds, m in source.items():
            by.setdefault(m, []).append(ds)
        for m in args.model:
            if m in by:
                print(f"# provenance {m}: {len(by[m])} rows -- {', '.join(by[m])}")
    src_cite = PUBLISHED_SOURCE.get(backbone or "", "published")
    for t, ds_ in deltas.items():
        worst = max(ds_, key=abs)
        print(f"# cross-check {t}: n={len(ds_)} meanΔ={sum(ds_)/len(ds_):+.2f} "
              f"max|Δ|={worst:+.1f} vs {src_cite}")
    if not pub_tbl:
        # Say it out loud. Silently dropping the columns would read as "the cross-check was
        # forgotten"; silently keeping them would be the bug this guard exists to stop.
        # Wording mirrors scripts/run_hest.py's own NO-published-counterpart note.
        if backbone:
            print(f"# backbone={backbone} has NO published counterpart here -- this table is "
                  "our own reference for")
            print("# checkpoint-to-checkpoint retention only, so no published columns and no "
                  "cross-check are")
            print(f"# emitted. arXiv:2507.07860v3's rows in this script are "
                  f"{PHIKONV2} and nothing else.")
        else:
            print(f"# backbone UNKNOWN for run name(s) {', '.join(args.model)} -- no prefix in "
                  "BACKBONE_RUN_PREFIXES")
            print("# matches, so published columns and the cross-check are withheld rather "
                  "than guessed. Pass")
            print("# --backbone (or set WAIV_BACKBONE) to name the encoder these results "
                  "came from.")

    # A mean over a partial roster is not the paper's mean; label it so nobody quotes it.
    for grp, names in (("classification", PAPER_CLS), ("segmentation", PAPER_SEG)):
        for t in cols:
            vals = [table[d][t][0] for d in names if d in table and table[d].get(t, (None,))[0] is not None]
            if vals:
                print(f"# mean {grp} {t} F1 over {len(vals)}/{len(names)} present = {sum(vals)/len(vals):.4f}")


if __name__ == "__main__":
    main()
