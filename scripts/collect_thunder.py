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
# F7 fix (2026-08-26): renamed.  Two modules used to export a symbol called
# `PAPER_SEG` with DIFFERENT contents -- 4 here, 2 in collect_final5 -- so which one a
# consumer got depended on which module it imported.  The names are now distinct:
#   PAPER_SEG_PUBLISHED = Waiv's published 4-dataset segmentation panel (this file)
#   collect_final5.PAPER_SEG = the 2-dataset panel WE actually submitted
# A mean over the 2 we ran is NOT comparable to Waiv's 4-dataset published mean.
PAPER_SEG_PUBLISHED = ["ocelot", "pannuke", "segpath_epithelial", "segpath_lymphocytes"]
PAPER_SEG_SUBMITTED = ["ocelot", "pannuke"]

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
# this file, so there is deliberately no MIDNIGHT key in PUBLISHED below -- PUBLISHED holds
# THUNDER-paper per-DATASET F1s and those exist here for phikon-v2 only.
#
# Midnight and Virchow2 published rows DO now exist in this script, but as a SEPARATE dict:
# PUBLISHED_TASKMEAN, sourced from the WAIV paper (arXiv:2607.22861 Table 2), which reports
# per-TASK means rather than per-dataset F1s. The two dicts are kept apart on purpose --
# see the long note on PUBLISHED_TASKMEAN for why merging them would be a category error.

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

# ---------------------------------------------------------------------------------------
# Published per-TASK means, KEYED BY BACKBONE. DIFFERENT PAPER, DIFFERENT SHAPE, DIFFERENT
# DICT -- read this before using it.
#
# PUBLISHED above comes from arXiv:2507.07860v3, THUNDER's own paper, and is per DATASET
# (the 12 classification + 4 segmentation F1s in appendix Tables S37/S39/S50). It exists
# for phikon-v2 only, because that is the only backbone whose per-dataset appendix rows
# anyone has transcribed.
#
# PUBLISHED_TASKMEAN below comes from arXiv:2607.22861 Table 2 -- the WAIV paper, a
# different group evaluating a different roster -- and is per TASK: one number per model
# per task, ALREADY averaged over THUNDER's datasets. Full transcription of all 20 models
# and all four of Waiv's tables lives in docs/waiv_published.json; the three backbones this
# repo actually runs are copied here so the cross-check works without loading that file.
#
# WHY NOT ONE DICT. A task mean is not a dataset score. Pasting Waiv's 80.0 kNN into the
# per-dataset PUBLISHED shape would put an average in a cell headed `bach` and the script
# would happily print a delta against it. Worse, the two dicts disagree by construction:
# they are two labs' independent evaluations, so phikon-v2 appears in BOTH with different
# numbers (THUNDER's own per-dataset rows vs Waiv's re-run task means), and averaging or
# silently preferring one would erase a real reproducibility signal. Hence: two dicts, two
# source citations, two clearly-labelled cross-check lines, never merged.
#
# TASK-NAME MAPPING. Waiv's task keys are not ours:
#     Waiv "knn"          -> our "knn"
#     Waiv "linear"       -> our "linear_probing"
#     Waiv "few_shot"     -> our "simple_shot"
#     Waiv "segmentation" -> our "segmentation"
# Waiv additionally score "calibration" and "adversarial", which this script does not
# compute; they are transcribed below for completeness and are excluded from every mean
# and delta the script prints. Any aggregate built here is over 4 tasks, NOT Waiv's 6, and
# is therefore NOT comparable to their rank sum.
#
# CAVEAT ON THE DELTA. Our task mean is over the datasets WE ran, which for most sweeps is
# a subset of THUNDER's 12+4. Waiv's is over their full set. The cross-check line prints n
# so an unequal-support comparison is visible rather than implied; treat it as a sanity
# check on magnitude, not as a matched-pairs delta.
#
# variant="base" rows only. Waiv also publish their own fine-tuned Virchow2 (RI 0.918,
# thunder knn 82.6 / linear 85.1 / few_shot 76.6 / segmentation 68.0); that is THEIR method's
# output, not a published base to reproduce, so it is deliberately not a cross-check target
# here. It is in docs/waiv_published.json under variant "fine-tuned" for the
# Delta-vs-Delta comparison in docs/RESULTS.md.
PUBLISHED_TASKMEAN = {
    VIRCHOW2: {  # Waiv Table 2, "Virchow2" variant=base
        "knn": 82.9, "linear": 84.8, "few_shot": 73.9,
        "segmentation": 68.2,
        # not computed by this script -- transcribed for completeness only:
        "calibration": 3.6, "adversarial": 31.1,
    },
    MIDNIGHT: {  # Waiv Table 2, "Midnight-12k" variant=base
        "knn": 80.0, "linear": 84.4, "few_shot": 71.5,
        "segmentation": 66.0,
        # not computed by this script -- transcribed for completeness only:
        "calibration": 2.4, "adversarial": 35.7,
    },
    PHIKONV2: {  # Waiv Table 2, "Phikon-v2" variant=base
        "knn": 74.0, "linear": 79.3, "few_shot": 71.8,
        "segmentation": 66.5,
        # not computed by this script -- transcribed for completeness only:
        "calibration": 4.5, "adversarial": 41.9,
    },
}

# Citation printed on the `# cross-check taskmean` line, per backbone. Distinct from
# PUBLISHED_SOURCE on purpose: these are not the same paper.
PUBLISHED_TASKMEAN_SOURCE = {
    PHIKONV2: "arXiv:2607.22861 Table 2 (Waiv)",
    MIDNIGHT: "arXiv:2607.22861 Table 2 (Waiv)",
    VIRCHOW2: "arXiv:2607.22861 Table 2 (Waiv)",
}

# Waiv task key -> this script's TASKS name. Waiv-only tasks map to None (not computed here).
WAIV_TASK_ALIAS = {
    "knn": "knn",
    "linear": "linear_probing",
    "few_shot": "simple_shot",
    "segmentation": "segmentation",
    "calibration": None,
    "adversarial": None,
}


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
# PUBLISHED still holds only PHIKONV2, and that is now a TRANSCRIPTION gap, not an absence
# in the literature. Waiv publish per-task THUNDER for every backbone here in
# arXiv:2607.22861 Table 2 -- including Virchow2 and Midnight -- and all four of their tables
# are transcribed in docs/waiv_published.json. Two things block a straight paste into
# PUBLISHED, both real:
#   1. PUBLISHED is keyed per DATASET (the 12+4 F1s from the phikon-v2 appendix, S37/S39/S50).
#      Table 2 is per TASK -- one kNN/linear/few-shot/segmentation number per model, already
#      averaged over the datasets. The two are not the same shape, and pasting task means into
#      a per-dataset dict would silently compare an average against a single dataset.
#   2. Table 2 scores 6 tasks; this script computes 4. Any mean built from it must exclude
#      calibration and adversarial attack, which is a different quantity from Waiv's rank sum.
# So the Delta-vs-Delta comparison lives in docs/RESULTS.md Sections 2 and 6, computed from
# waiv_published.json at the task level where the shapes match. Do NOT add a VIRCHOW2 key
# here until PUBLISHED grows a per-task variant; the "NO published counterpart" line this
# script prints means "none transcribed IN THIS DICT", and docs/RESULTS.md says so explicitly.
BACKBONE_RUN_PREFIXES = (
    ("mbase", MIDNIGHT),
    ("mft", MIDNIGHT),
    ("vbase", VIRCHOW2),
    ("vft", VIRCHOW2),
    ("base", PHIKONV2),
    ("ft", PHIKONV2),
)


def read_provenance(run_name: str, root: str | Path | None = None) -> dict | None:
    """The provenance sidecar for a run name, if scripts/write_thunder_provenance.py wrote one.

    THUNDER records nothing about the encoder (see the long comment above), so this sidecar
    -- ``outputs/provenance/<run_name>.json``, with per-results-dir copies named
    ``waiv_provenance.json`` -- is the only artifact that binds a results directory to the
    checkpoint that produced it (adapter path + sha256 + source training job). Absent for
    every run predating it, hence the None return and the prefix table below as fallback.
    """
    base = Path(root or os.environ.get("THUNDER_BASE_DATA_FOLDER", "/data/ryan.kim/thunder"))
    p = base / "outputs" / "provenance" / f"{run_name}.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def infer_backbone(run_name: str, root: str | Path | None = None) -> str | None:
    """Map a WAIV_RUN_NAME to its backbone, or None when nothing covers it.

    EVIDENCE FIRST: a provenance sidecar records the backbone that was actually exported
    for the run, so it wins over the name-prefix convention -- which is a convention, not a
    fact, and covers only the run names this repo happened to have produced by 2026-08.
    ``ph2mask_*`` matches no prefix at all and resolved to None before the sidecar existed.
    """
    prov = read_provenance(run_name, root)
    if prov and prov.get("backbone"):
        return prov["backbone"]
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
        guesses = {m: infer_backbone(m, args.root) for m in args.model}
        distinct = {b for b in guesses.values() if b is not None}
        if len(distinct) > 1:
            ap.error("--model names map to more than one backbone "
                     + ", ".join(f"{m}->{b}" for m, b in guesses.items())
                     + "; they cannot share one table. Pass --backbone if this is wrong.")
        backbone = next(iter(distinct)) if distinct else None
    pub_tbl: dict[str, dict[str, float]] = PUBLISHED.get(backbone or "", {})
    # Waiv's per-task means are available for all three backbones we run, so this one is
    # NOT gated the same way as pub_tbl -- a Midnight run gets a task-level cross-check even
    # though it has no per-dataset row.
    pub_task: dict[str, float] = PUBLISHED_TASKMEAN.get(backbone or "", {})

    res = Path(args.root) / "outputs" / "res"
    table: dict[str, dict[str, tuple]] = {}
    # Which run_name actually supplied each row. With one --model this is constant and the
    # column is suppressed so single-name output is unchanged; with several it is the whole
    # point of the merge and must be visible.
    source: dict[str, str] = {}
    for ds in PAPER_CLS + PAPER_SEG_PUBLISHED:
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
        for ds in PAPER_CLS + PAPER_SEG_PUBLISHED:
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
    for ds in PAPER_CLS + PAPER_SEG_PUBLISHED:
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
    if pub_task:
        # Our side of this comparison is a mean over the datasets present in `table`, which
        # is why n is printed: an n far below THUNDER's 12 (classification) or 4
        # (segmentation) means the two means do not rest on the same support.
        cite = PUBLISHED_TASKMEAN_SOURCE.get(backbone or "", "published")
        print(f"# taskmean source: {cite} -- per-TASK means, NOT the per-dataset "
              f"{PUBLISHED_SOURCE.get(backbone or '', 'THUNDER paper')} rows above")
        for waiv_key, ours in WAIV_TASK_ALIAS.items():
            pub = pub_task.get(waiv_key)
            if pub is None:
                continue
            if ours is None:
                print(f"# taskmean {waiv_key}: pub={pub:.1f} -- not computed by this script")
                continue
            vals = [table[ds][ours][0] * 100 for ds in table
                    if ours in table[ds] and table[ds][ours][0] is not None]
            if not vals:
                print(f"# taskmean {waiv_key}: pub={pub:.1f} ours=-- (no rows)")
                continue
            ours_mean = sum(vals) / len(vals)
            print(f"# taskmean {waiv_key}: n={len(vals)} ours={ours_mean:.1f} "
                  f"pub={pub:.1f} \u0394={ours_mean - pub:+.1f}")

    if not pub_tbl:
        # Say it out loud. Silently dropping the columns would read as "the cross-check was
        # forgotten"; silently keeping them would be the bug this guard exists to stop.
        # Wording mirrors scripts/run_hest.py's own NO-published-counterpart note.
        if backbone:
            print(f"# backbone={backbone} has NO PER-DATASET published counterpart -- the "
                  "table above is our own")
            print("# reference for checkpoint-to-checkpoint retention only, so no "
                  "per-dataset published columns")
            print(f"# and no per-dataset cross-check are emitted. arXiv:2507.07860v3's "
                  f"rows in this script are")
            print(f"# {PHIKONV2} and nothing else. The `# taskmean` lines above, when "
                  "present, are a DIFFERENT")
            print("# paper (Waiv, arXiv:2607.22861) at a DIFFERENT granularity (per-task "
                  "means) -- do not read")
            print("# them as the same cross-check.")
        else:
            print(f"# backbone UNKNOWN for run name(s) {', '.join(args.model)} -- no prefix in "
                  "BACKBONE_RUN_PREFIXES")
            print("# matches, so published columns and the cross-check are withheld rather "
                  "than guessed. Pass")
            print("# --backbone (or set WAIV_BACKBONE) to name the encoder these results "
                  "came from.")

    # A mean over a partial roster is not the paper's mean; label it so nobody quotes it.
    for grp, names in (("classification", PAPER_CLS),
                       ("segmentation[published-4]", PAPER_SEG_PUBLISHED),
                       ("segmentation[submitted-2]", PAPER_SEG_SUBMITTED)):
        for t in cols:
            vals = [table[d][t][0] for d in names if d in table and table[d].get(t, (None,))[0] is not None]
            if vals:
                print(f"# mean {grp} {t} F1 over {len(vals)}/{len(names)} present = {sum(vals)/len(vals):.4f}")


if __name__ == "__main__":
    main()
