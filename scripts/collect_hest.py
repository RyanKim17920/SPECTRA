#!/usr/bin/env python
"""Build the HEST retention table from the per-run summaries run_hest.py leaves behind.

Counterpart to collect_thunder.py. Reads <results-dir>/<exp_code>_summary.json and prints
one row per leaderboard task (plus Avg), one column pair per run: absolute Pearson and the
delta against the base row.

Two pooling families, and they are NOT interchangeable:

  cls      1024-d, HEST's own published protocol. Our base reproduces the published
           phikon_v2 row to 1e-16, so the published column is a real external gate.
  clsmean  2048-d, what the PathoROB numbers are computed on. There is NO published
           counterpart -- the base column is our own reference and the only thing the
           fine-tuned rows may be compared against.

The benchmark's dynamic range is narrow (0.3252 ResNet50 -> 0.4229 H-Optimus-1), so read
deltas against that span, not against zero.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from waivphaet.eval.hest_adapter import (  # noqa: E402
    HEST_RANGE,
    LEADERBOARD_TASKS,
    PUBLISHED_PHAET_AVG,
    PUBLISHED_PHIKONV2_AVG,
    PUBLISHED_PHIKONV2_CLS,
    WAIV_PHAET_HEST,
    WAIV_PHIKONV2_HEST,
)

# Env-overridable, same shape as collect_thunder.py's THUNDER_BASE_DATA_FOLDER default.
# WAIV_HEST_RESULTS points straight at the results dir; WAIV_HEST_WORK_DIR mirrors
# hest_adapter.DEFAULT_WORK_DIR and gets "/results" appended, so whichever of the two the
# caller already exports works. Value with neither set is the previous hardcoded path.
DEFAULT_RESULTS = Path(
    os.environ.get("WAIV_HEST_RESULTS")
    or (Path(os.environ.get("WAIV_HEST_WORK_DIR", "/data/ryan.kim/hest_work")) / "results")
)

PHIKONV2 = "owkin/phikon-v2"

# Summaries written before run_hest.py started recording "backbone" (base_cls, base_clsmean,
# ft2000_*, smoke_skcm) have no field to read. Their embed_dim is still a witness: phikon-v2
# is a ViT-L, so cls is 1024-d and clsmean 2048-d, while Midnight's ViT-g is 1536/3072 and
# no other backbone in this repo shares the ViT-L widths. A legacy summary matching this
# signature is treated as phikon-v2; anything else is left unknown, which withholds the
# published column rather than guessing it.
LEGACY_PHIKONV2_DIMS = {"cls": 1024, "clsmean": 2048}


def resolve_backbone(blob: dict | None, pooling: str, override: str | None) -> str | None:
    """Backbone of a run: explicit flag > recorded field > legacy embed_dim signature."""
    if override:
        return override
    if blob is None:
        return None
    recorded = blob.get("backbone")
    if recorded:
        return str(recorded)
    if blob.get("embed_dim") == LEGACY_PHIKONV2_DIMS.get(pooling):
        return PHIKONV2
    return None


def load(results_dir: Path, exp_code: str) -> dict | None:
    p = results_dir / f"{exp_code}_summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def discover(results_dir: Path, pooling: str, base: str) -> list[str]:
    """Every <something>_<pooling>_summary.json that is not the base, sorted by name."""
    out = []
    for p in sorted(results_dir.glob(f"*_{pooling}_summary.json")):
        code = p.name[: -len("_summary.json")]
        if code != base:
            out.append(code)
    return out


def fmt(v: float | None, digits: int = 4) -> str:
    return "--" if v is None else f"{v:.{digits}f}"


def signed(v: float | None, digits: int = 4) -> str:
    return "--" if v is None else f"{v:+.{digits}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--pooling", default="cls", choices=("cls", "clsmean"))
    ap.add_argument("--base", default=None, help="default base_<pooling>")
    ap.add_argument("--backbone", default=os.environ.get("WAIV_BACKBONE"),
                    help="override the backbone recorded in the base run's summary JSON; "
                         "only owkin/phikon-v2 has a published HEST row")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="exp_codes to compare; default: auto-discover all non-base "
                         "*_<pooling>_summary.json")
    ap.add_argument("--csv", action="store_true")
    args = ap.parse_args()

    base_code = args.base or f"base_{args.pooling}"
    runs = args.runs if args.runs is not None else discover(args.results_dir, args.pooling, base_code)

    base_blob = load(args.results_dir, base_code)
    missing = []
    if base_blob is None:
        missing.append(base_code)
    loaded: list[tuple[str, dict]] = []
    for r in runs:
        blob = load(args.results_dir, r)
        if blob is None:
            missing.append(r)
        else:
            loaded.append((r, blob))

    base_res = (base_blob or {}).get("results", {})
    rows = list(LEADERBOARD_TASKS) + ["avg"]

    def val(blob: dict, task: str) -> float | None:
        # F-E fix (2026-08-26): the "avg" ROW is the repo's headline HEST scalar, and it
        # must be THE headline scalar -- hest_perf_per_encoder.custom_encoder -- not
        # `results.avg`, which is a ROUNDED copy of it.  Reading the rounded field here
        # while collect_final5 / scoreboard / final_recipe_report read custom_encoder is
        # why this script and collect_final5 printed different HEST numbers for the SAME
        # run.  One field, repo-wide; collect_final5 owns the reader.
        if task == "avg":
            v = (blob.get("hest_perf_per_encoder") or {}).get("custom_encoder")
            if v is None:
                v = blob.get("results", {}).get(task)   # pre-custom_encoder summaries
        else:
            v = blob.get("results", {}).get(task)
        return None if v is None else float(v)

    # BOTH halves gate the published column. Pooling alone is not enough: 0.3747 is
    # phikon-v2's CLS row, so a Midnight (or any third backbone) run with cls pooling --
    # `--base mbase_cls` -- would otherwise print it and diff against it, which compares two
    # different encoders and calls the result a reproduction.
    backbone = resolve_backbone(base_blob, args.pooling, args.backbone)
    show_published = args.pooling == "cls" and backbone == PHIKONV2

    if args.csv:
        head = ["task", f"base:{base_code}"]
        if show_published:
            head.append("published_phikonv2")
        for code, _ in loaded:
            head += [code, f"delta_{code}"]
        print(",".join(head))
        for task in rows:
            b = base_res.get(task)
            cells = [task, fmt(b)]
            if show_published:
                pub = PUBLISHED_PHIKONV2_AVG if task == "avg" else PUBLISHED_PHIKONV2_CLS.get(task)
                cells.append(fmt(pub))
            for _code, blob in loaded:
                v = val(blob, task)
                d = None if (v is None or b is None) else v - float(b)
                cells += [fmt(v), signed(d)]
            print(",".join(cells))
        return 0

    print(f"# HEST retention -- pooling={args.pooling}  ({args.results_dir})")
    print(f"# base row: {base_code}")
    if missing:
        print(f"# MISSING (not yet on disk, columns omitted): {', '.join(missing)}")
    if not loaded:
        print("# no fine-tuned runs found yet -- base row only")
    if show_published:
        print("# 'published' is HEST's own phikon_v2 row (CLS, fp32); Waiv Table 1 quotes it verbatim.")
    elif args.pooling != "cls":
        print("# clsmean has NO published counterpart. The base column is OUR OWN reference,")
        print("# valid only for checkpoint-to-checkpoint retention. Do not compare it to 0.3747.")
    else:
        # cls pooling, wrong (or unrecorded) backbone -- wording mirrors run_hest.py's note.
        print(f"# backbone={backbone or 'UNKNOWN'} pooling=cls has NO published counterpart here --")
        print("# this is our own reference for checkpoint-to-checkpoint retention only.")
        print("# 0.3747 is phikon-v2 CLS and nothing else.")
    print(f"# benchmark dynamic range across all encoders: {HEST_RANGE[0]:.4f} - {HEST_RANGE[1]:.4f} "
          f"(span {HEST_RANGE[1] - HEST_RANGE[0]:.4f})")
    print()

    head = ["task", f"base ({base_code})"]
    if show_published:
        head.append("published phikon-v2")
    for code, _ in loaded:
        head += [code, f"Δ vs base"]
    print("| " + " | ".join(head) + " |")
    print("|" + "|".join(["---"] * len(head)) + "|")

    for task in rows:
        b = base_res.get(task)
        label = "**Avg**" if task == "avg" else task
        cells = [label, f"**{fmt(b)}**" if task == "avg" else fmt(b)]
        if show_published:
            pub = PUBLISHED_PHIKONV2_AVG if task == "avg" else PUBLISHED_PHIKONV2_CLS.get(task)
            cells.append(fmt(pub))
        for _code, blob in loaded:
            v = val(blob, task)
            d = None if (v is None or b is None) else v - float(b)
            cells += [f"**{fmt(v)}**" if task == "avg" else fmt(v), signed(d)]
        print("| " + " | ".join(cells) + " |")

    print()
    if show_published:
        print(f"Waiv's published Phaet HEST average: **{PUBLISHED_PHAET_AVG:.4f}** "
              f"(from {PUBLISHED_PHIKONV2_AVG:.4f} base, i.e. {PUBLISHED_PHAET_AVG - PUBLISHED_PHIKONV2_AVG:+.4f}).")
        print("Per-task Phaet row (arXiv:2607.22861 Table 3), for Δ-vs-Δ comparison:")
        for t, v in WAIV_PHAET_HEST.items():
            print(f"  {t:10s} {v:.4f}  (base {WAIV_PHIKONV2_HEST[t]:.4f}, "
                  f"their Δ {v - WAIV_PHIKONV2_HEST[t]:+.4f})")
    for code, blob in loaded:
        print(f"- {code}: embed_dim={blob.get('embed_dim')} precision={blob.get('precision')} "
              f"seconds={blob.get('seconds')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
