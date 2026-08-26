#!/usr/bin/env python3
"""Per-dataset THUNDER seed-1 minus seed-0 deltas for the Virchow2 ret0.01 pair.

Reports RAW F1 POINTS per (dataset, task), plus a partial mean over the datasets complete
on BOTH seeds. The partial mean is NOT the 12-dataset task mean the resolvability bar is
defined on -- see the banner.

METRIC EXTRACTION IS IMPORTED FROM collect_thunder._score, NOT REIMPLEMENTED. outputs.json
has three different shapes: linear_probing is flat, knn nests under the single selected k,
and simple_shot nests under EVERY shot count {1,2,4,8,16}. A hand-rolled reader that takes
the best of those shot counts would be a max-over-C, which has manufactured a false effect
in this project before. Reusing the collector also guarantees these deltas are in the same
units and use the same selection rule as the scoreboard number they are meant to calibrate.
"""
import importlib.util, json, os, sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "collect_thunder", Path(__file__).resolve().parent / "collect_thunder.py")
_ct = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ct)

ROOT = Path(os.environ.get("THUNDER_BASE_DATA_FOLDER", "/data/ryan.kim/thunder")) / "outputs" / "res"
S0 = "f5_ret0.01-virchow2-s0-t900-391059_s0000250"
S1 = "f5_ret0.01-virchow2-s1-t900-392045_s0000250"
PAPER_CLS = _ct.PAPER_CLS
TASKS = ["knn", "linear_probing", "simple_shot"]
# Waiv Table 2, Virchow2: base -> fine-tuned, F1 points.
WAIV = {"knn": (82.9, 82.6), "linear_probing": (84.8, 85.1), "simple_shot": (73.9, 76.6)}
SD_BUDGET_FRAC = 0.10  # scoreboard UNRESOLVABLE_SD_PCT_LIMIT = 10.0

def score(run, ds, task):
    p = ROOT / ds / run / task / "frozen" / "outputs.json"
    if not p.is_file():
        return None
    f1, _ece = _ct._score(json.loads(p.read_text()), task)
    return None if f1 is None else f1 * 100.0

for task in TASKS:
    rs = []
    for ds in PAPER_CLS:
        a, b = score(S0, ds, task), score(S1, ds, task)
        if a is not None and b is not None:
            rs.append((ds, a, b, b - a))
    base, ft = WAIV[task]
    gain = ft - base
    budget = SD_BUDGET_FRAC * gain
    print(f"\n=== {task}   Waiv gain {gain:+.1f} pts"
          + (f"; task-mean SD budget {budget:.3f} pts" if gain > 0
             else "  [NEGATIVE GAIN -- no bar exists, cell unusable at any noise level]"))
    if not rs:
        print("  (no complete pairs yet)")
        continue
    print(f"  {'dataset':<18}{'seed0':>9}{'seed1':>9}{'delta':>9}")
    for ds, a, b, d in rs:
        print(f"  {ds:<18}{a:>9.3f}{b:>9.3f}{d:>+9.3f}")
    ds_d = [d for *_, d in rs]
    mad = sum(abs(d) for d in ds_d) / len(ds_d)
    pm = sum(ds_d) / len(ds_d)
    worst = max(ds_d, key=abs)
    print(f"  n={len(rs)}/12   mean|delta|={mad:.3f}   max|delta|={abs(worst):.3f}"
          f"   PARTIAL-mean delta={pm:+.3f} pts")
    print(f"  per-dataset SD est (mean|d|/sqrt2) = {mad/2**0.5:.3f} pts")
    if gain > 0:
        print(f"  partial-mean SD est (|partial mean|/sqrt2) = {abs(pm)/2**0.5:.3f} pts"
              f"  vs budget {budget:.3f}")

print("\n" + "=" * 78)
print("PARTIAL MEAN IS NOT THE TASK MEAN. The bar is defined on the mean over all 12")
print("datasets; a 6-dataset mean is a different estimator with a larger SD, and the")
print("missing sets (esca, wilds, tcga_tils, tcga_uniform, patch_camelyon) are the large,")
print("heterogeneous ones. Per-dataset SD -> task-mean SD divides by sqrt(12)=3.46 ONLY")
print("under independence, which is the optimistic ceiling here, not the estimate.")
print("=" * 78)
