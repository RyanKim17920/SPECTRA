#!/usr/bin/env python3
"""Wait for base-Midnight PathoROB features, then score them against Waiv's published row.

Chained rather than run inline because the extraction is a SLURM job (369107) and the RI
metric is a CPU kNN -- this lets the metric fire the moment the features land instead of
being polled for. Waits on FILES, not on squeue.

Waiv arXiv:2607.22861 Table 1, Midnight-12k base row:
    TCGA 0.858 | Camelyon 0.478 | Tolkach 0.941 | Avg 0.759
Reproducing that on a ViT-g backbone -- different width, SwiGLU FFN, different
normalization -- is the second independent check that our harness is faithful, and the
prerequisite for a Midnight fine-tuning run being interpretable.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

FEATURES = REPO / "third_party/PathoROB/data/features/midnight_clsmean_ours"
LOG = Path("/data/ryan.kim/midnight_gate.log")
DATASETS = ("camelyon", "tolkach_esca", "tcga")
#: Waiv's published base row; tcga has 26 centers so it is the completeness sentinel.
WAIV_BASE = {"camelyon": 0.478, "tolkach_esca": 0.941, "tcga": 0.858}
WAIV_AVG = 0.759
MASCARET_AVG = 0.924


def log(msg: str) -> None:
    with LOG.open("a") as fh:
        fh.write(msg + "\n")
    print(msg, flush=True)


def ready() -> bool:
    for d in DATASETS:
        if not (FEATURES / d).is_dir():
            return False
    # tcga writes 26 npz (one per medical center); anything less means mid-write.
    return len(list((FEATURES / "tcga").glob("*.npz"))) >= 26


def main() -> int:
    log(f"[gate] waiting for features under {FEATURES}")
    for _ in range(360):  # up to 3h
        if ready():
            break
        time.sleep(30)
    else:
        log("MIDNIGHT_GATE_TIMEOUT features never completed")
        return 2

    from waivphaet.eval.pathorob_adapter import read_results, run_robustness_index

    model = "midnight_clsmean_ours"
    run_robustness_index(model, list(DATASETS))
    ris = []
    for d in DATASETS:
        v = float(read_results(model, d)["robustness_index"])
        ris.append(v)
        log(f"RESULT {d:14s} ours={v:.4f}  waiv_base={WAIV_BASE[d]:.3f}  delta={v - WAIV_BASE[d]:+.4f}")
    avg = sum(ris) / len(ris)
    log(f"RESULT {'AVG':14s} ours={avg:.4f}  waiv_base={WAIV_AVG:.3f}  delta={avg - WAIV_AVG:+.4f}")
    log(f"[gate] MASCARET target for a fine-tuned Midnight is {MASCARET_AVG} "
        f"(headroom from our base: {MASCARET_AVG - avg:+.4f})")
    log("MIDNIGHT_GATE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
