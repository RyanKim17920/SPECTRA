"""The R1 stop rule: pick the training step from a selection run's OWN in-run PLISM metric.

``P`` maps step -> in-run selection primary (``spectra.sweep.evaluate``) of a held-out
SELECTION run (scanner GT450, stain MY and 7 tissue cores never trained on), recorded every
25 steps. Nothing outside PLISM is read: no labels, no benchmark.

    R1 = argmax_{s >= 25} P(s)          (ties -> the earlier step)

over the saved checkpoint steps 25, 50, ..., 250. The release model is then trained on all 91
conditions and all cores with the same lr schedule, stopped at that step.

This module is pure python (no torch), so it can run anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

EVAL_EVERY = 25
#: candidate stop steps of a selection run (in-run eval + checkpoint every 25 steps to 250)
CANDIDATES = tuple(range(EVAL_EVERY, 251, EVAL_EVERY))


def r1_pick(P: dict, candidates: Iterable[int] = CANDIDATES) -> int | None:
    """argmax over the candidate steps that have a P (ties -> the earlier step); None if none has."""
    have = {int(s): float(P[s]) for s in candidates if s in P and P[s] is not None}
    if not have:
        return None
    return max(have, key=lambda s: (have[s], -s))


def complete(P: dict, candidates: Iterable[int] = CANDIDATES) -> bool:
    """True when P has every candidate step (the rule's inputs)."""
    return all(s in P for s in candidates)


def read_primary(run_dir: str | Path, metric: str = "primary") -> dict[int, float]:
    """{step: in-run primary} from ``metrics.jsonl`` and ``posthoc.jsonl`` of a run dir (later
    records win). Refuses runs whose eval cores were trained on (an all-data final run)."""
    P: dict[int, float] = {}
    for fn in ("metrics.jsonl", "posthoc.jsonl"):
        p = Path(run_dir) / fn
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("kind") != "eval" or metric not in r:
                continue
            if r.get("cores_in_train"):
                raise ValueError(f"{run_dir}: eval cores were trained on (final run?) -- "
                                 "the stop rule needs a held-out selection run")
            P[int(r["step"])] = float(r[metric])
    return dict(sorted(P.items()))
