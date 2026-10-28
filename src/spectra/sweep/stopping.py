"""Slope-rule stopping: pick the training step from the run's OWN in-run PLISM primary.

``P`` maps step -> in-run selection primary (``spectra.sweep.evaluate``) of a held-out
SELECTION run, with ``P[0]`` = the step-0 (frozen-base) eval. Nothing outside PLISM is read.

    slope(s) = [P(s) - P(s - 50)] / 2         (gain per 25 steps; P linearly interpolated
                                               between evals at most ``maxgap`` apart)

Walk the candidate steps ``H`` in order; the rule ARMS once slope(s) >= theta and stops at the
first armed step whose slope falls below theta, keeping the PREVIOUS candidate (the last step
that was still improving by >= theta per 25 steps). Never armed -> the first candidate. Armed
to the end -> the last candidate whose slope was >= theta.

Q8 defaults: theta = 0.016 per 25 steps, candidates H = 75, 100, ..., 250, evals every 25 steps.

This module is pure python (no torch), so it can run anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

THETA = 0.016
#: candidate stop steps of the Q8 selection runs (in-run eval every 25 steps up to 250)
CANDIDATES = tuple(range(75, 251, 25))
EVAL_EVERY = 25


def interp(P: dict, s: float, maxgap: int = 50) -> float | None:
    """P(s): exact value, else linear between the neighbouring evals if they are <= maxgap apart."""
    if s in P:
        return P[s]
    lo = [u for u in P if u < s]
    hi = [u for u in P if u > s]
    if not lo or not hi:
        return None
    a, b = max(lo), min(hi)
    if b - a > maxgap:
        return None
    return P[a] + (P[b] - P[a]) * (s - a) / (b - a)


def slope(P: dict, s: int, window: int = 50, maxgap: int = 50) -> float | None:
    """Gain of P per 25 steps over the last ``window`` steps; None when undefined."""
    a, c = interp(P, s - window, maxgap), interp(P, s, maxgap)
    return None if a is None or c is None else (c - a) * 25.0 / window


def slope_pick(P: dict, candidates: Iterable[int] = CANDIDATES, theta: float = THETA):
    """-> (pick, {step: slope}). pick is None when the slope is undefined at every candidate."""
    hs = sorted(int(h) for h in candidates)
    v = {s: slope(P, s) for s in hs}
    if all(x is None for x in v.values()):
        return None, v
    pk, armed = hs[0], False
    for s in hs:
        x = v[s]
        if x is None:
            continue
        if x >= theta:
            pk, armed = s, True
        elif armed:
            break
    return pk, v


def complete(P: dict, last: int = CANDIDATES[-1], every: int = EVAL_EVERY) -> bool:
    """True when P has step 0 and every ``every``-step eval up to ``last`` (the rule's inputs)."""
    return 0 in P and all(s in P for s in range(every, last + 1, every))


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
                                 "the stopping rule needs a held-out selection run")
            P[int(r["step"])] = float(r[metric])
    return dict(sorted(P.items()))
