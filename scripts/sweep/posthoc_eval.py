#!/usr/bin/env python
"""Score saved checkpoints of a run post hoc with the SAME in-run eval (primary + GTEx guard),
e.g. to fill an in-run eval that a preempted job skipped before running the slope rule.

    python scripts/sweep/posthoc_eval.py <run_dir> [<run_dir> ...] [--steps 125,150]

Appends ``{"kind": "eval", "posthoc": true, ...}`` records to <run_dir>/posthoc.jsonl
(metrics.jsonl is left untouched; spectra.sweep.stopping reads both). Steps already scored
are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from spectra.sweep.config import load_dumped  # noqa: E402
from spectra.sweep.model import SweepModel  # noqa: E402
from spectra.sweep.train import JsonlLog, build_evaluator, load_trainable, resolve_split  # noqa: E402


def scored_steps(rd: Path) -> set[int]:
    done = set()
    for f in ("metrics.jsonl", "posthoc.jsonl"):
        if (rd / f).exists():
            for line in (rd / f).read_text().splitlines():
                r = json.loads(line)
                if r.get("kind") == "eval":
                    done.add(int(r["step"]))
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--steps", default="", help="comma list; default = every saved ckpt not yet scored")
    a = ap.parse_args(argv)
    device = torch.device("cuda")
    from safetensors.torch import load_file

    for rd in a.run_dirs:
        cfg = load_dumped(rd / "config.json")
        done = scored_steps(rd)
        have = {int(p.stem.split("_")[-1]): p for p in (rd / "ckpt").glob("step_*.safetensors")}
        steps = [int(s) for s in a.steps.split(",") if s] if a.steps else sorted(have)
        steps = [s for s in steps if s in have and s not in done]
        if not steps:
            print(f"{rd.name}: nothing to score")
            continue
        model = SweepModel(cfg).to(device)
        ev = build_evaluator(cfg, model, device, resolve_split(cfg), rd)
        ev.ensure_base(model)
        log = JsonlLog(rd / "posthoc.jsonl")
        for s in steps:
            load_trainable(model, {k: v.to(device) for k, v in load_file(str(have[s])).items()})
            rec = ev(model, s)
            rec["posthoc"] = True
            log.write(rec)
            print(f"{rd.name} step {s}: primary {rec['primary']:.4f} "
                  f"guard {rec.get('guard_lin_bacc', float('nan')):.4f}", flush=True)
        del model
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
