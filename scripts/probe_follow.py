#!/usr/bin/env python
"""Tier-1 fast collapse detector for a full-FT run (PLAN.md 3 phase 8, tripwire half).

``eval_checkpoints.py`` is the Tier-2 follower: embedding probe **plus** the full PathoROB
RI on three datasets, ~15-20 min per checkpoint. That is the number we quote, but it is far
too slow to *stop* a run. Full fine-tuning can destroy a representation in tens of steps
(hence the 1e-5 LR and the ``25,50,75,...`` dense early schedule), so by the time Tier 2
reports on step 50 the run is already at step 400.

This is Tier 1: probe only, ~30 s per checkpoint, evaluated against three hard kill signals
taken from the LoRA control run (``runs/waiv-real-369043``, base = ``probe_before.json``):

1. ``heldout.within_condition_random > 0.70``, or a rise of ``+0.10`` between consecutive
   checkpoints. This is the collapse detector proper: it is the mean cosine between tiles
   of *different* conditions, i.e. how much everything is turning into the same vector.
   Base 0.470; the LoRA run sat at 0.577 by step 500 and that was already the least
   comfortable number in the run. Matched cosine cannot do this job -- it *rises* under
   collapse, which is exactly why it is not one of the signals.
2. ``heldout.cross_scanner.separation < 0.376`` (the base value) after warmup. Separation
   is matched-minus-random; dropping below base means the fine-tune has made scanner
   invariance *worse* than the backbone we started from, which is the whole point of the
   run. Suppressed during warmup because the LR ramp legitimately moves it around.
3. ``heldout.cross_stain.top1`` below base 0.696, or a fall of more than 0.05 between
   consecutive checkpoints. Stain top-1 is the retention side of the pair: a run can buy
   scanner separation by forgetting stain structure, and signal 2 alone would call that a
   success.

Signals 1 and 3 are deliberately two-sided (absolute threshold OR inter-checkpoint delta).
A slow drift past an absolute bound and a single violent step are different failures, and
the dense early schedule exists precisely to catch the second one.

Cost to Tier 2: **zero.** The probe json is written to the exact path
``eval_checkpoints.run_probe`` looks for (``probe_step_NNNNNNN.json`` in the run dir) with
the same probe arguments, and that function skips any step whose file already exists. So
Tier 1 does not duplicate work, it *removes* the probe from Tier 2's critical path.

This does NOT kill the job. It prints, loudly, on stdout, and records every evaluation in
``collapse_watch.json``. Deciding to scancel is a human call -- an automated kill on a
noisy 256-tile probe would be a worse failure mode than a wasted GPU-hour.

    python scripts/probe_follow.py --run-dir runs/waiv-fullft-NNNN \
        --conditions-file runs/waiv-fullft-NNNN/conditions_used.json \
        --stop-file runs/waiv-fullft-NNNN/TRAIN_DONE --poll-s 20
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: Base (un-fine-tuned phikon-v2) values, measured on the pinned condition set of the LoRA
#: control run: runs/waiv-real-369043/probe_before.json, groups.heldout. Not hand-copied
#: from the paper -- these are this repo's own numbers on this repo's own condition set,
#: which is the only way the comparison means anything.
BASE_WITHIN_CONDITION_RANDOM = 0.4701
BASE_CROSS_SCANNER_SEPARATION = 0.3760
BASE_CROSS_STAIN_TOP1 = 0.6958

#: Signal thresholds. See the module docstring for why each one is here.
WCR_ABS_MAX = 0.70          # signal 1, absolute
WCR_STEP_RISE = 0.10        # signal 1, inter-checkpoint
SCANNER_SEP_MIN = 0.376     # signal 2, absolute (== base)
STAIN_TOP1_MIN = 0.696      # signal 3, absolute (== base)
STAIN_TOP1_FALL = 0.05      # signal 3, inter-checkpoint


def discover(run_dir: Path) -> list[tuple[int, Path]]:
    """Complete checkpoints, ascending.

    ``metrics.json`` is written LAST by ``save_checkpoint``, so it is the completeness
    sentinel -- without it we could probe a half-flushed ``backbone.safetensors``. Same
    rule as ``eval_checkpoints.discover``; deliberately kept identical so the two
    followers can never disagree about what a finished checkpoint is.
    """
    out = []
    for d in sorted(run_dir.glob("step_*")):
        if not (d / "metrics.json").exists():
            continue
        if (d / "adapter").is_dir() or (d / "backbone.safetensors").exists():
            out.append((int(d.name.split("_")[1]), d))
    return out


def is_full_ft_checkpoint(ckpt: Path) -> bool:
    return (ckpt / "backbone.safetensors").exists() and not (ckpt / "adapter").is_dir()


def probe_path(run_dir: Path, step: int) -> Path:
    """The path eval_checkpoints.run_probe checks for and skips on. Must match exactly."""
    return run_dir / f"probe_step_{step:07d}.json"


def run_probe(args, ckpt: Path, step: int) -> dict:
    out = probe_path(args.run_dir, step)
    if not out.exists():
        # Write to a temp path and os.replace into place. eval_checkpoints tests
        # `out.exists()` and immediately json.loads it, so a probe json that is visible
        # while still being written is a crash in the OTHER follower. os.replace is
        # atomic within a filesystem, so the file only ever appears complete.
        tmp = out.with_suffix(".json.partial")
        cmd = [
            args.python, "scripts/embed_probe.py",
            "--packed-dir", str(args.packed_dir),
            "--out", str(tmp),
            "--proj-out-dim", str(args.proj_out_dim),
            "--n-tiles", str(args.probe_tiles),
        ]
        if is_full_ft_checkpoint(ckpt):
            cmd += ["--checkpoint", str(ckpt)]
        else:
            cmd += ["--adapter", str(ckpt),
                    "--lora-rank", str(args.lora_rank),
                    "--lora-alpha", str(args.lora_alpha)]
        if args.conditions_file:
            cmd += ["--conditions-file", str(args.conditions_file)]
        print("+ " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=str(REPO))
        os.replace(tmp, out)
    return json.loads(out.read_text())


def digest(p: dict) -> dict:
    """The three heldout numbers the signals are defined on."""
    g = p.get("groups", {}).get("heldout", {})
    scan = g.get("cross_scanner.embedding") or {}
    stain = g.get("cross_stain.embedding") or {}
    return {
        "within_condition_random": g.get("within_condition_random.embedding"),
        "cross_scanner.separation": scan.get("separation"),
        "cross_scanner.top1": scan.get("top1"),
        "cross_stain.separation": stain.get("separation"),
        "cross_stain.top1": stain.get("top1"),
    }


def check_signals(step: int, cur: dict, prev: dict | None, warmup_steps: int) -> list[str]:
    """Return a list of tripped-signal descriptions; empty means clean."""
    trips: list[str] = []

    # --- signal 1: representation collapse -------------------------------------------
    wcr = cur.get("within_condition_random")
    if wcr is not None:
        if wcr > WCR_ABS_MAX:
            trips.append(
                f"SIGNAL 1 (collapse): heldout.within_condition_random={wcr:.4f} "
                f"> {WCR_ABS_MAX} (base {BASE_WITHIN_CONDITION_RANDOM:.4f}). Different "
                f"conditions are converging on one vector."
            )
        if prev is not None and prev.get("within_condition_random") is not None:
            rise = wcr - prev["within_condition_random"]
            if rise > WCR_STEP_RISE:
                trips.append(
                    f"SIGNAL 1 (collapse rate): heldout.within_condition_random rose "
                    f"{rise:+.4f} since step {prev['step']} "
                    f"({prev['within_condition_random']:.4f} -> {wcr:.4f}), "
                    f"> +{WCR_STEP_RISE} in one checkpoint."
                )

    # --- signal 2: scanner invariance worse than the backbone we started from ---------
    sep = cur.get("cross_scanner.separation")
    if sep is not None:
        if step <= warmup_steps:
            pass  # the LR ramp moves this legitimately; not a signal yet
        elif sep < SCANNER_SEP_MIN:
            trips.append(
                f"SIGNAL 2 (scanner regression): heldout.cross_scanner.separation="
                f"{sep:.4f} < {SCANNER_SEP_MIN} (base "
                f"{BASE_CROSS_SCANNER_SEPARATION:.4f}), past warmup step {warmup_steps}. "
                f"The fine-tune is now WORSE than base at the thing it is for."
            )

    # --- signal 3: stain retention ----------------------------------------------------
    t1 = cur.get("cross_stain.top1")
    if t1 is not None:
        if t1 < STAIN_TOP1_MIN:
            trips.append(
                f"SIGNAL 3 (stain forgetting): heldout.cross_stain.top1={t1:.4f} "
                f"< {STAIN_TOP1_MIN} (base {BASE_CROSS_STAIN_TOP1:.4f})."
            )
        if prev is not None and prev.get("cross_stain.top1") is not None:
            fall = prev["cross_stain.top1"] - t1
            if fall > STAIN_TOP1_FALL:
                trips.append(
                    f"SIGNAL 3 (stain forgetting rate): heldout.cross_stain.top1 fell "
                    f"{fall:.4f} since step {prev['step']} "
                    f"({prev['cross_stain.top1']:.4f} -> {t1:.4f}), "
                    f"> {STAIN_TOP1_FALL} in one checkpoint."
                )
    return trips


def write_watch(run_dir: Path, records: list[dict]) -> None:
    out = run_dir / "collapse_watch.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({
        "run_dir": str(run_dir),
        "base": {
            "within_condition_random": BASE_WITHIN_CONDITION_RANDOM,
            "cross_scanner.separation": BASE_CROSS_SCANNER_SEPARATION,
            "cross_stain.top1": BASE_CROSS_STAIN_TOP1,
            "source": "runs/waiv-real-369043/probe_before.json :: groups.heldout",
        },
        "thresholds": {
            "wcr_abs_max": WCR_ABS_MAX, "wcr_step_rise": WCR_STEP_RISE,
            "scanner_sep_min": SCANNER_SEP_MIN,
            "stain_top1_min": STAIN_TOP1_MIN, "stain_top1_fall": STAIN_TOP1_FALL,
        },
        "points": records,
    }, indent=2))
    os.replace(tmp, out)


def report(step: int, cur: dict, trips: list[str]) -> None:
    def f(k):
        v = cur.get(k)
        return "-" if v is None else f"{v:.4f}"
    print(
        f"[probe-follow] step {step:>6} | wcr {f('within_condition_random')} "
        f"| scan-sep {f('cross_scanner.separation')} "
        f"| stain-top1 {f('cross_stain.top1')} "
        f"| {'CLEAN' if not trips else str(len(trips)) + ' TRIPPED'}",
        flush=True,
    )
    for t in trips:
        print(f"[probe-follow] *** WARN *** step {step}: {t}", flush=True)
    if trips:
        print(
            "[probe-follow] *** WARN *** consider `scancel` -- this is a tripwire, not an "
            "automatic kill; confirm against the next checkpoint and history.json's "
            "grad_norm before acting.", flush=True,
        )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--packed-dir", type=Path,
                    default=Path("/data/ryan.kim/plism/repacked"))
    ap.add_argument("--conditions-file", type=Path, default=None,
                    help="pin the probe's condition set, exactly as the BEFORE probe and "
                         "eval_checkpoints.py do -- otherwise a shrinking condition set "
                         "reads as a representation change")
    ap.add_argument("--proj-out-dim", type=int, default=512)
    ap.add_argument("--probe-tiles", type=int, default=256)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--warmup-steps", type=int, default=100,
                    help="signal 2 is suppressed at or below this step; match the "
                         "trainer's --warmup-steps")
    ap.add_argument("--python", default=str(REPO / ".venv" / "bin" / "python"))
    ap.add_argument("--stop-file", type=Path, default=None,
                    help="exit once this exists AND no unprocessed checkpoint remains")
    ap.add_argument("--poll-s", type=int, default=20)
    ap.add_argument("--max-wait-s", type=int, default=8 * 3600)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/data/ryan.kim/hf_home")

    # Resume: anything already in collapse_watch.json keeps its verdict, and its digest
    # stays available as the `prev` term so a restart cannot manufacture a fake delta.
    watch_path = args.run_dir / "collapse_watch.json"
    records: list[dict] = []
    if watch_path.exists():
        try:
            records = json.loads(watch_path.read_text()).get("points", [])
        except json.JSONDecodeError:
            print("[probe-follow] existing collapse_watch.json unreadable, starting over")
    done = {r["step"] for r in records}

    t0 = time.time()
    total_trips = 0
    while True:
        for step, ckpt in discover(args.run_dir):
            if step in done:
                continue
            t = time.time()
            cur = digest(run_probe(args, ckpt, step))
            prev = records[-1] if records else None
            trips = check_signals(step, cur, prev, args.warmup_steps)
            total_trips += len(trips)
            report(step, cur, trips)
            records.append({"step": step, "checkpoint": str(ckpt),
                            "probe_seconds": round(time.time() - t, 1),
                            "signals": trips, **cur})
            records.sort(key=lambda r: r["step"])
            done.add(step)
            write_watch(args.run_dir, records)

        stopped = args.stop_file is not None and args.stop_file.exists()
        if stopped and not [s for s, _ in discover(args.run_dir) if s not in done]:
            print("[probe-follow] training finished and every checkpoint is probed.")
            break
        if time.time() - t0 > args.max_wait_s:
            print("[probe-follow] max wait exceeded; exiting with what we have.")
            break
        time.sleep(args.poll_s)

    print(f"[probe-follow] {len(records)} checkpoints probed, "
          f"{total_trips} signal(s) tripped -> {watch_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
