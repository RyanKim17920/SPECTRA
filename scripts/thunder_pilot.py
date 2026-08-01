#!/usr/bin/env python3
"""Drain the held THUNDER job queue at a bounded concurrency, unattended.

Why this exists
---------------
The THUNDER sweep is ~30 SLURM jobs (base + fine-tuned x 15 datasets). Submitting
them all at once spikes to ~24 concurrent GPUs and crowds other work on this
cluster, so they were submitted then `scontrol hold`-ed. This process releases
them a few at a time as slots free, so the sweep completes over hours without
supervision and without a GPU spike.

Design notes
------------
* Emits ONE line per meaningful event on stdout, flushed, so a Monitor can watch
  it. Progress *and* failure lines both emit -- silence must not be ambiguous
  between "still working" and "died".
* Heartbeats periodically even when nothing changes, so a stalled pilot is
  distinguishable from a quiet one.
* Idempotent and restartable: state lives in SLURM, not here. Killing and
  relaunching this picks up wherever the queue is.
* Exits 0 only when no THUNDER job is held, pending, or running.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

USER = "ryan.kim"
# Job-name prefixes for the THUNDER sweep: `thd-` = base phikon-v2, `thdft1k-` =
# fine-tuned step-1000 adapter. Anything else in the queue (hest, speedlm, qwen,
# other users' work) is deliberately untouched.
PREFIXES = ("thd-", "thdft1k-")


def emit(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def squeue() -> list[dict]:
    """Current THUNDER jobs. Returns [] on transient squeue failure rather than
    raising -- one flaky call must not kill a multi-hour pilot."""
    try:
        out = subprocess.run(
            ["squeue", "-u", USER, "-h", "-o", "%i|%j|%t|%r"],
            capture_output=True, text=True, timeout=60,
        )
    except Exception as e:  # noqa: BLE001 - transient; log and continue
        emit(f"WARN squeue failed: {e}")
        return []
    if out.returncode != 0:
        emit(f"WARN squeue rc={out.returncode}: {out.stderr.strip()[:200]}")
        return []
    jobs = []
    for line in out.stdout.splitlines():
        parts = line.split("|")
        if len(parts) != 4:
            continue
        jid, name, state, reason = (p.strip() for p in parts)
        if name.startswith(PREFIXES):
            jobs.append({"id": jid, "name": name, "state": state, "reason": reason})
    return jobs


def fast_failures(since: str = "today", max_seconds: int = 150) -> list[str]:
    """THUNDER jobs that died suspiciously fast.

    The cold-embedding-cache bug killed jobs in ~37s with a FileNotFoundError. A pilot
    that keeps releasing into that burns the whole queue in minutes and leaves a pile of
    FAILED jobs that look like they ran. Any systematic breakage shows up as a cluster of
    sub-150s failures, so that is the circuit breaker: trip, stop releasing, and say so.
    """
    try:
        out = subprocess.run(
            ["sacct", "-u", USER, "-S", since, "-X", "-n", "-P",
             "--format=JobID,JobName,State,ElapsedRaw"],
            capture_output=True, text=True, timeout=60,
        )
    except Exception:  # noqa: BLE001 - never let the breaker itself kill the pilot
        return []
    bad = []
    for line in out.stdout.splitlines():
        parts = line.split("|")
        if len(parts) != 4:
            continue
        jid, name, state, elapsed = parts
        if not name.startswith(PREFIXES) or not state.startswith("FAILED"):
            continue
        try:
            if int(elapsed) <= max_seconds:
                bad.append(f"{name}({jid},{elapsed}s)")
        except ValueError:
            continue
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=4,
                    help="max concurrent THUNDER jobs (running + runnable-pending)")
    ap.add_argument("--interval", type=int, default=120, help="seconds between checks")
    ap.add_argument("--heartbeat", type=int, default=1800,
                    help="seconds between heartbeats when nothing changes")
    ap.add_argument("--max-fast-failures", type=int, default=3,
                    help="halt releases after this many sub-150s THUNDER failures")
    args = ap.parse_args()

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    seen_bad: set[str] = set()
    emit(f"START pilot cap={args.cap} interval={args.interval}s "
         f"breaker={args.max_fast_failures} since={started_at}")
    last_beat = 0.0
    last_sig = None

    while True:
        jobs = squeue()
        running = [j for j in jobs if j["state"] == "R"]
        held = [j for j in jobs if j["reason"] == "JobHeldUser"]
        # Pending-but-not-held still occupies our budget: it will start on its own.
        queued = [j for j in jobs if j["state"] == "PD" and j["reason"] != "JobHeldUser"]

        if not jobs:
            emit("DONE no THUNDER jobs held, pending, or running -- sweep complete")
            return 0

        # Circuit breaker: only counts failures from THIS pilot's lifetime, so the
        # pre-fix cold-cache casualties don't trip it on startup.
        bad = [b for b in fast_failures(since=started_at) if b not in seen_bad]
        if bad:
            seen_bad.update(bad)
        if len(seen_bad) >= args.max_fast_failures:
            emit(f"FATAL {len(seen_bad)} THUNDER jobs failed in <150s: "
                 f"{', '.join(sorted(seen_bad)[:6])} -- systematic breakage, "
                 f"halting releases with {len(held)} still held. Fix, then relaunch.")
            return 2

        active = len(running) + len(queued)
        slots = max(0, args.cap - active)
        if slots and held:
            batch = [j["id"] for j in held[:slots]]
            names = ", ".join(j["name"] for j in held[:slots])
            r = subprocess.run(["scontrol", "release", ",".join(batch)],
                               capture_output=True, text=True)
            if r.returncode == 0:
                emit(f"RELEASE {len(batch)} job(s): {names} "
                     f"(active {active}->{active + len(batch)}, {len(held) - len(batch)} still held)")
            else:
                emit(f"ERROR release failed rc={r.returncode}: {r.stderr.strip()[:200]}")

        sig = (len(running), len(queued), len(held))
        now = time.time()
        if sig != last_sig:
            emit(f"STATE running={sig[0]} queued={sig[1]} held={sig[2]}")
            last_sig, last_beat = sig, now
        elif now - last_beat >= args.heartbeat:
            emit(f"HEARTBEAT running={sig[0]} queued={sig[1]} held={sig[2]} (unchanged)")
            last_beat = now

        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
