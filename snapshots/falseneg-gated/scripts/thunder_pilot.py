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
import os
import subprocess
import sys
import time

# The account whose queue is piloted. Env-derived so the script is not pinned to one
# operator; the literal is only a fallback for cron/systemd contexts with no USER set.
USER = os.environ.get("USER") or os.environ.get("LOGNAME") or "ryan.kim"

# Job-name prefixes for the THUNDER sweep: `thd-` = base phikon-v2, `thdft1k-` =
# fine-tuned step-1000 adapter. Anything else in the queue (hest, speedlm, qwen,
# other users' work) is deliberately untouched.
#: phikon-v2 sweep = thd-/thdft1k-, Midnight sweep = mthd-/mthdft-, Virchow2 sweep =
#: vthd-/vthdft-. A prefix missing here
#: makes the pilot see an empty queue and declare DONE while jobs sit held forever, which
#: is exactly what happened on the first Midnight submission.
#:
#: A FOURTH backbone would reproduce that bug verbatim, so the tuple is overridable
#: without editing this file: `--prefixes thd-,thdft1k-,xthd-` or THUNDER_PREFIXES=...
#: The default is the six the three sweeps use; adding the Virchow2 pair cannot change
#: what the two live sweeps see, because `startswith` on a longer, disjoint prefix set is
#: purely additive and no phikon-v2 or Midnight job name begins with "vthd-".
DEFAULT_PREFIXES = ("thd-", "thdft1k-", "mthd-", "mthdft-", "vthd-", "vthdft-")


def _parse_prefixes(spec: str | None) -> tuple[str, ...]:
    """Comma/space separated -> tuple. Empty/blank spec falls back to the default rather
    than to (), because () makes str.startswith match NOTHING -- the pilot would print
    DONE against a full held queue, the precise failure this knob exists to prevent."""
    if not spec:
        return DEFAULT_PREFIXES
    parts = tuple(p.strip() for p in spec.replace(",", " ").split() if p.strip())
    return parts or DEFAULT_PREFIXES


PREFIXES = _parse_prefixes(os.environ.get("THUNDER_PREFIXES"))


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
    global PREFIXES
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=4,
                    help="max concurrent THUNDER jobs (running + runnable-pending)")
    ap.add_argument("--interval", type=int, default=120, help="seconds between checks")
    ap.add_argument("--heartbeat", type=int, default=1800,
                    help="seconds between heartbeats when nothing changes")
    ap.add_argument("--max-fast-failures", type=int, default=3,
                    help="halt releases after this many sub-150s THUNDER failures")
    ap.add_argument("--prefixes", default=None,
                    help="comma-separated job-name prefixes to pilot; overrides "
                         "THUNDER_PREFIXES. Default: " + ",".join(DEFAULT_PREFIXES))
    args = ap.parse_args()
    if args.prefixes:
        PREFIXES = _parse_prefixes(args.prefixes)

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    seen_bad: set[str] = set()
    emit(f"START pilot user={USER} prefixes={','.join(PREFIXES)} "
         f"cap={args.cap} interval={args.interval}s "
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
            # Release fine-tuned jobs first. GPU capacity here is the binding constraint
            # (all 8 H100 nodes allocated; SLURM estimates ~20 h to the next start), so
            # the ORDER decides what we actually learn if the sweep is cut short.
            # The ft1000 rows are irreplaceable -- they are the result. Our own base rows
            # are a nice-to-have control: THUNDER publishes per-dataset phikon-v2 values
            # and our mhist base already reproduced them (66.4 vs 66.1), so `base_cls`
            # runs can be backfilled later or substituted with the published row.
            # Derived from PREFIXES rather than the literal "thdft1k-" that used to be
            # here: that literal silently de-prioritised nothing on the Midnight sweep
            # (mthdft- never matched), and a third backbone would inherit the same
            # no-op. Convention: a fine-tuned prefix contains "ft".
            ft_prefixes = tuple(p for p in PREFIXES if "ft" in p)
            held.sort(key=lambda j: 0 if j["name"].startswith(ft_prefixes) else 1)
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
