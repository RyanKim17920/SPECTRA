#!/usr/bin/env python
"""Summarise a sweep: ledger + SLURM state + run-dir progress; find dead runs.

    python scripts/sweep/status.py <sweep>              # table + resubmit commands
    python scripts/sweep/status.py <sweep> --resubmit   # actually resubmit dead runs

Categories: DONE, RUNNING, PENDING, PREEMPTED (requeue-able: PREEMPTED / REQUEUED / NODE_FAIL /
TIMEOUT or a trainer that saved and exited on a signal), FAILED (FAILED / OUT_OF_MEMORY /
CANCELLED / trainer state failed), LOST (no live job, not done, no reason). PREEMPTED and LOST
are resubmitted by --resubmit; FAILED only with --include-failed (an OOM or a bug will just
fail again).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "sweep"))

from spectra.sweep.config import runs_root  # noqa: E402

REQUEUEABLE = {"PREEMPTED", "REQUEUED", "NODE_FAIL", "TIMEOUT", "BOOT_FAIL", "DEADLINE"}
FAILED = {"FAILED", "OUT_OF_MEMORY", "CANCELLED"}


def squeue() -> dict[str, str]:
    out = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i|%T|%r"],
                         capture_output=True, text=True).stdout
    res = {}
    for line in out.splitlines():
        jid, st, reason = line.split("|", 2)
        res[jid.split("_")[0]] = f"{st}" + (f"({reason})" if st == "PENDING" else "")
    return res


def sacct(ids: list[str]) -> dict[str, str]:
    if not ids:
        return {}
    out = subprocess.run(["sacct", "-n", "-P", "-X", "-j", ",".join(ids), "-o", "JobID,State,ExitCode"],
                         capture_output=True, text=True).stdout
    res = {}
    for line in out.splitlines():
        p = line.split("|")
        if len(p) >= 2:
            res[p[0].split("_")[0].split(".")[0]] = p[1].split()[0]
    return res


def last_eval(run_dir: Path) -> dict:
    p = run_dir / "metrics.jsonl"
    ev = {}
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("kind") == "eval":
                ev = r
    return ev


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep")
    ap.add_argument("--resubmit", action="store_true")
    ap.add_argument("--include-failed", action="store_true")
    ap.add_argument("--sbatch-args", default=os.environ.get("SPECTRA_SBATCH_ARGS", ""))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    sdir = runs_root() / a.sweep
    ledger = sdir / "ledger.jsonl"
    if not ledger.exists():
        print(f"no ledger at {ledger}")
        return 1
    latest: dict[str, dict] = {}
    for x in ledger.read_text().splitlines():
        if x.strip():
            r = json.loads(x)
            latest[r["run"]] = r
    q = squeue()
    acct = sacct([r["job_id"] for r in latest.values() if r.get("job_id")])
    rows = []
    for name, r in latest.items():
        rd = Path(r["run_dir"])
        jid = r.get("job_id") or ""
        st = json.loads((rd / "state.json").read_text()) if (rd / "state.json").exists() else {}
        if (rd / "DONE").exists():
            cat = "DONE"
        elif jid in q:
            cat = "RUNNING" if q[jid].startswith(("RUNNING", "COMPLETING")) else "PENDING"
        else:
            s = acct.get(jid, "")
            if s in REQUEUEABLE or st.get("status") == "preempted":
                cat = "PREEMPTED"
            elif s in FAILED or st.get("status") == "failed":
                cat = "FAILED"
            else:
                cat = "LOST"
        ev = last_eval(rd)
        rows.append({"run": name, "job": jid, "cat": cat, "slurm": q.get(jid) or acct.get(jid, "?"),
                     "step": st.get("step", 0), "steps": st.get("steps"), "error": st.get("error"),
                     "primary": ev.get("primary"), "guard": ev.get("guard_lin_bacc"),
                     "eval_step": ev.get("step"), "run_dir": str(rd)})
    rows.sort(key=lambda x: (x["cat"], x["run"]))
    if a.json:
        print(json.dumps(rows, indent=2))
    else:
        print(f"{'run':40s} {'job':>8s} {'cat':9s} {'step':>9s} {'primary@step':>16s} {'guard':>7s}  slurm")
        for x in rows:
            p = f"{x['primary']:.4f}@{x['eval_step']}" if isinstance(x["primary"], float) else "-"
            g = f"{x['guard']:.4f}" if isinstance(x["guard"], float) else "-"
            print(f"{x['run']:40s} {x['job']:>8s} {x['cat']:9s} {x['step']:>4}/{x['steps'] or '?':<4} "
                  f"{p:>16s} {g:>7s}  {x['slurm']}" + (f"  ERR {x['error'][:60]}" if x["error"] else ""))
        counts: dict[str, int] = {}
        for x in rows:
            counts[x["cat"]] = counts.get(x["cat"], 0) + 1
        print("summary:", " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    dead = [x for x in rows if x["cat"] in ("PREEMPTED", "LOST") or (a.include_failed and x["cat"] == "FAILED")]
    if dead:
        print(f"\n{len(dead)} run(s) to resubmit:")
        for x in dead:
            print(f"  python scripts/sweep/submit.py --resubmit-dir {x['run_dir']}")
        if a.resubmit:
            from submit import submit_run  # noqa: WPS433

            from spectra.sweep.config import load_dumped
            for x in dead:
                cfg = load_dumped(Path(x["run_dir"]) / "config.json")
                submit_run(cfg, a.sweep, shlex.split(a.sbatch_args), force=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
