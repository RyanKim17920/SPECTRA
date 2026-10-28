#!/usr/bin/env python
"""Resolve a config (x seeds), write each run dir, sbatch it, record the ledger.

    python scripts/sweep/submit.py configs/v2/q8_selection.yaml --sweep q8_selection \
        --seeds 0,1,2 -o backbone.name=hoptimus [--dry-run]
    python scripts/sweep/submit.py --resubmit-dir <runs root>/<sweep>/<run>

Run dirs: ``$SPECTRA_RUNS_DIR/<sweep>/<run name>`` (default ``<repo>/runs/sweeps``).
SLURM: ``scripts/sweep/run_one.sbatch`` (one GPU). Cluster-specific options come from
``--sbatch-args`` or ``$SPECTRA_SBATCH_ARGS`` (e.g. "--account=X --partition=Y --qos=Z") and
``--time``. Ledger: one JSON line per submission at ``<runs root>/<sweep>/ledger.jsonl``. A run
whose dir already has DONE, or whose last ledger job is still queued/running, is skipped
(``--force`` resubmits anyway).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from spectra.sweep.config import (SweepConfig, config_hash, dump, load_config, load_dumped,  # noqa: E402
                                  run_name, runs_root)

SBATCH = REPO / "scripts" / "sweep" / "run_one.sbatch"


def queued_jobs() -> set[str]:
    try:
        out = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i"],
                             capture_output=True, text=True, timeout=60).stdout
        return {x.strip().split("_")[0] for x in out.split()}
    except Exception:  # noqa: BLE001
        return set()


def read_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def sbatch_cmd(run_dir: Path, extra: list[str], time_limit: str | None) -> list[str]:
    cmd = ["sbatch", "--parsable", f"--job-name=q8-{run_dir.name[:30]}",
           f"--output={run_dir}/slurm-%j.out", "--open-mode=append",
           f"--export=ALL,SWEEP_RUN_DIR={run_dir},SPECTRA_REPO={REPO}"]
    if time_limit:
        cmd.append(f"--time={time_limit}")
    return cmd + list(extra) + [str(SBATCH)]


def submit_run(cfg: SweepConfig, sweep: str, extra: list[str], time_limit: str | None = None, *,
               dry: bool = False, force: bool = False) -> str | None:
    sdir = runs_root() / sweep
    name = run_name(cfg)
    run_dir = sdir / name
    ledger = sdir / "ledger.jsonl"
    if (run_dir / "DONE").exists() and not force:
        print(f"skip {name}: DONE")
        return None
    prev = [r for r in read_ledger(ledger) if r["run"] == name and r.get("job_id")]
    if prev and not force and prev[-1]["job_id"] in queued_jobs():
        print(f"skip {name}: job {prev[-1]['job_id']} still queued/running")
        return None
    cmd = sbatch_cmd(run_dir, extra, time_limit)
    if dry:
        print("DRY", run_dir, "\n   ", " ".join(shlex.quote(c) for c in cmd))
        return None
    run_dir.mkdir(parents=True, exist_ok=True)
    cp = run_dir / "config.json"
    if cp.exists() and config_hash(load_dumped(cp)) != config_hash(cfg):
        raise RuntimeError(f"{cp} holds a different config")
    dump(cfg, cp)
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    if r.returncode != 0:
        raise RuntimeError(f"sbatch failed: {r.stderr.strip()}")
    jid = r.stdout.strip().split(";")[0]
    with open(ledger, "a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "sweep": sweep, "run": name,
                            "hash": config_hash(cfg), "seed": cfg.run.seed, "job_id": jid,
                            "run_dir": str(run_dir), "sbatch_args": extra}) + "\n")
    print(f"submitted {name} job={jid}")
    return jid


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", help="config YAML (e.g. configs/v2/q8_selection.yaml)")
    ap.add_argument("-o", "--override", action="append", default=[], help="key.sub=value (repeatable)")
    ap.add_argument("--seeds", default=None, help="comma list of run.seed values (default: the config's)")
    ap.add_argument("--sweep", default="dev", help="sub-directory of the runs root")
    ap.add_argument("--resubmit-dir", type=Path, help="resubmit an existing run dir (resumes from resume.pt)")
    ap.add_argument("--sbatch-args", default=os.environ.get("SPECTRA_SBATCH_ARGS", ""),
                    help="extra sbatch options, e.g. '--account=X --partition=Y' (default $SPECTRA_SBATCH_ARGS)")
    ap.add_argument("--time", default=None, help="sbatch --time (default: the sbatch header's)")
    ap.add_argument("--smoke", action="store_true", help="run.smoke=true (20 steps, tiny eval)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    extra = shlex.split(a.sbatch_args)
    if a.resubmit_dir:
        run_dir = a.resubmit_dir.resolve()
        cfg = load_dumped(run_dir / "config.json")
        submit_run(cfg, run_dir.parent.name, extra, a.time, dry=a.dry_run, force=True)
        return 0
    if not a.config:
        ap.error("need a config (or --resubmit-dir)")
    ovr = list(a.override) + (["run.smoke=true"] if a.smoke else [])
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else [None]
    for s in seeds:
        cfg = load_config(a.config, ovr + ([f"run.seed={s}"] if s is not None else []))
        submit_run(cfg, a.sweep, extra, a.time, dry=a.dry_run, force=a.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
