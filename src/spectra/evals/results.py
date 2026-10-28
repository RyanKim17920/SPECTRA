"""Result writer: ``$SPECTRA_EVAL_OUT/<model_key>/<bench>.json`` (atomic).

Stdlib only. ``SPECTRA_EVAL_OUT`` is the results root (default ``runs/evals``).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import socket
import subprocess
from pathlib import Path

from .keys import validate_key

SCHEMA = "spectra-evals/1"
DEFAULT_ROOT = "runs/evals"
WORKTREE = Path(__file__).resolve().parents[3]


def evals_root() -> Path:
    return Path(os.environ.get("SPECTRA_EVAL_OUT", DEFAULT_ROOT))


def out_dir(model_key: str) -> Path:
    d = evals_root() / validate_key(model_key)
    d.mkdir(parents=True, exist_ok=True)
    return d


def git_commit(repo: Path = WORKTREE) -> str | None:
    try:
        sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=20).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=20).stdout.strip()
        return (sha + ("-dirty" if dirty else "")) if sha else None
    except Exception:  # noqa: BLE001
        return None


def _read_spec(model_key: str) -> dict | None:
    p = evals_root() / model_key / "spec.json"
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


def write_result(model_key: str, bench: str, payload: dict, spec: dict | None = None) -> Path:
    """Atomically write ``<out_dir>/<bench>.json`` = payload + provenance header.

    ``spec`` defaults to ``<out_dir>/spec.json``. If both exist and their identity differs,
    raise: a result must never be filed under another model's key."""
    if not bench or "/" in bench or bench.startswith("."):
        raise ValueError(f"bad bench name {bench!r}")
    d = out_dir(model_key)
    disk = _read_spec(model_key)
    if spec is not None and disk is not None and spec.get("identity_sha256") != disk.get("identity_sha256"):
        raise ValueError(f"{model_key}: spec identity {spec.get('identity_sha256')} != {d / 'spec.json'} "
                         f"{disk.get('identity_sha256')}")
    spec = spec if spec is not None else disk
    if spec is not None and spec.get("model_key") not in (None, model_key):
        raise ValueError(f"spec model_key {spec.get('model_key')!r} != {model_key!r}")
    rec = dict(payload)
    rec.update(
        schema=SCHEMA,
        model_key=model_key,
        bench=bench,
        spec=spec,
        written_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        host=socket.gethostname(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        git_commit=git_commit(),
    )
    out = d / f"{bench}.json"
    tmp = d / f".{bench}.json.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(rec, indent=2, sort_keys=True, default=_default) + "\n")
    os.replace(tmp, out)
    return out


def _default(o):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def read_result(model_key: str, bench: str) -> dict | None:
    p = evals_root() / model_key / f"{bench}.json"
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None
