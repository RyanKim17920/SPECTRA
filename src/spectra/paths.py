"""Every filesystem root this project reads or writes, resolved in ONE place.

Why this module exists
----------------------
Scripts here used to carry absolute paths from the machine they were written on -- one
author's home directory for the checkout and runs, one cluster's scratch volume for the
PLISM corpus and the THUNDER/HEST harness outputs, one LaTeX tree for the tables.  None
of them were wrong; all of them were unrunnable anywhere else, and there was no single
file to edit to move them.

So every root is now an environment variable with a repo-relative default:

    SPECTRA_REPO          the checkout itself                  (auto-detected)
    SPECTRA_RUNS          training run directories             <repo>/runs
    SPECTRA_DATA          shared corpus/scratch root           <repo>/data_root
    SPECTRA_PLISM         PLISM corpus (raw .h5)               <data>/plism
    SPECTRA_PLISM_PACKED  repacked PLISM tiles                 <plism>/repacked
    SPECTRA_THUNDER       THUNDER base data folder             <data>/thunder
    SPECTRA_HEST_WORK     HEST work dir (embeddings/results)   <data>/hest_work
    SPECTRA_HEST_BENCH    HEST-benchmark data                  <data>/hest_bench
    SPECTRA_EVALS         cell-harness eval outputs            <data>/full-evals
    SPECTRA_HF_HOME       HuggingFace cache root               <data>/huggingface
    SPECTRA_INPUTS        local base-model weight roots        <data>/inputs
    SPECTRA_CELLS         per-checkpoint eval "cells"          <repo>/cells
    SPECTRA_PAPER         LaTeX source dir (tables/, figures/) <repo>/paper
    SPECTRA_SNAPSHOTS     pinned code snapshots                <repo>/snapshots
    SPECTRA_BACKUPS       result backups off volatile storage  <repo>/result_backups

A root that is *derived* from another (``SPECTRA_PLISM`` from ``SPECTRA_DATA``) follows
it automatically, so relocating the whole corpus is one variable, not eight.

``.env``
--------
Setting fifteen variables in every shell and every sbatch is not a workflow, so an
optional ``.env`` at the repo root is read at import time.  It is a deliberately dumb
parser -- ``KEY=value`` lines, ``#`` comments, optional surrounding quotes, an optional
``export`` prefix -- because adding a dependency to read a config file is worse than the
problem it solves.  **A variable already present in the real environment always wins**,
so ``SPECTRA_RUNS=/tmp/x python scripts/foo.py`` overrides the file rather than fighting
it.  ``.env`` is gitignored (it describes one machine); ``.env.example`` is committed.

Nothing here creates directories or checks that they exist.  A root that is missing is a
missing corpus, and the script that needs it should say so at the point of use, naming
the variable -- not fail at import for a root it never touches.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "REPO",
    "RUNS",
    "DATA",
    "PLISM",
    "PLISM_PACKED",
    "THUNDER",
    "HEST_WORK",
    "HEST_BENCH",
    "EVALS",
    "HF_HOME",
    "INPUTS",
    "CELLS",
    "PAPER",
    "PAPER_TABLES",
    "PAPER_FIGURES",
    "SNAPSHOTS",
    "BACKUPS",
    "load_dotenv",
    "alias_legacy_env",
    "env_path",
    "export_legacy_env",
    "describe",
]


def _repo_root() -> Path:
    """The checkout this file lives in: ``src/spectra/paths.py`` -> up three."""
    return Path(__file__).resolve().parent.parent.parent


def load_dotenv(path: Path | None = None) -> dict[str, str]:
    """Read ``<repo>/.env`` into ``os.environ`` WITHOUT overriding the real environment.

    Returns the mapping that was applied (empty if the file does not exist).  Malformed
    lines are skipped silently rather than raising: a stray line in a convenience file
    must never be the reason a training job dies.
    """
    path = path or (_repo_root() / ".env")
    applied: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return applied
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not key or key in os.environ:
            continue
        os.environ[key] = value
        applied[key] = value
    return applied


def alias_legacy_env() -> dict[str, str]:
    """Mirror ``WAIV_*`` <-> ``SPECTRA_*`` in ``os.environ``, neither side clobbering.

    The pinned code snapshots under :data:`SNAPSHOTS` were frozen under an earlier
    internal package name, and those copies still read ``WAIV_PACKED_DIR``,
    ``WAIV_BACKBONE`` and friends.  A pin is frozen on purpose, so rather than rewrite
    one, both spellings are published.  Returns what it added.  ``scripts/_env.sh`` does
    the same thing for the shell.  Delete this once no pin in use reads the old names.
    """
    applied: dict[str, str] = {}
    for key in list(os.environ):
        if key.startswith("WAIV_"):
            other = "SPECTRA_" + key[len("WAIV_"):]
        elif key.startswith("SPECTRA_"):
            other = "WAIV_" + key[len("SPECTRA_"):]
        else:
            continue
        if other not in os.environ:
            os.environ[other] = os.environ[key]
            applied[other] = os.environ[key]
    return applied


load_dotenv()
alias_legacy_env()


def env_path(name: str, default: Path | str) -> Path:
    """``Path`` for environment variable ``name``, or ``default``. ``~`` is expanded."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return Path(default)
    return Path(os.path.expanduser(raw.strip()))


REPO: Path = env_path("SPECTRA_REPO", _repo_root())

#: Training runs: checkpoints, ``ri_curve.json``, per-benchmark summaries.
RUNS: Path = env_path("SPECTRA_RUNS", REPO / "runs")

#: Everything too big to live in the checkout: corpora, caches, harness outputs.
DATA: Path = env_path("SPECTRA_DATA", REPO / "data_root")

PLISM: Path = env_path("SPECTRA_PLISM", DATA / "plism")
PLISM_PACKED: Path = env_path("SPECTRA_PLISM_PACKED", PLISM / "repacked")

#: THUNDER's ``THUNDER_BASE_DATA_FOLDER``. Its own env var is honoured as a fallback so
#: an existing THUNDER shell keeps working without a SPECTRA_ variable being set.
THUNDER: Path = env_path(
    "SPECTRA_THUNDER",
    env_path("THUNDER_BASE_DATA_FOLDER", DATA / "thunder"),
)

HEST_WORK: Path = env_path("SPECTRA_HEST_WORK", DATA / "hest_work")
HEST_BENCH: Path = env_path("SPECTRA_HEST_BENCH", DATA / "hest_bench")

#: Output root of the per-checkpoint "cell" harness (pathorob/, thunder/, hest/, cptac/).
EVALS: Path = env_path("SPECTRA_EVALS", DATA / "full-evals")

#: HuggingFace cache. Never the default ``~/.cache`` -- on a cluster that fills /home.
HF_HOME: Path = env_path("SPECTRA_HF_HOME", DATA / "huggingface")

#: Root holding locally-served base-model weight directories (gated repos, converted
#: checkpoints). See ``spectra.models.backbones``.
INPUTS: Path = env_path("SPECTRA_INPUTS", DATA / "inputs")

#: One directory per evaluated checkpoint, each with its own ``model.py``.
CELLS: Path = env_path("SPECTRA_CELLS", REPO / "cells")

#: LaTeX source tree the table/figure generators write into.
PAPER: Path = env_path("SPECTRA_PAPER", REPO / "paper")
PAPER_TABLES: Path = env_path("SPECTRA_PAPER_TABLES", PAPER / "tables")
PAPER_FIGURES: Path = env_path("SPECTRA_PAPER_FIGURES", PAPER / "figures")

#: Frozen code snapshots a run was pinned to (``SPECTRA_PIN``).
SNAPSHOTS: Path = env_path("SPECTRA_SNAPSHOTS", REPO / "snapshots")

#: Durable copies of result JSON that otherwise live only on volatile scratch.
BACKUPS: Path = env_path("SPECTRA_BACKUPS", REPO / "result_backups")


def export_legacy_env() -> None:
    """Publish the roots under the variable names third-party harnesses read.

    HEST, THUNDER and huggingface_hub each read their own variable and know nothing
    about ``SPECTRA_*``.  Called by scripts that shell out to them.  Existing values are
    left alone -- an operator who set ``HF_HOME`` by hand meant it.
    """
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    os.environ.setdefault("THUNDER_BASE_DATA_FOLDER", str(THUNDER))
    os.environ.setdefault("HEST_WORK_DIR", str(HEST_WORK))


def describe() -> str:
    """Every resolved root, one per line -- for ``--print-paths`` style diagnostics."""
    rows = [
        ("SPECTRA_REPO", REPO),
        ("SPECTRA_RUNS", RUNS),
        ("SPECTRA_DATA", DATA),
        ("SPECTRA_PLISM", PLISM),
        ("SPECTRA_PLISM_PACKED", PLISM_PACKED),
        ("SPECTRA_THUNDER", THUNDER),
        ("SPECTRA_HEST_WORK", HEST_WORK),
        ("SPECTRA_HEST_BENCH", HEST_BENCH),
        ("SPECTRA_EVALS", EVALS),
        ("SPECTRA_HF_HOME", HF_HOME),
        ("SPECTRA_INPUTS", INPUTS),
        ("SPECTRA_CELLS", CELLS),
        ("SPECTRA_PAPER", PAPER),
        ("SPECTRA_SNAPSHOTS", SNAPSHOTS),
        ("SPECTRA_BACKUPS", BACKUPS),
    ]
    width = max(len(k) for k, _ in rows)
    return "\n".join(f"{k:<{width}}  {v}" for k, v in rows)


if __name__ == "__main__":  # pragma: no cover - diagnostic entrypoint
    print(describe())
