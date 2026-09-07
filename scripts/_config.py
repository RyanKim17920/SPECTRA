"""Path roots for ``scripts/``, importable without installing the package.

The generators and submitters in this directory are run as ``./.venv/bin/python
scripts/foo.py`` from four different virtualenvs (main, hest, thunder, pathorob), and
only some of those have ``spectra`` installed.  So this shim puts ``<repo>/src`` on
``sys.path`` first and then re-exports :mod:`spectra.paths`, which stays the single
definition of every root.  Import it as::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _config import RUNS, PAPER_TABLES

See :mod:`spectra.paths` for the variable names, defaults and the ``.env`` contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = str(REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from spectra.paths import (  # noqa: E402,F401  -- re-export is the point
    BACKUPS,
    CELLS,
    DATA,
    EVALS,
    HEST_BENCH,
    HEST_WORK,
    HF_HOME,
    INPUTS,
    PAPER,
    PAPER_FIGURES,
    PAPER_TABLES,
    PLISM,
    PLISM_PACKED,
    REPO,
    RUNS,
    SNAPSHOTS,
    THUNDER,
    describe,
    env_path,
    export_legacy_env,
    load_dotenv,
)

__all__ = [
    "REPO",
    "REPO_ROOT",
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
    "describe",
    "env_path",
    "export_legacy_env",
    "load_dotenv",
]

if __name__ == "__main__":  # pragma: no cover
    print(describe())
