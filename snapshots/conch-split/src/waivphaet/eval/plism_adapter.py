"""Adapter onto ``plismbench`` -- DIAGNOSTIC ONLY.

.. warning::
   We **train** on PLISM. Any number produced here is a training diagnostic and must be
   labelled as such; it is never leaderboard-comparable and must never be printed next
   to H0-mini's 0.541 (PLAN.md 1 and 6). Report cross-scanner and cross-stain
   separately, and never cosine similarity alone -- PLIP scores 0.878 cosine at 0.054
   top-10 (PLAN.md 6).

What plismbench actually is (inspected at ``third_party/plism-benchmark``)
---------------------------------------------------------------------------
* Distribution ``owkin-plismbench``, import ``plismbench``, one Typer CLI with
  ``extract`` / ``download`` / ``evaluate``.
* The extractor registry is a hand-written ``FeatureExtractorsEnum`` plus an ``init()``
  if/elif chain in ``plismbench/models/__init__.py`` -- registering a model means editing
  their source. **We do not.**
* ``plismbench evaluate`` never touches the registry (``--extractor`` is just a directory
  name), so we write features ourselves in their layout and call only ``evaluate``.

Their on-disk feature contract::

    <features_dir>/<extractor>/<slide_id>/features.npy
      slide_id  = "{stain}_{scanner}_to_GMH_S60.tif"   (exactly 91 such dirs)
      features  = float32 (16278, 3 + d)
                  cols 0:3 = tile coords parsed from `tile_id.split("_")[1:]`
                  cols 3:  = the embedding
                  rows sorted by (x, y)

Note their sort is by ``(x, y)``, which is NOT our repack's native HDF5 key order --
:func:`write_slide_features` does the reorder explicitly.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path("third_party/plism-benchmark")
DIAGNOSTIC_NOTICE = (
    "PLISM retrieval is a TRAINING DIAGNOSTIC for this project (we fine-tune on PLISM). "
    "Not leaderboard-comparable. See PLAN.md 1 and 6."
)


def tile_coords(keys: list[str]) -> np.ndarray:
    """``tile_{level}_{x}_{y}`` -> ``(N, 3)`` int array, matching their column 0:3."""
    return np.asarray([[int(v) for v in k.split("_")[1:]] for k in keys], dtype=np.int64)


def write_slide_features(
    features_dir: Path | str,
    extractor: str,
    slide_id: str,
    keys: list[str],
    embeddings: np.ndarray,
) -> Path:
    """Write one slide in plismbench's exact layout.

    Args:
        slide_id: e.g. ``GIVH_AT2_to_GMH_S60.tif`` (their glob requires ``_to_GMH_S60``).
        keys: tile keys in the row order of ``embeddings`` (our ``keys.json``).
        embeddings: ``(16278, d)``.
    """
    coords = tile_coords(keys)
    if len(coords) != len(embeddings):
        raise ValueError(f"{len(keys)} keys vs {len(embeddings)} embeddings")
    arr = np.concatenate([coords.astype(np.float32), np.asarray(embeddings, np.float32)], axis=1)
    order = np.lexsort((arr[:, 2], arr[:, 1]))  # sort by (x, y) as they do
    out = Path(features_dir) / extractor / slide_id
    out.mkdir(parents=True, exist_ok=True)
    path = out / "features.npy"
    np.save(path, arr[order])
    return path


def run_evaluate(
    extractor: str,
    features_dir: Path | str,
    metrics_dir: Path | str,
    *,
    root: Path = DEFAULT_ROOT,
    n_tiles: int = 8139,
    top_k: str = "1 3 5 10",
    device: str = "gpu",
    workers: int = 4,
) -> subprocess.CompletedProcess:
    """Call ``plismbench evaluate``. ``n_tiles`` must be one of 460/2713/5426/8139/16278.

    ``--device gpu`` needs ``cupy`` (not in their pyproject: ``pip install cupy-cuda12x``);
    ``cpu`` falls back to numpy.
    """
    print(f"[waivphaet] {DIAGNOSTIC_NOTICE}")
    cmd = [
        sys.executable, "-m", "plismbench.engine.cli", "evaluate",
        "--extractor", extractor,
        "--features-dir", str(Path(features_dir).resolve()),
        "--metrics-dir", str(Path(metrics_dir).resolve()),
        "--n-tiles", str(n_tiles),
        "--top-k", top_k,
        "--device", device,
        "--workers", str(workers),
    ]
    return subprocess.run(cmd, cwd=str(Path(root).resolve()), check=True)


def read_results(metrics_dir: Path | str, extractor: str, n_tiles: int = 8139):
    """Their ``results.csv``: rows ``inter-scanner`` / ``inter-staining`` /
    ``inter-scanner, inter-staining`` / ``all`` -- already split the way PLAN.md 6 wants."""
    import csv

    p = Path(metrics_dir) / f"{n_tiles}_tiles" / extractor / "results.csv"
    if not p.exists():
        raise FileNotFoundError(f"no plismbench results at {p}")
    with p.open() as f:
        return list(csv.DictReader(f))
