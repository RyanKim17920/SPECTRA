"""Adapter onto PathoROB -- our PRIMARY metric (PLAN.md 1).

What PathoROB actually is (inspected at ``third_party/PathoROB``)
------------------------------------------------------------------
* Package ``pathorob``, no console scripts; everything is ``python -m``.
* The robustness index itself is dead simple and never touches a model::

      # pathorob/robustness_index/robustness_index_utils.py:92
      RI = SO / (SO + OS)

  where ``SO`` = fraction of kNN neighbours with the *same biological class, other
  medical center* and ``OS`` = *other class, same center*. Everything upstream is a kNN
  over L2-normalised embeddings.
* **It consumes precomputed embeddings, not models.** ``FeatureDataManager.load_features``
  reads ``{features_dir}/{model}/{dataset}/{medical_center}.npz`` where each npz maps
  ``f"{slide_id}-{patch_id}" -> 1-D float vector``. ``load_model()`` is never called by
  the metric scripts.

So the whole adapter is: run *our* encoder over their patches, hand the array to *their*
``save_features``, then shell out to *their* metric. We add no math.

Their metadata CSVs ship in the repo (``data/metadata/*.csv``, columns
``subset,slide_id,patch_id,biological_class,medical_center``); the images stream from HF
(``bifold-pathomics/PathoROB-{dataset}``), so set ``HF_HOME=/data/huggingface``.

Reference numbers are committed under ``third_party/PathoROB/results/``, including
``phikonv2_clsmean`` -- that is the gate for PLAN.md 3 phase 5 (reproduce Avg RI 0.469,
Camelyon 0.019). Report cross-stain and cross-scanner separately (PLAN.md 6).
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATASETS = ("tcga", "camelyon", "tolkach_esca")
DEFAULT_ROOT = Path("third_party/PathoROB")

#: PathoROB hard-pins numpy 2.2.6 / pandas 2.3.2 / transformers 4.56.1; our venv runs
#: numpy 2.5 / pandas 3.0 / transformers 5.14. We do NOT downgrade ours -- their metric
#: gets its own venv (``.venv-pathorob``, created from their pyproject) so the number we
#: report comes off exactly the dependency set that produced their published leaderboard.
#: Only torch 2.8.0 is shared, and that pin is fixed by the cluster's CUDA 12.8 driver.
DEFAULT_PYTHON = Path(".venv-pathorob/bin/python")

#: Waiv Table 1 (PLAN.md 1). Our phase-5 gate is the base row.
TARGETS = {
    "phikon_v2_base": {"tcga": 0.619, "camelyon": 0.019, "tolkach_esca": 0.768, "avg": 0.469},
    "phaet_target": {"tcga": 0.785, "camelyon": 0.702, "tolkach_esca": 0.932, "avg": 0.806},
    # Second backbone (PLAN.md 5 lists it as the ungated 4.55 GB alternative).
    # kaiko-ai/midnight = "Midnight-12k" in Waiv's Table 1; MASCARET is their fine-tune of
    # it, and it carries the largest published gain in the table -- which is why it is the
    # right second test of whether our reconstruction generalises.
    "midnight_base": {"tcga": 0.858, "camelyon": 0.478, "tolkach_esca": 0.941, "avg": 0.759},
    "mascaret_target": {"tcga": 0.893, "camelyon": 0.907, "tolkach_esca": 0.972, "avg": 0.924},
    # Third backbone, paige-ai/Virchow2 (timm ViT-H/14). AVERAGE ONLY, ON PURPOSE.
    # Waiv Table 1 gives Virchow2 Avg RI 0.858 base -> 0.918 fine-tuned; the per-dataset
    # tcga / camelyon / tolkach_esca breakdown behind that average was not transcribed and
    # is not in this repo. Three numbers that average to 0.858 are trivial to invent and
    # impossible to distinguish from real ones once written down, so the keys are simply
    # absent. A caller that indexes TARGETS["virchow2_base"]["camelyon"] therefore gets a
    # loud KeyError, which is the correct outcome; use waiv_target() below for the
    # tolerant, None-returning read.
    "virchow2_base": {"avg": 0.858},
    "virchow2_target": {"avg": 0.918},
}

def waiv_target(key: str, dataset: str) -> float | None:
    """Waiv Table-1 value for ``key`` on ``dataset`` (or ``"avg"``), or None if unpublished.

    ``TARGETS[key][dataset]`` stays a hard KeyError for anything that assumes a full
    breakdown exists (scripts/pathorob_gate.py does, deliberately -- it prints a
    per-dataset comparison column and has nothing to print without one). This accessor is
    for code that can honestly render "not published".
    """
    if key not in TARGETS:
        raise KeyError(f"unknown Waiv Table-1 row {key!r}; have {sorted(TARGETS)}")
    return TARGETS[key].get(dataset)


@dataclass
class PathoRobPaths:
    """PathoROB resolves its default paths against *cwd*, so we always pass absolutes."""

    root: Path = DEFAULT_ROOT

    @property
    def features_dir(self) -> Path:
        return (self.root / "data" / "features").resolve()

    @property
    def metadata_dir(self) -> Path:
        return (self.root / "data" / "metadata").resolve()

    @property
    def results_dir(self) -> Path:
        """Their ``--results_dir`` default is ``results/robustness_index``, i.e. it already
        includes the metric name -- passing bare ``results/`` silently writes one level
        too high and the summary is then unreadable at the documented path."""
        return (self.root / "results" / "robustness_index").resolve()

    def check(self) -> None:
        if not (self.root / "pathorob").is_dir():
            raise FileNotFoundError(
                f"PathoROB not found at {self.root}. Clone it:\n"
                "  git clone --depth 1 https://github.com/bifold-pathomics/PathoROB "
                f"{self.root}\n  pip install -e {self.root}"
            )


def _data_manager(paths: PathoRobPaths):
    """Import their FeatureDataManager, adding the clone to ``sys.path`` if not installed."""
    paths.check()
    try:
        from pathorob.features.data_manager import FeatureDataManager
    except ImportError:
        sys.path.insert(0, str(paths.root.resolve()))
        from pathorob.features.data_manager import FeatureDataManager
    return FeatureDataManager(
        features_dir=str(paths.features_dir), metadata_dir=str(paths.metadata_dir)
    )


def load_metadata(dataset: str, paths: PathoRobPaths | None = None):
    """Load their metadata CSV. ``dataset`` is a *metadata* name, e.g. ``tcga_4x4``."""
    return _data_manager(paths or PathoRobPaths()).load_metadata(dataset)


def save_features(
    model_name: str,
    dataset: str,
    features: np.ndarray,
    metadata,
    paths: PathoRobPaths | None = None,
) -> None:
    """Write ``(N, D)`` embeddings row-aligned to ``metadata`` in their npz layout.

    ``model_name`` is a free-form directory name -- because the metric scripts never call
    ``load_model``, we never have to register a ``ModelWrapper``. Use e.g.
    ``waivphaet_step0005000``.
    """
    if len(features) != len(metadata):
        raise ValueError(f"features {len(features)} != metadata rows {len(metadata)}")
    _data_manager(paths or PathoRobPaths()).save_features(
        model_name, dataset, np.asarray(features), metadata
    )


def run_robustness_index(
    model_name: str,
    datasets: list[str] | None = None,
    *,
    paths: PathoRobPaths | None = None,
    k_opt_param: int = 0,
    paired_evaluation: bool | None = None,
    extra_args: list[str] | None = None,
    python_exe: str | Path | None = None,
) -> subprocess.CompletedProcess:
    """Shell out to ``python -m pathorob.robustness_index.robustness_index``.

    We invoke their module rather than importing ``compute()`` so that the metric we
    report is byte-identical to the one that produced their published leaderboard.
    ``k_opt_param=0`` uses their per-dataset default k (tcga 61 / camelyon 11 /
    tolkach 46); ``-1`` sweeps k by balanced accuracy.
    """
    paths = paths or PathoRobPaths()
    paths.check()
    if python_exe is None:
        # .absolute(), NOT .resolve(): the venv's python is a symlink to the interpreter
        # it was built from, and resolving it silently drops us out of the venv.
        cand = Path(DEFAULT_PYTHON).absolute()
        python_exe = cand if cand.exists() else sys.executable
    cmd = [
        str(python_exe), "-m", "pathorob.robustness_index.robustness_index",
        "--model", model_name,
        "--features_dir", str(paths.features_dir),
        "--metadata_dir", str(paths.metadata_dir),
        "--results_dir", str(paths.results_dir),
        "--k_opt_param", str(k_opt_param),
    ]
    if datasets:
        cmd += ["--datasets", *datasets]
    if paired_evaluation is not None:
        # Their flag is str2bool-valued, not store_true; leaving it unset selects their
        # per-dataset default (True for tcga, False elsewhere), which is what we want.
        cmd += ["--paired_evaluation", str(paired_evaluation).lower()]
    cmd += extra_args or []
    # PYTHONNOUSERSITE: ~/.local/lib/python3.12/site-packages is on this machine and its
    # (broken) pandas shadows the venv's pinned one. Isolate or the pins mean nothing.
    env = {**os.environ, "PYTHONNOUSERSITE": "1"}
    return subprocess.run(cmd, cwd=str(paths.root.resolve()), check=True, env=env)


def read_results(
    model_name: str,
    dataset: str,
    *,
    paths: PathoRobPaths | None = None,
    max_patches_per_combi: str = "-1",
    k_opt_param: int = 0,
) -> dict:
    """Read their ``results_summary.json`` (key ``robustness_index``, among others)."""
    import json

    paths = paths or PathoRobPaths()
    p = (
        paths.results_dir / model_name / dataset
        / f"{max_patches_per_combi}_{k_opt_param}" / "results_summary.json"
    )
    if not p.exists():
        raise FileNotFoundError(f"no PathoROB results at {p}; run run_robustness_index first")
    return json.loads(p.read_text())


def summarize(model_name: str, **kw) -> dict[str, float]:
    """``{dataset: RI, ..., "avg": mean}`` -- the shape of Waiv's Table 1 row."""
    out = {d: float(read_results(model_name, d, **kw)["robustness_index"]) for d in DATASETS}
    out["avg"] = float(np.mean(list(out.values())))
    return out
