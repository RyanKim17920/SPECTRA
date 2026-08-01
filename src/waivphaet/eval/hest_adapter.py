"""Adapter onto HEST-Benchmark -- our first RETENTION metric (PLAN.md §1, §3 risk 1).

Why this benchmark
------------------
PLAN.md §3 risk 1: "Forgetting is the default outcome, not a tail risk." PathoROB's
robustness index says nothing about whether the backbone still encodes general biology,
because RI is a *ratio* of same-class-other-center to other-class-same-center neighbours --
a model that collapses biology but keeps centers apart can score well. HEST asks a
question PathoROB structurally cannot: from a 224px tile, can you still regress the
expression of the 50 most variable genes? That is general biological content, measured by
Pearson correlation, on tissue that has nothing to do with PLISM.

What HEST-Benchmark actually is (inspected at ``third_party/HEST``)
--------------------------------------------------------------------
* **No WSIs.** ``hest.bench.benchmark`` only ever opens ``patches/*.h5`` (pre-extracted
  224x224 tiles at 0.5 um/px, one per Visium/Xenium spot) and ``adata/*.h5ad``
  (expression). The 2.01 TB ``MahmoodLab/hest`` dataset is NOT needed; the 42.25 GB
  ungated ``MahmoodLab/hest-bench`` is. No openslide, no pyvips at bench time.
* **It consumes a model, not features** -- the opposite of PathoROB. ``benchmark()`` takes
  an ``nn.Module`` + transform + precision, embeds tiles itself, then runs its own
  regression. We add no math: PCA(256) -> Ridge -> ``scipy.stats.pearsonr`` per gene is
  all theirs (``hest/bench/benchmark.py``, ``hest/bench/trainer.py``).
* 9 tasks (``IDC PRAD PAAD SKCM COAD READ CCRCC LUNG LYMPH_IDC``), leave-one-patient-out
  folds discovered from ``splits/``, mean over 50 genes -> mean over folds -> mean over
  tasks. ``HCC`` ships in the repo but is *not* in their leaderboard config -- excluded
  here, or the average is not comparable to 0.3747.

Pooling: this benchmark DEMANDS ``cls``, not ``clsmean``
---------------------------------------------------------
Everywhere else in this project we pool ``clsmean`` (2048-d) because PathoROB's committed
reference row is ``phikonv2_clsmean``. **HEST's published 0.3747 is a different pooling.**
Their ``phikon_v2`` baseline is TRIDENT's ``Phikonv2InferenceEncoder``, whose forward is
literally::

    # trident/patch_encoder_models/load.py:1288
    out = self.model(x)
    out = out.last_hidden_state[:, 0, :]     # CLS only -> 1024-d
    # precision = torch.float32

So reproducing 0.3747 requires ``pooling="cls"``. We run **both**:

* ``cls`` -- the reproduction gate against Waiv/HEST's published 0.3747.
* ``clsmean`` -- the number we track across our own checkpoints, so retention is measured
  on the *same* representation PathoROB scores. Its base value is ours to establish;
  it has no published counterpart and must never be compared to 0.3747.

The transform is unchanged from PathoROB's (Resize(224)/CenterCrop(224)/ToTensor/
Normalize(IMAGENET)) -- TRIDENT's phikon_v2 uses byte-for-byte the same one, so there is
no preprocessing divergence to declare. HEST tiles are already 224x224, so resize and crop
are no-ops there.
"""

from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[3]
HEST_ROOT = REPO / "third_party" / "HEST"

#: 42.25 GB, ungated (``MahmoodLab/hest-bench``). Lives on /data, never /admin.
DEFAULT_BENCH_DATA = Path("/data/ryan.kim/hest_bench")
DEFAULT_WORK_DIR = Path("/data/ryan.kim/hest_work")

#: HEST's own leaderboard task set. ``HCC`` exists on HF but is excluded from their
#: published average -- including it would silently make our number incomparable.
LEADERBOARD_TASKS = (
    "IDC", "PRAD", "PAAD", "SKCM", "COAD", "READ", "CCRCC", "LUNG", "LYMPH_IDC",
)

#: HEST leaderboard row for ``phikon_v2`` (CLS pooling, fp32). Waiv's Table 1 HEST column
#: (PLAN.md §1: phikon-v2 0.3747 -> Phaet 0.3943) quotes this verbatim; it is not an
#: independent measurement. Per-task values are the reproduction gate.
PUBLISHED_PHIKONV2_CLS = {
    "IDC": 0.5408, "PRAD": 0.3545, "PAAD": 0.4455, "SKCM": 0.5554, "COAD": 0.2500,
    "READ": 0.1749, "CCRCC": 0.2659, "LUNG": 0.5419, "LYMPH_IDC": 0.2437,
}
PUBLISHED_PHIKONV2_AVG = 0.3747
PUBLISHED_PHAET_AVG = 0.3943

#: Their whole dynamic range is ~0.10 Pearson (ResNet50 0.3252 -> H-Optimus-1 0.4229).
#: Phikon-v1 -> v2 is +0.0087. Quote any delta against this scale or it is meaningless.
HEST_RANGE = (0.3252, 0.4229)


class HestEncoderWrapper(nn.Module):
    """``forward(pixel_values) -> (B, D)``, which is the entire contract HEST imposes.

    ``hest.bench.benchmark.embed_tiles`` calls ``model(imgs)`` under
    ``torch.inference_mode()`` + ``autocast('cuda', dtype=precision)`` and expects a 2-D
    tensor back. Our ``PhikonEncoder.forward`` returns ``(embedding, projection)``, so it
    cannot be handed over directly -- this unwraps to the embedding, which is the same
    vector PathoROB scores.
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = int(encoder.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.embed(x)


def build_transform():
    """Identical to ``build_preprocess`` in the PathoROB extractor *and* to TRIDENT's
    ``Phikonv2InferenceEncoder`` eval transform. Keeping one transform across both
    benchmarks is what makes robustness-vs-retention a fair pair (PLAN.md §6)."""
    import torchvision.transforms as T

    from waivphaet.models.encoder import IMAGENET_MEAN, IMAGENET_STD

    return T.Compose(
        [
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def load_encoder(checkpoint=None, adapter=None, pooling: str = "cls", **kw):
    """Reuse the PathoROB extractor's ``build_model`` verbatim.

    Imported by path rather than copied so that "base", "--checkpoint" and "--adapter"
    mean exactly the same thing on both benchmarks. If the two loaders ever drifted, a
    retention regression could be a loader artefact and we would not be able to tell.
    """
    src = REPO / "scripts" / "extract_pathorob_features.py"
    spec = importlib.util.spec_from_file_location("_waiv_pathorob_extract", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_model(checkpoint, pooling, adapter, **kw)


@dataclass
class HestPaths:
    """HEST resolves ``./``-prefixed paths against its own source root, so pass absolutes."""

    bench_data: Path = DEFAULT_BENCH_DATA
    work_dir: Path = DEFAULT_WORK_DIR

    @property
    def embed_dir(self) -> Path:
        return self.work_dir / "embeddings"

    @property
    def results_dir(self) -> Path:
        return self.work_dir / "results"

    def check(self) -> None:
        if not (HEST_ROOT / "src" / "hest" / "bench").is_dir():
            raise FileNotFoundError(
                f"HEST not found at {HEST_ROOT}. Clone it:\n"
                f"  git clone --depth 1 https://github.com/mahmoodlab/HEST {HEST_ROOT}"
            )
        missing = [t for t in LEADERBOARD_TASKS if not (self.bench_data / t).is_dir()]
        if missing:
            raise FileNotFoundError(
                f"missing HEST-bench tasks under {self.bench_data}: {missing}\n"
                "  huggingface-cli download MahmoodLab/hest-bench --repo-type dataset "
                f"--local-dir {self.bench_data} --exclude 'fm_v1/*'"
            )


def import_hest_benchmark():
    """Import ``hest.bench.benchmark`` WITHOUT installing the ``hest`` distribution.

    ``pip install hest`` drags ultralytics, spatialdata, dask[complete], pyvips and
    openslide-python -- none of which the benchmark path touches (verified: the only
    file handles opened are ``patches/*.h5`` and ``adata/*.h5ad``). Its real imports are
    torch / numpy / pandas / sklearn / scipy / h5py / scanpy / yaml / loguru /
    huggingface_hub, plus ``trident.patch_encoder_models``. We put the source tree on
    ``sys.path`` instead so the metric code is theirs, unmodified, at a pinned commit.
    """
    import sys

    p = str(HEST_ROOT / "src")
    if p not in sys.path:
        sys.path.insert(0, p)
    from hest.bench.benchmark import benchmark  # noqa: PLC0415

    return benchmark


def read_results(exp_dir: Path) -> dict[str, float]:
    """``{task: pearson_mean, ..., "avg": mean}`` -- the shape of Waiv's Table 1 column.

    HEST writes ``<results_dir>/<exp_code>::<timestamp>/<task>/<enc>/results_kfold.json``.
    We read ``pearson_mean`` (already mean-over-genes-then-folds) per task and average
    over tasks unweighted, which is how their leaderboard "Average" column is built.
    """
    out: dict[str, float] = {}
    for task in LEADERBOARD_TASKS:
        f = exp_dir / task / "custom_encoder" / "results_kfold.json"
        if not f.exists():
            continue
        out[task] = float(json.loads(f.read_text())["pearson_mean"])
    if out:
        out["avg"] = sum(out.values()) / len(out)
    return out


def compare_to_published(results: dict[str, float]) -> dict[str, dict]:
    """Per-task and average delta vs the HEST leaderboard's ``phikon_v2`` row.

    Only meaningful for ``pooling="cls"``: the published row is CLS-only (see module
    docstring). Calling this on a ``clsmean`` run compares two different representations.
    """
    rows = {}
    for task, pub in PUBLISHED_PHIKONV2_CLS.items():
        if task in results:
            rows[task] = {"ours": results[task], "published": pub,
                          "delta": results[task] - pub}
    if "avg" in results:
        rows["avg"] = {"ours": results["avg"], "published": PUBLISHED_PHIKONV2_AVG,
                       "delta": results["avg"] - PUBLISHED_PHIKONV2_AVG}
    return rows


def env_defaults() -> dict[str, str]:
    """PYTHONNOUSERSITE: a broken pandas in ``~/.local`` shadows venv pins on this box."""
    return {"HF_HOME": "/data/ryan.kim/hf_home", "PYTHONNOUSERSITE": "1",
            "TOKENIZERS_PARALLELISM": "false"}


def apply_env_defaults() -> None:
    for k, v in env_defaults().items():
        os.environ.setdefault(k, v)
