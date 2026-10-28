"""Adapter onto CROMA (``third_party/croma``, clemsgrs/croma) -- DRAFT replacement for PathoROB RI.

What CROMA is
-------------
A metrics-only library (numpy/pandas/scikit-learn; never loads a model, never reads an
image). For each sample it looks at cosine neighbours, excluding the sample's own
``group_id`` (= PathoROB ``slide_id``), and asks whether the nearest *SO* (same label,
other center) neighbours are closer than the nearest *OS* (other label, same center)::

    CRoMa_i = (d_OS - d_SO) / (d_OS + d_SO)       in (-1, 1), >0 = biology wins

``d_SO`` / ``d_OS`` are mean cosine distances to the ``m`` (default 5) nearest SO / OS
neighbours, found by adaptive outward search (start_k=200, x2 growth), so every sample is
scored -- no fixed k, no ``support`` caveat. Reported per cohort:

* ``value``     -- median of the per-sample distribution (headline),
* ``ltm_alpha`` -- lower-tail mean of the worst 10% (tail severity),
* ``f0``        -- fraction of samples with CRoMa_i <= 0 (tail prevalence).

The library also ships a clean RI (identical definition to PathoROB's
``SO / (SO + OS)``, k chosen by biological kNN balanced accuracy) and MaRI, which we
record alongside for continuity with the old headline.

Data: identical to PathoROB
---------------------------
CROMA's tile benchmarks ARE the three PathoROB cohorts, on PathoROB's own RI row sets
(``scripts/prep/build_pathorob_views.py`` upstream):

===============  ==============================================  ======
cohort           rows                                            n
===============  ==============================================  ======
camelyon         ``camelyon.csv`` where ``subset == "ID"``        20400
tcga (4x4)       ``tcga_4x4.csv`` where ``subset == "ID"``         5760
tolkach_esca     ``tolkach_esca_reduced.csv`` in full              9000
===============  ==============================================  ======

So we reuse, unchanged, the features ``scripts/extract_pathorob_features.py`` already
writes (``{features_dir}/{model}/{dataset}/{center}.npz`` with arrays ``ids`` =
``f"{slide_id}-{patch_id}"`` and ``features``). There is NO new GPU work: CROMA is a CPU
post-processing step on the PathoROB embeddings.

Usage::

    python -m spectra.eval.croma_adapter --model phikonv2_clsmean_ours \
        --features-dir /path/to/pathorob/features --metadata-dir /path/to/metadata \
        --out results/croma/phikonv2_clsmean_ours.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import os
# croma is imported from an installed copy, else from the clone at $PATHOROB_CROMA_ROOT
# (default third_party/croma, pinned to CROMA_COMMIT below).
DEFAULT_CROMA_ROOT = Path(os.environ.get("PATHOROB_CROMA_ROOT", "third_party/croma"))
#: Commit the draft was written and smoke-tested against. croma.__version__ == "1.0.1".
CROMA_COMMIT = "873612e1bfc8090bd95de8ffbc6b69cd387f0c95"

CONFOUNDER = "medical_center"
HEADLINE_M = 5
M_SWEEP = (1, 5, 10)
ALPHA = 0.10
#: PathoROB's sparse k grid with k_max=100 (``robustness_index_utils.get_k_values``),
#: as reproduced by croma's ``run_config.resolve_sweep_k_values(100, "sparse")``.
RI_K_CANDIDATES = (1, 3, 5, 7, 9, 11, 21, 31, 41, 51, 61, 71, 81, 91)


@dataclass(frozen=True)
class Cohort:
    dataset: str          # PathoROB features dir name
    metadata_csv: str     # PathoROB metadata file holding the RI row set
    filter_id: bool       # keep only subset == "ID"


#: Must match croma's ``scripts/bench/benchmarks.py`` (all three use design="all").
COHORTS: dict[str, Cohort] = {
    "camelyon": Cohort("camelyon", "camelyon.csv", True),
    "tcga": Cohort("tcga", "tcga_4x4.csv", True),
    "tolkach_esca": Cohort("tolkach_esca", "tolkach_esca_reduced.csv", False),
}


def _import_croma(root: Path = DEFAULT_CROMA_ROOT):
    """Import croma, preferring an installed copy, else the pinned clone's ``src``."""
    try:
        import croma  # noqa: F401
    except ImportError:
        src = Path(root) / "src"
        if not (src / "croma" / "__init__.py").exists():
            raise FileNotFoundError(
                f"croma not installed and no clone at {root}. Clone it:\n"
                f"  git clone https://github.com/clemsgrs/croma {root} && "
                f"git -C {root} checkout {CROMA_COMMIT}"
            ) from None
        sys.path.insert(0, str(src))
        import croma  # noqa: F401
    return croma


def build_manifest(cohort: Cohort, metadata_dir: Path) -> pd.DataFrame:
    """CROMA manifest for one cohort, in PathoROB metadata row order."""
    meta = pd.read_csv(Path(metadata_dir) / cohort.metadata_csv, dtype=str)
    for col in ("slide_id", "patch_id"):
        meta[col] = meta[col].str.strip()
    if cohort.filter_id:
        meta = meta[meta["subset"] == "ID"].reset_index(drop=True)
    key = meta["slide_id"] + "-" + meta["patch_id"]
    if key.duplicated().any():
        raise ValueError(f"{cohort.metadata_csv}: duplicate slide_id-patch_id keys")
    return pd.DataFrame({
        "sample_id": key,
        "image_path": f"pathorob/{cohort.dataset}/" + key,  # provenance only; never read
        "label": meta["biological_class"],
        CONFOUNDER: meta["medical_center"],
        "group_id": meta["slide_id"],   # PathoROB's slide = independence unit (as upstream)
        "dataset": cohort.dataset,
    })


def load_features(features_dir: Path, model: str, cohort: Cohort,
                  manifest: pd.DataFrame) -> np.ndarray:
    """Gather the PathoROB npz features into manifest row order. Fails on any gap."""
    d = Path(features_dir) / model / cohort.dataset
    files = sorted(d.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no PathoROB features at {d}")
    index: dict[str, tuple[int, int]] = {}
    mats: list[np.ndarray] = []
    for fi, f in enumerate(files):
        z = np.load(f, allow_pickle=False)
        if "ids" in z.files and "features" in z.files:
            ids, feats = z["ids"], z["features"]
        else:
            # PathoROB FeatureDataManager.save_features layout -- what
            # extract_pathorob_features.py actually writes: one 1-D array per
            # f"{slide_id}-{patch_id}" key.
            ids = np.asarray(z.files)
            feats = np.stack([z[k] for k in z.files])
        mats.append(np.asarray(feats, dtype=np.float32))
        for ri, k in enumerate(ids.tolist()):
            index[str(k)] = (fi, ri)
    missing = [k for k in manifest["sample_id"] if k not in index]
    if missing:
        raise KeyError(f"{len(missing)} manifest rows have no feature in {d} (e.g. {missing[:3]})")
    out = np.stack([mats[fi][ri] for fi, ri in (index[k] for k in manifest["sample_id"])])
    if not np.isfinite(out).all():
        raise ValueError(f"non-finite features in {d}")
    return out


def score_cohort(features: np.ndarray, manifest: pd.DataFrame, *, with_ri: bool = True,
                 m_values=M_SWEEP, alpha: float = ALPHA) -> dict:
    croma = _import_croma()
    t0 = time.time()
    res = croma.CRoMa.compute(features, manifest, confounder_column=CONFOUNDER,
                              evaluation_design="all", m=list(m_values), alpha=alpha)
    out: dict = {"n": int(len(manifest)), "croma": {}}
    for mm, r in res.items():
        out["croma"][str(mm)] = {
            "median": float(r.value), "ltm_alpha": float(r.ltm_alpha),
            "q_alpha": float(r.q_alpha), "f0": float(r.f0),
            "mean": float(np.mean(r.sample_values)), "k_final": int(r.k_final),
        }
    if with_ri:
        ri = croma.RI.compute(features, manifest, confounder_column=CONFOUNDER,
                              k_candidates=list(RI_K_CANDIDATES))
        mari = croma.MaRI.compute(features, manifest, confounder_column=CONFOUNDER,
                                  k_candidates=list(RI_K_CANDIDATES))
        out["ri"] = {"value": float(ri.value), "k": int(ri.k), "support": float(ri.support)}
        out["mari"] = {"value": float(mari.value), "tau": float(mari.tau)}
    out["seconds"] = round(time.time() - t0, 1)
    return out


def evaluate(model: str, features_dir: Path, metadata_dir: Path,
             datasets=tuple(COHORTS), with_ri: bool = True) -> dict:
    """``{"datasets": {ds: ...}, "avg": {...}}`` -- the shape a checkpoint curve consumes."""
    per = {}
    for ds in datasets:
        c = COHORTS[ds]
        man = build_manifest(c, metadata_dir)
        per[ds] = score_cohort(load_features(features_dir, model, c, man), man, with_ri=with_ri)
    h = str(HEADLINE_M)
    avg = {
        # Selection scalar candidates; see the integration plan for which one gates.
        "croma_median": float(np.mean([v["croma"][h]["median"] for v in per.values()])),
        "croma_ltm10": float(np.mean([v["croma"][h]["ltm_alpha"] for v in per.values()])),
        "croma_f0": float(np.mean([v["croma"][h]["f0"] for v in per.values()])),
    }
    if with_ri:
        avg["ri"] = float(np.mean([v["ri"]["value"] for v in per.values()]))
    return {"model": model, "croma_commit": CROMA_COMMIT, "m_headline": HEADLINE_M,
            "alpha": ALPHA, "datasets": per, "avg": avg}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="PathoROB features dir name")
    ap.add_argument("--features-dir", type=Path, required=True)
    ap.add_argument("--metadata-dir", type=Path, required=True)
    ap.add_argument("--datasets", nargs="+", default=list(COHORTS), choices=list(COHORTS))
    ap.add_argument("--no-ri", action="store_true", help="skip RI/MaRI (CRoMa only)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    res = evaluate(a.model, a.features_dir, a.metadata_dir, a.datasets, with_ri=not a.no_ri)
    text = json.dumps(res, indent=2)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
