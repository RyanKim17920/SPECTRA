"""Shared constants / task enumeration for the CPTAC (Patho-Bench subset) harness.

Protocol:

* 11 CPTAC cohorts, official Patho-Bench splits ($PATHOBENCH_DIR/splits/<cohort>/<task>/k=all.tsv),
  38 classification tasks + 4 survival tasks (x 3 CoxNet alphas = 12 fits) = 50 fits.
* tiles: pre-extracted 20x 224px JPEGs ($PATHOBENCH_DIR/job_dir/20x_p224_tp0.6/...), fed
  WITHOUT resizing (ToTensor + Normalize, model's own mean/std).
* encoder: fp16 autocast, per-backbone official readout.
* slide = unweighted mean of patch embeddings, accumulated in fp64, stored fp32;
  case = mean of its slides' means (fp64 -> fp32).
* classification: ExperimentFactory.linprobe(cost=0.5, balanced=True, num_bootstraps=100),
  metric macro-ovr-auc; survival: ExperimentFactory.coxnet(alpha in {0.01,0.02,0.07},
  l1_ratio=0.5, num_bootstraps=100); the upstream "weights are too large" CoxNet failure is
  recorded as null, everything else fails loudly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

#: Patho-Bench data folder (splits + pre-extracted 20x 224px tiles): set PATHOBENCH_DIR
PATHOBENCH_DIR = os.environ.get("PATHOBENCH_DIR", "")
PATCH_DIR = os.path.join(PATHOBENCH_DIR, "job_dir/20x_p224_tp0.6/20x_224px_0px_overlap/patch_images")
SPLITS = os.path.join(PATHOBENCH_DIR, "splits")
DATASETS = [
    "cptac_brca", "cptac_ccrcc", "cptac_coad", "cptac_gbm", "cptac_hnsc",
    "cptac_lscc", "cptac_luad", "cptac_lung", "cptac_ov", "cptac_pda", "cptac_ucec",
]
N_SLIDES = 2083
N_PATCHES = 13_350_806
ALPHAS = (0.01, 0.02, 0.07)
UPSTREAM_COMMIT = "660e77044640e3d7d2f1150cc6721e97454993bf"
REPO = Path(__file__).resolve().parents[2]
UPSTREAM_DIR = REPO / "third_party" / "Patho-Bench"
PATCH_FILE = REPO / "scripts" / "evals" / "env" / "cptac_patho_bench.patch"
PATCH_SHA256 = "35afa3c316ba0933dc7fc27586fa9fc1c89493f683c0e2074875a52b3ffdcd24"
# Shared, model-independent patch-count manifest (built once by the first extract shard).
SHARED_DIR = Path(os.environ.get("SPECTRA_EVAL_OUT", "runs/evals")) / "_cptac"
PATCH_COUNTS = SHARED_DIR / "patch_counts.tsv"

# Per-backbone official readout. NOTE: this is the THUNDER/official readout table, which
# differs from HEST for kaiko-ai/midnight (clsmean here, cls in HEST).
CPTAC_READOUT = {
    "bioptimus/H-optimus-0": "cls",
    "MahmoodLab/UNI2-h": "cls",
    "SophontAI/OpenMidnight": "cls",
    "owkin/phikon-v2": "cls",
    "kaiko-ai/midnight": "clsmean",
    "paige-ai/Virchow2": "clsmean",
    "paige-ai/Virchow": "clsmean",
}

# "matched" subset: tasks with a metric-compatible row in
# data/reference_published.json -> table4_pathobench (mutation / MSI-H AUC tasks).
REFERENCE_JSON = REPO / "data" / "reference_published.json"


def pub_key(ours: str) -> str | None:
    """'cptac_coad/KRAS_mutation' -> '[CPTAC COAD][KRAS][AUC]' (= final_scoreboard._pub_cptac_key)."""
    cohort, _, task = ours.partition("/")
    if not cohort.startswith("cptac_"):
        return None
    coh = cohort[len("cptac_"):].upper()
    if task.endswith("_mutation"):
        return f"[CPTAC {coh}][{task[: -len('_mutation')]}][AUC]"
    if task == "MSI_H":
        return f"[CPTAC {coh}][MSI-H][AUC]"
    return None


def matched_tasks(task_keys) -> list[str]:
    t4 = json.loads(REFERENCE_JSON.read_text())["table4_pathobench"]
    pub = {t["task"] for t in t4["tasks"]}
    return [k for k in sorted(task_keys) if pub_key(k) in pub]


def tasks():
    """[(dataset, task, task_type, sample_col, split_tsv, config_yaml)] in the old order."""
    import yaml

    out = []
    for dataset in DATASETS:
        for task in sorted(os.listdir(os.path.join(SPLITS, dataset))):
            task_dir = os.path.join(SPLITS, dataset, task)
            split = os.path.join(task_dir, "k=all.tsv")
            config = os.path.join(task_dir, "config.yaml")
            with open(config) as handle:
                cfg = yaml.safe_load(handle)
            out.append((dataset, task, cfg["task_type"], cfg["sample_col"], split, config))
    assert sum(t[2] == "classification" for t in out) == 38
    assert sum(t[2] == "survival" for t in out) == 4
    assert all(t[3] == "case_id" for t in out)
    return out


def probe_tasks(task_list=None):
    """50 independent fits: (dataset, task, task_type, sample_col, split, config, alpha|None)."""
    task_list = task_list or tasks()
    out = []
    for dataset, task, task_type, sample_col, split, config in task_list:
        if task_type == "classification":
            out.append((dataset, task, task_type, sample_col, split, config, None))
        else:
            out.extend((dataset, task, task_type, sample_col, split, config, a) for a in ALPHAS)
    assert len(out) == 50
    return out


def required_slides(task_list=None) -> set[str]:
    import pandas as pd

    req = set()
    for dataset in DATASETS:
        for task in os.listdir(os.path.join(SPLITS, dataset)):
            req.update(pd.read_csv(os.path.join(SPLITS, dataset, task, "k=all.tsv"), sep="\t",
                                   usecols=["slide_id"], dtype=str)["slide_id"])
    return req


def patch_counts() -> dict[str, int]:
    """slide_id -> number of tiles; cached at PATCH_COUNTS (atomic, model independent)."""
    req = required_slides()
    assert len(req) == N_SLIDES, len(req)
    if not PATCH_COUNTS.exists():
        PATCH_COUNTS.parent.mkdir(parents=True, exist_ok=True)
        counts = {s: sum(n.lower().endswith((".jpg", ".jpeg", ".png"))
                         for n in os.listdir(os.path.join(PATCH_DIR, s))) for s in sorted(req)}
        tmp = PATCH_COUNTS.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text("".join(f"{s}\t{counts[s]}\n" for s in sorted(counts)))
        os.replace(tmp, PATCH_COUNTS)
    cached = {s: int(c) for s, c in (ln.split("\t") for ln in PATCH_COUNTS.read_text().splitlines())}
    assert req <= set(cached)
    counts = {s: cached[s] for s in req}
    assert sum(counts.values()) == N_PATCHES, sum(counts.values())
    return counts


def shard_slides(counts: dict[str, int], shards: int) -> list[list[str]]:
    """Greedy patch-balanced shards (largest slide first), identical to the old harness."""
    groups = [[] for _ in range(shards)]
    totals = [0] * shards
    for s in sorted(counts, key=lambda n: (-counts[n], n)):
        i = min(range(shards), key=lambda j: totals[j])
        groups[i].append(s)
        totals[i] += counts[s]
    return groups


def work_dir(model_key: str) -> Path:
    from spectra.evals.results import out_dir

    d = out_dir(model_key) / "cptac_work"
    d.mkdir(parents=True, exist_ok=True)
    return d
