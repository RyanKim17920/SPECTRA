"""CPTAC CPU stages: case pooling, the 50 Patho-Bench probe fits, finalize -> cptac.json.

    python scripts/evals/cptac_probe.py pool     --key K [--work DIR]
    python scripts/evals/cptac_probe.py probe I  --key K [--work DIR]     (I in 0..49)
    python scripts/evals/cptac_probe.py finalize --key K [--work DIR] [--retain] [--jobs JSON]
    python scripts/evals/cptac_probe.py summarize --work DIR              (no write_result; validation)

The Patho-Bench calls (ExperimentFactory.linprobe / .coxnet) and their arguments are verbatim.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cptac_common as C  # noqa: E402

COX_FAIL = "Numerical error, because weights are too large"


def _work(a) -> Path:
    return Path(a.work) if a.work else C.work_dir(a.key)


def do_pool(work: Path, shards: int):
    import h5py
    import numpy as np
    import pandas as pd

    assert all((work / f"extract_{s}_of_{shards}.done").exists() for s in range(shards)), "extract incomplete"
    mp, cd = work / "meanpool_embeddings", work / "case_meanpool_embeddings"
    case2slides: dict[str, set] = {}
    for *_, sample_col, split, _ in C.tasks():
        frame = pd.read_csv(split, sep="\t", dtype=str)
        for case_id, group in frame.groupby("case_id"):
            case2slides.setdefault(case_id, set()).update(group["slide_id"])
    cd.mkdir(parents=True, exist_ok=True)
    print(f"{len(case2slides)} cases", flush=True)
    for case_id, slide_ids in case2slides.items():
        out = cd / f"{case_id}.h5"
        if out.exists():
            with h5py.File(out) as h:
                assert h["features"].ndim == 2 and h["features"].shape[0] == 1
                assert np.isfinite(h["features"][:]).all()
            continue
        vecs = []
        for sid in sorted(slide_ids):
            with h5py.File(mp / f"{sid}.h5") as h:
                v = h["features"][:]
            assert v.ndim == 2 and v.shape[0] == 1 and np.isfinite(v).all(), sid
            vecs.append(v.astype(np.float64))
        emb = np.concatenate(vecs).mean(0).astype(np.float32)[None, :]
        assert np.isfinite(emb).all()
        tmp = out.with_suffix(f".h5.{os.getpid()}.tmp")
        with h5py.File(tmp, "w") as h:
            h.create_dataset("features", data=emb)
        os.replace(tmp, out)
    (work / ".pooled").touch()
    print("case pooling DONE", flush=True)


def do_probe(work: Path, index: int, emb_from: Path | None):
    assert (work / ".pooled").exists() or emb_from is not None
    dataset, task, task_type, sample_col, split, config, alpha = C.probe_tasks()[index]
    src = emb_from or work
    pooled_dir = str(src / ("meanpool_embeddings" if sample_col == "slide_id" else "case_meanpool_embeddings"))
    output = work / "experiment" / "meanpool" / dataset / task
    from patho_bench.ExperimentFactory import ExperimentFactory

    t0 = time.time()
    if task_type == "classification":
        assert alpha is None
        result = output / "test_metrics_summary.json"
        if not result.exists():
            exp = ExperimentFactory.linprobe(
                split=split, task_config=config, pooled_embeddings_dir=pooled_dir,
                saveto=str(output), combine_slides_per_patient=False,
                cost=0.5, balanced=True, num_bootstraps=100,
            )
            exp.train()
            exp.test()
            auc = exp.report_results(metric="macro-ovr-auc")
            print(f"{dataset}--{task}: macro-ovr-auc {auc:.4f}", flush=True)
        json.loads(result.read_text())
    else:
        ao = output / f"alpha_{alpha}"
        result, failed = ao / "test_metrics_summary.json", ao / "failed.json"
        if not result.exists() and not failed.exists():
            exp = ExperimentFactory.coxnet(
                split=split, task_config=config, pooled_embeddings_dir=pooled_dir,
                saveto=str(ao), combine_slides_per_patient=False,
                alpha=alpha, l1_ratio=0.5, num_bootstraps=100,
            )
            exp.train()
            if len(exp.models) == exp.dataset.num_folds:
                assert exp.train_error is None
                exp.test()
            if not result.exists():
                assert COX_FAIL in exp.train_error, exp.train_error
                failed.write_text(json.dumps({
                    "status": "upstream_numerical_failure", "trained_folds": len(exp.models),
                    "expected_folds": exp.dataset.num_folds, "message": exp.train_error}, indent=2) + "\n")
        if result.exists():
            json.loads(result.read_text())
            print(f"{dataset}--{task}: coxnet alpha={alpha} done", flush=True)
        else:
            assert json.loads(failed.read_text())["status"] == "upstream_numerical_failure"
            print(f"{dataset}--{task}: coxnet alpha={alpha} numerical failure", flush=True)
    (work / "probe_times").mkdir(exist_ok=True)
    (work / "probe_times" / f"{index}.json").write_text(json.dumps(
        {"index": index, "seconds": time.time() - t0, "slurm_job_id": os.environ.get("SLURM_JOB_ID")}))


def summarize(work: Path) -> dict:
    exp = work / "experiment" / "meanpool"
    cls, cls_se, surv, surv_se, failures = {}, {}, {}, {}, []
    for dataset, task, task_type, *_ in C.tasks():
        key = f"{dataset}/{task}"
        if task_type == "classification":
            r = json.loads((exp / dataset / task / "test_metrics_summary.json").read_text())
            cls[key] = float(r["macro-ovr-auc"]["mean"])
            cls_se[key] = float(r["macro-ovr-auc"]["se"])
        else:
            surv[key], surv_se[key] = {}, {}
            for alpha in C.ALPHAS:
                d = exp / dataset / task / f"alpha_{alpha}"
                rp, fp = d / "test_metrics_summary.json", d / "failed.json"
                assert rp.exists() + fp.exists() == 1, d
                if rp.exists():
                    r = json.loads(rp.read_text())
                    surv[key][str(alpha)] = float(r["cindex"]["mean"])
                    surv_se[key][str(alpha)] = float(r["cindex"]["se"])
                else:
                    surv[key][str(alpha)] = surv_se[key][str(alpha)] = None
                    failures.append(f"{key}@alpha={alpha}: " + json.loads(fp.read_text())["message"][:200])
    assert len(cls) == 38 and len(surv) == 4
    matched = C.matched_tasks(cls)
    return {
        "classification_macro_ovr_auc": cls,
        "classification_macro_ovr_auc_bootstrap_se": cls_se,
        "mean_classification_macro_ovr_auc": sum(cls.values()) / len(cls),
        "matched_tasks": matched,
        "mean_matched_macro_ovr_auc": sum(cls[k] for k in matched) / len(matched),
        "survival_cindex_by_alpha": surv,
        "survival_cindex_bootstrap_se": surv_se,
        "coxnet_failures": failures,
    }


def _sacct(jobs: list[dict]) -> tuple[list[dict], float, float, float]:
    """Per array element (all attempts incl. requeues): elapsed, gpus. -> rows, gpu_h, cpu_h, wall."""
    rows, gpu_h, cpu_h = [], 0.0, 0.0
    starts, ends = [], []
    for j in jobs:
        try:
            out = subprocess.run(["sacct", "-j", str(j["job_id"]), "-X", "-D", "-n", "-P", "--format",
                                  "JobID,State,ElapsedRaw,AllocTRES,NodeList,Start,End"],
                                 capture_output=True, text=True, timeout=60).stdout
        except Exception as e:  # noqa: BLE001
            print(f"sacct failed for {j}: {e}", flush=True)
            continue
        for ln in out.strip().splitlines():
            jid, state, el, tres, node, st, en = ln.split("|")
            if not el.isdigit() or int(el) == 0:
                continue
            tr = dict(kv.split("=", 1) for kv in tres.split(",") if "=" in kv)
            ngpu = int(tr.get("gres/gpu", 0) or 0)
            ncpu = int(tr.get("cpu", 0) or 0)
            el = int(el)
            rows.append({"job_id": jid, "stage": j["stage"], "state": state, "node": node,
                         "elapsed_s": el, "n_gpus": ngpu, "n_cpus": ncpu})
            gpu_h += el * ngpu / 3600
            if ngpu == 0:
                cpu_h += el * ncpu / 3600
            starts.append(st)
            ends.append(en)
    wall = 0.0
    try:
        import datetime as dt

        f = "%Y-%m-%dT%H:%M:%S"
        s = [dt.datetime.strptime(x, f) for x in starts if x[:1].isdigit()]
        e = [dt.datetime.strptime(x, f) for x in ends if x[:1].isdigit()]
        if s and e:
            wall = (max(e) - min(s)).total_seconds()
    except Exception:  # noqa: BLE001
        pass
    return rows, gpu_h, cpu_h, wall


def env_info() -> dict:
    import numpy
    import sklearn
    import torch

    try:
        import timm
        tv = timm.__version__
    except Exception:  # noqa: BLE001
        tv = None
    try:
        import sksurv
        sv = sksurv.__version__
    except Exception:  # noqa: BLE001
        sv = None
    return {"python": sys.version.split()[0], "torch": torch.__version__, "timm": tv, "numpy": numpy.__version__,
            "sklearn": sklearn.__version__, "sksurv": sv, "venv": sys.prefix,
            "upstream_commit": C.UPSTREAM_COMMIT, "upstream_dir": str(C.UPSTREAM_DIR)}


def do_finalize(a, work: Path):
    from spectra.evals.results import out_dir, write_result
    from spectra.evals.spec import load_spec

    spec = load_spec(out_dir(a.key) / "spec.json")
    summ = summarize(work)
    (work / "aggregate.json").write_text(json.dumps({k: summ[k] for k in (
        "classification_macro_ovr_auc", "mean_classification_macro_ovr_auc", "survival_cindex_by_alpha")},
        indent=2) + "\n")
    jobs = json.loads(Path(a.jobs).read_text()) if a.jobs and Path(a.jobs).exists() else []
    rows, gpu_h, cpu_h, wall = _sacct(jobs)
    ex = [json.loads(p.read_text()) for p in sorted(work.glob("extract_*_of_*.json"))]
    by_stage = {}
    for r in rows:
        s = by_stage.setdefault(r["stage"], {"gpu_hours": 0.0, "cpu_hours": 0.0, "elapsed_s": 0})
        s["gpu_hours"] += r["elapsed_s"] * r["n_gpus"] / 3600
        s["cpu_hours"] += r["elapsed_s"] * r["n_cpus"] / 3600 if r["n_gpus"] == 0 else 0.0
        s["elapsed_s"] += r["elapsed_s"]
    pooling = ex[0]["pooling"] if ex else None
    precision = ex[0]["precision"] if ex else None
    payload = {
        "protocol": {
            "name": "patho-bench-cptac-subset",
            "precision": precision, "encoder_autocast": precision, "weights_dtype": "fp32",
            "transform": "ToTensor + Normalize(model mean/std), no resize (tiles are 224px 20x)",
            "pooling": pooling, "slide_pool": "unweighted mean of patch features, fp64 accumulate, fp32 store",
            "case_pool": "mean of slide means (fp64 -> fp32)",
            "tiles": C.PATCH_DIR, "splits": C.SPLITS, "datasets": C.DATASETS,
            "n_slides": C.N_SLIDES, "n_patches": C.N_PATCHES,
            "classification": "ExperimentFactory.linprobe(cost=0.5, balanced=True, num_bootstraps=100), macro-ovr-auc",
            "survival": "ExperimentFactory.coxnet(alpha in {0.01,0.02,0.07}, l1_ratio=0.5, num_bootstraps=100), cindex",
            "n_classification_tasks": 38, "n_survival_fits": 12,
            "matched_subset": "the 25 mutation / MSI-H AUC tasks that have a published Patho-Bench reference row",
            "upstream_commit": C.UPSTREAM_COMMIT, "patch": str(C.PATCH_FILE), "patch_sha256": C.PATCH_SHA256,
            "extract_shards": len(ex),
        },
        "env": env_info(),
        "metrics": {
            "mean_classification_macro_ovr_auc": summ["mean_classification_macro_ovr_auc"],
            "mean_matched_macro_ovr_auc": summ["mean_matched_macro_ovr_auc"],
            "mean_matched_macro_ovr_auc_pct": 100 * summ["mean_matched_macro_ovr_auc"],
            "n_matched_tasks": len(summ["matched_tasks"]),
            "n_coxnet_failures": len(summ["coxnet_failures"]),
        },
        "per_task": {
            "classification_macro_ovr_auc": summ["classification_macro_ovr_auc"],
            "classification_macro_ovr_auc_bootstrap_se": summ["classification_macro_ovr_auc_bootstrap_se"],
            "survival_cindex_by_alpha": summ["survival_cindex_by_alpha"],
            "survival_cindex_bootstrap_se": summ["survival_cindex_bootstrap_se"],
            "matched_tasks": summ["matched_tasks"],
        },
        "gpu_hours": gpu_h, "cpu_hours": cpu_h, "wall_seconds": wall,
        "stage_hours": by_stage,
        "extract_shards": ex,
        "slurm_jobs": rows,
        "status": "complete",
        "notes": ([f"CoxNet upstream numerical failure (recorded null): {f}"
                   for f in summ["coxnet_failures"]]
                  + ["gpu_hours/cpu_hours = sacct ElapsedRaw x AllocTRES over all attempts (incl. requeues)"]),
    }
    path = write_result(a.key, "cptac", payload, spec=spec)
    print(f"wrote {path}: headline {summ['mean_classification_macro_ovr_auc']:.4f} "
          f"matched {100 * summ['mean_matched_macro_ovr_auc']:.2f} gpu_h {gpu_h:.2f}", flush=True)
    if not a.retain:
        for d in ("meanpool_embeddings", "case_meanpool_embeddings"):
            if (work / d).exists():
                shutil.rmtree(work / d)
    (work / ".complete").touch()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pool", "probe", "finalize", "summarize"])
    ap.add_argument("index", nargs="?", type=int)
    ap.add_argument("--key")
    ap.add_argument("--work")
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--emb-from", default=None, help="validation: read pooled embeddings from another work dir")
    ap.add_argument("--retain", action="store_true")
    ap.add_argument("--jobs", default=None)
    a = ap.parse_args(argv)
    work = _work(a)
    if a.mode == "pool":
        do_pool(work, a.shards)
    elif a.mode == "probe":
        assert a.index is not None and 0 <= a.index < 50
        do_probe(work, a.index, Path(a.emb_from) if a.emb_from else None)
    elif a.mode == "summarize":
        s = summarize(work)
        print(json.dumps({k: s[k] for k in ("mean_classification_macro_ovr_auc", "mean_matched_macro_ovr_auc",
                                            "coxnet_failures")}, indent=2))
    else:
        do_finalize(a, work)


if __name__ == "__main__":
    main()
