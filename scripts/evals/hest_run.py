#!/usr/bin/env python
"""HEST-Benchmark driver on the spectra.evals API (one model -> $SPECTRA_EVAL_OUT/<key>/hest.json).

Protocol:

* 9 leaderboard tasks IDC PRAD PAAD SKCM COAD READ CCRCC LUNG LYMPH_IDC (HCC excluded),
  official patient splits from ``<bench>/<task>/splits`` (train_i/test_i), HEST's own
  StandardScaler -> PCA(256, random_state=1) -> Ridge(lsqr, alpha=100/(256*50)) -> per-gene
  Pearson, mean over 50 genes -> mean over folds -> unweighted mean over tasks.
* encoder: frozen, fp32 weights, fp32 forward (no autocast) unless --precision says otherwise;
  pooling = protocol.pooling_for("hest", repo) (H-optimus-0 / UNI2-h / phikon-v2 / Midnight: cls,
  Virchow/Virchow2: clsmean).
* transform: Resize(224, bilinear) + CenterCrop(224) + ToTensor + Normalize(backbone stats)
  (HEST tiles are already 224 px so Resize/Crop are no-ops).
* batch 256, num_workers 0 (the HEST loader dies in shm teardown at >0 workers).
* the metric code is HEST's, unmodified apart from hest.patch (I/O only: no snapshot_download,
  single h5 handle per dataset, streamed embedding writes, no forced re-embedding), off the
  pinned clone third_party/HEST @ 3ddb5eaf.

Concurrency: every job writes ONLY under its own work dir ``$SPECTRA_EVAL_OUT/<key>/hest_work`` (held with an
exclusive flock for the job's lifetime); the shared bench h5 files are opened read-only, and HDF5
read-only opens take shared flocks that never conflict . Two jobs on the same key are refused instead of corrupting each other's embeddings --
that collision (same exp_code -> same embedding .h5 opened 'w' by one job while another reads it)
is what produced the old ``BlockingIOError: unable to lock file (errno 11)``.

Resumable: a task with a ``done/<task>.json`` marker is skipped; any other task's partial
embeddings/results are wiped and recomputed (a preempted embed .h5 is never trusted).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
HEST_ROOT = WORKTREE / "third_party" / "HEST"
HEST_COMMIT = "3ddb5eaf5bd2a8133e0c0e8015816489a3d99dc3"
TASKS = ("IDC", "PRAD", "PAAD", "SKCM", "COAD", "READ", "CCRCC", "LUNG", "LYMPH_IDC")
os.environ.setdefault("PYTHONNOUSERSITE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _git(*a, cwd=HEST_ROOT) -> str:
    try:
        return subprocess.run(["git", "-C", str(cwd), *a], capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def hest_provenance() -> dict:
    import hashlib

    commit = _git("rev-parse", "HEAD")
    diff = subprocess.run(["git", "-C", str(HEST_ROOT), "diff"], capture_output=True,
                          timeout=30).stdout
    return {"upstream": "https://github.com/mahmoodlab/HEST", "commit": commit,
            "commit_expected": HEST_COMMIT, "patch_sha256": hashlib.sha256(diff).hexdigest(),
            "patch_files": _git("diff", "--name-only").split()}


def env_info() -> dict:
    import numpy, scipy, sklearn, timm, torch  # noqa: E401

    try:
        import h5py
        h5 = {"h5py": h5py.__version__, "hdf5": h5py.version.hdf5_version}
    except Exception:  # noqa: BLE001
        h5 = {}
    return {"python": platform.python_version(), "torch": torch.__version__, "timm": timm.__version__,
            "numpy": numpy.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
            "cuda": torch.version.cuda, "venv": sys.prefix, "upstream_commit": _git("rev-parse", "HEAD"),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, **h5}


def slurm_usage(job_id: str | None) -> tuple[list[dict], float, float]:
    """[{job_id, stage, node, elapsed_s, n_gpus}], gpu_hours, cpu_hours over every attempt of
    this job (requeues included, sacct -D); the running attempt is counted up to now."""
    if not job_id:
        return [], 0.0, 0.0
    try:
        out = subprocess.run(["sacct", "-j", job_id, "-D", "-X", "-n", "-P", "-o",
                              "JobID,NodeList,ElapsedRaw,AllocTRES,State"],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception:  # noqa: BLE001
        out = ""
    jobs, gh, ch = [], 0.0, 0.0
    for line in out.strip().splitlines():
        jid, node, el, tres, state = (line.split("|") + [""] * 5)[:5]
        tr = dict(kv.split("=", 1) for kv in tres.split(",") if "=" in kv)
        ng = int(tr.get("gres/gpu", 0) or 0)
        nc = int(tr.get("cpu", 0) or 0)
        el = float(el or 0)
        jobs.append({"job_id": jid, "stage": "hest", "node": node, "elapsed_s": el, "n_gpus": ng,
                     "n_cpus": nc, "state": state})
        gh += el * ng / 3600
        ch += el * nc / 3600
    return jobs, round(gh, 4), round(ch, 4)


def build_module(m, pooling: str):
    import torch.nn as nn

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.m = m

        def forward(self, x):          # HEST: model(imgs) -> [B, D]
            return self.m.features(x, pooling)

    return _M()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", required=True, help="spec.json from python -m spectra.evals.spec")
    ap.add_argument("--precision", default="fp32", choices=("fp32", "fp16", "bf16"))
    ap.add_argument("--pooling", default=None, help="override protocol.pooling_for (not comparable!)")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--bench-data", default=os.environ.get("HEST_DATA_DIR"),
                    help="MahmoodLab/hest-bench download (default: $HEST_DATA_DIR)")
    ap.add_argument("--work-dir", default=None, help="default $SPECTRA_EVAL_OUT/<key>/hest_work")
    ap.add_argument("--out-name", default="hest", help="result file $SPECTRA_EVAL_OUT/<key>/<out-name>.json")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--retain", action="store_true", help="keep embeddings after the result is written")
    ap.add_argument("--force", action="store_true", help="recompute even if the result is complete")
    a = ap.parse_args(argv)

    from spectra.evals import spec as S
    from spectra.evals.protocol import pooling_for
    from spectra.evals.results import out_dir, read_result, write_result

    spec = S.load_spec(a.spec)
    key = spec["model_key"]
    tasks = [t for t in TASKS if t in a.tasks] + [t for t in a.tasks if t not in TASKS]
    full = tasks == list(TASKS)
    prev = read_result(key, a.out_name)
    if (prev and prev.get("status") == "complete" and not a.force
            and (prev.get("spec") or {}).get("identity_sha256") == spec.get("identity_sha256")):
        print(f"[hest] {key}: {a.out_name}.json already complete -- skipping (use --force)", flush=True)
        return 0

    pooling = pooling_for("hest", spec["backbone_repo"], a.pooling)
    work = Path(a.work_dir) if a.work_dir else out_dir(key) / "hest_work"
    work.mkdir(parents=True, exist_ok=True)
    lockf = open(work / ".lock", "w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(f"[hest] {work} is held by another job (same model key). Refusing to share "
                         "an embedding cache -- that is the old errno-11 collision.")
    lockf.write(f"{socket.gethostname()} pid={os.getpid()} job={os.environ.get('SLURM_JOB_ID')}\n")
    lockf.flush()

    if not a.bench_data:
        raise SystemExit("[hest] set HEST_DATA_DIR or pass --bench-data (the MahmoodLab/hest-bench download)")
    bench = Path(a.bench_data)
    missing = [t for t in tasks if not (bench / t / "splits").is_dir() or not (bench / t / "patches").is_dir()]
    if missing:
        raise SystemExit(f"[hest] bench data missing tasks {missing} under {bench}")
    prov = hest_provenance()
    if prov["commit"] != HEST_COMMIT:
        raise SystemExit(f"[hest] {HEST_ROOT} is at {prov['commit']}, expected {HEST_COMMIT}")

    import torch

    from spectra.evals.model import load_eval_model

    sys.path.insert(0, str(HEST_ROOT / "src"))
    from hest.bench.benchmark import benchmark  # noqa: E402

    t_start = time.time()
    m = load_eval_model(spec, device="cuda", precision=a.precision)
    module = build_module(m, pooling)
    transform = m.transform(resize=True, size=224, interpolation="bilinear")
    expect = m.feat_dim(pooling)
    with torch.no_grad():
        probe = module(torch.zeros(2, 3, 224, 224, device=m.device))
    assert probe.shape == (2, expect) and probe.dtype == torch.float32, (probe.shape, probe.dtype, expect)
    print(f"[hest] key={key} kind={spec['kind']} repo={spec['backbone_repo']} pooling={pooling} "
          f"dim={expect} precision={a.precision} mean={m.mean} std={m.std} tasks={tasks}", flush=True)

    done_dir = work / "done"
    done_dir.mkdir(exist_ok=True)
    per_task, per_fold, task_seconds = {}, {}, {}
    for task in tasks:
        marker = done_dir / f"{task}.json"
        if marker.exists() and not a.force:
            rec = json.loads(marker.read_text())
            if rec.get("identity_sha256") == spec.get("identity_sha256") and rec.get("pooling") == pooling \
                    and rec.get("precision") == a.precision:
                print(f"[hest] {task}: done marker present ({rec['pearson_mean']:.5f}) -- reusing", flush=True)
                per_task[task], per_fold[task], task_seconds[task] = rec["pearson_mean"], rec["folds"], rec["seconds"]
                continue
        emb, res = work / "embeddings" / task, work / "results" / task
        shutil.rmtree(emb, ignore_errors=True)       # a preempted .h5 is never trusted
        shutil.rmtree(res, ignore_errors=True)
        t0 = time.time()
        # HEST precision = fp32 -> its autocast stays disabled; EvalModel applies --precision itself.
        dataset_perfs, _ = benchmark(
            module, transform, torch.float32,
            bench_data_root=str(bench), embed_dataroot=str(emb), results_dir=str(res),
            exp_code=task, datasets=[task], encoders=[], batch_size=a.batch_size,
            num_workers=a.num_workers,
        )
        r = dataset_perfs[0]["results"][0]
        assert r["encoder_name"] == "custom_encoder"
        kf = sorted(res.glob(f"{task}::*/{task}/custom_encoder/results_kfold.json"))[-1]
        kfd = json.loads(kf.read_text())
        rec = {"task": task, "pearson_mean": float(r["pearson_mean"]), "pearson_std": float(r["pearson_std"]),
               "folds": [float(v) for v in kfd["mean_per_split"]], "seconds": round(time.time() - t0, 1),
               "identity_sha256": spec.get("identity_sha256"), "pooling": pooling, "precision": a.precision,
               "results_kfold": str(kf)}
        marker.write_text(json.dumps(rec, indent=2))
        per_task[task], per_fold[task], task_seconds[task] = rec["pearson_mean"], rec["folds"], rec["seconds"]
        print(f"[hest] {task}: {rec['pearson_mean']:.5f} folds={rec['folds']} ({rec['seconds']:.0f}s)", flush=True)
        if not a.retain:
            shutil.rmtree(emb, ignore_errors=True)

    headline = sum(per_task.values()) / len(per_task)
    jobs, gpu_h, cpu_h = slurm_usage(os.environ.get("SLURM_JOB_ID"))
    payload = {
        "protocol": {
            "name": "HEST-Benchmark v1 leaderboard (docs protocol: scripts/run_hest.py / hest_final5.sbatch)",
            "tasks": tasks, "splits": "official patient-level splits <bench>/<task>/splits/{train,test}_i.csv",
            "metric": "Pearson per gene (50 most variable genes, var_50genes.json), mean over genes -> folds -> tasks",
            "readout": "StandardScaler -> PCA(256, random_state=1) -> Ridge(solver=lsqr, alpha=100/(256*50))",
            "precision": a.precision, "weights_dtype": "fp32", "pooling": pooling,
            "transform": "Resize(224, bilinear) + CenterCrop(224) + ToTensor + Normalize(mean, std)",
            "normalize_mean": list(m.mean), "normalize_std": list(m.std),
            "batch_size": a.batch_size, "num_workers": a.num_workers, "bench_data": str(bench),
            "upstream": prov,
        },
        "env": env_info(),
        "metrics": {"hest_mean_pearson": headline, "n_tasks": len(per_task), "full_9_tasks": full},
        "per_task": per_task,
        "per_fold": per_fold,
        "task_seconds": task_seconds,
        "gpu_hours": gpu_h, "cpu_hours": cpu_h, "wall_seconds": round(time.time() - t_start, 1),
        "slurm_jobs": jobs,
        "status": "complete" if full else "partial",
        "notes": ["per_task values are HEST's unrounded pearson_mean; per_fold values come from "
                  "results_kfold.json (HEST rounds them to 4 dp)",
                  "gpu_hours = sum over all attempts of this SLURM job (sacct -D) of Elapsed x n_gpus, "
                  "the running attempt counted up to write time"],
        "work_dir": str(work), "embeddings_retained": bool(a.retain),
    }
    path = write_result(key, a.out_name, payload, spec=spec)
    print(json.dumps({"key": key, "hest_mean_pearson": headline, "per_task": per_task,
                      "gpu_hours": gpu_h}, indent=2), flush=True)
    print(f"[hest] wrote {path}", flush=True)
    if not a.retain:
        shutil.rmtree(work / "embeddings", ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
