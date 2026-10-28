#!/usr/bin/env python
"""PathoROB robustness index + CROMA for any spectra.evals model spec.

Driven by scripts/evals/run_pathorob.sbatch.  Stages (each resumable, each in the right venv):

  prepare   resolve the spec once, guard spec.json identity, emit shell vars   (extraction venv, CPU)
  extract   GPU: clsmean features -> PathoROB per-center npz layout in the work dir (extraction venv)
  metrics   CPU: PathoROB robustness_index (+bootstrap) subprocess + CROMA      (.venv-pathorob)
  finalize  CPU: assemble payload, write_result(key, "pathorob"/"croma"), purge  (extraction venv)

Environment (see run_pathorob.sbatch and env/pathorob_*.txt):
  extraction venv   the interpreter running this script (torch, timm, peft, pyarrow)
  SPECTRA_VENV_PATHOROB_METRICS  directory of the metrics venv (numpy 2.2.6 pandas 2.3.2: PathoROB's own pins)
  PATHOROB_ROOT / PATHOROB_CROMA_ROOT  clones of PathoROB @6583cf0 (+ env/pathorob_bootstrap_ri.patch) and croma @873612e
  SPECTRA_EVAL_OUT               results root (default runs/evals); SPECTRA_PATHOROB_FEATURES  feature work dir root

Protocols (--protocol):
  resize224 (default) fp32, Resize(224, bilinear, antialias PIL) -> CenterCrop(224) -> ToTensor -> Normalize(backbone mean/std),
                      cat(cls, mean patch) = clsmean, PathoROB `robustness_index` k_opt_param=0.
  resize256           fp16, bicubic Resize(256) -> CenterCrop(224) (a centre crop, smaller field of view than `resize224`).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PATHOROB_ROOT = Path(os.environ.get("PATHOROB_ROOT", "third_party/PathoROB"))
CROMA_ROOT = Path(os.environ.get("PATHOROB_CROMA_ROOT", "third_party/croma"))
EXTRACT_PY = Path(sys.executable)
METRICS_PY = Path(os.environ.get("SPECTRA_VENV_PATHOROB_METRICS", "")) / "bin" / "python"
PATCH = HERE / "env" / "pathorob_bootstrap_ri.patch"
FEATURES_ROOT = Path(os.environ.get("SPECTRA_PATHOROB_FEATURES", "runs/pathorob_features"))
EVALS_ROOT = Path(os.environ.get("SPECTRA_EVAL_OUT", "runs/evals"))
DATASETS = ("camelyon", "tcga", "tolkach_esca")
META_COLUMNS = ("slide_id", "patch_id", "biological_class", "medical_center")

PROTOCOLS = {
    "resize224": dict(precision="fp32", resize=224, interpolation="bilinear", crop=224, autocast=None),
    "resize256": dict(precision="fp16", resize=256, interpolation="bicubic", crop=224, autocast=None),
}


def work_dir(key: str) -> Path:
    return FEATURES_ROOT / f"evals_{key}"


def _sha256(p: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _git_rev(p: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(p), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _append_stage(wd: Path, rec: dict) -> None:
    wd.mkdir(parents=True, exist_ok=True)
    with open(wd / "stages.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")


def _stage_rec(stage: str, t0: float, n_gpus: int, ncpu: int | None = None) -> dict:
    return {"job_id": os.environ.get("SLURM_JOB_ID"), "stage": stage, "node": os.uname().nodename,
            "elapsed_s": round(time.time() - t0, 1), "n_gpus": n_gpus,
            "cpus": ncpu or int(os.environ.get("SLURM_CPUS_ON_NODE") or os.cpu_count() or 1)}


# --------------------------------------------------------------------------------------
# prepare
def cmd_prepare(a) -> int:
    sys.path.insert(0, str(REPO / "src"))
    from spectra.evals.spec import resolve_spec
    spec = resolve_spec(backbone=a.backbone, ckpt=a.ckpt, step=a.step, key=a.key, ema=a.ema)
    key = spec["model_key"]
    out = EVALS_ROOT / key
    out.mkdir(parents=True, exist_ok=True)
    sp = out / "spec.json"
    if sp.exists():
        old = json.loads(sp.read_text())
        if old.get("identity_sha256") != spec.get("identity_sha256"):
            if not a.force:
                raise SystemExit(f"{sp} has identity {old.get('identity_sha256')} != new {spec.get('identity_sha256')}; "
                                 "pass --force to overwrite (this also invalidates stored results)")
            print(f"[prepare] --force: replacing spec for {key}", file=sys.stderr)
            (work_dir(key)).exists() and shutil.rmtree(work_dir(key))
            for b in ("pathorob", "croma"):
                (out / f"{b}.json").unlink(missing_ok=True)
            sp.write_text(json.dumps(spec, indent=2))
    else:
        sp.write_text(json.dumps(spec, indent=2))
    res = out / "pathorob.json"
    done = False
    if res.exists() and not a.force:
        try:
            r = json.loads(res.read_text())
            done = r.get("status") == "complete" and r.get("protocol", {}).get("name") == a.protocol
        except Exception:
            done = False
    wd = work_dir(key)
    ext_done = all((wd / f"extract_{d}.done").exists() for d in DATASETS)
    if a.force and res.exists():
        done = False
    for line in (f"KEY={key}", f"RESULT_DONE={int(done)}", f"EXTRACT_DONE={int(ext_done)}",
                 f"SPEC_JSON={sp}"):
        print(line)
    return 0


# --------------------------------------------------------------------------------------
# extract
def cmd_extract(a) -> int:
    import numpy as np
    import torch
    import torchvision.transforms as T
    sys.path.insert(0, str(REPO / "src"))
    from spectra.evals.model import load_eval_model
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    proto = PROTOCOLS[a.protocol]
    spec = json.loads((EVALS_ROOT / a.key / "spec.json").read_text())
    wd = work_dir(a.key)
    wd.mkdir(parents=True, exist_ok=True)
    todo = [d for d in DATASETS if not (wd / f"extract_{d}.done").exists()]
    if not todo:
        print("[extract] all datasets already extracted"); return 0
    t0 = time.time()
    sys.path.insert(0, str(PATHOROB_ROOT))
    from pathorob.features.data_manager import FeatureDataManager
    import io, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
    from PIL import Image
    from huggingface_hub import snapshot_download
    from torch.utils.data import DataLoader, Dataset

    prec = {"fp32": "fp32", "fp16": "fp16", "bf16": "bf16"}[a.precision or proto["precision"]]
    m = load_eval_model(spec, device="cuda", precision=prec)
    mean, std = tuple(float(x) for x in m.mean), tuple(float(x) for x in m.std)
    print(f"[extract] key={a.key} backbone={m.backbone_repo} kind={m.kind} precision={prec} protocol={a.protocol} "
          f"mean={mean} std={std}", flush=True)
    interp = {"bilinear": T.InterpolationMode.BILINEAR, "bicubic": T.InterpolationMode.BICUBIC}[proto["interpolation"]]
    tf = T.Compose([T.Resize(proto["resize"], interpolation=interp), T.CenterCrop(proto["crop"]),
                    T.ToTensor(), T.Normalize(mean, std)])
    # equivalence check vs the model's own transform  on a synthetic 256px tile
    try:
        g = np.random.RandomState(0).randint(0, 255, (256, 256, 3), dtype=np.uint8)
        im = Image.fromarray(g)
        mt = m.transform(resize=True, size=224, interpolation=proto["interpolation"]) if a.protocol == "resize224" else None
        if mt is not None:
            d = float((mt(im) - tf(im)).abs().max())
            print(f"[extract] max|m.transform - PathoROB-protocol transform| on a synthetic tile = {d:.3e}", flush=True)
    except Exception as e:  # informational only
        print(f"[extract] transform equivalence check skipped: {type(e).__name__}: {e}", flush=True)

    class DS(Dataset):
        def __init__(self, dataset):
            snap = snapshot_download(f"bifold-pathomics/PathoROB-{dataset}", repo_type="dataset",
                                     allow_patterns=["data/*.parquet"])
            files = sorted((Path(snap) / "data").glob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"no parquet under {snap}/data")
            self.table = pa.concat_tables([pq.read_table(f) for f in files])
            self.images = self.table.column("image")
            self.meta = {c: self.table.column(c).to_pylist() for c in META_COLUMNS}
        def __len__(self): return self.table.num_rows
        def __getitem__(self, i):
            img = Image.open(io.BytesIO(self.images[i]["bytes"].as_py())).convert("RGB")
            return tf(img), i

    ac = proto["autocast"]
    ac_dtype = {"float16": torch.float16, None: None}[ac]
    info = {}
    for dataset in todo:
        ds = DS(dataset)
        n = len(ds)
        loader = DataLoader(ds, batch_size=a.batch_size, num_workers=a.num_workers, shuffle=False, pin_memory=True)
        feats = None
        t1 = time.time(); done = 0
        with torch.inference_mode():
            for images, idx in loader:
                images = images.to("cuda", non_blocking=True)
                if ac_dtype is not None:
                    with torch.autocast("cuda", dtype=ac_dtype):
                        out = m.clsmean_features(images)
                else:
                    out = m.clsmean_features(images)
                out = out.float().cpu().numpy()
                if feats is None:
                    feats = np.empty((n, out.shape[1]), dtype=np.float32)
                feats[idx.numpy()] = out
                done += len(idx)
                if done % (a.batch_size * 20) < a.batch_size:
                    r = done / (time.time() - t1)
                    print(f"[extract] {dataset} {done}/{n} {r:.0f} img/s eta {(n - done) / r:.0f}s", flush=True)
        if not np.isfinite(feats).all():
            raise RuntimeError("non-finite features")
        norms = np.linalg.norm(feats, axis=1)
        print(f"[extract] {dataset}: {n} x {feats.shape[1]} in {time.time() - t1:.0f}s  |f| min {norms.min():.3f} "
              f"mean {norms.mean():.3f} max {norms.max():.3f}", flush=True)
        metadata = pd.DataFrame({c: ds.meta[c] for c in META_COLUMNS})
        # write via PathoROB's own FeatureDataManager into a tmp root then move into place (atomic per dataset)
        tmp = wd / f"_tmp_{dataset}"
        shutil.rmtree(tmp, ignore_errors=True)
        dm = FeatureDataManager(features_dir=str(tmp), metadata_dir=str(PATHOROB_ROOT / "data" / "metadata"))
        dm.save_features(a.key, dataset, feats, metadata)
        dest = wd / "features" / a.key / dataset
        shutil.rmtree(dest, ignore_errors=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp / a.key / dataset, dest)
        shutil.rmtree(tmp, ignore_errors=True)
        (wd / f"extract_{dataset}.done").write_text(json.dumps({"rows": n, "dim": int(feats.shape[1])}))
        info[dataset] = {"rows": n, "dim": int(feats.shape[1]), "feat_norm_mean": float(norms.mean())}
        del ds, loader, feats
    import timm
    env = {"torch": torch.__version__, "timm": timm.__version__, "python": sys.version.split()[0],
           "extract_venv": str(EXTRACT_PY), "cuda_device": torch.cuda.get_device_name(0)}
    (wd / "extract_env.json").write_text(json.dumps({"env": env, "info": info, "precision": prec,
                                                     "protocol": a.protocol, "backbone": m.backbone_repo}))
    _append_stage(wd, _stage_rec("extract", t0, 1))
    return 0


# --------------------------------------------------------------------------------------
def cmd_metrics(a) -> int:
    wd = work_dir(a.key)
    fdir = wd / "features"
    t0 = time.time()
    rdir = wd / "ri_results"
    mdir = PATHOROB_ROOT / "data" / "metadata"
    env = {**os.environ, "PYTHONNOUSERSITE": "1", "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "4")}
    nw = str(a.num_workers)
    for ds in DATASETS:
        summ = rdir / a.key / ds / "-1_0" / "results_summary.json"
        if summ.exists():
            print(f"[metrics] RI {ds}: cached"); continue
        cmd = [str(METRICS_PY), "-m", "pathorob.robustness_index.robustness_index", "--model", a.key,
               "--datasets", ds, "--features_dir", str(fdir), "--metadata_dir", str(mdir),
               "--results_dir", str(rdir), "--k_opt_param", "0", "--num_workers", nw]
        if a.bootstrap:
            cmd.append("--compute_bootstrapped_robustness_index")
        t1 = time.time()
        subprocess.run(cmd, cwd=str(PATHOROB_ROOT), check=True, env=env)
        print(f"[metrics] RI {ds}: {time.time() - t1:.0f}s", flush=True)
    croma_p = wd / "croma.json"
    if not croma_p.exists():
        t1 = time.time()
        sys.path.insert(0, str(HERE))
        import pathorob_croma_adapter as ca
        res = ca.evaluate(a.key, fdir, mdir, tuple(DATASETS), with_ri=True)
        croma_p.write_text(json.dumps(res, indent=2))
        print(f"[metrics] CROMA {time.time() - t1:.0f}s avg={res['avg']}", flush=True)
    import numpy, pandas, sklearn
    menv = {"python": sys.version.split()[0], "numpy": numpy.__version__, "pandas": pandas.__version__,
            "sklearn": sklearn.__version__, "metrics_venv": str(METRICS_PY)}
    (wd / "metrics_env.json").write_text(json.dumps(menv))
    _append_stage(wd, _stage_rec("metrics", t0, 0, ncpu=int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)))
    return 0


# --------------------------------------------------------------------------------------
def _sacct(job_id: str) -> dict:
    try:
        o = subprocess.check_output(["sacct", "-j", str(job_id), "-X", "-n", "-P", "-o",
                                     "JobID,State,ElapsedRaw,AllocTRES,TotalCPU"], text=True, timeout=30).strip()
        j, st, el, tres, tcpu = o.splitlines()[0].split("|")
        ng = 0
        for part in tres.split(","):
            if part.startswith("gres/gpu=") and not part.startswith("gres/gpu:"):
                ng = int(part.split("=")[1])
        cpus = next((int(p.split("=")[1]) for p in tres.split(",") if p.startswith("cpu=")), 0)
        return {"state": st, "elapsed_s": float(el), "n_gpus": ng, "cpus": cpus, "total_cpu": tcpu}
    except Exception:
        return {}


def cmd_finalize(a) -> int:
    sys.path.insert(0, str(REPO / "src"))
    from spectra.evals.results import write_result
    wd = work_dir(a.key)
    spec = json.loads((EVALS_ROOT / a.key / "spec.json").read_text())
    per = {}
    for ds in DATASETS:
        r = json.loads((wd / "ri_results" / a.key / ds / "-1_0" / "results_summary.json").read_text())
        per[ds] = {"robustness_index": r["robustness_index"], **{k: v for k, v in r.items() if k != "robustness_index"}}
    croma = json.loads((wd / "croma.json").read_text())
    for ds in DATASETS:
        per[ds]["croma"] = croma["datasets"][ds]
    ris = [per[d]["robustness_index"] for d in DATASETS]
    metrics = {"avg_robustness_index": sum(ris) / 3, **{f"ri_{d}": per[d]["robustness_index"] for d in DATASETS}}
    for k in ("balanced_accuracy", "generalization_index", "bio_vs_confounding", "confounder_insensitivity",
              "normalized_confounder_insensitivity"):
        vals = [per[d].get(k) for d in DATASETS]
        if all(v is not None for v in vals):
            metrics[f"avg_{k}"] = sum(vals) / 3
    metrics.update({f"croma_{k}": v for k, v in croma["avg"].items()})
    stages = [json.loads(l) for l in (wd / "stages.jsonl").read_text().splitlines() if l.strip()]
    gpu_s = cpu_s = 0.0
    for s in stages:
        sa = _sacct(s["job_id"]) if s.get("job_id") else {}
        # in-process wall time is the per-attempt truth (a requeued job id spans several attempts)
        if s["stage"] == "extract":
            gpu_s += s["elapsed_s"] * s["n_gpus"]
        cpu_s += s["elapsed_s"] * s["cpus"] if s["stage"] == "metrics" else 0.0
        s["sacct"] = sa
    ex = json.loads((wd / "extract_env.json").read_text())
    menv = json.loads((wd / "metrics_env.json").read_text())
    proto = PROTOCOLS[a.protocol]
    ex_env = ex["env"]
    payload = {
        "protocol": {"name": a.protocol, **proto, "precision_used": ex["precision"], "pooling": "clsmean (cat(cls, mean patch tokens))",
                     "tasks": list(DATASETS), "splits": "PathoROB ID/OOD defaults (tcga paired_evaluation default), k_opt_param=0 (per-dataset default k)",
                     "ri_args": {"k_opt_param": 0, "max_patches_per_combi": -1, "bootstrap": bool(a.bootstrap)},
                     "pathorob_commit": _git_rev(PATHOROB_ROOT), "pathorob_patch": str(PATCH), "pathorob_patch_sha256": _sha256(PATCH),
                     "croma_commit": _git_rev(CROMA_ROOT), "croma_adapter": "scripts/evals/pathorob_croma_adapter.py (m=5 headline, alpha=0.10)",
                     "features_dim": {d: ex["info"][d]["dim"] for d in ex["info"]} if ex.get("info") else None},
        "env": {**ex_env, **{f"metrics_{k}": v for k, v in menv.items()}, "pathorob_commit": _git_rev(PATHOROB_ROOT),
                "croma_commit": _git_rev(CROMA_ROOT)},
        "metrics": metrics,
        "per_task": per,
        "croma": croma,
        "gpu_hours": gpu_s / 3600.0,
        "cpu_hours": cpu_s / 3600.0,
        "wall_seconds": sum(s["elapsed_s"] for s in stages),
        "slurm_jobs": [{k: s.get(k) for k in ("job_id", "stage", "node", "elapsed_s", "n_gpus", "cpus", "sacct")} for s in stages],
        "status": "complete",
        "notes": ["RI and CROMA are reporting-only (never used for selection).",
                  "gpu_hours = extract elapsed x n_gpus (in-process wall per attempt); cpu_hours = metrics elapsed x allocated cpus."],
    }
    p = write_result(a.key, "pathorob", payload)
    write_result(a.key, "croma", {"protocol": payload["protocol"], "env": payload["env"], "metrics": {k: v for k, v in metrics.items() if k.startswith("croma_")},
                                  "per_task": {d: per[d]["croma"] for d in DATASETS}, "croma": croma,
                                  "gpu_hours": 0.0, "cpu_hours": payload["cpu_hours"], "wall_seconds": payload["wall_seconds"],
                                  "slurm_jobs": payload["slurm_jobs"], "status": "complete",
                                  "notes": ["computed on the same PathoROB features as pathorob.json (no extra GPU work)"]})
    print(f"[finalize] wrote {p}  avg_RI={metrics['avg_robustness_index']:.6f}  gpu_h={payload['gpu_hours']:.3f} cpu_h={payload['cpu_hours']:.3f}")
    if a.retain:
        print(f"[finalize] --retain: keeping {wd}")
    else:
        shutil.rmtree(wd, ignore_errors=True)
        print(f"[finalize] purged {wd}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    def common(p):
        p.add_argument("--protocol", choices=sorted(PROTOCOLS), default="resize224")
        p.add_argument("--retain", action="store_true")
        p.add_argument("--force", action="store_true")
        p.add_argument("--no-bootstrap", dest="bootstrap", action="store_false", default=True)
        p.add_argument("--num-workers", type=int, default=8)
        p.add_argument("--batch-size", type=int, default=256)
        p.add_argument("--precision", default=None, choices=[None, "fp32", "fp16", "bf16"])
    p = sub.add_parser("prepare"); common(p)
    p.add_argument("--backbone", required=True); p.add_argument("--ckpt"); p.add_argument("--step", type=int)
    p.add_argument("--key"); p.add_argument("--ema", action="store_true")
    for n in ("extract", "metrics", "finalize"):
        q = sub.add_parser(n); common(q); q.add_argument("--key", required=True)
    a = ap.parse_args()
    return {"prepare": cmd_prepare, "extract": cmd_extract, "metrics": cmd_metrics, "finalize": cmd_finalize}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
