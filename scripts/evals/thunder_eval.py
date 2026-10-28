# THUNDER stage driver. The current THUNDER website leaderboard = 16 classification datasets x
# {KNN, linear probing (+ calibration ECE), 16-shot SimpleShot, PGD adversarial attack} + 4
# segmentation datasets with the official SegPath epoch overrides.
# https://mics-lab.github.io/thunder/leaderboards/
#
#   * the model is spectra.evals.load_eval_model(spec);
#   * the run manifest binds the THUNDER model id to spec identity_sha256 + the explicit
#     PROTOCOL dict below + upstream commit/patch hashes;
#   * all THUNDER state for one model lives under $SPECTRA_EVAL_OUT/<key>/thunder_work (one
#     results.csv per model) and the summary writes <key>/thunder.json via spectra.evals.results.
# Numerics: fp16 autocast encoder, fp32 frozen features, int8 segmentation tokens (patched
# upstream), batch sizes 1024 / attack 128, adaptation batch 64, SegPath epochs 9/21,
# per-backbone pooling (spectra.evals.protocol.THUNDER_POOLING).
#
# usage (normally via scripts/evals/run_thunder.sbatch):
#   THUNDER_SPEC=<spec.json> python thunder_eval.py MODE [INDEX]
#   MODE: preflight | precompute_group I | cached_group I | cleanup | adversarial_group I |
#         segmentation I | summary | status
import atexit
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[1]
UPSTREAM = Path(os.environ.get("THUNDER_UPSTREAM", WORKTREE / "third_party/thunder"))
UPSTREAM_COMMIT = "3d1cc9513fb2cfd8c4afb0d7bb9f5c4f6b69117f"
PATCHES = [HERE / "env/thunder.patch", HERE / "env/thunder_local.patch"]
DATA_ROOT = os.environ.get("THUNDER_DATA") or sys.exit("set THUNDER_DATA (the THUNDER data folder)")

PROTOCOL_VERSION = "spectra-thunder-v1"
CLS_DATASETS = [
    "bach", "bracs", "break_his", "ccrcc", "crc", "esca", "mhist", "patch_camelyon",
    "spider_breast", "spider_colorectal", "spider_skin", "spider_thorax",
    "wilds", "tcga_crc_msi", "tcga_tils", "tcga_uniform",
]
SEG_DATASETS = ["pannuke", "ocelot", "segpath_epithelial", "segpath_lymphocytes"]
SEG_EPOCHS = {"segpath_epithelial": 9, "segpath_lymphocytes": 21}
CACHED_TASKS = ["linear_probing", "knn", "simple_shot"]
# Greedy balance by official split size (2,297,274 images total) and PGD sample cap.
PRECOMPUTE_GROUPS = [[12, 14, 10, 4, 11, 13, 3], [5, 7, 15, 8, 9, 1, 6, 2, 0]]
ADVERSARIAL_GROUPS = [[5, 8, 10, 12, 14, 4, 2, 0], [7, 9, 11, 13, 15, 3, 6, 1]]
# Online work items, in run order.
ONLINE_TASKS = [("segmentation", 1), ("segmentation", 2), ("segmentation", 3),
                ("segmentation", 0), ("adversarial_group", 0), ("adversarial_group", 1)]
# Cap each extracted dense-feature tensor near the tested ViT-S batch footprint.
MAX_SEGMENTATION_BATCH_VALUES = 2048 * 256 * 384
ADAPTATION_BATCH = 64
PRE_COMP_WORKERS = 16

PRECISION = os.environ.get("THUNDER_PRECISION", "fp16")
ADV_PRECISION = os.environ.get("THUNDER_ADV_PRECISION", "fp16")
EXTRACTION_BATCH = int(os.environ.get("EXTRACTION_BATCH", "1024"))
ATTACK_BATCH = int(os.environ.get("ATTACK_BATCH", "128" if ADV_PRECISION != "fp32" else "32"))
RETAIN_EMBEDDINGS = os.environ.get("RETAIN_EMBEDDINGS", "0") == "1"
assert PRECISION in ("fp16", "bf16", "fp32") and ADV_PRECISION in ("fp16", "bf16", "fp32")

SPEC_PATH = Path(os.environ["THUNDER_SPEC"])
SPEC = json.loads(SPEC_PATH.read_text())
MODEL_KEY = SPEC["model_key"]
MODEL_ID = MODEL_KEY + "_optimized"
assert len(MODEL_ID) <= 64, MODEL_ID
WORK_ROOT = Path(os.environ.get("THUNDER_WORK", SPEC_PATH.parent / "thunder_work"))
WORK_ROOT.mkdir(parents=True, exist_ok=True)
RESULTS = WORK_ROOT / "outputs/res"
EMBEDDINGS = WORK_ROOT / "embeddings"
MANIFEST = WORK_ROOT / "protocol_manifest.json"
JOBS_LEDGER = WORK_ROOT / "slurm_jobs.jsonl"

os.environ["THUNDER_BASE_DATA_FOLDER"] = str(WORK_ROOT)
os.environ["THUNDER_DATA_FOLDER"] = DATA_ROOT
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["THUNDER_WANDB_MODE"] = "disabled"
os.environ["WANDB_MODE"] = "disabled"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

mode = sys.argv[1] if len(sys.argv) > 1 else ""
assert mode in ("preflight", "precompute_group", "cached_group", "cleanup", "adversarial_group",
                "segmentation", "summary", "status"), mode
SEGMENTATION_EMBEDDINGS = Path(
    f"/tmp/{os.environ.get('USER', 'nouser')}/spectra-thunder/{os.environ.get('SLURM_JOB_ID', 'local')}"
    f"_{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}"
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 23), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_pooling():
    p = os.environ.get("THUNDER_POOLING")
    if p:
        return p
    from spectra.evals.protocol import pooling_for
    return pooling_for("thunder", SPEC["backbone_repo"])


POOLING = resolve_pooling()
assert POOLING in ("cls", "mean", "clsmean"), POOLING


def protocol():
    """Every identity-bearing knob. Batch sizes are tuning, not identity ."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "upstream": "https://github.com/MICS-Lab/thunder",
        "upstream_commit": UPSTREAM_COMMIT,
        "patch_sha256": {p.name: sha256_file(p) for p in PATCHES},
        "data_root": DATA_ROOT,
        "cls_datasets": CLS_DATASETS, "seg_datasets": SEG_DATASETS, "seg_epochs": SEG_EPOCHS,
        "cached_tasks": CACHED_TASKS, "adaptation_batch_size": ADAPTATION_BATCH,
        "max_segmentation_batch_values": MAX_SEGMENTATION_BATCH_VALUES,
        "segmentation_token_cache": "int8 symmetric, one fp16 scale per token (thunder.patch)",
        "segmentation_matmul_precision": "high (TF32)",
        "encoder_precision": PRECISION, "adversarial_precision": ADV_PRECISION,
        # PGD's random start / subset are drawn per batch: an attack-batch change moved the
        # H-optimus-0 PGD drop by 2.2 points (batch 64: 34.6 vs batch 128: 32.4),
        # so the attack batch is identity-bearing (extraction batch is not: knn/lin/ECE match).
        "attack_batch_size": ATTACK_BATCH,
        "pooling": POOLING, "transform": "EvalModel.transform(timm_style=True)",
        "model_id": MODEL_ID, "identity_sha256": SPEC["identity_sha256"], "spec_kind": SPEC["kind"],
    }


def write_or_verify_manifest(extra=None):
    current = protocol()
    if MANIFEST.exists():
        saved = json.loads(MANIFEST.read_text())
        saved_core = {k: v for k, v in saved.items() if k in current}
        assert saved_core == current, (
            f"{MODEL_ID}: THUNDER work dir {WORK_ROOT} is bound to a different model/protocol.\n"
            + "\n".join(f"  {k}: saved={saved_core.get(k)!r} now={v!r}" for k, v in current.items()
                        if saved_core.get(k) != v)
            + "\nUse a new --key, or --force to wipe the work dir.")
        return saved
    existing = [str(p) for p in list(RESULTS.glob(f"*/{MODEL_ID}")) + list(EMBEDDINGS.glob(f"*/{MODEL_ID}"))]
    assert not existing, f"unverifiable outputs exist without a manifest: {existing[:5]}"
    rec = dict(current, **(extra or {}))
    tmp = MANIFEST.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, MANIFEST)
    return rec


def verify_manifest():
    assert MANIFEST.exists(), f"run preflight before evaluation stages ({MANIFEST} missing)"
    return write_or_verify_manifest()


def outputs_json(dataset, task):
    return RESULTS / dataset / MODEL_ID / task / "frozen" / "outputs.json"


def readable(path):
    try:
        json.loads(Path(path).read_text())
        return True
    except Exception:  # noqa: BLE001
        return False


def task_done(dataset, task):
    return readable(outputs_json(dataset, task))


# ------------------------------------------------------------------------------------------
# status / summary (no torch model needed)
# ------------------------------------------------------------------------------------------
def status():
    st = {
        "cached": {t: sum(task_done(d, t) for d in CLS_DATASETS) for t in CACHED_TASKS},
        "adversarial_attack": sum(task_done(d, "adversarial_attack") for d in CLS_DATASETS),
        "segmentation": sum(task_done(d, "segmentation") for d in SEG_DATASETS),
    }
    st["cached_complete"] = all(v == len(CLS_DATASETS) for v in st["cached"].values())
    st["online_complete"] = (st["adversarial_attack"] == len(CLS_DATASETS)
                             and st["segmentation"] == len(SEG_DATASETS))
    return st


if mode == "status":
    print(json.dumps(status()))
    sys.exit(0)


def sacct_jobs():
    """[{job_id, stage, node, elapsed_s, n_gpus, n_cpus, start, end, state}] for every array task
    recorded in the ledger (all submissions; -D: preempted/requeued attempts are counted too)."""
    if not JOBS_LEDGER.exists():
        return []
    ledger = [json.loads(line) for line in JOBS_LEDGER.read_text().splitlines() if line.strip()]
    stage_of = {str(r["job_id"]): r["stage"] for r in ledger}
    if not stage_of:
        return []
    out = subprocess.run(
        ["sacct", "-n", "-P", "-X", "-D", "-j", ",".join(stage_of), "--format",
         "JobID,JobName,State,ElapsedRaw,NodeList,AllocTRES,Start,End"],
        capture_output=True, text=True, timeout=120).stdout
    rows = []
    for line in out.splitlines():
        jid, name, state, elapsed, node, tres, start, end = line.split("|")
        base = jid.split("_")[0]
        tres_d = dict(kv.split("=", 1) for kv in tres.split(",") if "=" in kv)
        rows.append({"job_id": jid, "stage": stage_of.get(base, name), "state": state, "node": node,
                     "elapsed_s": int(elapsed or 0), "n_gpus": int(tres_d.get("gres/gpu", 0)),
                     "n_cpus": int(tres_d.get("cpu", 0)), "start": start, "end": end})
    return rows


def summary():
    import pandas as pd
    from thunder.utils.results import gather_results

    from spectra.evals.results import write_result

    manifest = verify_manifest()
    gather_results()
    frame = pd.read_csv(RESULTS / "results.csv")
    frame = frame[frame["model"] == MODEL_ID]
    # pandas parses the simple_shot shot-count setting as a float (16.0); normalise to "16"
    frame["setting"] = frame["setting"].map(
        lambda v: "" if pd.isna(v) else (str(int(v)) if isinstance(v, float) and float(v).is_integer() else str(v)))

    def score(dataset, task, metric, setting=""):
        sub = frame[(frame["dataset"] == dataset) & (frame["task"] == task)
                    & (frame["metric"] == metric) & (frame["setting"] == setting)]
        return None if sub.empty else float(sub["metric_score"].iloc[0])

    rollups = {  # name -> (roll-up row, task, metric, setting, datasets, higher-is-better)
        "knn": ("benchmark_knn", "knn", "f1", "", CLS_DATASETS, True),
        "linear_probing": ("benchmark_linear_probing", "linear_probing", "f1", "", CLS_DATASETS, True),
        "simple_shot": ("benchmark_simple_shot", "simple_shot", "f1", "16", CLS_DATASETS, True),
        "segmentation": ("benchmark_segmentation", "segmentation", "f1", "", SEG_DATASETS, True),
        "calibration": ("benchmark_calibration", "linear_probing", "ECE", "", CLS_DATASETS, False),
        "adversarial_attack": ("benchmark_adversarial_attack", "adversarial_attack", "f1", "drop",
                               CLS_DATASETS, False),
    }
    metrics, per_task, notes = {}, {}, []
    for name, (row, task, metric, setting, datasets, higher) in rollups.items():
        per_ds = {d: score(d, task, metric, setting) for d in datasets}
        per_task[name] = {"task": task, "metric": metric, "setting": setting,
                          "higher_is_better": higher, "per_dataset": per_ds}
        metrics[name] = score(row, task, metric, setting)
        if metrics[name] is None:
            have = [v for v in per_ds.values() if v is not None]
            notes.append(f"{name}: {len(have)}/{len(datasets)} datasets scored; no roll-up")
    for d in CLS_DATASETS:  # adversarial clean / adversarial f1 alongside the drop
        for s in ("clean", "adversarial"):
            per_task["adversarial_attack"].setdefault(f"per_dataset_{s}", {})[d] = score(
                d, "adversarial_attack", "f1", s)
    per_task["simple_shot"]["per_dataset_all_shots"] = {
        d: {k: score(d, "simple_shot", "f1", k) for k in ("1", "2", "4", "8", "16")} for d in CLS_DATASETS}
    complete = all(v is not None for v in metrics.values())

    jobs = sacct_jobs()
    gpu_h, cpu_h, stage_h = 0.0, 0.0, {}
    for j in jobs:
        if j["n_gpus"]:
            h = j["elapsed_s"] * j["n_gpus"] / 3600
            gpu_h += h
            stage_h[j["stage"]] = stage_h.get(j["stage"], 0.0) + h
        else:
            cpu_h += j["elapsed_s"] * max(j["n_cpus"], 1) / 3600
    starts = [j["start"] for j in jobs if j["start"] not in ("", "Unknown", "None")]
    ends = [j["end"] for j in jobs if j["end"] not in ("", "Unknown", "None")]
    wall = 0.0
    if starts and ends:
        from datetime import datetime
        wall = (max(datetime.fromisoformat(e) for e in ends)
                - min(datetime.fromisoformat(s) for s in starts)).total_seconds()
    import numpy, sklearn, timm, torch  # noqa: E401
    payload = {
        "protocol": {**{k: v for k, v in manifest.items()}, "extraction_batch": EXTRACTION_BATCH,
                     "attack_batch": ATTACK_BATCH, "splits": f"{DATA_ROOT}/data_splits/<dataset>.json (official)",
                     "tasks": list(rollups)},
        "env": {"python": sys.version.split()[0], "torch": torch.__version__, "timm": timm.__version__,
                "numpy": numpy.__version__, "sklearn": sklearn.__version__, "venv": sys.prefix,
                "upstream_commit": UPSTREAM_COMMIT, "upstream_path": str(UPSTREAM)},
        "metrics": metrics,
        "per_task": per_task,
        "gpu_hours": round(gpu_h, 3), "gpu_hours_by_stage": {k: round(v, 3) for k, v in stage_h.items()},
        "cpu_hours": round(cpu_h, 3), "wall_seconds": wall,
        "slurm_jobs": jobs,
        "status": "complete" if complete else "partial",
        "notes": notes + [
            "metrics are THUNDER's own benchmark_* roll-ups (mean of per-dataset scores rounded to 0.1, "
            "x100); calibration = ECE and adversarial_attack = PGD f1 drop are lower-is-better.",
            f"work dir {WORK_ROOT} (results.csv, per-dataset outputs.json)",
        ],
    }
    out = write_result(MODEL_KEY, "thunder", payload, spec=SPEC)
    print(json.dumps({"written": str(out), "status": payload["status"], "metrics": metrics,
                      "gpu_hours": payload["gpu_hours"], "by_stage": payload["gpu_hours_by_stage"]}, indent=2))


if mode == "summary":
    summary()
    sys.exit(0)

# ------------------------------------------------------------------------------------------
# model + THUNDER wrappers
# ------------------------------------------------------------------------------------------
import torch  # noqa: E402
from thunder import benchmark  # noqa: E402
from thunder.models.pretrained_models import PretrainedModel  # noqa: E402

KWARGS = {
    "dataset.base_data_folder": DATA_ROOT,
    "task.pre_comp_emb_num_workers": PRE_COMP_WORKERS,
    "task.pre_comp_emb_batch_size": EXTRACTION_BATCH,
    "task.attack_batch_size": ATTACK_BATCH,
    "adaptation.batch_size": ADAPTATION_BATCH,
}


class ThunderModel(PretrainedModel):
    """THUNDER custom model around spectra.evals.EvalModel ."""

    def __init__(self, precision=PRECISION):
        super().__init__()
        from spectra.evals.model import load_eval_model

        self.encoder = load_eval_model(SPEC, device="cuda", precision=precision)
        assert self.encoder.name == MODEL_KEY
        self.name = MODEL_ID
        self.vlm = False
        self._transform = self.encoder.transform(timm_style=True)
        probe = torch.zeros(1, 3, 224, 224, device="cuda")
        with torch.inference_mode():
            self.emb_dim = int(self._cls(probe).shape[-1])
            tok = self.encoder.patch_tokens(probe)
            self.segmentation_tokens, self.emb_dim_seg = int(tok.shape[1]), int(tok.shape[2])

    def _cls(self, images):
        return self.encoder.features(images, POOLING)

    def get_transform(self):
        return self._transform

    def get_linear_probing_embeddings(self, images):
        return self._cls(images).float()

    def get_segmentation_embeddings(self, images):
        return self.encoder.patch_tokens(images).float()

    def forward(self, images):
        return self.get_linear_probing_embeddings(images)


class CachedThunderModel(torch.nn.Module):
    # Cached probes only need the model name and embedding width; never reload the backbone.
    def __init__(self, embedding_dim, segmentation_dim=None):
        super().__init__()
        self.name = MODEL_ID
        self.emb_dim = embedding_dim
        if segmentation_dim is not None:
            self.emb_dim_seg = segmentation_dim
        self.vlm = False

    def get_transform(self):
        raise RuntimeError("cached probes must not request an image transform")

    def get_linear_probing_embeddings(self, images):
        raise RuntimeError("cached probes must not run the backbone")

    def get_segmentation_embeddings(self, images):
        raise RuntimeError("cached probes must not run the backbone")


def preflight():
    """Upstream pin + patches, data, spec identity, model load, transform, shapes, finiteness,
    PGD input gradients, and synthetic regressions for the optimized THUNDER ops."""
    import numpy as np

    head = subprocess.run(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    assert head == UPSTREAM_COMMIT, (head, UPSTREAM_COMMIT)
    import thunder
    assert Path(thunder.__file__).resolve().is_relative_to(UPSTREAM.resolve()), thunder.__file__
    # working tree == commit + thunder.patch + thunder_local.patch, exactly
    probe = subprocess.run(
        f"set -e; t=$(mktemp -d); git -C {UPSTREAM} archive HEAD src | tar -x -C $t; cd $t; "
        f"git init -q .; git apply {PATCHES[0]}; git apply {PATCHES[1]}; "
        f"diff -r -x __pycache__ -x '*.egg-info' $t/src {UPSTREAM}/src && echo SAME; rm -rf $t",
        shell=True, capture_output=True, text=True)
    assert probe.stdout.strip().endswith("SAME"), f"upstream tree != commit+patches:\n{probe.stdout}{probe.stderr}"
    for ds in CLS_DATASETS + SEG_DATASETS:
        assert (Path(DATA_ROOT) / "data_splits" / f"{ds}.json").is_file(), ds
        assert (Path(DATA_ROOT) / ds).is_dir(), ds

    from thunder.tasks.knn_classification import knn_predict
    rng = np.random.default_rng(1337)
    train = rng.normal(size=(97, 31)).astype(np.float32)
    query = rng.normal(size=(23, 31)).astype(np.float32)
    labels = rng.integers(0, 5, size=len(train), dtype=np.int64)
    predictions = knn_predict(train, labels, query, [1, 5, 17])
    tn = train / np.linalg.norm(train, axis=1, keepdims=True)
    qn = query / np.linalg.norm(query, axis=1, keepdims=True)
    neighbors = np.argsort(-(qn @ tn.T), axis=1)
    for k in (1, 5, 17):
        exp = np.array([np.bincount(labels[row[:k]], minlength=5).argmax() for row in neighbors])
        assert np.array_equal(predictions[k], exp), k
    feats = torch.randn(4, 256, 384)
    scales = feats.abs().amax(dim=-1, keepdim=True).clamp_min_(1e-12).div_(127)
    q = torch.clamp(torch.round(feats / scales), -127, 127).to(torch.int8)
    rest = q.float() * scales
    assert torch.nn.functional.cosine_similarity(feats.flatten(), rest.flatten(), dim=0) > 0.9999

    from PIL import Image
    model = ThunderModel()
    transform = model.get_transform()
    image = Image.new("RGB", (300, 260), (32, 128, 224))
    x = transform(image)
    assert tuple(x.shape) == (3, 224, 224) and x.min() < 0 < x.max()
    images = x.unsqueeze(0).repeat(2, 1, 1, 1).cuda()
    with torch.inference_mode():
        cls = model.get_linear_probing_embeddings(images)
        seg = model.get_segmentation_embeddings(images) if model.emb_dim_seg else None
    assert cls.shape == (2, model.emb_dim) and torch.isfinite(cls).all()
    if seg is not None:
        assert seg.shape == (2, model.segmentation_tokens, model.emb_dim_seg) and torch.isfinite(seg).all()
    adv = model if ADV_PRECISION == PRECISION else None
    if adv is None:
        model.encoder.autocast_dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[ADV_PRECISION]
        adv = model
    images = images.clone().requires_grad_(True)
    adv.get_linear_probing_embeddings(images).sum().backward()
    assert images.grad is not None and torch.isfinite(images.grad).all() and images.grad.abs().sum() > 0
    rec = write_or_verify_manifest({
        "transform_repr": repr(transform), "emb_dim": model.emb_dim, "emb_dim_seg": model.emb_dim_seg,
        "segmentation_tokens": model.segmentation_tokens, "describe": model.encoder.describe(),
        "preflight_host": socket.gethostname(),
    })
    print(f"PASS preflight {MODEL_ID}: pooling={POOLING} emb_dim={model.emb_dim} "
          f"seg={model.segmentation_tokens}x{model.emb_dim_seg} precision={PRECISION}/{ADV_PRECISION}\n"
          f"transform={transform}\nmanifest={MANIFEST} (identity {rec['identity_sha256'][:12]})", flush=True)


if mode == "preflight":
    preflight()
    sys.exit(0)

verify_manifest()

if mode == "cleanup":
    assert len(sys.argv) == 2
    for dataset in CLS_DATASETS:
        for task in CACHED_TASKS:
            assert task_done(dataset, task), (dataset, task)
        cache = EMBEDDINGS / dataset / MODEL_ID
        if cache.exists() and not RETAIN_EMBEDDINGS:
            shutil.rmtree(cache)
            print(f"[{MODEL_ID}/{dataset}] cached probes complete; cache deleted", flush=True)
        elif cache.exists():
            print(f"[{MODEL_ID}/{dataset}] cached probes complete; cache retained", flush=True)
    sys.exit(0)

assert len(sys.argv) == 3 and sys.argv[2].isdigit(), sys.argv
index = int(sys.argv[2])

if mode == "precompute_group":
    model = None
    for i in PRECOMPUTE_GROUPS[index]:
        dataset = CLS_DATASETS[i]
        cache = EMBEDDINGS / dataset / MODEL_ID
        if all(task_done(dataset, t) for t in CACHED_TASKS):
            if cache.exists() and not RETAIN_EMBEDDINGS:
                shutil.rmtree(cache)
            print(f"[{MODEL_ID}/{dataset}] cached probe outputs already complete", flush=True)
            continue
        if not (cache / ".done").exists():
            if cache.exists():
                shutil.rmtree(cache)
            model = ThunderModel() if model is None else model
            started = time.time()
            benchmark(model, dataset=dataset, task="pre_computing_embeddings", **KWARGS)
            (cache / ".done").touch()
            (cache / ".dim").write_text(f"{model.emb_dim}\n")
            print(f"[{MODEL_ID}/{dataset}] embeddings ready in {(time.time() - started) / 60:.1f} min", flush=True)
        else:
            print(f"[{MODEL_ID}/{dataset}] embeddings already cached", flush=True)

elif mode == "cached_group":
    for i in PRECOMPUTE_GROUPS[index]:
        dataset = CLS_DATASETS[i]
        cache = EMBEDDINGS / dataset / MODEL_ID
        for task in CACHED_TASKS:
            if task_done(dataset, task):
                print(f"[{MODEL_ID}/{dataset}/{task}] already complete", flush=True)
                continue
            assert (cache / ".done").exists() and (cache / ".dim").exists(), cache
            model = CachedThunderModel(int((cache / ".dim").read_text()))
            loading = "embedding_pre_loading" if task == "linear_probing" else "online_loading"
            started = time.time()
            benchmark(model, dataset=dataset, task=task, loading_mode=loading, **KWARGS)
            assert task_done(dataset, task), (dataset, task)
            print(f"[{MODEL_ID}/{dataset}/{task}] done in {(time.time() - started) / 60:.1f} min", flush=True)

elif mode == "adversarial_group":
    model = None
    for i in ADVERSARIAL_GROUPS[index]:
        dataset = CLS_DATASETS[i]
        if task_done(dataset, "adversarial_attack"):
            print(f"[{MODEL_ID}/{dataset}/adversarial_attack] already complete", flush=True)
            continue
        assert outputs_json(dataset, "linear_probing").exists(), dataset
        model = ThunderModel(precision=ADV_PRECISION) if model is None else model
        started = time.time()
        benchmark(model, dataset=dataset, task="adversarial_attack", loading_mode="online_loading", **KWARGS)
        assert task_done(dataset, "adversarial_attack"), dataset
        print(f"[{MODEL_ID}/{dataset}/adversarial_attack] done in {(time.time() - started) / 60:.1f} min",
              flush=True)

else:
    assert mode == "segmentation"
    torch.set_float32_matmul_precision("high")
    dataset = SEG_DATASETS[index]
    cache = SEGMENTATION_EMBEDDINGS / dataset / MODEL_ID
    legacy_cache = EMBEDDINGS / dataset / MODEL_ID
    if task_done(dataset, "segmentation"):
        shutil.rmtree(SEGMENTATION_EMBEDDINGS, ignore_errors=True)
        shutil.rmtree(legacy_cache, ignore_errors=True)
        print(f"[{MODEL_ID}/{dataset}/segmentation] already complete", flush=True)
        sys.exit(0)
    shutil.rmtree(legacy_cache, ignore_errors=True)
    atexit.register(lambda: shutil.rmtree(SEGMENTATION_EMBEDDINGS, ignore_errors=True))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(128 + signal.SIGTERM))
    signal.signal(signal.SIGINT, lambda *_: sys.exit(128 + signal.SIGINT))
    started = time.time()
    model = ThunderModel()
    if model.emb_dim_seg is None:
        print(f"[{MODEL_ID}/{dataset}/segmentation] SKIPPED: model has no single patch-token grid", flush=True)
        sys.exit(0)
    segmentation_batch = min(EXTRACTION_BATCH,
                             MAX_SEGMENTATION_BATCH_VALUES // (model.segmentation_tokens * model.emb_dim_seg))
    assert segmentation_batch > 0
    if not (cache / ".done").exists():
        shutil.rmtree(cache, ignore_errors=True)
        print(f"[{MODEL_ID}/{dataset}/segmentation] extraction batch={segmentation_batch}", flush=True)
        benchmark(model, dataset=dataset, task="pre_computing_embeddings",
                  **{**KWARGS, "task.pre_comp_emb_batch_size": segmentation_batch,
                     "task.base_embeddings_folder": str(SEGMENTATION_EMBEDDINGS)})
        (cache / ".done").touch()
    for split in ("train", "val", "test"):
        for filename in ("embeddings.h5", "labels.h5"):
            p = cache / split / filename
            assert p.exists() and p.stat().st_size > 0, p
    cached_model = CachedThunderModel(model.emb_dim, model.emb_dim_seg)
    del model
    torch.cuda.empty_cache()
    epochs = {"adaptation.epochs": SEG_EPOCHS[dataset]} if dataset in SEG_EPOCHS else {}
    benchmark(cached_model, dataset=dataset, task="segmentation", loading_mode="embedding_pre_loading",
              **{**KWARGS, **epochs, "adaptation.num_workers": 0,
                 "task.base_embeddings_folder": str(SEGMENTATION_EMBEDDINGS)})
    assert task_done(dataset, "segmentation"), dataset
    shutil.rmtree(SEGMENTATION_EMBEDDINGS, ignore_errors=True)
    print(f"[{MODEL_ID}/{dataset}/segmentation] done in {(time.time() - started) / 60:.1f} min; cache deleted",
          flush=True)
