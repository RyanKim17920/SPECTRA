#!/usr/bin/env python3
"""Write a PROVENANCE SIDECAR for a THUNDER run name.

WHY THIS EXISTS
---------------
THUNDER writes nothing about the encoder into its results tree. The config.json it emits
at ``outputs/res/<ds>/<run>/<task>/frozen/config.json`` carries exactly

    adaptation, ckpt_saving, dataset, data_loading, task, wandb,
    embedding_recomputing, model_retraining

and not one of those names the model -- THUNDER only ever sees ``custom:<path>.py`` and
``src/waivphaet/eval/thunder_model.py`` resolves WAIV_BACKBONE / WAIV_ADAPTER internally
from the environment at import time. So a finished THUNDER cell cannot be attributed to
the checkpoint that produced it from its own artifacts: attribution otherwise rests on
timing inference against the slurm log, which is not evidence.

This script closes that gap by writing a small JSON next to the results recording the
run name, backbone, the ABSOLUTE adapter path, the source training job, the step, the
pooling protocol, the datasets/protocols covered, the submitting job ids, and a sha256 of
``adapter/adapter_model.safetensors`` -- so the binding between a results directory and a
checkpoint is cryptographic rather than chronological.

It is deliberately a SIDECAR and not a rename. THUNDER's embedding cache key is
``embeddings/<dataset>/<run_name>`` and NOTHING else (see run_thunder.sbatch's EMB_DIR),
so renaming an in-flight run would orphan every warm cache and silently re-run the whole
roster. Adding a file changes no key.

Usage::

    python scripts/write_thunder_provenance.py \
        --run-name ph2mask_midnight_s250_v2 \
        --backbone kaiko-ai/midnight \
        --adapter /admin/home/ryan.kim/waiv/runs/ph2-midnight-s0-t900-391061/step_0000250 \
        --pooling auto --jobs 391839-391852

Called with no arguments it regenerates the sidecars for the 2026-08-24 mask roster from
the ROSTER table below (derived from ``sacct --format=SubmitLine``, not from assumption).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

THUNDER_ROOT = Path(os.environ.get("THUNDER_BASE_DATA_FOLDER", "/data/ryan.kim/thunder"))
BACKUP_ROOT = Path("/admin/home/ryan.kim/waiv_result_backups/thunder_provenance")
SIDECAR_NAME = "waiv_provenance.json"

#: The 2026-08-24 negative-masking THUNDER roster, transcribed from
#: ``sacct --format=JobID,JobName,State,SubmitLine -j 391839-391866``. The adapter path is
#: the 6th positional of run_thunder.sbatch on each submit line; the 5th (ckpt) is empty.
ROSTER = {
    "ph2mask_midnight_s250_v2": {
        "backbone": "kaiko-ai/midnight",
        "adapter": "/admin/home/ryan.kim/waiv/runs/ph2-midnight-s0-t900-391061/step_0000250",
        "pooling_arg": "auto",
        "jobs": list(range(391839, 391853)),
    },
    "ph2mask_virchow2_s250_v2": {
        "backbone": "paige-ai/Virchow2",
        "adapter": "/admin/home/ryan.kim/waiv/runs/ph2-virchow2-s0-t900-391062/step_0000250",
        "pooling_arg": "auto",
        "jobs": list(range(391853, 391867)),
    },
}


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sacct_rows(jobs: list[int]) -> list[dict]:
    """(job_id, name, state, submit_line) for each roster job, or [] if sacct is absent."""
    try:
        out = subprocess.run(
            ["sacct", "-X", "-P", "-n", "--format=JobID,JobName%64,State,SubmitLine%512",
             "-j", ",".join(str(j) for j in jobs)],
            capture_output=True, text=True, timeout=120,
        ).stdout
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 4 or "." in parts[0]:
            continue
        submit = parts[3]
        dataset = None
        toks = submit.split()
        # Match on BASENAME, not the literal relative path. Rosters submitted with an
        # absolute `$REPO/scripts/run_thunder.sbatch` (the seed-1 Virchow2 seed-floor
        # roster, jobs 392194-392205) are the same script, but an equality test against
        # "scripts/run_thunder.sbatch" misses them and silently records dataset=None for
        # every row -- which is exactly the attribution gap this sidecar exists to close.
        for i, tok in enumerate(toks):
            if os.path.basename(tok) == "run_thunder.sbatch":
                if i + 1 < len(toks):
                    dataset = toks[i + 1]
                break
        rows.append({"job_id": parts[0], "job_name": parts[1], "state": parts[2],
                     "dataset": dataset, "submit_line": submit})
    return rows


def observed_results(run_name: str) -> dict[str, list[str]]:
    """dataset -> [task,...] actually present under outputs/res for this run name."""
    res = THUNDER_ROOT / "outputs" / "res"
    found: dict[str, list[str]] = {}
    if not res.is_dir():
        return found
    for ds_dir in sorted(res.iterdir()):
        run_dir = ds_dir / run_name
        if not run_dir.is_dir():
            continue
        tasks = sorted(
            t.name for t in run_dir.iterdir()
            if (t / "frozen" / "outputs.json").is_file()
        )
        if tasks:
            found[ds_dir.name] = tasks
    return found


def build(run_name: str, backbone: str, adapter: str, pooling_arg: str,
          jobs: list[int]) -> dict:
    ad = Path(adapter)
    step = None
    if ad.name.startswith("step_"):
        try:
            step = int(ad.name.split("_", 1)[1])
        except ValueError:
            pass
    run_dir = ad.parent
    src_job = run_dir.name.rsplit("-", 1)[-1] if "-" in run_dir.name else None

    rows = sacct_rows(jobs)
    adapter_cfg = ad / "adapter" / "adapter_config.json"
    cfg = json.loads(adapter_cfg.read_text()) if adapter_cfg.is_file() else {}
    train_cfg_path = run_dir / "config.json"
    train_cfg = json.loads(train_cfg_path.read_text()) if train_cfg_path.is_file() else {}

    return {
        "schema": "waiv.thunder.provenance/1",
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_name": run_name,
        "backbone": backbone,
        "step": step,
        "pooling_arg": pooling_arg,
        "pooling_effective": {
            "classification": "clsmean" if backbone in (
                "kaiko-ai/midnight", "paige-ai/Virchow2") else "cls",
            "segmentation": "cls",
            "note": "resolve_pooling() in src/waivphaet/eval/thunder_model.py: "
                    "clsmean backbones fall back to cls for segmentation.",
        },
        "adapter": {
            "checkpoint_dir": str(ad),
            "adapter_dir": str(ad / "adapter"),
            "safetensors": str(ad / "adapter" / "adapter_model.safetensors"),
            "sha256": sha256_file(ad / "adapter" / "adapter_model.safetensors"),
            "adapter_config_sha256": sha256_file(adapter_cfg),
            "pool_head_pt_sha256": sha256_file(ad / "pool_head.pt"),
            "lora_r": cfg.get("r"),
            "lora_alpha": cfg.get("lora_alpha"),
            "base_model_name_or_path": cfg.get("base_model_name_or_path"),
        },
        "source_training": {
            "run_dir": str(run_dir),
            "slurm_job_id": src_job,
            "seed": train_cfg.get("seed"),
            "max_steps": train_cfg.get("max_steps"),
            "config_json": str(train_cfg_path) if train_cfg_path.is_file() else None,
            "config_sha256": sha256_file(train_cfg_path),
        },
        "eval_env": {
            "WAIV_BACKBONE": backbone,
            "WAIV_ADAPTER": str(ad),
            "WAIV_RUN_NAME": run_name,
            "WAIV_LORA_RANK": str(cfg.get("r", 32)),
            "WAIV_LORA_ALPHA": str(cfg.get("lora_alpha", 64)),
            "WAIV_PROJ_OUT_DIM": "512",
        },
        "submitting_jobs": rows,
        "results_present": observed_results(run_name),
        "embedding_cache_key": f"embeddings/<dataset>/{run_name}",
        "collector_note": (
            f"scripts/collect_thunder.py --model {run_name} reads these results, and its "
            f"infer_backbone() now resolves the backbone from THIS sidecar (it matches no "
            f"entry in BACKBONE_RUN_PREFIXES, so before the sidecar it returned None and "
            f"--backbone had to be passed by hand). scripts/collect_final5.py does NOT see "
            f"this run: _thunder_score() builds the model dir as f5_<run>_s<step:07d> and "
            f"_parse_run_name() only accepts final5-<arm>-s<seed>-t<T>-<jobid>, neither of "
            f"which this roster uses. That is deliberate -- renaming would change the "
            f"embedding cache key embeddings/<dataset>/{run_name} and orphan every warm "
            f"cache mid-roster. Use collect_thunder.py for this roster; the checkpoint "
            f"binding is the adapter sha256 above."
        ),
    }


def write(run_name: str, blob: dict) -> list[Path]:
    written: list[Path] = []
    canonical = THUNDER_ROOT / "outputs" / "provenance" / f"{run_name}.json"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_text(json.dumps(blob, indent=2) + "\n")
    written.append(canonical)

    # A copy inside each existing per-dataset results dir, so a results directory is
    # self-describing. Purely additive: THUNDER never enumerates this directory.
    res = THUNDER_ROOT / "outputs" / "res"
    if res.is_dir():
        for ds_dir in sorted(res.iterdir()):
            run_dir = ds_dir / run_name
            if run_dir.is_dir():
                p = run_dir / SIDECAR_NAME
                p.write_text(json.dumps(blob, indent=2) + "\n")
                written.append(p)

    BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    backup = BACKUP_ROOT / f"{run_name}.json"
    shutil.copy2(canonical, backup)
    written.append(backup)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name")
    ap.add_argument("--backbone")
    ap.add_argument("--adapter")
    ap.add_argument("--pooling", default="auto")
    ap.add_argument("--jobs", default="", help="comma list and/or a-b ranges of job ids")
    args = ap.parse_args()

    if args.run_name:
        jobs: list[int] = []
        for tok in filter(None, args.jobs.split(",")):
            if "-" in tok:
                a, b = tok.split("-", 1)
                jobs.extend(range(int(a), int(b) + 1))
            else:
                jobs.append(int(tok))
        targets = {args.run_name: {"backbone": args.backbone, "adapter": args.adapter,
                                   "pooling_arg": args.pooling, "jobs": jobs}}
    else:
        targets = ROSTER

    for run_name, spec in targets.items():
        blob = build(run_name, spec["backbone"], spec["adapter"],
                     spec["pooling_arg"], spec["jobs"])
        for p in write(run_name, blob):
            print(f"wrote {p}")
        print(f"  {run_name}: sha256={blob['adapter']['sha256']} "
              f"step={blob['step']} src_job={blob['source_training']['slurm_job_id']}")


if __name__ == "__main__":
    main()
