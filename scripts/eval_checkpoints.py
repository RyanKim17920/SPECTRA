#!/usr/bin/env python
"""Evaluate every training checkpoint and emit the RI-vs-step curve (PLAN.md 3 phase 8).

Phase 8 says "checkpoint often; evaluate retention at every checkpoint, not just at the
end". This is that. It follows a run directory and, for each ``step_*`` that appears,
runs the two things that are allowed to be believed:

1. ``embed_probe.py`` -- embedding-space **separation** (matched minus random) and
   rank-based top-1, on a pinned condition set. Matched cosine alone is not a signal:
   it rises under representation collapse (see the script's own docstring), so the
   curve records ``separation`` and ``within_condition_random`` next to it.
2. **PathoROB RI on all three datasets**, through the exact extractor that produced the
   Phase-2 gate row -- same Resize(224)->CenterCrop(224)->ToTensor->ImageNet Normalize,
   same clsmean 2048-d, same fp32, same npz layout, same ``robustness_index`` module in
   ``.venv-pathorob``. The LoRA adapter is the only difference, and
   ``extract_pathorob_features.py`` proves it applied (``rel_l2_delta`` against
   ``disable_adapter()``) or exits non-zero. We parse that number out and store it in
   the curve, because a silently-unloaded adapter reproduces the baseline exactly and
   would otherwise read as "the fine-tune had no effect".

Why a follower process and not the ``on_checkpoint`` hook: extraction plus the CPU kNN
is ~15-20 min per checkpoint. Inline, that roughly doubles wall time with the training
GPU idle. Here it runs on a second GPU of the same allocation while training keeps the
first one saturated.

    python scripts/eval_checkpoints.py --run-dir runs/waiv-lora-NNNN --lora-rank 32 \
        --lora-alpha 64 --conditions-file runs/waiv-lora-NNNN/conditions_used.json \
        --follow --stop-file runs/waiv-lora-NNNN/TRAIN_DONE
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from waivphaet.eval.pathorob_adapter import (  # noqa: E402
    TARGETS,
    PathoRobPaths,
    read_results,
    run_robustness_index,
)

DATASETS = ("camelyon", "tolkach_esca", "tcga")
#: Keys of their ``results_summary.json`` worth carrying into the curve. ``robustness_index``
#: is the headline; ``balanced_accuracy`` / ``prediction_performance`` are the forgetting
#: detector (PLAN.md 3 risk 1: a robustness win that costs biology is a failed
#: reproduction); ``confounder_insensitivity`` is where the smoke run's entire gain landed.
RESULT_KEYS = (
    "robustness_index", "balanced_accuracy", "k_opt", "generalization_index",
    "confounder_insensitivity", "prediction_performance", "ID_performance",
    "OOD_performance",
)
_REL_L2 = re.compile(r"rel_l2_delta=([0-9.eE+-]+)")


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], check=True, cwd=str(REPO), **kw)


def discover(run_dir: Path) -> list[tuple[int, Path]]:
    """Complete checkpoints, ascending. ``metrics.json`` is written last by
    ``save_checkpoint``, so it is the completeness sentinel -- without it we could read
    a half-flushed ``adapter_model.safetensors``.

    Handles both checkpoint formats:
    - LoRA: ``step_*/adapter/`` directory + metrics.json
    - Full FT: ``step_*/backbone.safetensors`` + metrics.json
    """
    out = []
    for d in sorted(run_dir.glob("step_*")):
        if not (d / "metrics.json").exists():
            continue
        # LoRA checkpoint has an adapter/ dir; full FT has backbone.safetensors.
        if (d / "adapter").is_dir() or (d / "backbone.safetensors").exists():
            out.append((int(d.name.split("_")[1]), d))
    return out


def is_full_ft_checkpoint(ckpt: Path) -> bool:
    """Check whether a checkpoint dir is full-FT (backbone.safetensors) vs LoRA (adapter/)."""
    return (ckpt / "backbone.safetensors").exists() and not (ckpt / "adapter").is_dir()


def run_probe(args, ckpt: Path, step: int) -> dict:
    out = args.run_dir / f"probe_step_{step:07d}.json"
    if not out.exists():
        if is_full_ft_checkpoint(ckpt):
            cmd = [
                args.python, "scripts/embed_probe.py",
                "--packed-dir", args.packed_dir,
                "--checkpoint", ckpt,
                "--out", out,
                "--proj-out-dim", args.proj_out_dim,
                "--n-tiles", args.probe_tiles,
            ]
        else:
            cmd = [
                args.python, "scripts/embed_probe.py",
                "--packed-dir", args.packed_dir, "--adapter", ckpt, "--out", out,
                "--lora-rank", args.lora_rank, "--lora-alpha", args.lora_alpha,
                "--proj-out-dim", args.proj_out_dim, "--n-tiles", args.probe_tiles,
            ]
        # Only appended when explicitly asked for. embed_probe.py's own default is
        # DEFAULT_BACKBONE, and for a checkpoint/adapter it cross-checks the value against
        # the one recorded at train time and hard-fails on a mismatch -- so passing nothing
        # keeps the pre-existing behaviour exactly.
        if args.backbone:
            cmd += ["--backbone", args.backbone]
        if args.conditions_file:
            cmd += ["--conditions-file", args.conditions_file]
        sh(cmd)
    return json.loads(out.read_text())


def probe_digest(p: dict) -> dict:
    """Pull out the numbers we gate on. Separation and top-1, never matched alone."""
    d: dict = {}
    for grp, g in p.get("groups", {}).items():
        for axis in ("cross_scanner", "cross_stain"):
            e = g.get(f"{axis}.embedding")
            if e:
                d[f"{grp}.{axis}.separation"] = e["separation"]
                d[f"{grp}.{axis}.matched"] = e["matched"]
                d[f"{grp}.{axis}.random"] = e["random"]
                d[f"{grp}.{axis}.top1"] = e["top1"]
        if "within_condition_random.embedding" in g:
            d[f"{grp}.within_condition_random"] = g["within_condition_random.embedding"]
    return d


def run_pathorob(args, ckpt: Path, step: int, paths: PathoRobPaths) -> dict:
    model_name = f"{args.model_prefix}_s{step:07d}"
    feat_dir = paths.features_dir / model_name
    adapter_checks: dict[str, float] = {}

    for ds in args.datasets:
        ds_dir = feat_dir / ds
        if ds_dir.exists():
            npz_count = len(list(ds_dir.glob("*.npz")))
            if npz_count > 0:
                print(f"[eval] features for {model_name}/{ds} already present "
                      f"({npz_count} npz), skipping extract")
                continue
            # The directory exists but contains no *.npz files -- this is the same
            # presence-vs-completeness bug that hit THUNDER embedding caches
            # (fixed in run_thunder.sbatch).  Concretely: backfill job 376305 died in 46s
            # because waiv_waiv_v2rank128_376088_s0001250/ already had camelyon/ (5 npz)
            # and tolkach_esca/ (4 npz) complete but tcga/ EMPTY (0 files) -- left behind
            # by the extractor process that was killed mid-run.  The old existence check
            # saw tcga/ exists, skipped extraction, and run_robustness_index/read_results
            # then failed: "Cannot find features for chunk 'Asterand' at .../tcga/Asterand.npz".
            # Remove the stale empty dir so the extractor starts from a clean slate.
            print(f"[eval] {model_name}/{ds} dir exists but has no *.npz files "
                  f"(stale partial extract); removing and re-extracting")
            shutil.rmtree(ds_dir)

        if is_full_ft_checkpoint(ckpt):
            cmd = [args.python, "scripts/extract_pathorob_features.py",
                   "--dataset", ds, "--model-name", model_name,
                   "--checkpoint", str(ckpt),
                   "--proj-out-dim", str(args.proj_out_dim),
                   "--batch-size", str(args.batch_size), "--num-workers", str(args.num_workers)]
        else:
            cmd = [args.python, "scripts/extract_pathorob_features.py",
                   "--dataset", ds, "--model-name", model_name, "--adapter", str(ckpt),
                   "--lora-rank", str(args.lora_rank), "--lora-alpha", str(args.lora_alpha),
                   "--proj-out-dim", str(args.proj_out_dim),
                   "--batch-size", str(args.batch_size), "--num-workers", str(args.num_workers)]
        # Same rule as run_probe: only appended when explicitly asked for.
        # extract_pathorob_features.py already falls back to $WAIV_BACKBONE and then to the
        # checkpoint's own recorded backbone, and refuses an override that contradicts the
        # checkpoint -- so passing nothing keeps the pre-existing behaviour exactly.
        if args.backbone:
            cmd += ["--backbone", str(args.backbone)]
        # capture_output + check=True raises BEFORE anything is written, so a failing
        # extractor used to leave zero diagnostics in the follower log (job 369922).
        # Echo both streams on failure, then re-raise.
        proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
        if proc.returncode != 0:
            sys.stdout.write(proc.stdout)
            sys.stderr.write(proc.stderr)
            sys.stdout.flush()
            raise subprocess.CalledProcessError(proc.returncode, cmd,
                                                output=proc.stdout, stderr=proc.stderr)
        sys.stdout.write(proc.stdout)
        # extract_pathorob_features emits TWO adapter/checkpoint checks per run, by design:
        #   1. inside build_model, on a deterministic *synthetic* batch (seed 1234) -- a cheap
        #      CPU-side proof before the caller ever sees the model. Same input every time, so
        #      its value tracks only the checkpoint (~0.25), never the dataset.
        #   2. in main(), on *real tiles* from this dataset -- the sharper, dataset-specific
        #      number (camelyon ~0.75, tolkach ~0.93, tcga ~0.79).
        # Take the last match: the real-tile check is always emitted after the synthetic one.
        matches = _REL_L2.findall(proc.stdout)
        # extract_pathorob_features exits non-zero below 1e-4, so reaching here already
        # means the adapter/checkpoint changed the embeddings; we record how much.
        adapter_checks[ds] = float(matches[-1]) if matches else float("nan")

    run_robustness_index(model_name, list(args.datasets), paths=paths)

    rec: dict = {"model": model_name, "adapter_rel_l2_delta": adapter_checks, "datasets": {}}
    ris = []
    for ds in args.datasets:
        res = read_results(model_name, ds, paths=paths)
        rec["datasets"][ds] = {k: res[k] for k in RESULT_KEYS if k in res}
        ris.append(float(res["robustness_index"]))
    rec["avg_robustness_index"] = sum(ris) / len(ris)
    rec["avg_balanced_accuracy"] = sum(
        float(rec["datasets"][d]["balanced_accuracy"]) for d in args.datasets
    ) / len(args.datasets)

    if args.purge_features and feat_dir.exists():
        # 804 MB per checkpoint per full sweep; the RI json is what we keep.
        shutil.rmtree(feat_dir)
        print(f"[eval] purged {feat_dir}")
    return rec


def write_curve(args, curve: list[dict]) -> None:
    out = args.run_dir / "ri_curve.json"
    # Atomic: a plain write_text leaves a truncated file visible to anyone reading the
    # curve mid-run, and a job killed at the wall clock mid-write would leave unparseable
    # JSON that the backfill job needs in order to know which steps are already done.
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(
        {"run_dir": str(args.run_dir), "datasets": list(args.datasets),
         "targets": TARGETS, "points": curve}, indent=2))
    os.replace(tmp, out)
    print(f"\n[eval] --- RI vs step ({out}) ---")
    hdr = f"{'step':>7}" + "".join(f"{d:>15}" for d in args.datasets) + \
          f"{'AVG RI':>9}{'AVG bal-acc':>13}{'sep(scan)':>11}{'top1(scan)':>12}"
    print(hdr)
    for p in curve:
        cells = "".join(f"{p['datasets'][d]['robustness_index']:>15.6f}" for d in args.datasets)
        pr = p.get("probe", {})
        sep = pr.get("heldout.cross_scanner.separation")
        t1 = pr.get("heldout.cross_scanner.top1")
        sep_s = "-" if sep is None else f"{sep:.4f}"
        t1_s = "-" if t1 is None else f"{t1:.4f}"
        print(f"{p['step']:>7}{cells}{p['avg_robustness_index']:>9.4f}"
              f"{p['avg_balanced_accuracy']:>13.4f}{sep_s:>11}{t1_s:>12}")
    print(f"{'target':>7}" + "".join(
        f"{TARGETS['phaet_target'][d]:>15.3f}" for d in args.datasets) +
        f"{TARGETS['phaet_target']['avg']:>9.3f}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--packed-dir",
                    default=os.environ.get("WAIV_PACKED_DIR", "/data/plism/repacked"))
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--model-prefix", default=None,
                    help="features/results dir prefix; defaults to waiv_<run-dir name>")
    # 32/64 are the phikon-v2 run geometry (job 369043) and are only meaningful for a LoRA
    # checkpoint; they are ignored for a full-FT one. Pass the values the run was TRAINED
    # with -- build_model hard-fails on a rank/alpha mismatch rather than silently
    # evaluating a differently-shaped adapter.
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    # Default None (not DEFAULT_BACKBONE) so nothing is forwarded unless asked: the probe
    # and the extractor both derive the backbone from the checkpoint when told nothing, and
    # both refuse an override that contradicts it. $WAIV_BACKBONE is the same knob the
    # training sbatch files and thunder_model.py use, so exporting it once is enough.
    ap.add_argument("--backbone", default=os.environ.get("WAIV_BACKBONE") or None,
                    help="HF id of the base backbone, e.g. kaiko-ai/midnight. Defaults to "
                         "$WAIV_BACKBONE, else unset (each sub-tool falls back to the "
                         "checkpoint's own recorded backbone / owkin/phikon-v2).")
    ap.add_argument("--proj-out-dim", type=int, default=512)
    ap.add_argument("--conditions-file", type=Path, default=None)
    ap.add_argument("--probe-tiles", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--python", default=str(REPO / ".venv" / "bin" / "python"))
    ap.add_argument("--follow", action="store_true",
                    help="keep polling for new checkpoints until --stop-file appears")
    ap.add_argument("--stop-file", type=Path, default=None)
    ap.add_argument("--poll-s", type=int, default=60)
    ap.add_argument("--max-wait-s", type=int, default=8 * 3600)
    ap.add_argument("--purge-features", action="store_true", default=False,
                    help="Delete extracted features after scoring. Makes independent "
                         "re-readout impossible without re-extracting on a GPU. "
                         "Exists because /data runs near-full (94%% as of 2026-08-14).")
    ap.add_argument("--keep-features", dest="purge_features", action="store_false",
                    help="Keep extracted features after scoring (now the default). "
                         "Retained for backward compatibility only.")
    args = ap.parse_args()

    # Every path below is handed to a subprocess that sh()/run_pathorob spawn with
    # cwd=REPO, so a RELATIVE one is resolved by the CHILD against REPO -- not against the
    # cwd the caller typed it in. When REPO is a pinned read-only snapshot (which has no
    # runs/ subtree) that silently retargets the argument and the child dies far from the
    # cause: jobs 381012/381013 got
    #   FileNotFoundError: 'runs/gridcmp2-ctrlseed-380889/conditions_used.json'
    # from embed_probe.py, after training had already finished, leaving a full set of
    # checkpoints and an empty ri_curve.json. Absolutise against the INVOCATION cwd, which
    # is what the caller meant; an already-absolute path is unchanged, so this is a no-op
    # for every existing caller and the eval itself is untouched.
    args.run_dir = args.run_dir.resolve()
    args.packed_dir = str(Path(args.packed_dir).resolve())
    if args.conditions_file is not None:
        args.conditions_file = args.conditions_file.resolve()
    if args.stop_file is not None:
        args.stop_file = args.stop_file.resolve()

    os.environ.setdefault("HF_HOME", "/data/huggingface")
    if args.model_prefix is None:
        args.model_prefix = "waiv_" + args.run_dir.name.replace("-", "_")
    paths = PathoRobPaths(root=REPO / "third_party" / "PathoROB")
    paths.check()

    curve_path = args.run_dir / "ri_curve.json"
    curve: list[dict] = []
    if curve_path.exists():
        curve = json.loads(curve_path.read_text()).get("points", [])
    done = {p["step"] for p in curve}

    t0 = time.time()
    while True:
        for step, ckpt in discover(args.run_dir):
            if step in done:
                continue
            print(f"\n[eval] ===== step {step} :: {ckpt} =====", flush=True)
            t = time.time()
            probe = probe_digest(run_probe(args, ckpt, step))
            rec = run_pathorob(args, ckpt, step, paths)
            rec.update(step=step, checkpoint=str(ckpt), probe=probe,
                       eval_seconds=round(time.time() - t, 1))
            try:
                rec["train_metrics"] = json.loads((ckpt / "metrics.json").read_text())
            except Exception:  # pragma: no cover - metrics are advisory here
                pass
            curve.append(rec)
            curve.sort(key=lambda p: p["step"])
            done.add(step)
            write_curve(args, curve)

        if not args.follow:
            break
        stopped = args.stop_file is not None and args.stop_file.exists()
        if stopped and not [s for s, _ in discover(args.run_dir) if s not in done]:
            print("[eval] training finished and every checkpoint is evaluated.")
            break
        if time.time() - t0 > args.max_wait_s:
            print("[eval] max wait exceeded; exiting with what we have.")
            break
        time.sleep(args.poll_s)

    if not curve:
        print("[eval] no checkpoints evaluated")
        return 1
    write_curve(args, curve)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
