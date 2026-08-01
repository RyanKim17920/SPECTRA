#!/usr/bin/env python
"""Run HEST-Benchmark on our encoder -- the first RETENTION detector (PLAN.md §3 risk 1).

    # base phikon-v2, CLS pooling -> the reproduction gate against 0.3747
    python scripts/run_hest.py --pooling cls  --exp-code base_cls
    # base phikon-v2, clsmean -> the number we track across checkpoints
    python scripts/run_hest.py --pooling clsmean --exp-code base_clsmean
    # a LoRA checkpoint
    python scripts/run_hest.py --adapter runs/xxx/step5000 --exp-code step5000_clsmean

All the maths -- tile embedding, PCA(256), Ridge, per-gene ``pearsonr``, fold and task
averaging -- is ``hest.bench.benchmark``, unmodified, imported off the pinned clone in
``third_party/HEST``. This file only builds the encoder, hands over the transform, and
prints the comparison against the published row.

See ``waivphaet.eval.hest_adapter`` for why ``--pooling cls`` is what reproduces 0.3747
while ``--pooling clsmean`` is what stays comparable to our PathoROB row.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from waivphaet.eval import hest_adapter as H  # noqa: E402

H.apply_env_defaults()

import torch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pooling", default="cls", choices=("cls", "mean", "clsmean"),
                    help="cls reproduces HEST's published phikon_v2 row (CLS-only, 1024-d); "
                         "clsmean matches our PathoROB representation but has no published "
                         "counterpart -- never compare a clsmean number to 0.3747")
    ap.add_argument("--checkpoint", default=None, help="omit for the base backbone")
    ap.add_argument("--backbone", default=None,
                    help="HF id of the base backbone (default owkin/phikon-v2, or "
                         "WAIV_BACKBONE). Note the published 0.3747 row is phikon-v2 "
                         "CLS only -- another backbone has no such counterpart")
    ap.add_argument("--adapter", type=Path, default=None,
                    help="LoRA checkpoint dir (adapter/ + projector.pt)")
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--proj-out-dim", type=int, default=512)
    ap.add_argument("--exp-code", default=None)
    ap.add_argument("--tasks", nargs="+", default=list(H.LEADERBOARD_TASKS),
                    help="default = HEST's 9 leaderboard tasks. HCC is deliberately "
                         "excluded: it ships on HF but is not in their published average")
    ap.add_argument("--bench-data", type=Path, default=H.DEFAULT_BENCH_DATA)
    ap.add_argument("--work-dir", type=Path, default=H.DEFAULT_WORK_DIR)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--precision", default="float32", choices=("float32", "float16", "bfloat16"),
                    help="TRIDENT's phikon_v2 baseline runs fp32; keep it for the gate run")
    ap.add_argument("--also-encoders", nargs="*", default=[],
                    help="extra TRIDENT-registry encoders to run alongside ours, e.g. "
                         "'phikon_v2' (their own baseline, an end-to-end harness control) "
                         "or 'resnet50' (their published 0.3252 floor). Empty by default: "
                         "BenchmarkConfig would otherwise silently append resnet50 to every "
                         "run and double the GPU time")
    args = ap.parse_args()

    if args.checkpoint and args.adapter:
        raise SystemExit("--checkpoint and --adapter are mutually exclusive")

    paths = H.HestPaths(bench_data=args.bench_data.resolve(), work_dir=args.work_dir.resolve())
    paths.check()
    paths.embed_dir.mkdir(parents=True, exist_ok=True)
    paths.results_dir.mkdir(parents=True, exist_ok=True)

    encoder = H.load_encoder(args.checkpoint, args.adapter, args.pooling,
                             lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
                             proj_out_dim=args.proj_out_dim, backbone=args.backbone)
    wrapped = H.HestEncoderWrapper(encoder)
    # Derived from the backbone's own hidden size, not a literal: 1024/2048 on phikon-v2,
    # 1536/3072 on midnight. Still asserted -- a pooling/embed_dim desync would write
    # half-width embeddings that HEST would happily regress on.
    expected = encoder.hidden_size * (2 if args.pooling == "clsmean" else 1)
    if wrapped.embed_dim != expected:
        raise RuntimeError(f"pooling={args.pooling} should give {expected}-d, "
                           f"got {wrapped.embed_dim}")
    print(f"[hest] backbone={encoder.cfg.backbone} pooling={args.pooling} "
          f"embed_dim={wrapped.embed_dim} "
          f"precision={args.precision} tasks={len(args.tasks)}", flush=True)

    exp_code = args.exp_code or f"{args.pooling}_{'base' if not (args.checkpoint or args.adapter) else 'ckpt'}"
    benchmark = H.import_hest_benchmark()

    t0 = time.time()
    # Embeddings are cached per (task, encoder) under embed_dataroot and only extracted on
    # fold 0, so a re-run of the regression half is cheap -- but a *different* checkpoint
    # under the same exp_code would silently reuse them. HEST forces overwrite=True for
    # 'custom_encoder', which is why every checkpoint must get its own --work-dir or
    # --exp-code is not enough on its own.
    _, per_enc = benchmark(
        wrapped,
        H.build_transform(encoder.cfg.backbone),
        getattr(torch, args.precision),
        bench_data_root=str(paths.bench_data),
        embed_dataroot=str(paths.embed_dir / exp_code),
        results_dir=str(paths.results_dir),
        exp_code=exp_code,
        datasets=list(args.tasks),
        encoders=list(args.also_encoders),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dt = time.time() - t0

    exp_dirs = sorted(paths.results_dir.glob(f"{exp_code}::*"))
    results = H.read_results(exp_dirs[-1]) if exp_dirs else {}
    payload = {
        "exp_code": exp_code, "backbone": encoder.cfg.backbone,
        "pooling": args.pooling, "embed_dim": wrapped.embed_dim,
        "precision": args.precision, "seconds": round(dt, 1),
        "results": results, "hest_perf_per_encoder": per_enc,
        "results_dir": str(exp_dirs[-1]) if exp_dirs else None,
    }
    # The published 0.3747 row is *phikon-v2, CLS*. Both halves matter: on any other
    # backbone the comparison is meaningless, so do not emit it.
    if args.pooling == "cls" and encoder.cfg.backbone == "owkin/phikon-v2":
        payload["vs_published_phikonv2"] = H.compare_to_published(results)
        payload["note"] = (
            "published row is HEST's own phikon_v2 (CLS, fp32); Waiv Table 1 quotes it "
            "verbatim. Benchmark dynamic range is only "
            f"{H.HEST_RANGE[0]}-{H.HEST_RANGE[1]} Pearson."
        )
    else:
        payload["note"] = (
            f"backbone={encoder.cfg.backbone} pooling={args.pooling} has NO published "
            "counterpart here -- this is our own reference for checkpoint-to-checkpoint "
            "retention only. 0.3747 is phikon-v2 CLS and nothing else."
        )
    print(json.dumps(payload, indent=2))
    out = paths.results_dir / f"{exp_code}_summary.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"[hest] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
