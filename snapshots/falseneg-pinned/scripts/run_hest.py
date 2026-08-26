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
    ap.add_argument("--pool-head", default=None,
                    choices=("gem", "gem_clamp", "attn", "lse"),
                    help="apply this run's TRAINED pooling head inside the exported "
                         "embedding, restoring it from the checkpoint's pool_head.pt. "
                         "Off by default: without it the eval representation is the "
                         "protocol constant and the trained pooling reaches the readout "
                         "only via the LoRA weights it shaped. Must match what the run "
                         "was trained with. Dimensionality is unchanged.")
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

    if args.pool_head and not args.adapter:
        raise SystemExit("--pool-head needs --adapter: pool_head.pt lives in the "
                         "checkpoint dir alongside adapter/")

    encoder = H.load_encoder(args.checkpoint, args.adapter, args.pooling,
                             lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
                             proj_out_dim=args.proj_out_dim, backbone=args.backbone,
                             pool_head=args.pool_head,
                             infer_pool_head=bool(args.pool_head))
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
    # ------------------------------------------------------------------------------
    # CACHE KEY WARNING. The embedding cache key is exactly `embed_dir / exp_code`.
    # It does NOT include the backbone, the checkpoint/adapter, the precision, or
    # --pool-head. Embeddings are cached per (task, encoder) under embed_dataroot and
    # only extracted on fold 0, so a re-run of the regression half is cheap -- but a
    # *different* checkpoint, or the SAME checkpoint read with a different --pool-head,
    # under the same exp_code will silently reuse whatever is already on disk. HEST
    # forces overwrite=True for 'custom_encoder', which is why every distinct encoder
    # configuration must get its own --work-dir or --exp-code; --exp-code is not enough
    # on its own.
    #
    # The --pool-head case is the nastiest one because it fails SILENTLY IN THE
    # DIRECTION OF A NULL RESULT: --pool-head flips infer_pool_head on, which changes
    # what _pool() returns, but if the exp_code was previously used by a plain
    # arithmetic-mean run you score the STALE mean features and conclude the pool head
    # "does nothing". That is a manufactured null, not a measurement.
    #
    # The key format is deliberately NOT changed here: several caches already on disk
    # under /data/ryan.kim/hest_work/embeddings/ (sub5_gem_clsmean, sub5_gem500_clsmean,
    # sub5_g3*_clsmean, ...) were produced by real --pool-head runs, and re-keying would
    # orphan those valid caches and force a full re-extraction. Warn loudly instead.
    cache_dir = paths.embed_dir / exp_code
    if args.pool_head:
        reused = cache_dir.exists() and any(cache_dir.rglob("*.h5"))
        banner = "!" * 78
        print(f"\n{banner}", file=sys.stderr, flush=True)
        print(f"[hest] WARNING: --pool-head={args.pool_head} is set, but the embedding "
              f"cache key is exp_code ONLY:", file=sys.stderr, flush=True)
        print(f"[hest]   {cache_dir}", file=sys.stderr, flush=True)
        print("[hest] The pool-head setting is NOT part of that key. If this exp_code was "
              "ever used by a run", file=sys.stderr, flush=True)
        print("[hest] with a different --pool-head (including the default arithmetic mean), "
              "HEST will reuse the", file=sys.stderr, flush=True)
        print("[hest] STALE embeddings and you will score the wrong features -- typically "
              "manufacturing a null", file=sys.stderr, flush=True)
        print("[hest] result for the pool head. Use a pool-head-specific --exp-code (or a "
              "fresh --work-dir).", file=sys.stderr, flush=True)
        if reused:
            print(f"[hest] >>> THIS CACHE DIRECTORY ALREADY EXISTS AND CONTAINS .h5 FILES. "
                  f"They WILL be reused. <<<", file=sys.stderr, flush=True)
            print("[hest] >>> Unless you know they came from this exact --pool-head setting, "
                  "STOP and re-key. <<<", file=sys.stderr, flush=True)
        print(f"{banner}\n", file=sys.stderr, flush=True)
    # ------------------------------------------------------------------------------
    _, per_enc = benchmark(
        wrapped,
        H.build_transform(encoder.cfg.backbone),
        getattr(torch, args.precision),
        bench_data_root=str(paths.bench_data),
        embed_dataroot=str(cache_dir),
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
        "pool_head": args.pool_head, "infer_pool_head": bool(args.pool_head),
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
