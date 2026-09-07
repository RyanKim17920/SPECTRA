#!/usr/bin/env python
"""Measure what actually fits, per backbone, instead of guessing (PLAN.md §5).

Why this exists
---------------
The negative count in our objective is ``group_size - 1`` and negatives may only come
from the anchor's own condition-homogeneous group (PLAN.md §2). Gradient accumulation
therefore **cannot** buy negatives -- they have to be in the same *forward* batch. So the
only question that matters for a new backbone is: how many images per step fit, and how
slow is a step at that size.

The phikon-v2 answer (ViT-L/16, 24 blocks, 1024-d) is on record in README: 2 groups x 192
= 191 negatives/anchor = 768 images/step, 21.77 GiB peak with gradient checkpointing,
1.91 s/step; 2 x 256 measured at 28.55 GiB / 2.91 s. midnight is ViT-g/14 -- ~3x the
parameters *and* 256 tokens instead of 196 -- so none of those numbers transfer. Measure.

The probe runs the real training step (encoder forward -> masked InfoNCE -> backward ->
step) on synthetic uint8 tiles. Synthetic input is fine for a *sizing* number: memory and
step time depend on shapes and dtypes, not pixel values, and the real loader reads a
repacked memmap at ~900k tiles/s so it is nowhere near the bottleneck.

    ./.venv/bin/python scripts/sizing_probe.py \\
        --backbone kaiko-ai/midnight --grad-checkpointing \\
        --shapes 2x48 2x64 2x96 2x128 2x192

Each shape is measured in a fresh process-local state with the CUDA caching allocator
reset, and a shape that OOMs is reported as ``oom`` rather than killing the run -- the
point is to find the ceiling, so hitting it is a result.
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("HF_HOME", "/data/huggingface")

import torch

from waivphaet.models.encoder import DEFAULT_BACKBONE, build_encoder
from waivphaet.train.contrastive import masked_info_nce

GIB = 1024 ** 3


def parse_shape(s: str) -> tuple[int, int]:
    g, n = s.lower().split("x")
    return int(g), int(n)


def probe(backbone: str, n_groups: int, group_size: int, *, lora_rank: int, lora_alpha: int,
          grad_checkpointing: bool, amp: str, steps: int, lr: float) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_encoder(
        backbone=backbone, lora_rank=lora_rank, lora_alpha=lora_alpha,
        pooling="clsmean", grad_checkpointing=grad_checkpointing,
    ).cuda()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    amp_dtype = {"none": None, "float16": torch.float16, "bfloat16": torch.bfloat16}[amp]

    batch = n_groups * group_size
    # The step sees 2*batch images: anchors and their registered positives.
    anchors = torch.randint(0, 255, (batch, 224, 224, 3), dtype=torch.uint8, device="cuda")
    positives = torch.randint(0, 255, (batch, 224, 224, 3), dtype=torch.uint8, device="cuda")
    group_id = torch.arange(batch, device="cuda") // group_size

    times = []
    for i in range(steps):
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        ctx = (torch.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None
               else torch.autocast("cuda", enabled=False))
        with ctx:
            _, za = model(anchors)
            _, zp = model(positives)
        loss, _ = masked_info_nce(za, zp, group_id, temperature=0.07)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        dt = time.time() - t0
        if i:  # drop the first step: cudnn autotune + allocator warmup
            times.append(dt)

    peak = torch.cuda.max_memory_allocated() / GIB
    reserved = torch.cuda.max_memory_reserved() / GIB
    summary = model.trainable_parameter_summary()
    del model, opt, anchors, positives
    torch.cuda.empty_cache()
    step_s = sum(times) / len(times)
    return {
        "n_groups": n_groups, "group_size": group_size,
        "negatives_per_anchor": group_size - 1,
        "anchors_per_step": batch, "images_per_step": 2 * batch,
        "peak_gib": round(peak, 2), "reserved_gib": round(reserved, 2),
        "gib_per_image": round(peak / (2 * batch), 4),
        "step_s": round(step_s, 3),
        "images_per_s": round(2 * batch / step_s, 1),
        "trainable": summary["trainable"], "total": summary["total"],
        "lora_targets": summary["lora_targets"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE)
    ap.add_argument("--shapes", nargs="+", default=["2x48", "2x64", "2x96", "2x128", "2x192"],
                    help="groups x group_size, e.g. 2x192 = the phikon-v2 real-run shape")
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--amp", default="bfloat16", choices=("none", "float16", "bfloat16"))
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("sizing probe needs a GPU; submit it through SLURM")
    dev = torch.cuda.get_device_properties(0)
    print(f"[probe] backbone={args.backbone} gpu={dev.name} {dev.total_memory / GIB:.1f} GiB "
          f"amp={args.amp} grad_ckpt={args.grad_checkpointing} rank={args.lora_rank}",
          flush=True)

    rows = []
    for s in args.shapes:
        g, n = parse_shape(s)
        try:
            r = probe(args.backbone, g, n, lora_rank=args.lora_rank,
                      lora_alpha=args.lora_alpha, grad_checkpointing=args.grad_checkpointing,
                      amp=args.amp, steps=args.steps, lr=args.lr)
        except torch.OutOfMemoryError:
            # The ceiling is the answer we came for, so record it and keep going.
            torch.cuda.empty_cache()
            r = {"n_groups": g, "group_size": n, "negatives_per_anchor": n - 1,
                 "images_per_step": 2 * g * n, "status": "oom"}
        r.setdefault("status", "ok")
        rows.append(r)
        print(json.dumps(r), flush=True)

    print()
    hdr = f"{'shape':>9}{'neg/anchor':>12}{'img/step':>10}{'peak GiB':>10}{'step s':>9}{'img/s':>9}   status"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        shape = f"{r['n_groups']}x{r['group_size']}"
        if r["status"] != "ok":
            print(f"{shape:>9}{r['negatives_per_anchor']:>12}{r['images_per_step']:>10}"
                  f"{'-':>10}{'-':>9}{'-':>9}   {r['status']}")
            continue
        print(f"{shape:>9}{r['negatives_per_anchor']:>12}{r['images_per_step']:>10}"
              f"{r['peak_gib']:>10.2f}{r['step_s']:>9.3f}{r['images_per_s']:>9.1f}   ok")
    payload = {"backbone": args.backbone, "gpu": dev.name,
               "gpu_gib": round(dev.total_memory / GIB, 1), "amp": args.amp,
               "grad_checkpointing": args.grad_checkpointing, "lora_rank": args.lora_rank,
               "rows": rows}
    print()
    print(json.dumps(payload, indent=2))
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
