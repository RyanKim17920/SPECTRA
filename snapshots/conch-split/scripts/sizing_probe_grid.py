#!/usr/bin/env python
"""How large can T get at C=2 before the grid step OOMs? (docs/RESULTS.md 8.6)

Why this exists
---------------
``scripts/sizing_probe.py`` sizes the PAIR path, where the batch is ``n_groups x
group_size`` and negatives are ``group_size - 1``. The GRID sampler
(:mod:`waivphaet.data.grid`) has a different arithmetic: with ``C`` conditions and ``T``
shared tiles the step forwards ``B = C*T`` images, every row gets ``T - 1`` negatives and
there are ``B*(C-1)`` query rows. Negatives per row therefore depend ONLY on ``T``, so at
a fixed memory budget they are maximised by spending the whole budget on tiles -- i.e.
``C = 2``, the smallest the sampler allows (``C = 1`` raises: a row's candidates must come
from a *different* condition group). See docs/RESULTS.md 8.

That turns the geometry question into a single sizing question: **how large can T get at
C=2 before OOM?** This probe answers it by running the real grid training step (chunked
encoder forward -> ``grid_info_nce`` -> backward -> optimiser step) on synthetic uint8
tiles, exactly as the pair probe does and for the same reason: peak memory and step time
are a function of shapes and dtypes, not of pixel values.

    ./.venv/bin/python scripts/sizing_probe_grid.py \\
        --geoms 2x1200 2x1400 2x1600 --chunk 600 --steps 4

An OOM is a *result*, not a crash: it is the ceiling we came to find. Each geometry is
caught, reported as ``oom``, and the sweep continues.

``--loss-only`` additionally measures ``grid_info_nce`` in isolation (forward + backward
on a detached leaf) so the ``C^2*T^2 = B^2`` logit tensor can be priced separately from
the activations, which grow only linearly in ``B``. Those two curves have very different
slopes and the whole point of recording both is to know whether they ever cross.
"""

from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("HF_HOME", "/data/huggingface")

import torch

from waivphaet.models.encoder import DEFAULT_BACKBONE, build_encoder
from waivphaet.train.contrastive import _chunked_forward, grid_info_nce

GIB = 1024 ** 3


def parse_geom(s: str) -> tuple[int, int]:
    """``"2x1200"`` -> ``(C, T) = (2, 1200)``."""
    c, t = s.lower().split("x")
    return int(c), int(t)


def loss_only(n_cond: int, n_tiles: int, dim: int, temperature: float) -> dict:
    """Price ``grid_info_nce`` alone: the ``(C,C,T,T)`` logits and their backward.

    The projections are a detached leaf with ``requires_grad``, so the measured peak is
    exactly the loss's own tensors -- no encoder activations in sight.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    z = torch.randn(n_cond * n_tiles, dim, device="cuda", requires_grad=True)
    base = torch.cuda.memory_allocated()
    loss, _ = grid_info_nce(z, n_cond, n_tiles, temperature)
    loss.backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del z, loss
    torch.cuda.empty_cache()
    b = n_cond * n_tiles
    return {
        "loss_peak_gib": round((peak - base) / GIB, 4),
        # what the (C,C,T,T) fp32 logit tensor costs on its own, for the analytic model
        "logits_gib": round(n_cond * n_cond * n_tiles * n_tiles * 4 / GIB, 4),
        "b_squared": b * b,
    }


def probe(backbone: str, n_cond: int, n_tiles: int, *, chunk: int, lora_rank: int,
          lora_alpha: int, grad_checkpointing: bool, amp: str, steps: int, lr: float,
          temperature: float, per_step: bool = False, fresh_batch: bool = False,
          offload: bool = False) -> dict:
    """One geometry, ``steps`` real training steps. Returns peak memory and step time.

    ``fresh_batch`` allocates the device batch anew every step (host pinned -> H2D), which
    is what the real loader does. It costs nothing in steady state if the caching allocator
    reuses the block -- and if it does NOT, that is exactly the fragmentation this probe is
    supposed to catch, so the honest long-horizon run wants it on.
    """
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

    batch = n_cond * n_tiles
    # The grid step forwards ONE tensor of B images -- every image is both an anchor in its
    # own condition group and a query against every other group. Contrast the pair path,
    # which forwards 2*batch (anchors + positives) for the same row count.
    if fresh_batch:
        host = torch.randint(0, 255, (batch, 224, 224, 3), dtype=torch.uint8).pin_memory()
        images = host.to("cuda", non_blocking=True)
    else:
        host = None
        images = torch.randint(0, 255, (batch, 224, 224, 3), dtype=torch.uint8,
                               device="cuda")

    times: list[float] = []
    trace: list[dict] = []
    for i in range(steps):
        t0 = time.time()
        if host is not None:
            # Drop the old device batch first, exactly as the training loop does when the
            # next iterate rebinds ``images``.
            del images
            images = host.to("cuda", non_blocking=True)
        opt.zero_grad(set_to_none=True)
        ctx = (torch.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None
               else torch.autocast("cuda", enabled=False))
        with ctx:
            _, gz = _chunked_forward(model, images, chunk, offload)
        loss, metrics = grid_info_nce(gz, n_cond, n_tiles, temperature)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        dt = time.time() - t0
        if i:  # drop step 0: cudnn autotune + allocator warmup
            times.append(dt)
        if per_step:
            # Creep is the whole question for the long verification run, so record the
            # running peak per step rather than only the final one.
            trace.append({
                "step": i,
                "peak_gib": round(torch.cuda.max_memory_allocated() / GIB, 3),
                "reserved_gib": round(torch.cuda.max_memory_reserved() / GIB, 3),
                "step_s": round(dt, 3),
            })

    peak = torch.cuda.max_memory_allocated() / GIB
    reserved = torch.cuda.max_memory_reserved() / GIB
    neg = metrics["negatives_per_anchor"]
    rows = metrics["n_rows"]
    del model, opt, images, loss, gz
    torch.cuda.empty_cache()
    step_s = sum(times) / len(times)
    out = {
        "n_cond": n_cond, "n_tiles": n_tiles, "chunk": chunk,
        "negatives_per_anchor": neg, "query_rows": rows,
        "images_per_step": batch,
        "peak_gib": round(peak, 2), "reserved_gib": round(reserved, 2),
        "gib_per_image": round(peak / batch, 5),
        "step_s": round(step_s, 3),
        "images_per_s": round(batch / step_s, 1),
    }
    if per_step:
        out["trace"] = trace
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE)
    ap.add_argument("--geoms", nargs="+", default=["2x1200", "2x1400", "2x1600"],
                    help="conditions x tiles, e.g. 2x1200 = the measured gridcmp2 point")
    ap.add_argument("--chunk", type=int, nargs="+", default=[600],
                    help="--grid-forward-chunk: micro-chunk of the BACKBONE forward. "
                         "Several values sweep the chunk at every geometry (geoms x chunks).")
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--activation-offload", action="store_true",
                    help="park the backbone's SAVED activations in pinned host RAM via "
                         "torch.autograd.graph.save_on_cpu. Composes with "
                         "--grad-checkpointing: checkpointing shrinks the recompute "
                         "buffer, this moves what checkpointing still saves. Exact, and "
                         "the whole point of the probe is what it does to the ceiling.")
    ap.add_argument("--amp", default="bfloat16", choices=("none", "float16", "bfloat16"))
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--loss-only", action="store_true",
                    help="also price grid_info_nce in isolation at each geometry")
    ap.add_argument("--per-step", action="store_true",
                    help="record the running peak every step (creep check)")
    ap.add_argument("--fresh-batch", action="store_true",
                    help="reallocate the device batch each step, as the real loader does")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("sizing probe needs a GPU; submit it through SLURM")
    dev = torch.cuda.get_device_properties(0)
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    print(f"[grid-probe] backbone={args.backbone} gpu={dev.name} "
          f"{dev.total_memory / GIB:.2f} GiB amp={args.amp} "
          f"grad_ckpt={args.grad_checkpointing} rank={args.lora_rank} chunk={args.chunk} "
          f"alloc_conf={alloc_conf!r}", flush=True)

    rows = []
    for g in args.geoms:
      c, t = parse_geom(g)
      for chunk in args.chunk:
        try:
            r = probe(args.backbone, c, t, chunk=chunk, lora_rank=args.lora_rank,
                      lora_alpha=args.lora_alpha,
                      grad_checkpointing=args.grad_checkpointing, amp=args.amp,
                      steps=args.steps, lr=args.lr, temperature=args.temperature,
                      per_step=args.per_step, fresh_batch=args.fresh_batch,
                      offload=args.activation_offload)
        except torch.OutOfMemoryError as e:
            # The ceiling is the answer we came for, so record it and keep going.
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            r = {"n_cond": c, "n_tiles": t, "chunk": chunk,
                 "activation_offload": args.activation_offload,
                 "negatives_per_anchor": float(t - 1), "images_per_step": c * t,
                 "status": "oom", "err": str(e).split("\n")[0][:200]}
        r.setdefault("status", "ok")
        r["activation_offload"] = args.activation_offload
        if args.loss_only:
            try:
                r.update(loss_only(c, t, 512, args.temperature))
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                r["loss_peak_gib"] = None
        rows.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "trace"}), flush=True)

    print()
    hdr = (f"{'geom':>10}{'neg/row':>9}{'B':>7}{'chunk':>7}{'peak':>9}{'resv':>9}"
           f"{'tiles/s':>9}{'s/step':>9}   status")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        geom = f"{r['n_cond']}x{r['n_tiles']}"
        if r["status"] != "ok":
            print(f"{geom:>10}{r['negatives_per_anchor']:>9.0f}{r['images_per_step']:>7}"
                  f"{r['chunk']:>7}{'-':>9}{'-':>9}{'-':>9}{'-':>9}   {r['status']}")
            continue
        print(f"{geom:>10}{r['negatives_per_anchor']:>9.0f}{r['images_per_step']:>7}"
              f"{r['chunk']:>7}{r['peak_gib']:>9.2f}{r['reserved_gib']:>9.2f}"
              f"{r['images_per_s']:>9.1f}{r['step_s']:>9.3f}   ok")

    payload = {"backbone": args.backbone, "gpu": dev.name,
               "gpu_gib": round(dev.total_memory / GIB, 2), "amp": args.amp,
               "grad_checkpointing": args.grad_checkpointing, "lora_rank": args.lora_rank,
               "chunk": args.chunk, "activation_offload": args.activation_offload,
               "alloc_conf": alloc_conf, "rows": rows}
    print()
    print(json.dumps(payload, indent=2))
    if args.out:
        from pathlib import Path
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
