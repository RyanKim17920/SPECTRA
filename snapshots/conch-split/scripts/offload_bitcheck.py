#!/usr/bin/env python
"""Prove ``--activation-offload`` changes memory and NOTHING else.

Why this exists
---------------
``torch.autograd.graph.save_on_cpu`` is *claimed* to be exact: it is a pair of
saved-tensor hooks that move a saved tensor to pinned host RAM on the way out and bring
the identical bytes back on the way in. Nothing about the objective, the batch, or the
op boundaries moves. But "claimed exact" is exactly the kind of assertion this codebase
has been burned by before: chunking the PROJECTOR was also "obviously" just a memory
device, and it silently computed BatchNorm over the wrong batch and shifted the loss by
1.65 magnitudes (see ``_chunked_forward``'s docstring). That bug was invisible in the
loss curve of a single run -- it only showed up against a matched control.

So this is the matched control. Same seed, same synthetic batch, same model weights, one
flag flipped:

    loss          must match to the bit (or to <1e-6 under a nondeterministic backend)
    every grad    must match likewise

There is a second, subtler thing being checked. ``save_on_cpu`` and non-reentrant
gradient checkpointing BOTH work by installing saved-tensor hooks, and nesting them is
the part of this change that is not obvious by inspection. If those two mechanisms
interfere, the failure mode is not necessarily a crash -- it could be a silently wrong
gradient. Comparing gradients, not just the loss, is what would catch that.

    ./.venv/bin/python scripts/offload_bitcheck.py --n-cond 2 --n-tiles 64 --chunk 16
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("HF_HOME", "/data/huggingface")

import torch

from waivphaet.models.encoder import DEFAULT_BACKBONE, build_encoder
from waivphaet.train.contrastive import _chunked_forward, grid_info_nce

GIB = 1024 ** 3


def run(backbone: str, n_cond: int, n_tiles: int, chunk: int, offload: bool,
        lora_rank: int, lora_alpha: int, amp: str, seed: int, temperature: float):
    """One forward+backward at a fixed seed. Returns (loss, {name: grad}, peak_gib)."""
    # Rebuild the model from the same seed rather than reusing one, so the two runs are
    # independent all the way down to LoRA init -- a shared model would hide an in-place
    # corruption of the weights by the first run.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = build_encoder(backbone=backbone, lora_rank=lora_rank, lora_alpha=lora_alpha,
                          pooling="clsmean", grad_checkpointing=True).cuda()
    model.train()

    # Same pixels both times: seeded separately so it cannot drift with model init.
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    images = torch.randint(0, 255, (n_cond * n_tiles, 224, 224, 3),
                           dtype=torch.uint8, generator=g).cuda()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    amp_dtype = {"none": None, "float16": torch.float16, "bfloat16": torch.bfloat16}[amp]
    ctx = (torch.autocast("cuda", dtype=amp_dtype) if amp_dtype is not None
           else torch.autocast("cuda", enabled=False))
    with ctx:
        _, gz = _chunked_forward(model, images, chunk, offload)
    loss, _ = grid_info_nce(gz, n_cond, n_tiles, temperature)
    loss.backward()
    torch.cuda.synchronize()

    grads = {n: p.grad.detach().float().clone()
             for n, p in model.named_parameters() if p.grad is not None}
    peak = torch.cuda.max_memory_allocated() / GIB
    out = (float(loss.detach()), grads, peak)
    del model, images, gz, loss
    torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE)
    ap.add_argument("--n-cond", type=int, default=2)
    ap.add_argument("--n-tiles", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--amp", default="bfloat16", choices=("none", "float16", "bfloat16"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="max allowed |offload - plain|. The hook restores identical "
                         "bytes, so the honest expectation is 0.0; the tolerance exists "
                         "only for backend nondeterminism in the reduction order.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("bitcheck needs a GPU; submit it through SLURM")

    common = dict(backbone=args.backbone, n_cond=args.n_cond, n_tiles=args.n_tiles,
                  chunk=args.chunk, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
                  amp=args.amp, seed=args.seed, temperature=args.temperature)

    print(f"[bitcheck] C={args.n_cond} T={args.n_tiles} B={args.n_cond * args.n_tiles} "
          f"chunk={args.chunk} amp={args.amp} seed={args.seed}", flush=True)

    loss_a, grads_a, peak_a = run(offload=False, **common)
    print(f"[bitcheck] plain   loss={loss_a!r} peak={peak_a:.3f} GiB "
          f"({len(grads_a)} grads)", flush=True)
    loss_b, grads_b, peak_b = run(offload=True, **common)
    print(f"[bitcheck] offload loss={loss_b!r} peak={peak_b:.3f} GiB "
          f"({len(grads_b)} grads)", flush=True)

    d_loss = abs(loss_a - loss_b)
    print(f"[bitcheck] |dloss| = {d_loss:.3e}")

    assert grads_a.keys() == grads_b.keys(), "different parameters received gradients"
    assert grads_a, "no gradients at all -- the check would be vacuous"

    worst, worst_name = 0.0, ""
    for n in grads_a:
        d = float((grads_a[n] - grads_b[n]).abs().max())
        if d > worst:
            worst, worst_name = d, n
    print(f"[bitcheck] worst |dgrad| = {worst:.3e}  ({worst_name})")

    saved = peak_a - peak_b
    pct = 100.0 * saved / peak_a if peak_a else 0.0
    print(f"[bitcheck] peak {peak_a:.3f} -> {peak_b:.3f} GiB "
          f"({saved:+.3f} GiB, {pct:+.1f}%)")

    ok = d_loss <= args.tol and worst <= args.tol
    # The memory result is REPORTED, never asserted: at this deliberately tiny geometry
    # the offload can easily be a wash or worse, because the hook's own bookkeeping is a
    # fixed cost while the tensors it moves are small. The ceiling question belongs to
    # sizing_probe_grid.py at real sizes. What must hold HERE is exactness.
    print(f"[bitcheck] {'PASS' if ok else 'FAIL'} "
          f"(tol={args.tol:.1e}; exactness is the gate, memory is informational)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
