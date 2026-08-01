#!/usr/bin/env python
"""Did the embedding actually move, and did it move the *right* way?

Run before and after a fine-tune. For a fixed set of tile indices and a fixed set of
acquisition conditions it measures two numbers:

* **matched** -- mean cosine between the *same* tile under two *different* conditions.
  This is the quantity the objective is supposed to raise: it is cross-scanner /
  cross-stain agreement on identical tissue.
* **random** -- mean cosine between *different* tiles under two different conditions.
  This is the quantity that must NOT rise: it is the null.

The interpretation rule is the whole point of the script. A contrastive fine-tune can
"succeed" on matched similarity in two very different ways:

    matched up, random flat/down   -> real invariance; the gap widened
    matched up, random up equally  -> REPRESENTATION COLLAPSE. Every embedding is
                                      drifting toward one vector, cosine goes to 1 for
                                      everything, and the number is meaningless.

So the headline is the **separation** ``matched - random``, never matched alone. PLAN.md 6
says the same thing for the eval suite: "never report cosine similarity alone -- PLIP
scores 0.878 cosine at 0.054 top-10".

We also report top-1 retrieval of the matched tile among the sampled tiles (same
condition pair), which is the rank-based counterpart that collapse cannot fake, and the
mean pairwise cosine of a single condition against itself over different tiles
(``within_condition_random``) as a direct collapse gauge.

Reported separately for cross-SCANNER (same stain) and cross-STAIN (same scanner) pairs,
and separately for train-split vs held-out-split conditions -- PLAN.md 6: "report
cross-stain and cross-scanner separately; the composite hides the hard axis".

    python scripts/embed_probe.py --out probe_before.json
    python scripts/embed_probe.py --adapter runs/x/step_0000300 --out probe_after.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from waivphaet.data.conditions import available_conditions, make_split
from waivphaet.data.repack import open_slide, present_filenames
from waivphaet.models.encoder import DEFAULT_BACKBONE, build_encoder


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packed-dir", type=Path, default=Path("/data/ryan.kim/plism/repacked"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, default=None,
                    help="checkpoint dir written by save_checkpoint (contains adapter/ + projector.pt)")
    ap.add_argument("--heldout-scanners", nargs="*", default=["GT450", "S210"])
    ap.add_argument("--heldout-stains", nargs="*", default=["HRH", "KR", "MY"])
    ap.add_argument("--conditions-file", type=Path, default=None,
                    help="JSON list of condition keys ('GIVH_AT2') to pin the probe set. "
                         "REQUIRED for a valid before/after comparison while the acquisition "
                         "job is still streaming -- otherwise the two probes score different "
                         "condition sets and the delta mixes a real change with a set change.")
    ap.add_argument("--n-tiles", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--proj-out-dim", type=int, default=512)
    ap.add_argument("--pooling", default="clsmean", choices=["cls", "mean", "clsmean"])
    ap.add_argument("--backbone", default=DEFAULT_BACKBONE,
                    help="HF id of the base backbone; must match what --adapter was trained on")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def load_adapter(model, ckpt: Path) -> None:
    """Restore a LoRA checkpoint written by ``waivphaet.train.contrastive.save_checkpoint``."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    sd_path = ckpt / "adapter" / "adapter_model.safetensors"
    state = (
        load_file(str(sd_path)) if sd_path.exists()
        else torch.load(ckpt / "adapter" / "adapter_model.bin", map_location="cpu")
    )
    out = set_peft_model_state_dict(model.backbone, state)
    missing = getattr(out, "unexpected_keys", None)
    if missing:
        raise RuntimeError(f"adapter keys not consumed: {list(missing)[:5]}")
    model.projector.load_state_dict(torch.load(ckpt / "projector.pt", map_location="cpu"))


@torch.no_grad()
def embed_condition(model, slide, tiles: np.ndarray, device, batch_size: int):
    """-> (embedding (N, D), projection (N, P)) for one condition, both L2-normalised."""
    embs, projs = [], []
    for i in range(0, tiles.size, batch_size):
        chunk = tiles[i : i + batch_size]
        imgs = np.stack([np.asarray(slide[int(t)]) for t in chunk])
        x = torch.from_numpy(imgs).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            e, p = model(x)
        embs.append(F.normalize(e.float(), dim=-1).cpu())
        projs.append(F.normalize(p.float(), dim=-1).cpu())
    return torch.cat(embs), torch.cat(projs)


def pair_stats(x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    """``x``/``y`` are (N, D) L2-normalised embeddings of the SAME tiles, two conditions."""
    sim = x @ y.t()  # (N, N); diagonal = matched tile, off-diagonal = the null
    n = sim.shape[0]
    off = ~torch.eye(n, dtype=torch.bool)
    matched = float(sim.diagonal().mean())
    random = float(sim[off].mean())
    top1 = float((sim.argmax(dim=1) == torch.arange(n)).float().mean())
    return {"matched": matched, "random": random, "separation": matched - random, "top1": top1}


def main() -> int:
    args = parse_args()
    os.environ.setdefault("HF_HOME", "/data/ryan.kim/hf_home")
    # the projection head is randomly initialised, so without this the "before" and
    # "after" runs get different heads and the projection-space delta is meaningless.
    # (The backbone needs no seed: zero-init LoRA B makes an unloaded adapter exactly
    # the base model, so embedding-space "before" is base phikon-v2 by construction.)
    torch.manual_seed(args.seed)

    split = make_split(args.heldout_scanners, args.heldout_stains)
    present = present_filenames(args.packed_dir)
    groups = {
        "train": available_conditions(split.train, present),
        "heldout": available_conditions(split.heldout, present),
    }
    if args.conditions_file is not None:
        pinned = set(json.loads(args.conditions_file.read_text()))
        groups = {k: [c for c in v if c.key in pinned] for k, v in groups.items()}
        got = {c.key for v in groups.values() for c in v}
        if got != pinned:
            raise SystemExit(f"pinned conditions not all available: missing {sorted(pinned - got)}")

    rng = np.random.default_rng(args.seed)
    from waivphaet.data.conditions import NUM_TILES

    tiles = np.sort(rng.choice(NUM_TILES, size=args.n_tiles, replace=False))

    device = torch.device(args.device)
    # A checkpoint must carry its own backbone. eval_checkpoints.py deliberately does not
    # pass --backbone (see its run_pathorob comment) because extract_pathorob_features
    # derives it from the adapter; the probe did not, so a Midnight adapter (1536-d) was
    # loaded into a phikon-v2 model (1024-d) and died on a size mismatch across all 24
    # layers. Derive it here too, and hard-fail on an explicit contradiction rather than
    # silently preferring one source.
    backbone = args.backbone
    if args.adapter is not None:
        acfg = Path(args.adapter) / "adapter" / "adapter_config.json"
        if acfg.exists():
            saved = json.loads(acfg.read_text()).get("base_model_name_or_path")
            if saved:
                if args.backbone != DEFAULT_BACKBONE and args.backbone != saved:
                    raise SystemExit(
                        f"--backbone {args.backbone} contradicts the adapter's own "
                        f"base_model_name_or_path={saved} ({acfg})"
                    )
                backbone = saved
    print(f"[probe] backbone={backbone}", flush=True)
    model = build_encoder(
        backbone=backbone,
        lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
        proj_out_dim=args.proj_out_dim, pooling=args.pooling,
    )
    if args.adapter is not None:
        load_adapter(model, args.adapter)
        print(f"[probe] loaded adapter from {args.adapter}")
    else:
        print("[probe] BASE model (no adapter)")
    model.to(device).eval()

    report: dict = {
        "adapter": str(args.adapter) if args.adapter else None,
        "n_tiles": int(tiles.size),
        "seed": args.seed,
        "groups": {},
    }

    for gname, conds in groups.items():
        if len(conds) < 2:
            print(f"[probe] skipping {gname}: only {len(conds)} condition(s) repacked")
            continue
        cache = {}
        for c in conds:
            slide = open_slide(args.packed_dir, c.slide_id.replace(".tif", ""))
            cache[c.key] = embed_condition(model, slide, tiles, device, args.batch_size)
            print(f"[probe] {gname}: embedded {c.key}")

        # cross-SCANNER = same stain, different scanner; cross-STAIN = the transpose.
        axes: dict[str, list[tuple] ] = {"cross_scanner": [], "cross_stain": []}
        for i, a in enumerate(conds):
            for b in conds[i + 1 :]:
                if a.stain == b.stain and a.scanner != b.scanner:
                    axes["cross_scanner"].append((a, b))
                elif a.scanner == b.scanner and a.stain != b.stain:
                    axes["cross_stain"].append((a, b))

        gres: dict = {}
        for axis, pairs in axes.items():
            for space, idx in (("embedding", 0), ("projection", 1)):
                acc = [pair_stats(cache[a.key][idx], cache[b.key][idx]) for a, b in pairs]
                if not acc:
                    continue
                gres[f"{axis}.{space}"] = {
                    k: float(np.mean([d[k] for d in acc])) for k in acc[0]
                } | {"n_pairs": len(acc)}

        # collapse gauge: different tiles, SAME condition. If this rises toward 1 the
        # model is mapping everything to one vector and every "matched" gain is fake.
        for space, idx in (("embedding", 0), ("projection", 1)):
            vals = []
            for c in conds:
                z = cache[c.key][idx]
                s = z @ z.t()
                vals.append(float(s[~torch.eye(s.shape[0], dtype=torch.bool)].mean()))
            gres[f"within_condition_random.{space}"] = float(np.mean(vals))

        gres["conditions"] = sorted(c.key for c in conds)
        report["groups"][gname] = gres

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
