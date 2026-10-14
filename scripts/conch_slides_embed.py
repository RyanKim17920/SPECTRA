"""Embed TCGA .svs tiles with base CONCH and SPECTRA-CONCH checkpoints at TRUE 20x.

Geometry: level-0 tile side S0 = round(448 * 0.5 / mpp-x) (true 20x, 448 px), read at level 0,
resized to 448 and fed through CONCH's own PIL preprocess (Resize 448 bicubic + CenterCrop 448 +
OpenAI norm) -> encode_image(proj_contrast=False, normalize=False). Slides with no mpp are
skipped. Tissue filter = code/conch15/slides_embed.py (thumbnail HSV sat > 25 and min < 225,
tile tissue frac >= 0.4). Up to --max-tiles tiles per slide, uniformly sampled (seed 0;
0 = all). Every tile is decoded ONCE and pushed through every model.

Output: <out>/<slide>.npz with xy (grid coords), S0, mpp, type, tss, patient and f_<model> (n,512) f16.

    python scripts/conch_slides_embed.py --slides sel.json --out DIR --max-tiles 500 \
        --models base anc1_s100=<run>/step_0000100/adapter ...
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import openslide
import torch
from PIL import Image

TILE = 448


def plan(rec, max_tiles):
    try:
        sl = openslide.OpenSlide(rec["svs"]); p = sl.properties
        mpp = float(p["openslide.mpp-x"])
    except Exception as e:  # noqa: BLE001
        return dict(rec, err=str(e)[:80])
    S0 = int(round(TILE * 0.5 / mpp)); W, H = sl.dimensions; nc, nr = W // S0, H // S0
    # Thumbnail from the pyramid level nearest 16 px/tile, read directly: get_thumbnail() on a slide
    # without a usable low-res level decodes level 0 whole (one such slide hit 180 GB RSS).
    lv = sl.get_best_level_for_downsample(S0 / 16); lw, lh = sl.level_dimensions[lv]
    if lw * lh > 4e8:
        return dict(rec, err=f"no usable pyramid level (best level {lv} is {lw}x{lh})")
    th = np.asarray(sl.read_region((0, 0), lv, (lw, lh)).convert("RGB").resize((nc * 16, nr * 16), Image.BILINEAR))
    hsv = np.asarray(Image.fromarray(th).convert("HSV"))
    tis = ((hsv[..., 1] > 25) & (th.min(2) < 225)).astype(np.float32)
    frac = tis.reshape(nr, 16, nc, 16).mean((1, 3)); ys, xs = np.where(frac >= 0.4)
    xy = np.stack([xs, ys], 1)
    n_tissue = len(xy)
    if max_tiles and len(xy) > max_tiles:
        xy = xy[np.sort(np.random.default_rng(0).choice(len(xy), max_tiles, replace=False))]
    return dict(rec, mpp=mpp, S0=S0, xy=xy, n_tissue=n_tissue)


class Tiles(torch.utils.data.Dataset):
    def __init__(self, items, pre):
        self.items, self.pre, self.h = items, pre, {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        si, path, S0, x, y = self.items[i]
        if path not in self.h:
            if len(self.h) > 8:
                self.h.clear()
            self.h[path] = openslide.OpenSlide(path)
        im = self.h[path].read_region((int(x * S0), int(y * S0)), 0, (S0, S0)).convert("RGB")
        if S0 != TILE:
            im = im.resize((TILE, TILE), Image.BILINEAR)
        return self.pre(im), si


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", type=Path, required=True, help="json list of manifest records")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--max-tiles", type=int, default=500)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--chunk-slides", type=int, default=20)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    from spectra.models.conch_visual import ConchVisual, load_conch_full
    from conch.open_clip_custom import create_model_from_pretrained  # noqa: F401  (preprocess)
    full = load_conch_full()
    from torchvision import transforms as T
    pre = T.Compose([T.Resize(448, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(448), T.ToTensor(),
                     T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    base = ConchVisual(full.visual).eval()
    models = {}
    for s in a.models:
        n, ad = (s.split("=", 1) + [None])[:2]
        if ad:
            from peft import PeftModel
            models[n] = PeftModel.from_pretrained(copy.deepcopy(base), ad).merge_and_unload().cuda().eval()
        else:
            models[n] = base
    base.cuda().eval()
    recs = json.load(open(a.slides))
    todo = []
    for r in recs:
        p = a.out / f"{r['slide']}.npz"
        if p.exists():
            have = {k[2:] for k in np.load(p).files if k.startswith("f_")}
            if set(models) <= have:
                continue
        todo.append(r)
    print(f"{len(todo)}/{len(recs)} slides to embed with {list(models)}", flush=True)
    t0 = time.time()
    with mp.Pool(min(32, a.workers)) as pool:
        plans = pool.starmap(plan, [(r, a.max_tiles) for r in todo])
    bad = [p for p in plans if "err" in p]
    plans = [p for p in plans if "err" not in p and len(p["xy"])]
    print(f"planned {len(plans)} slides ({len(bad)} errors) in {time.time()-t0:.0f}s; tiles={sum(len(p['xy']) for p in plans)}", flush=True)
    for b in bad:
        print("  SKIP", b["slide"], b["err"], flush=True)
    for c0 in range(0, len(plans), a.chunk_slides):
        chunk = plans[c0:c0 + a.chunk_slides]; t1 = time.time()
        items = [(si, p["svs"], p["S0"], int(x), int(y)) for si, p in enumerate(chunk) for x, y in p["xy"]]
        dl = torch.utils.data.DataLoader(Tiles(items, pre), batch_size=128, num_workers=a.workers)
        feats = {n: [] for n in models}; sids = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for x, si in dl:
                x = x.cuda(non_blocking=True); sids.append(si.numpy())
                for n, m in models.items():
                    feats[n].append(m.pooled(x).float().cpu().numpy().astype(np.float16))
        sids = np.concatenate(sids); feats = {n: np.concatenate(v) for n, v in feats.items()}
        for si, p in enumerate(chunk):
            path = a.out / f"{p['slide']}.npz"; m = sids == si
            old = dict(np.load(path)) if path.exists() else {}
            if old and not np.array_equal(old["xy"], p["xy"]):
                raise RuntimeError(f"tile plan changed for {p['slide']}")
            d = dict(old, xy=p["xy"], S0=p["S0"], mpp=p["mpp"], n_tissue=p["n_tissue"], type=p["type"], tss=p["tss"],
                     patient=p["patient"], **{f"f_{n}": feats[n][m] for n in models})
            np.savez(path, **d)
        n = len(items)
        print(f"chunk {c0//a.chunk_slides}: {len(chunk)} slides {n} tiles {n/(time.time()-t1):.0f} tiles/s "
              f"({time.time()-t0:.0f}s total)", flush=True)
    print("EMBED DONE", flush=True)


if __name__ == "__main__":
    main()
