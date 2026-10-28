"""GTEx UTILITY-GUARD tile cache -> (N,224,224,3) uint8 + JSON (per-tile split / tissue / donor).

GTEx is in no SPECTRA test benchmark (PathoROB / CROMA / HEST / THUNDER / CPTAC). Three
DONOR-disjoint splits of level-0 tiles (224 source px, ~0.5 um/px) from a directory of GTEx
tile parquet shards (``shard-*.parquet`` with columns tissue, slide_id, subject_id, level, jpeg):

* ``train`` -- linear-probe / kNN fitting set: per tissue ``--train-slides`` slides x
  ``--tiles-per-slide`` tiles (40 tissues),
* ``val``   -- probe scoring set (balanced accuracy over the 40 tissue types): per tissue
  ``--val-slides`` slides x ``--tiles-per-slide``,
* ``ret``   -- ``--n-ret`` retention tiles (CKA / cosine / kNN overlap to the frozen base),
  spread evenly over tissues.

Donors are assigned by a seeded permutation: 25 % train, 15 % val, 10 % ret; the remaining
50 % are never used (listed as ``reserved_donors``), so the cache is reproducible from the seed.

    python -m spectra.sweep.build_guard_cache --src <gtex tile parquet dir>
        (writes $SPECTRA_CACHE_DIR/gtex_guard_v1.npy + .json, where the trainer looks for it)
"""

from __future__ import annotations

import argparse
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np



def donor_of(slide_id: str) -> str:
    return "-".join(slide_id.split("-")[:2])


def main(argv=None) -> int:
    import pandas as pd
    import pyarrow.parquet as pq
    from PIL import Image

    from .config import cache_root

    ap = argparse.ArgumentParser()
    ap.add_argument("--train-slides", type=int, default=32)
    ap.add_argument("--val-slides", type=int, default=16)
    ap.add_argument("--tiles-per-slide", type=int, default=6)
    ap.add_argument("--n-ret", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--src", type=Path, default=os.environ.get("GTEX_TILE_DIR"),
                    help="GTEx tile parquet directory (default $GTEX_TILE_DIR)")
    ap.add_argument("--out", type=Path, default=None, help="default <cache root>/gtex_guard_v1.npy")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args(argv)
    if a.src is None:
        ap.error("--src (or $GTEX_TILE_DIR) is required")
    a.out = a.out or cache_root() / "gtex_guard_v1.npy"
    SRC = Path(a.src)
    rng = np.random.default_rng(a.seed)
    shards = sorted(SRC.glob("shard-*.parquet"))

    def meta(si):
        f = pq.ParquetFile(shards[si])
        t = f.read(columns=["tissue", "slide_id", "subject_id", "level"]).to_pandas()
        rg = np.repeat(np.arange(f.metadata.num_row_groups),
                       [f.metadata.row_group(i).num_rows for i in range(f.metadata.num_row_groups)])
        off = np.concatenate([np.arange(f.metadata.row_group(i).num_rows)
                              for i in range(f.metadata.num_row_groups)])
        t["shard"], t["rg"], t["off"] = si, rg, off
        return t[t.level == 0]

    with ThreadPoolExecutor(a.threads) as ex:
        df = pd.concat(list(ex.map(meta, range(len(shards)))), ignore_index=True)
    df["donor"] = [s if isinstance(s, str) and s else donor_of(x) for s, x in zip(df.subject_id, df.slide_id)]
    print(f"level-0 tiles {len(df)} slides {df.slide_id.nunique()} donors {df.donor.nunique()} "
          f"tissues {df.tissue.nunique()}", flush=True)

    donors = np.array(sorted(df.donor.unique()))
    perm = rng.permutation(len(donors))
    cuts = np.cumsum([int(f * len(donors)) for f in (0.25, 0.15, 0.10)])
    split_of = {}
    for i, d in enumerate(donors[perm]):
        split_of[d] = "train" if i < cuts[0] else "val" if i < cuts[1] else "ret" if i < cuts[2] else "reserved"
    df["split"] = df.donor.map(split_of)
    bad = {d for d, sp in split_of.items() if sp == "reserved"}
    df = df[df.split != "reserved"]
    print(f"{len(donors)} donors: {len(bad)} reserved (unused), {len(donors) - len(bad)} guard donors",
          flush=True)
    tissues = sorted(df.tissue.unique())
    picks = []

    def take(sub, n_slides, per_slide, quota):
        slides = sub.slide_id.unique()
        slides = slides[rng.permutation(len(slides))][:n_slides]
        rows = []
        for s in slides:
            r = sub[sub.slide_id == s]
            rows.append(r.iloc[rng.permutation(len(r))[:per_slide]])
        out = pd.concat(rows) if rows else sub.iloc[:0]
        if len(out) < quota:  # top up from any remaining tile of the split
            rest = sub.drop(out.index)
            out = pd.concat([out, rest.iloc[rng.permutation(len(rest))[:quota - len(out)]]])
        return out.iloc[:quota]

    per_t_ret = int(np.ceil(a.n_ret / len(tissues)))
    for t in tissues:
        sub = df[df.tissue == t]
        for sp, ns in (("train", a.train_slides), ("val", a.val_slides)):
            x = take(sub[sub.split == sp], ns, a.tiles_per_slide, ns * a.tiles_per_slide)
            picks.append(x.assign(split=sp))
        x = take(sub[sub.split == "ret"], per_t_ret, 1, per_t_ret)
        picks.append(x.assign(split="ret"))
    sel = pd.concat(picks)
    ret_idx = sel.index[sel.split == "ret"]
    if len(ret_idx) > a.n_ret:
        drop = rng.choice(ret_idx, size=len(ret_idx) - a.n_ret, replace=False)
        sel = sel.drop(drop)
    order = {"train": 0, "val": 1, "ret": 2}
    sel = sel.assign(_o=sel.split.map(order)).sort_values(["_o", "tissue", "slide_id"], kind="stable")
    print(sel.groupby("split").agg(n=("slide_id", "size"), slides=("slide_id", "nunique"),
                                   donors=("donor", "nunique"), tissues=("tissue", "nunique")), flush=True)
    assert not (set(sel.donor[sel.split == "train"]) & set(sel.donor[sel.split == "val"]))
    assert not (set(sel.donor[sel.split == "ret"]) & set(sel.donor[sel.split != "ret"]))
    assert not (set(sel.donor) & bad)

    imgs = np.zeros((len(sel), 224, 224, 3), np.uint8)
    pos = {k: i for i, k in enumerate(sel.index)}
    jobs = {}
    for k, r in sel.iterrows():
        jobs.setdefault((int(r.shard), int(r.rg)), []).append((k, int(r.off)))

    def load(item):
        (si, rg), lst = item
        f = pq.ParquetFile(shards[si])
        col = f.read_row_group(rg, columns=["jpeg"]).column("jpeg")
        n_bad = 0
        for k, off in lst:
            im = np.asarray(Image.open(io.BytesIO(col[off].as_py())).convert("RGB"))
            if im.shape != (224, 224, 3):
                from PIL import Image as _I
                im = np.asarray(_I.fromarray(im).resize((224, 224)))
                n_bad += 1
            imgs[pos[k]] = im
        return n_bad

    with ThreadPoolExecutor(a.threads) as ex:
        nb = sum(ex.map(load, list(jobs.items())))
    print(f"decoded {len(sel)} tiles ({nb} resized)", flush=True)
    lab = {t: i for i, t in enumerate(tissues)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(f".tmp{os.getpid()}.npy")
    np.save(tmp, imgs)
    os.replace(tmp, a.out)
    a.out.with_suffix(".json").write_text(json.dumps({
        "source": str(SRC), "seed": a.seed, "n": len(sel), "tissues": tissues,
        "reserved_donors": sorted(bad),
        "split": sel.split.tolist(), "label": [lab[t] for t in sel.tissue],
        "slide_id": sel.slide_id.tolist(), "donor": sel.donor.tolist(),
        "args": {k: str(v) for k, v in vars(a).items()}}))
    print(f"wrote {imgs.shape} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
