"""One-time uint8 tile caches for scripts/conch_eval.py (CPU only).

Writes under $CONCH_EVAL_CACHE (default /data/ryan.kim/ashby/scratch/spectra_conch_eval):
  plism_heldout.npy   (C_held, 256, 224, 224, 3) uint8 + plism_heldout.json (conditions)
      The SAME 256 locations and held-out conditions as the base-CONCH reference
      (code/plism/analyze.py: loc_idx.npy -> rng(1234).choice(1000, 256); held-out =
      scanner in {GT450, S210} or stain in {HRH, KR, MY}).
  crc.npy             (7420, 224, 224, 3) uint8 -- the concept_search crc.npz tile list
                      (CRC-VAL-HE-7K = split 0, plus a 100K top-up), same order/labels.
  nct.npy             (N, 224, 224, 3) uint8 memmap + nct_labels.npy -- NCT-CRC-HE-100K.
"""
import glob
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(os.environ.get("CONCH_EVAL_CACHE", "/data/ryan.kim/ashby/scratch/spectra_conch_eval"))
OUT.mkdir(parents=True, exist_ok=True)
PACK = Path(os.environ.get("SPECTRA_PACKED_DIR", "/data/plism/repacked"))
PL = Path("/data/ryan.kim/ashby/scratch/plism")
CRC = "/data/ryan.kim/ashby/data/concept_search/crc.npz"
NCT = "/data/thunder-data/crc/NCT-CRC-HE-100K"
HELD_SC, HELD_ST = {"GT450", "S210"}, {"HRH", "KR", "MY"}


def load_rgb(p):
    a = np.asarray(Image.open(p).convert("RGB"))
    if a.shape != (224, 224, 3):
        a = np.asarray(Image.open(p).convert("RGB").resize((224, 224), Image.BICUBIC))
    return a


def plism():
    if (OUT / "plism_heldout.npy").exists():
        return
    conds = json.load(open(PL / "conds.json"))
    sel = np.load(PL / "loc_idx.npy")
    idx = np.sort(np.random.default_rng(1234).choice(len(sel), 256, replace=False))
    locs = sel[idx]
    held = [c for c in conds if c.split("_")[1] in HELD_SC or c.split("_")[0] in HELD_ST]
    arr = np.zeros((len(held), len(locs), 224, 224, 3), np.uint8)
    for ci, c in enumerate(held):
        st, sc = c.split("_")
        s = np.load(PACK / f"{st}_{sc}_to_GMH_S60.npy", mmap_mode="r")
        arr[ci] = np.stack([np.asarray(s[int(i)]) for i in locs])
        print("plism", c, flush=True)
    np.save(OUT / "plism_heldout.npy", arr)
    json.dump({"conditions": held, "locations": locs.tolist()}, open(OUT / "plism_heldout.json", "w"))


def crc():
    if (OUT / "crc.npy").exists():
        return
    z = np.load(CRC)
    with Pool(32) as p:
        ims = p.map(load_rgb, list(z["files"]), chunksize=64)
    np.save(OUT / "crc.npy", np.stack(ims))
    np.savez(OUT / "crc_meta.npz", label=z["label"], split=z["split"], classes=z["classes"], files=z["files"])
    print("crc", len(ims), flush=True)


def nct():
    if (OUT / "nct_labels.npy").exists():
        return
    classes = sorted(os.listdir(NCT))
    paths, labels = [], []
    for ci, c in enumerate(classes):
        ps = sorted(glob.glob(f"{NCT}/{c}/*.tif"))
        paths += ps
        labels += [ci] * len(ps)
    mm = np.lib.format.open_memmap(OUT / "nct.npy", mode="w+", dtype=np.uint8, shape=(len(paths), 224, 224, 3))
    B = 4096
    with Pool(32) as p:
        for b in range(0, len(paths), B):
            mm[b:b + B] = np.stack(p.map(load_rgb, paths[b:b + B], chunksize=64))
            print("nct", b, len(paths), flush=True)
    mm.flush()
    np.save(OUT / "nct_classes.npy", np.array(classes))
    np.save(OUT / "nct_labels.npy", np.array(labels))


if __name__ == "__main__":
    for st in sys.argv[1:] or ["plism", "crc", "nct"]:
        globals()[st]()
    print("CACHE DONE", flush=True)
