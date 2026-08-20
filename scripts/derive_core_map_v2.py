"""Tile -> tissue map for PLISM, coordinate-first.

The TMA cores are physically separated on the slide, so 8-connectivity on the
tile lattice recovers them EXACTLY -- no model, no clustering, no k to choose.
That yields ~87 cores. But the slide holds only 46 tissue types, i.e. roughly
two replicate cores per tissue, and two cores of the SAME tissue are exactly as
false a negative as two tiles of the same core. So a second pass merges cores by
mean-embedding similarity down to --n-tissues groups.
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from waivphaet.models.encoder import build_encoder


def coord_cores(keys):
    rc = np.array([[int(m.group(1)), int(m.group(2))]
                   for m in (re.match(r"tile_16_(\d+)_(\d+)", x) for x in keys)])
    r0, c0 = rc.min(0)
    R, C = rc.max(0) - rc.min(0) + 1
    occ = np.zeros((R, C), bool)
    occ[rc[:, 0] - r0, rc[:, 1] - c0] = True
    lab, n = ndimage.label(occ, structure=np.ones((3, 3), bool))
    tile_lab = lab[rc[:, 0] - r0, rc[:, 1] - c0]
    sizes = np.bincount(tile_lab)
    # Fold fragments (< 25% of median core) into the nearest real core.
    med = np.median(sizes[1:][sizes[1:] > 0])
    small = {i for i in range(1, n + 1) if sizes[i] < 0.25 * med}
    if small:
        big = np.array([i for i in range(1, n + 1) if i not in small])
        cent = {i: rc[tile_lab == i].mean(0) for i in range(1, n + 1)}
        for i in small:
            d = [np.linalg.norm(cent[i] - cent[j]) for j in big]
            tile_lab[tile_lab == i] = big[int(np.argmin(d))]
    _, tile_lab = np.unique(tile_lab, return_inverse=True)
    return tile_lab.astype(np.int32), rc


def embed_all(packed_dir, device, bs, backbone):
    model = build_encoder(backbone=backbone, use_lora=False, proj_out_dim=512, pooling="clsmean")
    model.eval().to(device)
    ref = np.load(str(Path(packed_dir) / "GMH_S60_to_GMH_S60.npy"), mmap_mode="r")
    N = ref.shape[0]
    out = None
    with torch.no_grad():
        for s in range(0, N, bs):
            e = min(s + bs, N)
            imgs = torch.from_numpy(np.ascontiguousarray(ref[s:e])).to(device)
            emb = model.embed(imgs).float()
            if out is None:
                out = np.empty((N, emb.shape[1]), np.float32)
            out[s:e] = emb.cpu().numpy()
            if (s // bs) % 20 == 0:
                print(f"[embed] {e}/{N}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed-dir", default="/data/plism/repacked")
    ap.add_argument("--backbone", default="paige-ai/Virchow2")
    ap.add_argument("--n-tissues", type=int, default=46)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-labels", default="/admin/home/ryan.kim/waiv/runs/.plism_core_labels.npy")
    a = ap.parse_args()

    keys = json.load(open(Path(a.packed_dir) / "keys.json"))
    cores, rc = coord_cores(keys)
    n_cores = int(cores.max()) + 1
    print(f"[coord] {n_cores} cores from 8-connectivity; sizes "
          f"min={np.bincount(cores).min()} max={np.bincount(cores).max()}", flush=True)

    print(f"[embed] {a.backbone}", flush=True)
    emb = embed_all(a.packed_dir, a.device, a.batch_size, a.backbone)
    np.save("/admin/home/ryan.kim/waiv/runs/.plism_ref_emb.npy", emb)

    # Mean embedding per core, L2-normalised, then agglomerative merge to n_tissues.
    cm = np.stack([emb[cores == i].mean(0) for i in range(n_cores)])
    cm /= np.linalg.norm(cm, axis=1, keepdims=True) + 1e-8
    Z = linkage(pdist(cm, metric="cosine"), method="average")
    core2tis = fcluster(Z, t=a.n_tissues, criterion="maxclust") - 1
    labels = core2tis[cores].astype(np.int32)

    # Validation: spatial contiguity of the FINAL labels vs a random-label null.
    r0, c0 = rc.min(0)
    R, C = rc.max(0) - rc.min(0) + 1
    def contig(lb):
        g = -np.ones((R, C), np.int64)
        g[rc[:, 0] - r0, rc[:, 1] - c0] = lb
        same = tot = 0
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            a_ = g[max(0,dr):R+min(0,dr), max(0,dc):C+min(0,dc)]
            b_ = g[max(0,-dr):R+min(0,-dr), max(0,-dc):C+min(0,-dc)]
            m = (a_ >= 0) & (b_ >= 0)
            same += int((a_[m] == b_[m]).sum()); tot += int(m.sum())
        return same / max(tot, 1)
    rng = np.random.default_rng(0)
    null = float(np.mean([contig(rng.permutation(labels)) for _ in range(3)]))
    got = contig(labels)
    sz = np.bincount(labels)
    fn = float(sum((s / len(labels)) * ((s - 1) / (len(labels) - 1)) for s in sz))

    meta = {"n_cores_coord": n_cores, "n_tissues": int(a.n_tissues),
            "backbone": a.backbone, "contiguity": round(got, 4),
            "contiguity_null": round(null, 4), "lift": round(got / max(null, 1e-9), 2),
            "tissue_sizes_min": int(sz.min()), "tissue_sizes_max": int(sz.max()),
            "measured_fn_rate": round(fn, 5), "cores_per_tissue": round(n_cores / a.n_tissues, 2)}
    np.save(a.out_labels, labels)
    Path(str(a.out_labels).replace(".npy", "_meta.json")).write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
