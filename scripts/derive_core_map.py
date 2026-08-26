#!/usr/bin/env python3
"""Derive tile->core/organ map from phikon-v2 embeddings + k-means + spatial compactness check.

Usage: python scripts/derive_core_map.py --out-labels runs/.plism_core_labels.npy
"""
from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed-dir", type=Path, default=Path("/data/plism/repacked"))
    ap.add_argument("--out-labels", type=Path, default=Path("/admin/home/ryan.kim/waiv/runs/.plism_core_labels.npy"))
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--backbone", default="owkin/phikon-v2")
    ap.add_argument("--out-meta", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()

def embed_all(packed_dir, device, batch_size, backbone="owkin/phikon-v2"):
    import os
    os.environ.setdefault("HF_HOME", "/data/huggingface")
    # Add waiv source to path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from waivphaet.models.encoder import build_encoder
    import torchvision.transforms.functional as TF

    print("[embed] loading {backbone} base model (no LoRA)...", flush=True)
    model = build_encoder(backbone=backbone, use_lora=False, proj_out_dim=512, pooling="clsmean")
    model.eval().to(device)
    print(f"[embed] model loaded; norm_mean={model.norm_mean}, norm_std={model.norm_std}", flush=True)

    ref = np.load(str(packed_dir / "GMH_S60_to_GMH_S60.npy"), mmap_mode='r')
    N = ref.shape[0]
    print(f"[embed] embedding {N} tiles (shape {ref.shape}, dtype {ref.dtype})...", flush=True)

    # NOTE: hand-rolled ImageNet normalisation was WRONG -- WaivEncoder.tokens() applies
    # normalization_for(backbone) itself when handed uint8 NHWC. Feed it uint8 NHWC so the
    # tiles are normalised with exactly the constants training/eval uses.
    embeddings = None
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            imgs = torch.from_numpy(np.ascontiguousarray(ref[start:end])).to(device)  # uint8 NHWC
            emb = model.embed(imgs).float()  # (B, embed_dim); clsmean => 2*hidden
            if embeddings is None:
                D = int(emb.shape[1])
                print(f"[embed] embed_dim={D}", flush=True)
                embeddings = np.empty((N, D), dtype=np.float32)
            embeddings[start:end] = emb.cpu().numpy()
            if (start // batch_size) % 10 == 0:
                print(f"[embed] {end}/{N}  {time.time()-t0:.0f}s", flush=True)
    print(f"[embed] done in {time.time()-t0:.0f}s", flush=True)
    return embeddings

def parse_coords(keys):
    rows, cols = [], []
    for k in keys:
        m = re.search(r'tile_16_(\d+)_(\d+)', k)
        rows.append(int(m.group(1)))
        cols.append(int(m.group(2)))
    return np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32)

def spatial_compactness(labels, rows, cols, k):
    """Fraction of tiles whose 8-neighbours (in lattice coords) share their label."""
    # build coord->label lookup
    coord2lbl = {}
    for i, (r, c) in enumerate(zip(rows.tolist(), cols.tolist())):
        coord2lbl[(r, c)] = int(labels[i])

    neighbour_same = 0
    neighbour_total = 0
    for i, (r, c) in enumerate(zip(rows.tolist(), cols.tolist())):
        lbl = int(labels[i])
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = coord2lbl.get((r+dr, c+dc))
                if nb is not None:
                    neighbour_total += 1
                    if nb == lbl:
                        neighbour_same += 1

    frac_same = neighbour_same / max(neighbour_total, 1)

    # RANDOM-LABEL NULL for the same 8-neighbour statistic: shuffle the labels over the
    # SAME lattice (preserves cluster sizes exactly) and recompute. Without this the raw
    # frac_same is uninterpretable -- with k clusters chance alone gives ~sum(p_i^2).
    _rng_nb = np.random.default_rng(1234)
    null_fracs = []
    for _ in range(5):
        perm = _rng_nb.permutation(labels)
        c2l = {}
        for i, (r, c) in enumerate(zip(rows.tolist(), cols.tolist())):
            c2l[(r, c)] = int(perm[i])
        ns = nt = 0
        for i, (r, c) in enumerate(zip(rows.tolist(), cols.tolist())):
            lbl = int(perm[i])
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nb = c2l.get((r + dr, c + dc))
                    if nb is not None:
                        nt += 1
                        if nb == lbl:
                            ns += 1
        null_fracs.append(ns / max(nt, 1))
    null_frac_same = float(np.mean(null_fracs))

    # spatial variance per cluster vs random null
    per_cluster_var = []
    for ki in range(k):
        mask = labels == ki
        if mask.sum() < 2:
            continue
        r_k = rows[mask].astype(float)
        c_k = cols[mask].astype(float)
        per_cluster_var.append(r_k.var() + c_k.var())
    mean_cluster_var = float(np.mean(per_cluster_var)) if per_cluster_var else 0.0

    # random null: same cluster sizes, random assignment
    rng = np.random.default_rng(42)
    rand_labels = rng.permutation(labels)
    rand_var = []
    for ki in range(k):
        mask = rand_labels == ki
        if mask.sum() < 2:
            continue
        rand_var.append(rows[mask].astype(float).var() + cols[mask].astype(float).var())
    mean_rand_var = float(np.mean(rand_var)) if rand_var else 1.0

    compactness_ratio = mean_cluster_var / max(mean_rand_var, 1e-6)  # <1 = more compact than random

    return {
        "frac_neighbour_same_label": round(frac_same, 4),
        "frac_neighbour_same_label_NULL": round(null_frac_same, 4),
        "neighbour_lift_over_null": round(frac_same / max(null_frac_same, 1e-9), 2),
        "mean_cluster_spatial_var": round(mean_cluster_var, 2),
        "mean_random_spatial_var": round(mean_rand_var, 2),
        "compactness_ratio": round(compactness_ratio, 4),  # <1 = spatially contiguous
    }

def main():
    args = parse_args()
    keys = json.load(open(args.packed_dir / "keys.json"))
    rows, cols = parse_coords(keys)
    print(f"[map] grid: rows {rows.min()}-{rows.max()}, cols {cols.min()}-{cols.max()}", flush=True)

    embeddings = embed_all(args.packed_dir, args.device, args.batch_size, args.backbone)

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_n = embeddings / np.maximum(norms, 1e-8)

    best_k = None
    best_labels = None
    best_stats = None
    results = {}

    for k in [30, 46, 60, 80]:
        print(f"\n[kmeans] k={k} ...", flush=True)
        km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048, n_init=10, max_iter=300)
        labels = km.fit_predict(embeddings_n).astype(np.int32)
        sizes = np.bincount(labels, minlength=k).tolist()
        stats = spatial_compactness(labels, rows, cols, k)
        stats["cluster_sizes_min"] = int(min(sizes))
        stats["cluster_sizes_max"] = int(max(sizes))
        stats["cluster_sizes_mean"] = round(float(np.mean(sizes)), 1)
        results[k] = {"stats": stats, "labels": labels}
        print(f"  k={k}: frac_neighbour_same={stats['frac_neighbour_same_label']:.3f}  compactness_ratio={stats['compactness_ratio']:.4f}  sizes [{stats['cluster_sizes_min']}, {stats['cluster_sizes_max']}]", flush=True)

    # HARD GATE: pick best k by frac_neighbour_same_label (highest = most contiguous)
    # Must exceed 0.5 to be considered spatially contiguous
    CONTIGUITY_THRESHOLD = 0.5
    valid_ks = [(k, results[k]["stats"]["frac_neighbour_same_label"]) for k in [30,46,60,80]]
    valid_ks.sort(key=lambda x: -x[1])
    print(f"\n[gate] contiguity scores: {valid_ks}", flush=True)

    best_k_val, best_frac = valid_ks[0]
    if best_frac < CONTIGUITY_THRESHOLD:
        print(f"\n[HARD GATE FAIL] Best frac_neighbour_same={best_frac:.3f} < {CONTIGUITY_THRESHOLD}.", flush=True)
        print("Clusters are NOT spatially contiguous -- tracking texture/stain, not cores.", flush=True)
        print("DO NOT USE for false-negative masking. Stopping.", flush=True)
        sys.exit(1)

    best_k = best_k_val
    best_labels = results[best_k]["labels"]
    best_stats = results[best_k]["stats"]
    print(f"\n[map] CHOSEN k={best_k} (frac_neighbour_same={best_frac:.3f} >= {CONTIGUITY_THRESHOLD})", flush=True)

    # False-negative rate: for random anchor, fraction of other tiles sharing its cluster
    sizes = np.bincount(best_labels, minlength=best_k)
    fn_rates = []
    for ki in range(best_k):
        n_same = int(sizes[ki])
        fn_rates.append((n_same - 1) / (len(best_labels) - 1))
    mean_fn_rate = float(np.mean(fn_rates))
    print(f"[map] mean false-negative rate: {mean_fn_rate:.4f} ({mean_fn_rate*100:.2f}%) vs 1/46={1/46:.4f}", flush=True)

    # Save
    args.out_labels.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(args.out_labels), best_labels)

    meta = {
        "k": best_k,
        "n_tiles": int(len(best_labels)),
        "contiguity_threshold": CONTIGUITY_THRESHOLD,
        "chosen_stats": best_stats,
        "all_k_stats": {k: results[k]["stats"] for k in [30,46,60,80]},
        "mean_fn_rate": round(mean_fn_rate, 5),
        "expected_fn_rate_1_over_46": round(1/46, 5),
        "cluster_sizes": np.bincount(best_labels, minlength=best_k).tolist(),
    }
    meta_path = args.out_labels.parent / ".plism_core_labels_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[map] saved labels -> {args.out_labels}", flush=True)
    print(f"[map] saved meta   -> {meta_path}", flush=True)
    print(json.dumps(meta, indent=2), flush=True)

if __name__ == "__main__":
    main()
