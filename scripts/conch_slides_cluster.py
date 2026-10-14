"""Per-slide clustering coherence, base CONCH vs SPECTRA-CONCH (code/conch15/slides_cluster.py protocol).

Input: <emb>/<slide>.npz from conch_slides_embed.py --max-tiles 0 (all tissue tiles, true 20x 448px)
for the 8 slides of results/tests/clustering/per_slide_selection.json. Per model: center + L2 + PCA50
(code/clustering/common.py prep), Leiden (target 6-10 clusters) and k-means k=8; silhouette (cosine,
<=10k tiles) and 8-neighbour spatial coherence (vs label-shuffled), ARI of each model's Leiden vs base.

    python scripts/conch_slides_cluster.py --emb DIR --out results.json [--maps base,anc1_s300]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data/ryan.kim/ashby/code/clustering")
from common import MAN, PAL, leiden, openslide, overlay, prep, relabel_by_size  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import adjusted_rand_score, silhouette_score  # noqa: E402

SLIDES = json.load(open("/data/ryan.kim/ashby/results/tests/clustering/per_slide_selection.json"))


def coherence(xy, lab, rng):
    idx = {tuple(p): i for i, p in enumerate(xy.tolist())}

    def run(l):
        v = []
        for i, (x, y) in enumerate(xy.tolist()):
            nb = [idx[(x + dx, y + dy)] for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx or dy) and (x + dx, y + dy) in idx]
            if nb:
                v.append(np.mean(l[nb] == l[i]))
        return float(np.mean(v))
    return run(lab), run(rng.permutation(lab))


def sil(P, lab, rng):
    if len(np.unique(lab)) < 2:
        return float("nan")
    ii = rng.choice(len(P), min(len(P), 10000), replace=False)
    return float(silhouette_score(P[ii], lab[ii], metric="cosine"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", type=Path, default=Path("/data/ryan.kim/ashby/scratch/spectra_conch_cluster8"))
    ap.add_argument("--out", type=Path, default=Path("/data/ryan.kim/ashby/results/spectra_conch/cluster8/slides_cluster.json"))
    ap.add_argument("--maps", default="")
    a = ap.parse_args(); a.out.parent.mkdir(parents=True, exist_ok=True)
    res = {}; models = None
    for slide in SLIDES:
        z = np.load(a.emb / f"{slide}.npz"); xy = z["xy"]; S0 = int(z["S0"])
        ms = sorted(k[2:] for k in z.files if k.startswith("f_")); models = models or ms
        row = dict(type=str(z["type"]), n_tiles=int(len(xy))); labs = {}
        for m in models:
            P, _, _ = prep(z[f"f_{m}"].astype(np.float32)); rng = np.random.default_rng(0)
            ld, rs = leiden(P); km = relabel_by_size(KMeans(8, n_init=10, random_state=0).fit_predict(P))
            o = {}
            for name, lab in (("leiden", ld), ("kmeans8", km)):
                c, cs = coherence(xy, lab, rng)
                o[name] = dict(n_clusters=int(lab.max() + 1), silhouette=sil(P, lab, rng), coherence=c, coherence_shuffled=cs)
            o["leiden"]["res"] = rs; row[m] = o; labs[m] = ld
        for m in models:
            row[m]["ari_vs_base_leiden"] = float(adjusted_rand_score(labs["base"], labs[m]))
        res[slide] = row
        print(slide, row["type"], len(xy), {m: (round(row[m]["leiden"]["silhouette"], 3), round(row[m]["leiden"]["coherence"], 3),
                                                round(row[m]["kmeans8"]["silhouette"], 3), round(row[m]["kmeans8"]["coherence"], 3)) for m in models}, flush=True)
        if a.maps:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            mm = a.maps.split(","); r = MAN[slide]; sl = openslide.OpenSlide(r["svs"])
            th = sl.get_thumbnail((1600, 1600)).convert("RGB"); d = max(sl.dimensions) / th.size[0]
            fig, ax = plt.subplots(1, 1 + len(mm), figsize=(9 * (1 + len(mm)), 9)); ax[0].imshow(th); ax[0].set_title(f"{r['type']} {slide}")
            for x, m in zip(ax[1:], mm):
                x.imshow(overlay(th, d, S0, xy, labs[m])); x.set_title(f"{m} leiden sil={row[m]['leiden']['silhouette']:.3f} coh={row[m]['leiden']['coherence']:.3f}")
            for x in ax: x.axis("off")
            plt.tight_layout(); plt.savefig(a.out.parent / f"{slide}_map.png", dpi=50); plt.close()
    keys = ("n_clusters", "silhouette", "coherence", "coherence_shuffled")
    res["_mean"] = {m: {n: {k: float(np.mean([res[s][m][n][k] for s in SLIDES])) for k in keys} for n in ("leiden", "kmeans8")} for m in models}
    for m in models:
        res["_mean"][m]["ari_vs_base_leiden"] = float(np.mean([res[s][m]["ari_vs_base_leiden"] for s in SLIDES]))
    a.out.write_text(json.dumps(res, indent=1)); print(json.dumps(res["_mean"], indent=1)); print("CLUSTER DONE")


if __name__ == "__main__":
    main()
