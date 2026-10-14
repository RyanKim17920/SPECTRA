"""TCGA site shortcut + unseen-site cancer-type accuracy, base CONCH vs SPECTRA-CONCH.

Input: <emb>/<slide>.npz from conch_slides_embed.py (true 20x, 448px, <=500 tiles/slide).
Slide embedding = mean of tile features (also max, meanmax as in code/clustering/slide_level.py).

Metrics per model (same definitions as code/clustering/slide_level.py; LOO kNN on standardised +
L2-normalised slide embeddings, same-patient neighbours excluded):
  type_acc_k{1,5,10} / type_bacc_k*   cancer-type kNN across the 4 types.
  site_*                              WITHIN-TYPE tissue-source-site kNN (k=5) over sites with >= 5
                                      slides; acc, balanced acc, majority baseline, label-shuffle null.
                                      LOWER = weaker site shortcut.
  lp_type_grouped                     slide-level logistic regression for cancer type with
                                      site-grouped CV: folds split by TSS, so every test slide comes
                                      from a hospital never seen in training ("unseen hospitals").
                                      10 random group->fold assignments x 5 folds; mean +- sd.
  lp_type_strat                       same probe, plain stratified 5-fold (sites shared) - contrast.
  lp_site_within_type                 logistic regression predicting site within each type
                                      (stratified 5-fold), mean balanced acc - a linear site shortcut.

    python scripts/conch_tcga_slide_level.py --emb DIR --out results.json
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def norm(X):
    X = X - X.mean(0); X = X / (X.std(0) + 1e-6); return X / np.linalg.norm(X, axis=1, keepdims=True)


def knn_pred(Z, y, groups, k):
    S = Z @ Z.T; S[groups[:, None] == groups[None, :]] = -9
    nn = np.argsort(-S, 1)[:, :k]
    return np.array([collections.Counter(y[r]).most_common(1)[0][0] for r in nn])


def bal(y, p):
    return float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)]))


def lp_cv(X, y, folds, C):
    pred = np.empty_like(y)
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(X[te]))
    return float((pred == y).mean()), bal(y, pred)


def group_folds(groups, y, n, seed):
    """Random assignment of whole sites to n folds, balanced by slide count (greedy)."""
    rng = np.random.default_rng(seed); g = np.unique(groups); rng.shuffle(g)
    size = collections.Counter(groups); load = np.zeros(n); fold_of = {}
    for s in sorted(g, key=lambda s: -size[s] + rng.random() * 0.5):
        f = int(np.argmin(load)); fold_of[s] = f; load[f] += size[s]
    fa = np.array([fold_of[s] for s in groups])
    return [(np.where(fa != f)[0], np.where(fa == f)[0]) for f in range(n)]


def analyse(X, typ, tss, pat, rng):
    out = {}
    Z = norm(X)
    for k in (1, 5, 10):
        p = knn_pred(Z, typ, pat, k); out[f"type_acc_k{k}"] = float((p == typ).mean()); out[f"type_bacc_k{k}"] = bal(typ, p)
    site = {}
    for t in sorted(set(typ)):
        ii = np.where(typ == t)[0]; cnt = collections.Counter(tss[ii]); keep = ii[np.array([cnt[s] >= 5 for s in tss[ii]])]
        if len(set(tss[keep])) < 2:
            continue
        Zt = norm(X[keep]); yt = tss[keep]; p = knn_pred(Zt, yt, pat[keep], 5)
        null = np.mean([(lambda yp: (knn_pred(Zt, yp, pat[keep], 5) == yp).mean())(rng.permutation(yt)) for _ in range(20)])
        sk = StratifiedKFold(5, shuffle=True, random_state=0)
        lpa, lpb = lp_cv(X[keep], yt, list(sk.split(X[keep], yt)), 0.1)
        site[t] = dict(n=int(len(keep)), nsites=len(set(yt)), acc=float((p == yt).mean()), bacc=bal(yt, p),
                       majority=float(collections.Counter(yt).most_common(1)[0][1] / len(yt)), null_acc=float(null),
                       lp_acc=lpa, lp_bacc=lpb)
    out["site"] = site
    out["site_acc_mean"] = float(np.mean([s["acc"] for s in site.values()]))
    out["site_bacc_mean"] = float(np.mean([s["bacc"] for s in site.values()]))
    out["site_null_mean"] = float(np.mean([s["null_acc"] for s in site.values()]))
    out["site_lp_bacc_mean"] = float(np.mean([s["lp_bacc"] for s in site.values()]))
    for C in (0.01, 0.1, 1.0):
        g = [lp_cv(X, typ, group_folds(tss, typ, 5, s), C) for s in range(10)]
        out[f"lp_type_grouped_C{C}"] = dict(acc=float(np.mean([a for a, _ in g])), acc_sd=float(np.std([a for a, _ in g])),
                                             bacc=float(np.mean([b for _, b in g])))
        st = [lp_cv(X, typ, list(StratifiedKFold(5, shuffle=True, random_state=s).split(X, typ)), C) for s in range(10)]
        out[f"lp_type_strat_C{C}"] = dict(acc=float(np.mean([a for a, _ in st])), bacc=float(np.mean([b for _, b in st])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", type=Path, default=Path("/data/ryan.kim/ashby/scratch/spectra_conch_tcga"))
    ap.add_argument("--out", type=Path, default=Path("/data/ryan.kim/ashby/results/spectra_conch/tcga/slide_level.json"))
    a = ap.parse_args()
    files = sorted(a.emb.glob("*.npz"))
    zs = [np.load(f) for f in files]
    models = sorted(set.intersection(*[{k[2:] for k in z.files if k.startswith("f_")} for z in zs]))
    typ = np.array([str(z["type"]) for z in zs]); tss = np.array([str(z["tss"]) for z in zs]); pat = np.array([str(z["patient"]) for z in zs])
    print(len(zs), "slides", collections.Counter(typ), "models", models, flush=True)
    res = dict(n_slides=len(zs), types=dict(collections.Counter(typ)), n_sites=len(set(tss)),
               mean_tiles=float(np.mean([len(z["xy"]) for z in zs])))
    for m in models:
        F = [z[f"f_{m}"].astype(np.float32) for z in zs]
        for pool in ("mean", "max"):
            X = np.stack([f.mean(0) if pool == "mean" else f.max(0) for f in F])
            r = analyse(X, typ, tss, pat, np.random.default_rng(0)); res[f"{m}|{pool}"] = r
            g = r["lp_type_grouped_C0.1"]
            print(f"{m:12s} {pool:4s} type_k10={r['type_acc_k10']:.3f} site_acc={r['site_acc_mean']:.3f} "
                  f"(bacc {r['site_bacc_mean']:.3f}, null {r['site_null_mean']:.3f}, lp_bacc {r['site_lp_bacc_mean']:.3f}) "
                  f"LP-type grouped={g['acc']:.3f}+-{g['acc_sd']:.3f} strat={r['lp_type_strat_C0.1']['acc']:.3f}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(res, indent=1))
    print("SLIDE LEVEL DONE", a.out)


if __name__ == "__main__":
    main()
