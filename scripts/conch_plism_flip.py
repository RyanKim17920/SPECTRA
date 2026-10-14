"""PLISM prediction stability for base CONCH vs SPECTRA-CONCH checkpoints.

Protocol (code/plism/analyze.py flip_stats, restricted to the conditions SPECTRA never trained on):
  * Embeddings: encode_image(proj_contrast=False, normalize=False) via the conch_eval tensor path
    (uint8 224 -> /255 -> OpenAI norm -> bicubic 448), identical for base and LoRA.
  * Classifier: per model, logistic regression (C=0.1, standardised, max_iter=300) trained on THAT
    model's NCT-CRC-HE-100K embeddings (scored on CRC-VAL-HE-7K for sanity).
  * PLISM: the 1000 tissue locations of scratch/plism/loc_idx.npy x 91 conditions (registered to
    GMH_S60). Predicted 9-class label per (condition, location); also TUM-vs-rest.
  * HELD-OUT = scanner in {GT450, S210} or stain in {HRH, KR, MY} (train_lora --heldout-*), 41 of 91.
    Pairwise flip = mean over condition pairs (BOTH held-out) of frac locations whose label differs;
    cross-scanner = same stain, different scanner; cross-stain = same scanner, different stain.
    Per-location any-flip = frac locations whose label is not constant across a group of held-out
    conditions sharing the stain (cross-scanner groups) / the scanner (cross-stain groups),
    averaged over groups with >= 2 members; plus across all 41 held-out conditions.
  * Paired bootstrap over locations (2000x) for model - base differences of the pairwise rates.
  * The same numbers over all 91 conditions are reported for comparison with analyze.py.

    python scripts/conch_plism_flip.py --models base anc1_s100=<run>/step_0000100/adapter ...
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conch_eval import CACHE, HELD_SC, HELD_ST, embed  # noqa: E402

PACK = Path("/data/plism/repacked")
PL = Path("/data/ryan.kim/ashby/scratch/plism")
dev = "cuda"


def load_base():
    from spectra.models.conch_visual import ConchVisual, load_conch_full
    return ConchVisual(load_conch_full().visual).to(dev).eval()


def embed_model(name, adapter, base, out):
    fp = out / f"{name}_plism.npy"
    if fp.exists() and (out / f"{name}_nct.npy").exists():
        return
    if adapter:
        from peft import PeftModel
        m = PeftModel.from_pretrained(copy.deepcopy(base), resolve_adapter(adapter)).merge_and_unload().to(dev).eval()
    else:
        m = base
    conds = json.load(open(PL / "conds.json")); loc = np.load(PL / "loc_idx.npy")
    X = np.zeros((len(conds), len(loc), 512), np.float16)
    for ci, c in enumerate(conds):
        a = np.load(PACK / f"{c}_to_GMH_S60.npy", mmap_mode="r")
        X[ci] = embed(m, a[loc]).numpy().astype(np.float16)
    nct = np.load(CACHE / "nct.npy", mmap_mode="r")
    Xn = embed(m, nct).numpy().astype(np.float16)
    crc = np.load(CACHE / "crc.npy", mmap_mode="r"); meta = np.load(CACHE / "crc_meta.npz")
    Xv = embed(m, crc[np.where(meta["split"] == 0)[0]]).numpy().astype(np.float16)
    np.save(out / f"{name}_nct.npy", Xn); np.save(out / f"{name}_crcval.npy", Xv); np.save(fp, X)
    print("embedded", name, flush=True)
    if m is not base:
        del m; torch.cuda.empty_cache()


def predict(name, out):
    from sklearn.linear_model import LogisticRegression
    Xn = np.load(out / f"{name}_nct.npy").astype(np.float32); yn = np.load(CACHE / "nct_labels.npy")
    meta = np.load(CACHE / "crc_meta.npz"); yv = meta["label"][meta["split"] == 0]
    mu, sd = Xn.mean(0), Xn.std(0) + 1e-6
    clf = LogisticRegression(max_iter=300, C=0.1, n_jobs=16).fit((Xn - mu) / sd, yn)
    va = float(clf.score((np.load(out / f"{name}_crcval.npy").astype(np.float32) - mu) / sd, yv))
    X = np.load(out / f"{name}_plism.npy").astype(np.float32); C, L, D = X.shape
    pr = clf.predict_proba((X.reshape(-1, D) - mu) / sd).reshape(C, L, -1)
    return pr.argmax(-1), pr, va


def groups(conds, sel):
    st = [c.split("_")[0] for c in conds]; sc = [c.split("_")[1] for c in conds]
    cs = [(a, b) for a in sel for b in sel if a < b and st[a] == st[b] and sc[a] != sc[b]]
    ct = [(a, b) for a in sel for b in sel if a < b and sc[a] == sc[b] and st[a] != st[b]]
    g_st = [[i for i in sel if st[i] == s] for s in sorted({st[i] for i in sel})]
    g_sc = [[i for i in sel if sc[i] == s] for s in sorted({sc[i] for i in sel})]
    return cs, ct, [g for g in g_st if len(g) >= 2], [g for g in g_sc if len(g) >= 2]


def per_loc(P, pairs):
    """(L,) per-location flip frac averaged over pairs."""
    return np.mean([(P[a] != P[b]) for a, b in pairs], 0)


def anyflip(P, gs):
    return np.mean([(P[g] != P[g[0]][None]).any(0) for g in gs], 0)


def flip_stats(P, conds, sel):
    cs, ct, g_st, g_sc = groups(conds, sel)
    r = dict(n_conditions=len(sel), n_pairs_cross_scanner=len(cs), n_pairs_cross_stain=len(ct),
             n_groups_fixed_stain=len(g_st), n_groups_fixed_scanner=len(g_sc))
    r["pairwise_flip_cross_scanner"] = float(per_loc(P, cs).mean())
    r["pairwise_flip_cross_stain"] = float(per_loc(P, ct).mean())
    r["loc_anyflip_cross_scanner"] = float(anyflip(P, g_st).mean())
    r["loc_anyflip_cross_stain"] = float(anyflip(P, g_sc).mean())
    r["loc_anyflip_all_selected"] = float((P[sel] != P[sel[0]][None]).any(0).mean())
    return r


def resolve_adapter(path):
    """Accept a run step dir (<run>/step_N, containing adapter/) or the adapter dir itself."""
    from pathlib import Path as _P
    p = _P(path)
    if (p / "adapter" / "adapter_config.json").exists():
        return str(p / "adapter")
    if (p / "adapter_config.json").exists():
        return str(p)
    raise FileNotFoundError(f"no PEFT adapter at {p} (expected {p}/adapter_config.json or {p}/adapter/)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name[=adapter_dir]; 'base' = base CONCH")
    ap.add_argument("--out", type=Path, default=Path("/data/ryan.kim/ashby/results/spectra_conch/plism_flip"))
    a = ap.parse_args()
    feats = a.out / "feats"; feats.mkdir(parents=True, exist_ok=True)
    specs = [(s.split("=", 1)[0], s.split("=", 1)[1] if "=" in s else None) for s in a.models]
    if "base" not in [n for n, _ in specs]:
        specs = [("base", None)] + specs
    base = load_base()
    for n, ad in specs:
        embed_model(n, ad, base, feats)
    conds = json.load(open(PL / "conds.json"))
    held = [i for i, c in enumerate(conds) if c.split("_")[1] in HELD_SC or c.split("_")[0] in HELD_ST]
    allc = list(range(len(conds)))
    tum = 8  # NCT classes: ADI BACK DEB LYM MUC MUS NORM STR TUM
    assert list(np.load(CACHE / "nct_classes.npy"))[tum] == "TUM"
    res_p = a.out / "flip_results.json"
    res = json.loads(res_p.read_text()) if res_p.exists() else {}
    preds = {}
    for n, _ in specs:
        P, pr, va = predict(n, feats); preds[n] = P
        r = dict(crc_val7k_acc=va, mean_max_prob_heldout=float(pr[held].max(-1).mean()),
                 pred_dist_heldout={str(k): float((P[held] == k).mean()) for k in range(pr.shape[-1])})
        for tag, PP in (("9class", P), ("tum_vs_rest", (P == tum).astype(int))):
            r[f"heldout_{tag}"] = flip_stats(PP, conds, held)
            r[f"all91_{tag}"] = flip_stats(PP, conds, allc)
        res[n] = r
    # paired bootstrap over locations vs base (held-out, pairwise rates)
    rng = np.random.default_rng(0); cs, ct, g_st, g_sc = groups(conds, held)
    L = preds["base"].shape[1]; B = rng.integers(0, L, (2000, L))
    for n in preds:
        if n == "base":
            continue
        d = {}
        for tag, f in (("9class", lambda P: P), ("tum_vs_rest", lambda P: (P == tum).astype(int))):
            Pb, Pm = f(preds["base"]), f(preds[n])
            for ax, pairs in (("cross_scanner", cs), ("cross_stain", ct)):
                diff = per_loc(Pm, pairs) - per_loc(Pb, pairs)
                bs = diff[B].mean(1)
                d[f"{tag}_{ax}"] = dict(delta=float(diff.mean()), ci95=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))])
        res[n]["heldout_delta_vs_base"] = d
    res_p.write_text(json.dumps(res, indent=1))
    for n in preds:
        h, t = res[n]["heldout_9class"], res[n]["heldout_tum_vs_rest"]
        print(f"{n:14s} crc={res[n]['crc_val7k_acc']:.3f} | 9c pw sc={h['pairwise_flip_cross_scanner']:.3f} st={h['pairwise_flip_cross_stain']:.3f}"
              f" any sc={h['loc_anyflip_cross_scanner']:.3f} st={h['loc_anyflip_cross_stain']:.3f} all={h['loc_anyflip_all_selected']:.3f}"
              f" | TUM pw sc={t['pairwise_flip_cross_scanner']:.3f} st={t['pairwise_flip_cross_stain']:.3f}"
              f" | all91 9c pw sc={res[n]['all91_9class']['pairwise_flip_cross_scanner']:.3f} st={res[n]['all91_9class']['pairwise_flip_cross_stain']:.3f}", flush=True)
    print("FLIP DONE", res_p, flush=True)


if __name__ == "__main__":
    main()
