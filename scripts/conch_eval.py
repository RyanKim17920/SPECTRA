"""Per-checkpoint evaluation of SPECTRA-CONCH runs (and the base model as step 0).

Metrics, all on the exported vector encode_image(proj_contrast=False, normalize=False):
  plism_*      PLISM held-out retrieval, embed_probe.pair_stats protocol as reproduced in
               code/plism/analyze.py:spectra_pairwise (held-out conditions only, n=256
               locations): matched-tile top-1 among the other condition's tiles, averaged
               over cross-scanner (same stain) / cross-stain (same scanner) pairs.
  drift_cos    mean cos(f_lora(x), f_base(x)) on 2000 CRC tiles (CRC-VAL-HE-7K + top-up).
  zs_acc       CONCH zero-shot 9-way on the 7420-tile CRC set, 18 templates x multi-name
               ensemble (code/concept_eval.py 'ens_templates_ens_names'); image side =
               normalize(f_lora(x) @ proj_contrast), text side = the UNTOUCHED base tower.
  probe_acc    logistic regression (C=0.1, standardised, max_iter=300) trained on
               NCT-CRC-HE-100K, scored on CRC-VAL-HE-7K (code/plism/analyze.py).

Preprocessing: uint8 224px tile -> /255 -> CONCH (OpenAI) normalisation -> bicubic to
448 -> model, bf16 autocast. Identical for base and LoRA (it is the training path); the
base row is reported so the harness can be checked against the PIL-preprocess reference
(0.880 / 0.554 / 0.753 / 0.964).

    python scripts/conch_eval.py --run-dir /data/ryan.kim/ashby/runs/spectra_conch/anc1 [--base]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CACHE = Path(os.environ.get("CONCH_EVAL_CACHE", "/data/ryan.kim/ashby/scratch/spectra_conch_eval"))
dev = "cuda"

TEMPL = ['{}.', 'a photomicrograph showing {}.', 'a photomicrograph of {}.', 'an image of {}.', 'an image showing {}.',
         'an example of {}.', '{} is shown.', 'this is {}.', 'there is {}.', 'a histopathological image showing {}.',
         'a histopathological image of {}.', 'a histopathological photograph of {}.', 'a histopathological photograph showing {}.',
         'shows {}.', 'presence of {}.', '{} is present.', 'an H&E stained image of {}.', 'an H&E image of {}.']
ZS = {'ADI': ['adipose tissue', 'fat cells', 'adipocytes'], 'BACK': ['background', 'empty glass slide'],
      'DEB': ['necrosis', 'necrotic debris', 'debris'], 'LYM': ['lymphocytes', 'dense lymphocytic infiltrate'],
      'MUC': ['mucin', 'extracellular mucus'], 'MUS': ['smooth muscle', 'muscularis propria'],
      'NORM': ['normal colonic mucosa', 'normal colon glands'], 'STR': ['cancer-associated stroma', 'fibrous stroma'],
      'TUM': ['colorectal adenocarcinoma epithelium', 'tumor epithelium', 'adenocarcinoma']}
HELD_SC, HELD_ST = {"GT450", "S210"}, {"HRH", "KR", "MY"}


def load_models():
    from spectra.models.conch_visual import ConchVisual, load_conch_full
    full = load_conch_full()
    base = ConchVisual(full.visual).eval()  # moved to GPU after the text classifier
    return full, base


@torch.no_grad()
def text_classifier(full, classes):
    from conch.open_clip_custom import get_tokenizer
    tok = get_tokenizer()
    full = full.to(dev)
    out = []
    for c in classes:
        s = [t.format(p) for p in ZS[c] for t in TEMPL]
        tk = tok(s, max_length=127, add_special_tokens=True, return_token_type_ids=False, truncation=True,
                 padding='max_length', return_tensors='pt')
        ids = F.pad(tk['input_ids'], (0, 1), value=tok.pad_token_id).to(dev)
        e = full.encode_text(ids, normalize=True).float().mean(0)
        out.append(F.normalize(e, dim=-1))
    full.cpu()
    return torch.stack(out).cpu()  # (9, 512)


MEAN = torch.tensor((0.48145466, 0.4578275, 0.40821073)).view(1, 3, 1, 1)
STD = torch.tensor((0.26862954, 0.26130258, 0.27577711)).view(1, 3, 1, 1)


@torch.no_grad()
def embed(model, arr, bs=256):
    """arr: (N,224,224,3) uint8 (np / memmap) -> (N,512) float32 CPU."""
    m, s = MEAN.to(dev), STD.to(dev)
    out = []
    for b in range(0, len(arr), bs):
        x = torch.from_numpy(np.ascontiguousarray(arr[b:b + bs])).to(dev, non_blocking=True)
        x = (x.permute(0, 3, 1, 2).float() / 255 - m) / s
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = model.pooled(x)
        out.append(f.float().cpu())
    return torch.cat(out)


def pairwise(X, conds):
    """analyze.py spectra_pairwise on held-out conditions (all of `conds` are held-out)."""
    st = [c.split("_")[0] for c in conds]; sc = [c.split("_")[1] for c in conds]
    L = X.shape[1]; tgt = torch.arange(L); out = {"cross_scanner": [], "cross_stain": []}
    for a in range(len(conds)):
        for b in range(a + 1, len(conds)):
            if st[a] == st[b] and sc[a] != sc[b]: ax = "cross_scanner"
            elif sc[a] == sc[b] and st[a] != st[b]: ax = "cross_stain"
            else: continue
            sim = X[a] @ X[b].T
            out[ax].append((float((sim.argmax(1) == tgt).float().mean()),
                            float((sim.topk(5, 1).indices == tgt[:, None]).any(1).float().mean())))
    return {k: dict(top1=float(np.mean([v[0] for v in vs])), top5=float(np.mean([v[1] for v in vs])), n_pairs=len(vs))
            for k, vs in out.items()}


class Data:
    def __init__(self):
        self.plism = np.load(CACHE / "plism_heldout.npy", mmap_mode="r")
        self.plism_conds = json.load(open(CACHE / "plism_heldout.json"))["conditions"]
        assert all(c.split("_")[1] in HELD_SC or c.split("_")[0] in HELD_ST for c in self.plism_conds)
        self.crc = np.load(CACHE / "crc.npy", mmap_mode="r")
        meta = np.load(CACHE / "crc_meta.npz")
        self.crc_y, self.crc_split, self.classes = meta["label"], meta["split"], list(meta["classes"])
        self.drift_idx = np.sort(np.random.default_rng(0).choice(len(self.crc), 2000, replace=False))
        self.nct = np.load(CACHE / "nct.npy", mmap_mode="r")
        self.nct_y = np.load(CACHE / "nct_labels.npy")
        assert list(np.load(CACHE / "nct_classes.npy")) == self.classes


def evaluate(model, data, T, base_ref=None, probe=True):
    from sklearn.linear_model import LogisticRegression
    t0 = time.time(); r = {}
    C, L = data.plism.shape[:2]
    Xp = embed(model, data.plism.reshape(C * L, 224, 224, 3)).view(C, L, -1)
    pw = pairwise(F.normalize(Xp, dim=-1), data.plism_conds)
    r["plism_cross_scanner_top1"] = pw["cross_scanner"]["top1"]; r["plism_cross_stain_top1"] = pw["cross_stain"]["top1"]
    r["plism_cross_scanner_top5"] = pw["cross_scanner"]["top5"]; r["plism_cross_stain_top5"] = pw["cross_stain"]["top5"]
    r["plism_n_pairs"] = {k: v["n_pairs"] for k, v in pw.items()}
    Ec = embed(model, data.crc)  # (7420, 512) raw export
    if base_ref is not None:
        r["drift_cos"] = float(F.cosine_similarity(Ec[data.drift_idx], base_ref["crc"][data.drift_idx], dim=-1).mean())
        r["drift_cos_p05"] = float(torch.quantile(F.cosine_similarity(Ec[data.drift_idx], base_ref["crc"][data.drift_idx], dim=-1), 0.05))
        r["drift_cos_plism"] = float(F.cosine_similarity(Xp.reshape(C * L, -1), base_ref["plism"].reshape(C * L, -1), dim=-1).mean())
    with torch.no_grad():
        P = F.normalize(model.project(Ec.to(dev)), dim=-1).float().cpu()
    pr = (P @ T.T).argmax(1).numpy(); y = data.crc_y
    r["zs_acc"] = float((pr == y).mean()); m = data.crc_split == 0; r["zs_acc_val7k"] = float((pr[m] == y[m]).mean())
    if probe:
        Xn = embed(model, data.nct).numpy()
        mu, sd = Xn.mean(0), Xn.std(0) + 1e-6
        clf = LogisticRegression(max_iter=300, C=0.1, n_jobs=16).fit((Xn - mu) / sd, data.nct_y)
        Xv = Ec[m].numpy()
        r["probe_acc"] = float(clf.score((Xv - mu) / sd, y[m]))
    r["eval_s"] = time.time() - t0
    return r, {"crc": Ec, "plism": Xp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--base", action="store_true", help="also evaluate base CONCH (step 0)")
    ap.add_argument("--steps", default="", help="comma list; default all complete step_* dirs")
    ap.add_argument("--no-probe", action="store_true")
    a = ap.parse_args()
    from peft import PeftModel
    data = Data(); full, base = load_models(); T = text_classifier(full, data.classes)
    # text_classifier parks `full` on CPU, and base SHARES full.visual's modules -- move after.
    base = base.to(dev).eval()
    out_p = a.run_dir / "conch_eval.json"
    res = json.loads(out_p.read_text()) if out_p.exists() else {}
    base_cache = a.run_dir / "base_ref.pt"
    if base_cache.exists() and not a.base:
        base_ref = torch.load(base_cache)
    else:
        r0, base_ref = evaluate(base, data, T, probe=not a.no_probe)
        r0["drift_cos"] = 1.0
        torch.save(base_ref, base_cache)
        res["0"] = r0; print("step 0 (base)", json.dumps(r0), flush=True)
        out_p.write_text(json.dumps(res, indent=1))
    steps = sorted(d for d in a.run_dir.glob("step_*") if (d / "metrics.json").exists())
    if a.steps:
        want = {int(s) for s in a.steps.split(",")}; steps = [d for d in steps if int(d.name.split("_")[1]) in want]
    for d in steps:
        st = str(int(d.name.split("_")[1]))
        if st in res:
            continue
        m = PeftModel.from_pretrained(copy.deepcopy(base), str(d / "adapter")).merge_and_unload().to(dev).eval()
        r, _ = evaluate(m, data, T, base_ref=base_ref, probe=not a.no_probe)
        res[st] = r; print("step", st, json.dumps(r), flush=True)
        # merge-on-write: a concurrent eval of the same run dir may have added other steps
        cur = json.loads(out_p.read_text()) if out_p.exists() else {}
        res = {**cur, **res}
        out_p.write_text(json.dumps(res, indent=1))
        del m; torch.cuda.empty_cache()
    print("EVAL DONE", out_p, flush=True)


if __name__ == "__main__":
    main()
