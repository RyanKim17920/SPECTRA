"""PathoROB robustness index for base CONCH and SPECTRA-CONCH checkpoints.

Extraction mirrors what a CONCH user does: CONCH's own PIL preprocess (Resize 448 bicubic,
CenterCrop 448, OpenAI normalisation) -> encode_image(proj_contrast=False, normalize=False).
Features are written by PathoROB's own FeatureDataManager and scored by PathoROB's own
robustness_index.compute (repo: $PATHOROB_REPO), on the three datasets (tcga = paired).

    python scripts/conch_pathorob.py --name conch_base
    python scripts/conch_pathorob.py --name anc1_s150 --adapter <run>/step_0000150/adapter
    python scripts/conch_pathorob.py --name conch_base_clsmean --readout clsmean
"""
import argparse
import glob
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from PIL import Image

PR = Path(os.environ.get("PATHOROB_REPO", "/data/nanopath_full_evals/repos/PathoROB"))
sys.path.insert(0, str(PR))
DATA = Path(os.environ.get("PATHOROB_DATA", "/data/pathorob"))
OUT = Path(os.environ.get("CONCH_PATHOROB_OUT", "/data/ryan.kim/ashby/scratch/spectra_conch_pathorob"))
DATASETS = ("camelyon", "tcga", "tolkach_esca")


class PQ(torch.utils.data.Dataset):
    def __init__(self, files, preprocess):
        self.tabs = [pq.read_table(f) for f in files]
        self.idx = [(t, i) for t, tab in enumerate(self.tabs) for i in range(tab.num_rows)]
        self.pre = preprocess

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        t, i = self.idx[k]
        row = self.tabs[t].slice(i, 1).to_pylist()[0]
        im = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        return self.pre(im), {c: row[c] for c in ("slide_id", "patch_id", "biological_class", "medical_center")}


def collate(b):
    return torch.stack([x[0] for x in b]), {k: [x[1][k] for x in b] for k in b[0][1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--readout", default="export", choices=["export", "clsmean", "both"],
                    help="export = encode_image(proj_contrast=False) 512-d (default, what downstream "
                         "tools read); clsmean = cat(trunk CLS, mean trunk patch tokens) 1536-d, the "
                         "canonical SPECTRA/PathoROB readout; both = one trunk pass, scored as <name> "
                         "(export) and <name>_clsmean. Use a distinct --name per readout.")
    a = ap.parse_args()
    from conch.open_clip_custom import create_model_from_pretrained
    from spectra.models.conch_visual import ConchVisual, conch_weights_path
    from pathorob.features.data_manager import FeatureDataManager
    import pathorob.robustness_index.robustness_index as RI

    orig = torch.load
    torch.load = lambda *x, **k: orig(*x, **{**k, "weights_only": True})
    full, preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=conch_weights_path())
    torch.load = orig
    vis = ConchVisual(full.visual).eval()
    if a.adapter:
        from peft import PeftModel
        vis = PeftModel.from_pretrained(vis, a.adapter).merge_and_unload()
    vis = vis.cuda().eval()
    feats_dir = OUT / "features"
    dm = FeatureDataManager(features_dir=str(feats_dir), metadata_dir=str(PR / "data" / "metadata"))
    ros = ["export", "clsmean"] if a.readout == "both" else [a.readout]
    names = {ro: (a.name + "_clsmean" if (a.readout == "both" and ro == "clsmean") else a.name) for ro in ros}

    def feats(x):
        t = vis.trunk_tokens(x)
        out = {}
        if "export" in ros:
            out["export"] = vis.ln_contrast(vis.attn_pool_contrast(t)[:, 0])
        if "clsmean" in ros:
            out["clsmean"] = torch.cat([t[:, 0], t[:, 1:].mean(dim=1)], dim=1)
        return out
    for ds in a.datasets:
        if all((feats_dir / names[ro] / ds).exists() for ro in ros):
            continue
        dl = torch.utils.data.DataLoader(PQ(sorted(glob.glob(str(DATA / ds / "data" / "*.parquet"))), preprocess),
                                         batch_size=256, num_workers=16, collate_fn=collate)
        F_, M = {ro: [] for ro in ros}, {}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for x, meta in dl:
                for ro, f in feats(x.cuda(non_blocking=True)).items():
                    F_[ro].append(f.float().cpu())
                for k, v in meta.items():
                    M.setdefault(k, []).extend(v)
        for ro in ros:
            dm.save_features(names[ro], ds, torch.cat(F_[ro]), pd.DataFrame(M))
            print("extracted", names[ro], ds, flush=True)
    res = {}
    captured = {}
    _rs = RI.results_summary

    def grab(*args, **kw):
        r = _rs(*args, **kw)
        captured["r"] = r
        return r
    RI.results_summary = grab
    p = OUT / "ri_summary.json"
    for nm in names.values():
        res = {}
        for ds in a.datasets:
            rm = RI.compute(model=nm, dataset=ds, features_dir=str(feats_dir), metadata_dir=str(PR / "data" / "metadata"),
                       results_dir=str(OUT / "results"), figures_subdir=str(OUT / "results" / "fig"), num_workers=int(os.environ.get("PATHOROB_WORKERS", "16")))
            res[ds] = {"robustness_metrics": rm, "summary": captured.pop("r", None)}
            print("RI", nm, ds, res[ds], flush=True)
        allr = json.loads(p.read_text()) if p.exists() else {}
        allr[nm] = res
        p.write_text(json.dumps(allr, indent=1, default=str))
    print("PATHOROB DONE", flush=True)


if __name__ == "__main__":
    main()
