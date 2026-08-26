#!/usr/bin/env python
"""Bit-exactness check: base features from TWO checkouts of this repo, same backbone.

    git worktree add /tmp/old <pre-refactor-sha>
    ./.venv/bin/python scripts/regression_bitcheck.py /tmp/old .

Why this exists
---------------
The PathoROB gate costs a GPU and ~20 minutes of queue plus ~8 minutes of metric. When a
refactor is supposed to be behaviour-preserving on the existing backbone, this answers the
same question in ~2 CPU-minutes and answers it *more* sharply.

PathoROB's robustness_index is a deterministic function of the written .npz features, so
byte-identical features => byte-identical RI. That is a stronger statement than "the RI
matched to 6 decimals" -- it removes float-summation-order drift from the comparison
entirely, which is what the README had to explain away for tolkach and tcga.

It does NOT replace the full-benchmark rerun: that also exercises the parquet read, the
FeatureDataManager npz layout and the metric itself. It isolates exactly the part a model
refactor can break, and it does so before burning a GPU slot.
"""
import importlib.util, os, sys
os.environ.setdefault("HF_HOME", "/data/huggingface")
import numpy as np, torch

def load(repo, n=64, pooling="clsmean"):
    for m in list(sys.modules):
        if m.startswith("waivphaet") or m.startswith("_waiv"):
            del sys.modules[m]
    sys.path.insert(0, os.path.join(repo, "src"))
    src = os.path.join(repo, "scripts", "extract_pathorob_features.py")
    spec = importlib.util.spec_from_file_location("_waiv_ex", src)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    try:
        tf = mod.build_preprocess("owkin/phikon-v2")   # new signature
    except TypeError:
        tf = mod.build_preprocess()                     # old signature
    ds = mod.PathoRobParquet("camelyon", tf)
    x = torch.stack([ds[i][0] for i in range(n)])
    model = mod.build_model(None, pooling).eval()
    with torch.inference_mode():
        f = model.embed(x).float().numpy()
    sys.path.remove(os.path.join(repo, "src"))
    return x.numpy(), f

OLD, NEW = sys.argv[1], sys.argv[2]
for pooling in ("clsmean", "cls"):
    xo, fo = load(OLD, pooling=pooling)
    xn, fn = load(NEW, pooling=pooling)
    print(f"--- pooling={pooling} dim={fo.shape[1]}")
    print(f"  input tensors identical : {np.array_equal(xo, xn)}")
    print(f"  features  identical     : {np.array_equal(fo, fn)}  max|d|={np.abs(fo-fn).max():.3e}")
