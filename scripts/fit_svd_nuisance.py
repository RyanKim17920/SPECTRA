#!/usr/bin/env python
"""Fit a linear *nuisance* subspace from PLISM -- training-free, post-hoc.

The idea in one line: PLISM images **the same physical tile** under 91 (stain, scanner)
conditions, so for a fixed tile index the only thing that varies across the 91 embeddings
is acquisition. The spread of those 91 vectors around their own mean is therefore *pure
nuisance*, with the tissue term cancelled exactly rather than approximately. Diagonalise
that spread and the leading eigenvectors are the directions the encoder spends on "which
scanner / which stain", which :mod:`scripts.apply_svd_nuisance` then projects out of
already-cached PathoROB features.

Nothing here trains anything. The backbone is the frozen base encoder -- the same
``build_model(checkpoint=None, ...)`` that produced the PathoROB baseline row -- so the
only difference between the k=0 control and a k>0 arm is a rank-k projection applied to
features that were already on disk.

Why this is not the ``--center-embeddings`` finding again
--------------------------------------------------------
That one removed a *single* shared shift (a rank-1, dataset-global mean). This removes a
rank-k subspace estimated on a corpus where the confound is *identified by construction*,
and it is estimated on PLISM but applied to PathoROB -- a genuine transfer, not a
per-dataset recentering.

The go/no-go number is the printed cumulative explained-variance curve. If the top handful
of directions do not concentrate the condition variance, there is no low-rank nuisance
subspace to remove and the apply step cannot help.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/huggingface")

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]

# Module level rather than inside main(): the helpers below are imported by
# scripts/diagnose_nuisance.py, which must be able to reach ``waivphaet`` without
# going through this file's CLI. The heavy imports themselves stay lazy (inside the
# helpers) so importing this module is still cheap.
for _p in (str(REPO / "src"), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_PACKED_DIR = Path("/data/plism/repacked")

#: Ranks the spectrum diagnostic reports. Powers of two up to a quarter of the 2048-d
#: clsmean width -- past that "explained variance" stops being a low-rank claim.
SPECTRUM_RANKS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


# --------------------------------------------------------------------------------------
# the math


def nuisance_basis(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(C, N, D)`` per-condition embeddings -> ``(V, eigvals)``, both descending.

    ``deltas[c, i] = F[c, i] - mean_c F[c, i]``. Because the tile is held fixed along
    axis 1, the mean over axis 0 is the tissue term and the residual is acquisition only.

    The deltas are **already exactly zero-mean per tile by construction**, so the Gram
    matrix ``deltas.T @ deltas`` *is* the (unnormalised) covariance -- re-centering it
    would subtract a second, spurious mean. That is asserted numerically below rather
    than asserted in a comment, because a silently non-centered Gram produces a
    plausible-looking spectrum and a meaningless top eigenvector.

    Accumulated one condition at a time: the flattened ``(C*N, D)`` design matrix is
    ~1.5 GB at C=91, N=2000, D=2048 and is never materialised.
    """
    if F.ndim != 3:
        raise ValueError(f"expected (C, N, D), got {F.shape}")
    C, N, D = F.shape
    if C < 2:
        raise ValueError(f"need >=2 conditions to have any cross-condition spread, got {C}")

    F = np.asarray(F, dtype=np.float32)
    tile_mean = F.mean(axis=0, keepdims=True)  # (1, N, D) -- the held-fixed tissue term

    # float64 Gram: eigh on a 2048x2048 float32 Gram of ~180k rows loses the tail of the
    # spectrum, which is exactly the part the explained-variance ratio divides by.
    gram = np.zeros((D, D), dtype=np.float64)
    resid_mean = np.zeros((N, D), dtype=np.float64)
    for c in range(C):
        d = (F[c] - tile_mean[0]).astype(np.float64)
        gram += d.T @ d
        resid_mean += d
    resid_mean /= C

    # The zero-mean-per-tile assertion. Scaled by the deltas' own magnitude so it means
    # the same thing at any embedding norm.
    scale = float(np.sqrt(np.trace(gram) / max(C * N, 1))) or 1.0
    off = float(np.abs(resid_mean).max()) / scale
    if off > 1e-4:
        raise AssertionError(
            f"deltas are not zero-mean per tile (max |mean|/rms = {off:.3e}); the "
            "per-tile mean subtraction did not do what this function assumes"
        )

    evals, evecs = torch.linalg.eigh(torch.from_numpy(gram))
    # eigh returns ASCENDING. Flip both, together -- reordering the values and not the
    # vectors is the classic silent bug here and it yields a perfectly shaped, wrong basis.
    order = torch.argsort(evals, descending=True)
    evals = evals[order]
    evecs = evecs[:, order]
    V = evecs.T.contiguous().numpy().astype(np.float32)  # eigenvectors as ROWS
    return V, evals.numpy().astype(np.float64)


def spectrum_report(eigvals: np.ndarray, ranks=SPECTRUM_RANKS) -> dict[int, float]:
    """Cumulative explained-variance ratio at each rank. This is the go/no-go number."""
    total = float(eigvals.sum())
    if total <= 0:
        raise ValueError("degenerate spectrum: total variance is not positive")
    csum = np.cumsum(np.clip(eigvals, 0.0, None))
    return {k: float(csum[k - 1] / total) for k in ranks if k <= len(eigvals)}


# --------------------------------------------------------------------------------------
# shared setup helpers
#
# These three are factored out of main() so scripts/diagnose_nuisance.py can run the
# *identical* loader, condition roster and tile sample rather than reimplementing them --
# a diagnostic that disagrees with the fit about which tiles or which normalisation is in
# play is measuring a different quantity than the one it claims to explain.


def load_frozen_encoder(backbone: str, pooling: str, device: str):
    """The frozen base encoder, via the same ``build_model`` the PathoROB baseline used."""
    from extract_pathorob_features import build_model  # lazy: pulls torch/transformers

    return build_model(None, pooling, backbone=backbone).to(device)


def select_conditions(
    *,
    conditions_file: Path | None = None,
    heldout_scanners: str = "",
    heldout_stains: str = "",
    limit_conditions: int = 0,
    verbose: bool = True,
):
    """Resolve the CLI's condition selectors to a concrete list of :class:`Condition`."""
    from waivphaet.data.conditions import all_conditions, make_split, parse_filename

    if conditions_file is not None:
        wanted = [ln.strip() for ln in Path(conditions_file).read_text().splitlines()
                  if ln.strip() and not ln.startswith("#")]
        by_key = {c.key: c for c in all_conditions()}
        conditions = [parse_filename(w) if w.endswith(".h5") else by_key[w] for w in wanted]
    elif heldout_scanners or heldout_stains:
        split = make_split(
            [s for s in heldout_scanners.split(",") if s],
            [s for s in heldout_stains.split(",") if s],
        )
        if verbose:
            print(f"[fit] {split.summary()}", flush=True)
        conditions = split.train
    else:
        conditions = all_conditions()
    if limit_conditions:
        conditions = conditions[:limit_conditions]
    return conditions


def sample_tile_idx(seed: int, n_tiles: int) -> np.ndarray:
    """``n_tiles`` distinct tile indices, SORTED.

    Sorted because each condition is a 3.3 GB memmap and a shuffled index turns one
    sequential scan into ``n_tiles`` random seeks.
    """
    from waivphaet.data.conditions import NUM_TILES

    rng = np.random.default_rng(seed)
    n = min(n_tiles, NUM_TILES)
    return np.sort(rng.choice(NUM_TILES, size=n, replace=False))


# --------------------------------------------------------------------------------------
# embedding


def embed_condition(model, path: Path, tile_idx: np.ndarray, device: str,
                     batch_size: int, amp: str) -> np.ndarray:
    """Embed ``tile_idx`` of one PLISM condition -> ``(N, D)`` float32.

    The array is a plain uint8 memmap of shape ``(16278, 224, 224, 3)``; ``tile_idx`` is
    pre-sorted so the reads walk it forwards.

    **No Resize/CenterCrop.** ``build_preprocess`` exists for PathoROB's 256x256 PIL tiles,
    where the resize is load-bearing. PLISM tiles are already exactly 224x224 uint8 HWC, so
    the only preprocessing they need is the normalisation, and ``WaivEncoder.tokens``
    applies precisely that -- ``normalize_uint8(x, self.norm_mean, self.norm_std)`` with
    the stats from ``normalization_for(backbone)``, i.e. the same stats
    ``build_preprocess`` would have used. Re-resizing an already-224 tile would resample it
    for no reason and put this fit on a different field of view than the eval path.
    """
    arr = np.load(str(path), mmap_mode="r")
    amp_dtype = {"none": None, "float16": torch.float16, "bfloat16": torch.bfloat16}[amp]
    out = None
    with torch.inference_mode():
        for start in range(0, len(tile_idx), batch_size):
            chunk = tile_idx[start:start + batch_size]
            images = torch.from_numpy(np.ascontiguousarray(arr[chunk])).to(device)
            if amp_dtype is not None:
                with torch.autocast(device_type=device.split(":")[0], dtype=amp_dtype):
                    emb = model.embed(images)
            else:
                emb = model.embed(images)
            emb = emb.float().cpu().numpy()
            if out is None:
                out = np.empty((len(tile_idx), emb.shape[1]), dtype=np.float32)
            out[start:start + len(chunk)] = emb
    return out


# --------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", default="owkin/phikon-v2")
    ap.add_argument("--pooling", default="clsmean", choices=("cls", "mean", "clsmean"))
    ap.add_argument("--n-tiles", type=int, default=2000,
                    help="tiles sampled from the 16278; the fit sees C x n_tiles vectors")
    ap.add_argument("--conditions-file", type=Path, default=None,
                    help="newline-separated STAIN_SCANNER keys (or PLISM filenames); "
                         "overrides --heldout-*")
    ap.add_argument("--heldout-scanners", default="",
                    help="comma-separated; fit on the TRAIN side of the split. Empty "
                         "(the default) = fit on all 91 conditions.")
    ap.add_argument("--heldout-stains", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--amp", default="none", choices=("none", "float16", "bfloat16"))
    ap.add_argument("--packed-dir", type=Path, default=DEFAULT_PACKED_DIR)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True, help="npz to write (V + eigvals + meta)")
    ap.add_argument("--limit-conditions", type=int, default=0,
                    help="smoke: use only the first N conditions")
    args = ap.parse_args()

    from waivphaet.data.repack import npy_path

    # --- which conditions -------------------------------------------------------------
    conditions = select_conditions(
        conditions_file=args.conditions_file,
        heldout_scanners=args.heldout_scanners,
        heldout_stains=args.heldout_stains,
        limit_conditions=args.limit_conditions,
    )
    if len(conditions) < 2:
        raise SystemExit("need >=2 conditions: with one condition there is no spread to fit")

    # --- which tiles ------------------------------------------------------------------
    tile_idx = sample_tile_idx(args.seed, args.n_tiles)
    n_tiles = len(tile_idx)

    t0 = time.time()
    model = load_frozen_encoder(args.backbone, args.pooling, args.device)
    print(f"[fit] backbone={model.cfg.backbone} pooling={args.pooling} "
          f"embed_dim={model.embed_dim} norm mean={model.norm_mean} std={model.norm_std}",
          flush=True)
    print(f"[fit] {len(conditions)} conditions x {n_tiles} tiles, seed={args.seed}, "
          f"device={args.device} ({time.time() - t0:.1f}s to load)", flush=True)

    F = np.empty((len(conditions), n_tiles, model.embed_dim), dtype=np.float32)
    t1 = time.time()
    for ci, cond in enumerate(conditions):
        # slide_id keeps the ".tif" infix; repack strips it, so go through its helper
        # rather than rebuilding the name here and drifting from the on-disk layout.
        path = npy_path(args.packed_dir, Path(cond.filename))
        if not path.exists():
            raise SystemExit(f"missing PLISM condition {path}")
        F[ci] = embed_condition(model, path, tile_idx, args.device, args.batch_size, args.amp)
        done = ci + 1
        rate = done / (time.time() - t1)
        print(f"[fit] {done}/{len(conditions)} {cond.key}  "
              f"eta {(len(conditions) - done) / rate:.0f}s", flush=True)
    if not np.isfinite(F).all():
        raise RuntimeError("non-finite embeddings")

    V, eigvals = nuisance_basis(F)
    spec = spectrum_report(eigvals)

    print("[fit] cumulative explained variance of the nuisance spread "
          "(THE go/no-go number):", flush=True)
    for k, v in spec.items():
        print(f"[fit]   top {k:>4d}: {v:.4f}", flush=True)
    # How big the nuisance spread is at all, relative to the embeddings themselves. A tiny
    # ratio would mean there is nothing worth projecting out no matter how concentrated it is.
    tot_nuis = float(eigvals.sum())
    tot_emb = float(((F - F.mean(axis=(0, 1))) ** 2).sum())
    print(f"[fit] nuisance variance / total embedding variance = {tot_nuis / tot_emb:.4f}",
          flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        V=V,
        eigvals=eigvals.astype(np.float32),
        backbone=np.array(model.cfg.backbone),
        pooling=np.array(args.pooling),
        n_tiles=np.array(n_tiles),
        seed=np.array(args.seed),
        conditions=np.array([c.key for c in conditions]),
        tile_idx=tile_idx.astype(np.int32),
        D=np.array(model.embed_dim),
        timestamp=np.array(time.strftime("%Y-%m-%dT%H:%M:%S")),
    )
    print(f"[fit] wrote {args.out}  V={V.shape} eigvals={eigvals.shape}", flush=True)
    print(json.dumps({
        "out": str(args.out), "backbone": model.cfg.backbone, "pooling": args.pooling,
        "conditions": len(conditions), "n_tiles": int(n_tiles), "D": int(model.embed_dim),
        "explained_variance": {str(k): round(v, 6) for k, v in spec.items()},
        "seconds": round(time.time() - t1, 1),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
