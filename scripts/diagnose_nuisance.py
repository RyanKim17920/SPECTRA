#!/usr/bin/env python
"""Why is the PLISM nuisance spectrum *diffuse*? Four diagnostics, one embedding pass.

:mod:`scripts.fit_svd_nuisance` found that on Virchow2/clsmean the cross-condition spread
is 31.7% of the total embedding variance -- large -- but that its spectrum is flat
(cumulative EV: top-1 0.194, top-8 0.433, top-64 0.637, top-128 0.728, top-256 0.826).
A flat spectrum is what kills the rank-k projection: there is no small subspace to remove.

**The observation this script exists to explain.** If acquisition were a pure per-condition
offset -- ``d(i, c) = m_c``, the same vector added to every tile of condition ``c`` -- then
the delta matrix would be a sum of 91 rank-1 outer products with one linear constraint
(the deltas are zero-mean per tile by construction), so it would have rank <= 90 and every
eigenvalue past index 90 would be *exactly* zero. They are not: the curve keeps climbing
from 0.637 at 64 to 0.826 at 256. So a large share of the nuisance is a tile x condition
**interaction** -- the encoder's response to a scanner depends on what tissue is in the
tile. This script measures that share and then asks where it comes from.

The four diagnostics
--------------------
1. **Offset vs interaction.** Split ``d`` exactly into its per-condition mean part
   ``broadcast(m_c)`` and the residual, report the two energy shares (they are orthogonal,
   so they sum to 1) and the two spectra separately. The offset spectrum must die at rank
   90 -- that is a correctness check on the decomposition, asserted numerically.
2. **Scanner vs stain.** Re-form the deltas along one axis at a time (deltas to the mean
   over the 7 scanners at fixed stain; deltas to the mean over the 13 stains at fixed
   scanner). Hypothesis under test: scanner is pure optics -- one sensor, one illuminant,
   one JPEG pipeline -- and should be *concentrated*; stain is chemistry that binds
   different tissue components in different amounts, so it is partly biological and should
   be *diffuse*. Also reports the overlap between the two leading subspaces: if scanner and
   stain live in the same directions, "nuisance" is one phenomenon, not two.
3. **A synthetic misregistration control.** PLISM's 91 slides are Elastix-warped onto
   ``GMH_S60``; registration is never exact. Translate one condition's tiles by a few
   pixels and embed them again: that delta is *pure* misregistration, with no optics and no
   chemistry in it. If a 2-8 px shift produces deltas of comparable magnitude AND
   comparable diffuseness to the real cross-condition deltas, then residual registration
   error is a live explanation for the high-rank tail -- and the tail is an artefact of the
   corpus, not a property of the encoder's acquisition response worth projecting out.
4. **Is the interaction tile heterogeneity?** Per-tile nuisance magnitude vs (a) the tile's
   tissue fraction and (b) its raw pixel variance. An edge-rich, high-variance tile has more
   structure for a resampling or a stain difference to act on, so a positive correlation
   says the interaction term is carried by *which* tiles rather than spread evenly.

Everything runs off ONE embedding pass over the 91 conditions (plus one short extra pass
per shift in diagnostic 3). Nothing here trains, fits or writes a basis -- the output is a
single JSON.
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
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reused, not reimplemented: same loader, same condition roster, same sorted tile sample,
# same no-resize uint8-224 embedding convention as the fit whose result we are explaining.
from fit_svd_nuisance import (  # noqa: E402
    DEFAULT_PACKED_DIR,
    embed_condition,
    load_frozen_encoder,
    sample_tile_idx,
    select_conditions,
    spectrum_report,
)

#: Same ladder as the fit, plus **90** -- the rank a pure per-condition offset cannot
#: exceed, and therefore the single most diagnostic entry in the whole table.
DIAG_RANKS = (1, 2, 4, 8, 16, 32, 64, 90, 128, 256)

DEFAULT_TISSUE_FRACTION = Path("/data/ryan.kim/waiv_runs/.plism_tissue_fraction.npy")


# --------------------------------------------------------------------------------------
# small linear-algebra helpers


def gram_eigh(gram: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(D, D)`` Gram -> ``(eigvals desc, V)`` with eigenvectors as ROWS of ``V``.

    Same convention and same float64 discipline as :func:`fit_svd_nuisance.nuisance_basis`:
    ``eigh`` returns ascending, and reordering the values without the vectors is the classic
    silent bug that yields a perfectly shaped, wrong basis.
    """
    evals, evecs = torch.linalg.eigh(torch.from_numpy(np.asarray(gram, dtype=np.float64)))
    order = torch.argsort(evals, descending=True)
    evals = evals[order]
    evecs = evecs[:, order]
    return evals.numpy().astype(np.float64), evecs.T.contiguous().numpy().astype(np.float32)


def subspace_overlap(Va: np.ndarray, Vb: np.ndarray, k: int) -> float:
    """``||Va_k Vb_k^T||_F^2 / k`` -- mean squared cosine of the principal angles.

    Rows of ``Va``/``Vb`` are orthonormal eigenvectors (the layout :func:`gram_eigh`
    returns). The value is 1 when the two top-``k`` subspaces are identical (the Frobenius
    norm of a ``k x k`` orthogonal matrix is ``sqrt(k)``) and 0 when they are orthogonal.
    Normalising by ``k`` rather than reporting raw principal angles keeps it one number
    that is comparable across ``k``.
    """
    if k > Va.shape[0] or k > Vb.shape[0]:
        raise ValueError(f"k={k} exceeds available components ({Va.shape[0]}, {Vb.shape[0]})")
    M = Va[:k].astype(np.float64) @ Vb[:k].astype(np.float64).T
    return float((M ** 2).sum() / k)


def _sq(x: np.ndarray) -> float:
    return float((np.asarray(x, dtype=np.float64) ** 2).sum())


def _spectrum(gram: np.ndarray) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Cumulative-EV table + eigvals + basis. A term with *no* energy is a legitimate
    answer here (a pure per-condition offset has an exactly-zero interaction term), so a
    degenerate Gram reports zeros instead of raising the way the fit's go/no-go does."""
    evals, V = gram_eigh(gram)
    if float(np.clip(evals, 0.0, None).sum()) <= 0:
        return {str(k): 0.0 for k in DIAG_RANKS if k <= len(evals)}, evals, V
    return {str(k): round(v, 6) for k, v in spectrum_report(evals, DIAG_RANKS).items()}, evals, V


def _rank_conc(evals: np.ndarray, rank: int) -> float:
    """Fraction of the spectrum's mass sitting *past* ``rank``. ~0 = exhausted by ``rank``."""
    total = float(np.clip(evals, 0.0, None).sum())
    if total <= 0:
        return 0.0
    return float(np.clip(evals[rank:], 0.0, None).sum() / total)


# --------------------------------------------------------------------------------------
# diagnostic 1 -- per-condition offset vs tile x condition interaction


def offset_interaction_split(F: np.ndarray) -> dict:
    """``(C, N, D)`` -> the exact energy split of ``d = F - mean_c F`` and both spectra.

    ``d[c, i] = F[c, i] - mean_c F[c, i]``. Write ``m_c = mean_i d[c, i]``; then

        d = broadcast(m_c) + r,   r[c, i] = d[c, i] - m_c

    and the two terms are **orthogonal** (``sum_i r[c, i] = 0`` for every c), so the energies
    add exactly. That is asserted below rather than assumed -- if the split leaked, both
    shares would still look plausible.

    The offset part has rank <= C - 1 (the ``m_c`` sum to zero, because the deltas are
    zero-mean over c by construction), so ``offset.tail_past_rank_90`` must be ~0 at C=91.
    Anything the offset cannot explain is the interaction the docstring at the top is about.

    Accumulated one condition at a time: the flattened ``(C*N, D)`` matrix is ~1.9 GB at
    C=91, N=2000, D=2560 and is never materialised.
    """
    if F.ndim != 3:
        raise ValueError(f"expected (C, N, D), got {F.shape}")
    C, N, D = F.shape
    if C < 2:
        raise ValueError(f"need >=2 conditions to have any cross-condition spread, got {C}")

    F = np.asarray(F, dtype=np.float32)
    tile_mean = F.mean(axis=0)  # (N, D) -- the tissue term, held fixed along axis 1

    m = np.empty((C, D), dtype=np.float64)
    gram_r = np.zeros((D, D), dtype=np.float64)
    e_total = e_offset = e_inter = 0.0
    per_condition = []
    for c in range(C):
        d = (F[c] - tile_mean).astype(np.float64)
        m[c] = d.mean(axis=0)
        r = d - m[c]
        gram_r += r.T @ r
        e_d, e_o, e_r = _sq(d), N * _sq(m[c]), _sq(r)
        e_total += e_d
        e_offset += e_o
        e_inter += e_r
        per_condition.append({"delta_sq": e_d, "offset_sq": e_o, "interaction_sq": e_r})

    if e_total <= 0:
        raise ValueError("degenerate: the cross-condition deltas carry no energy at all")
    leak = abs(e_offset + e_inter - e_total) / e_total
    if leak > 1e-6:
        raise AssertionError(
            f"offset/interaction split is not orthogonal (relative leak {leak:.3e}); the "
            "two energies must add to the total or neither share means anything"
        )

    gram_m = N * (m.T @ m)
    spec_off, ev_off, _ = _spectrum(gram_m)
    spec_int, ev_int, _ = _spectrum(gram_r)

    for pc in per_condition:  # normalise only after the assertion, so the check sees raw sums
        pc["share_of_total_delta"] = round(pc["delta_sq"] / e_total, 6)
        pc["interaction_share_within_condition"] = round(pc["interaction_sq"] / pc["delta_sq"], 6)
        for key in ("delta_sq", "offset_sq", "interaction_sq"):
            pc[key] = round(pc[key], 4)

    return {
        "energy": {
            "delta_total": e_total,
            "offset_share": e_offset / e_total,
            "interaction_share": e_inter / e_total,
        },
        "offset": {
            "explained_variance": spec_off,
            # THE number: a pure per-condition offset is rank <= C-1, so this is ~0 by
            # construction. A non-zero value means the decomposition is broken.
            "tail_past_rank_90": _rank_conc(ev_off, C - 1),
        },
        "interaction": {
            "explained_variance": spec_int,
            # And THIS is the observation under test: for a pure offset there would be no
            # interaction term at all, so any mass here past rank 90 is the diffuse tail.
            "tail_past_rank_90": _rank_conc(ev_int, C - 1),
        },
        "per_condition": per_condition,
    }


# --------------------------------------------------------------------------------------
# diagnostic 2 -- one axis at a time


def axis_deltas(G: np.ndarray, axis: int) -> tuple[np.ndarray, float]:
    """Gram and total energy of ``f - mean_over_axis f`` for a ``(n_stain, n_scan, N, D)`` grid.

    ``axis=1`` (mean over the 7 scanners, at fixed stain) isolates **scanner**; ``axis=0``
    (mean over the 13 stains, at fixed scanner) isolates **stain**. Deltas are pooled over
    the other axis, so both estimates see the full 91-slide corpus rather than one slice.
    Accumulated per condition to avoid materialising a second copy of the grid.
    """
    if G.ndim != 4:
        raise ValueError(f"expected (n_stain, n_scan, N, D), got {G.shape}")
    n_stain, n_scan, _, D = G.shape
    gram = np.zeros((D, D), dtype=np.float64)
    total = 0.0
    outer = n_scan if axis == 0 else n_stain
    inner = n_stain if axis == 0 else n_scan
    for o in range(outer):
        block = G[:, o] if axis == 0 else G[o]  # (inner, N, D)
        mean = block.mean(axis=0)
        for j in range(inner):
            d = (block[j] - mean).astype(np.float64)
            gram += d.T @ d
            total += _sq(d)
    return gram, total


# --------------------------------------------------------------------------------------
# diagnostic 3 -- synthetic misregistration


def shift_tiles(tiles: np.ndarray, pixels: int) -> np.ndarray:
    """Translate ``(N, H, W, C)`` uint8 tiles diagonally by ``pixels`` along H and W.

    **np.roll (circular), not crop-and-pad.** A crop-and-pad would introduce a constant
    band along two edges, and a hard synthetic edge against tissue is exactly the kind of
    high-contrast structure a ViT responds to strongly -- the measured delta would then be
    dominated by the padding artefact rather than by the translation, which is the thing
    being modelled. Rolling wraps a ``pixels``-wide strip of real tissue instead: still an
    artefact, but a far smaller one at the 2-8 px scale of a registration residual, and it
    keeps every pixel of the tile real. Diagonal (both axes) because a registration residual
    has no reason to be axis-aligned.
    """
    if tiles.ndim != 4:
        raise ValueError(f"expected (N, H, W, C) tiles, got {tiles.shape}")
    return np.roll(tiles, shift=(pixels, pixels), axis=(1, 2))


def embed_shifted(model, path: Path, tile_idx: np.ndarray, pixels: int, device: str,
                  batch_size: int, amp: str) -> np.ndarray:
    """``embed_condition`` on translated tiles -- same normalisation, same no-resize path."""
    arr = np.load(str(path), mmap_mode="r")
    amp_dtype = {"none": None, "float16": torch.float16, "bfloat16": torch.bfloat16}[amp]
    out = None
    with torch.inference_mode():
        for start in range(0, len(tile_idx), batch_size):
            chunk = tile_idx[start:start + batch_size]
            tiles = shift_tiles(np.ascontiguousarray(arr[chunk]), pixels)
            images = torch.from_numpy(np.ascontiguousarray(tiles)).to(device)
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
# diagnostic 4 -- per-tile heterogeneity


def tile_pixel_variance(path: Path, tile_idx: np.ndarray, batch: int = 256) -> np.ndarray:
    """Per-tile variance of the raw uint8 pixels, over all pixels and channels.

    Deliberately the simplest possible heterogeneity measure: a flat, uniformly-stained
    tile has low variance; an edge-rich tile with nuclei, lumen and background has high
    variance. No filtering, no gradient operator -- anything fancier would need its own
    justification and this is only ever used as a correlate.
    """
    arr = np.load(str(path), mmap_mode="r")
    out = np.empty(len(tile_idx), dtype=np.float64)
    for start in range(0, len(tile_idx), batch):
        chunk = tile_idx[start:start + batch]
        block = np.asarray(arr[chunk], dtype=np.float32).reshape(len(chunk), -1)
        out[start:start + len(chunk)] = block.var(axis=1)
    return out


def _correlations(x: np.ndarray, y: np.ndarray) -> dict:
    from scipy import stats

    if np.std(x) == 0 or np.std(y) == 0:
        return {"pearson_r": None, "pearson_p": None, "spearman_r": None, "spearman_p": None,
                "note": "degenerate: one series is constant"}
    pr = stats.pearsonr(x, y)
    sr = stats.spearmanr(x, y)
    return {
        "pearson_r": float(pr[0]), "pearson_p": float(pr[1]),
        "spearman_r": float(sr[0]), "spearman_p": float(sr[1]),
        "n": int(len(x)),
    }


# --------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", default="paige-ai/Virchow2")
    ap.add_argument("--pooling", default="clsmean", choices=("cls", "mean", "clsmean"))
    ap.add_argument("--n-tiles", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--amp", default="bfloat16", choices=("none", "float16", "bfloat16"))
    ap.add_argument("--packed-dir", type=Path, default=DEFAULT_PACKED_DIR)
    ap.add_argument("--shift-pixels", type=int, nargs="+", default=[2, 4, 8],
                    help="diagnostic 3: pixel translations of the reference condition")
    ap.add_argument("--tissue-fraction", type=Path, default=DEFAULT_TISSUE_FRACTION,
                    help="diagnostic 4: per-tile tissue fraction over all 16278 tiles")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True, help="JSON report to write")
    args = ap.parse_args()

    from waivphaet.data.conditions import (
        REGISTRATION_REFERENCE,
        SCANNERS,
        STAINS,
        Condition,
        all_conditions,
    )
    from waivphaet.data.repack import npy_path

    # The whole of diagnostic 2 depends on F being reshapeable to the (stain, scanner)
    # grid, which is true only for the full roster in its canonical stain-major order.
    conditions = select_conditions()
    expected = all_conditions()
    if conditions != expected:
        raise SystemExit("diagnose_nuisance needs all 91 conditions in canonical order")
    n_stain, n_scan = len(STAINS), len(SCANNERS)
    assert n_stain * n_scan == len(conditions)

    tile_idx = sample_tile_idx(args.seed, args.n_tiles)
    n_tiles = len(tile_idx)

    t0 = time.time()
    model = load_frozen_encoder(args.backbone, args.pooling, args.device)
    print(f"[diag] backbone={model.cfg.backbone} pooling={args.pooling} "
          f"embed_dim={model.embed_dim} ({time.time() - t0:.1f}s to load)", flush=True)
    print(f"[diag] {len(conditions)} conditions x {n_tiles} tiles, seed={args.seed}, "
          f"device={args.device}, amp={args.amp}", flush=True)

    # --- the single embedding pass ----------------------------------------------------
    F = np.empty((len(conditions), n_tiles, model.embed_dim), dtype=np.float32)
    t1 = time.time()
    for ci, cond in enumerate(conditions):
        path = npy_path(args.packed_dir, Path(cond.filename))
        if not path.exists():
            raise SystemExit(f"missing PLISM condition {path}")
        F[ci] = embed_condition(model, path, tile_idx, args.device, args.batch_size, args.amp)
        done = ci + 1
        rate = done / (time.time() - t1)
        print(f"[diag] embed {done}/{len(conditions)} {cond.key}  "
              f"eta {(len(conditions) - done) / rate:.0f}s", flush=True)
    if not np.isfinite(F).all():
        raise RuntimeError("non-finite embeddings")

    # Denominator shared by every "nuisance / total" ratio below, and the same one the fit
    # prints, so the numbers here are directly comparable to its output. Accumulated per
    # condition: ``_sq(F - F.mean(axis=(0, 1)))`` in one expression would materialise a
    # float64 copy of the whole (91, 2000, 2560) block, ~3.7 GB on top of F itself.
    global_mean = F.mean(axis=(0, 1))
    total_embedding_var = sum(_sq(F[c] - global_mean) for c in range(len(conditions)))

    report: dict = {
        "backbone": model.cfg.backbone, "pooling": args.pooling, "D": int(model.embed_dim),
        "n_tiles": int(n_tiles), "n_conditions": len(conditions), "seed": args.seed,
        "amp": args.amp, "device": args.device,
        "total_embedding_variance": total_embedding_var,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # --- diagnostic 1 -----------------------------------------------------------------
    print("\n[diag 1] per-condition offset vs tile x condition interaction", flush=True)
    d1 = offset_interaction_split(F)
    d1["nuisance_over_total_variance"] = d1["energy"]["delta_total"] / total_embedding_var
    report["d1_offset_vs_interaction"] = d1
    print(f"[diag 1]   nuisance / total embedding variance = "
          f"{d1['nuisance_over_total_variance']:.4f}", flush=True)
    print(f"[diag 1]   offset share      = {d1['energy']['offset_share']:.4f}", flush=True)
    print(f"[diag 1]   interaction share = {d1['energy']['interaction_share']:.4f}"
          "   <- a pure per-condition offset would put this at 0", flush=True)
    for name in ("offset", "interaction"):
        print(f"[diag 1]   {name} cumulative EV: "
              + "  ".join(f"{k}:{v:.3f}" for k, v in d1[name]["explained_variance"].items()),
              flush=True)
        print(f"[diag 1]   {name} mass past rank 90 = {d1[name]['tail_past_rank_90']:.2e}",
              flush=True)

    # --- diagnostic 2 -----------------------------------------------------------------
    print("\n[diag 2] scanner-only vs stain-only deltas", flush=True)
    G = F.reshape(n_stain, n_scan, n_tiles, model.embed_dim)
    d2: dict = {}
    bases = {}
    for name, axis in (("scanner", 1), ("stain", 0)):
        gram, total = axis_deltas(G, axis)
        spec, evals, V = _spectrum(gram)
        bases[name] = V
        d2[name] = {
            "explained_variance": spec,
            "nuisance_over_total_variance": total / total_embedding_var,
            "delta_total": total,
            "n_levels": n_scan if axis == 1 else n_stain,
        }
        print(f"[diag 2]   {name:8s} var/total = {total / total_embedding_var:.4f}   "
              + "  ".join(f"{k}:{v:.3f}" for k, v in spec.items()), flush=True)
    d2["subspace_overlap"] = {
        str(k): subspace_overlap(bases["scanner"], bases["stain"], k) for k in (8, 32)
    }
    print(f"[diag 2]   scanner/stain top-k subspace overlap "
          f"(1 = same subspace, 0 = orthogonal): {d2['subspace_overlap']}", flush=True)
    report["d2_scanner_vs_stain"] = d2

    # --- diagnostic 3 -----------------------------------------------------------------
    print("\n[diag 3] synthetic misregistration control", flush=True)
    ref = Condition(*REGISTRATION_REFERENCE.rsplit("_", 1))
    ref_i = conditions.index(ref)
    ref_path = npy_path(args.packed_dir, Path(ref.filename))
    # Per-row so it is comparable: the cross-condition deltas have C*N rows, the shift
    # deltas only N, and the raw ratio would make any shift look tiny for that reason alone.
    cross_per_row = d1["energy"]["delta_total"] / (len(conditions) * n_tiles)
    d3: dict = {"reference_condition": ref.key, "shifts": {},
                "cross_condition_mean_sq_per_row": cross_per_row}
    for px in args.shift_pixels:
        emb = embed_shifted(model, ref_path, tile_idx, px, args.device,
                            args.batch_size, args.amp)
        d_shift = (emb - F[ref_i]).astype(np.float64)
        e_shift = _sq(d_shift)
        spec, _, _ = _spectrum(d_shift.T @ d_shift)
        d3["shifts"][str(px)] = {
            "delta_total": e_shift,
            # THE comparison: >= ~1 means a few pixels of registration slop moves the
            # embedding as far as changing scanner and stain does.
            "magnitude_ratio_per_row": (e_shift / n_tiles) / cross_per_row,
            "magnitude_ratio_raw": e_shift / d1["energy"]["delta_total"],
            "over_total_embedding_variance": e_shift / total_embedding_var,
            "explained_variance": spec,
        }
        print(f"[diag 3]   {px:>2d}px  mag/cross(per row) = "
              f"{d3['shifts'][str(px)]['magnitude_ratio_per_row']:.4f}   "
              + "  ".join(f"{k}:{v:.3f}" for k, v in spec.items()), flush=True)
    report["d3_shift_control"] = d3

    # --- diagnostic 4 -----------------------------------------------------------------
    print("\n[diag 4] per-tile nuisance magnitude vs tile heterogeneity", flush=True)
    tile_mean = F.mean(axis=0)
    per_tile = np.zeros(n_tiles, dtype=np.float64)
    for c in range(len(conditions)):
        per_tile += ((F[c] - tile_mean).astype(np.float64) ** 2).sum(axis=1)

    d4: dict = {"per_tile_nuisance_sq": {
        "mean": float(per_tile.mean()), "std": float(per_tile.std()),
        "min": float(per_tile.min()), "max": float(per_tile.max()),
    }}
    if args.tissue_fraction.exists():
        tf_all = np.load(str(args.tissue_fraction))
        if len(tf_all) != 16278:
            raise SystemExit(
                f"{args.tissue_fraction} has {len(tf_all)} entries, expected 16278 -- it "
                "must be indexed by the SAME tile index as the packed conditions")
        d4["tissue_fraction"] = _correlations(per_tile, tf_all[tile_idx].astype(np.float64))
        print(f"[diag 4]   vs tissue fraction: {d4['tissue_fraction']}", flush=True)
    else:
        d4["tissue_fraction"] = {"note": f"missing {args.tissue_fraction}"}
        print(f"[diag 4]   tissue fraction unavailable ({args.tissue_fraction})", flush=True)

    pv = tile_pixel_variance(ref_path, tile_idx)
    d4["pixel_variance"] = _correlations(per_tile, pv)
    d4["pixel_variance"]["reference_condition"] = ref.key
    print(f"[diag 4]   vs pixel variance:  {d4['pixel_variance']}", flush=True)
    report["d4_tile_heterogeneity"] = d4

    report["seconds"] = round(time.time() - t1, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\n[diag] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
