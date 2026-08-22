#!/usr/bin/env python
"""Project a fitted nuisance subspace out of already-cached PathoROB features.

Reads ``{features-dir}/{src-model}/{dataset}/{center}.npz`` (PathoROB's layout: one npz per
medical center, mapping ``f"{slide_id}-{patch_id}"`` -> 1-D float vector), applies

    f  ->  f - (f @ Vk.T) @ Vk        with Vk = V[:k], the top-k rows of the fit

and writes the result to ``{features-dir}/{dst-model}/{dataset}/{same filename}.npz``.
No model is loaded and no image is read -- this is arithmetic on cached vectors, which is
what makes the intervention training-free and the whole sweep cheap.

``k=0`` is an **exact byte-for-byte passthrough**, on purpose: it is the control arm. If
k=0 did not reproduce the source RI to the last digit, every delta in the sweep would be
measuring the copy rather than the projection, so the k=0 path skips the arithmetic
entirely rather than multiplying by an empty matrix and trusting float associativity.

The destination model dir is WIPED before writing, for the reason
``extract_pathorob_features.py:548`` refuses to write into an existing one: PathoROB's
``save_features`` MERGES into whatever npz is already there, so a stale 1024-d or
stale-k file would survive and be silently scored.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATASETS = ("camelyon", "tcga", "tolkach_esca")
DEFAULT_FEATURES_DIR = REPO / "third_party" / "PathoROB" / "data" / "features"


def load_basis(fit_npz: Path, k: int) -> tuple[np.ndarray, dict]:
    """``(Vk, meta)`` -- the rank-``k`` eigenvector ROWS plus the fit's metadata.

    Two packings, told apart by the ``basis`` field the fit writes. An npz written before
    that flag existed has no such field and is read back as ``joint``, which is what it is.

    ``joint``
        ``V`` is one ``(D, D)`` basis over the whole embedding; rank ``k`` is ``V[:k]``,
        i.e. **k directions removed**.
    ``split``
        ``V`` is block-diagonal -- rows ``0..H-1`` supported on the cls half, rows
        ``H..2H-1`` on the mean half (see ``fit_svd_nuisance.nuisance_basis_split``). Rank
        ``k`` is the top ``k`` of *each* half, ``V[:k]`` stacked on ``V[H:H+k]``, i.e.
        **2k directions removed**, and ``k`` is therefore capped at ``H`` and not ``D``.
        This is the only place the packing is decoded; a split sweep at k is not
        comparable to a joint sweep at k, which is why ``meta`` carries the true count.
    """
    z = np.load(fit_npz, allow_pickle=False)
    V = z["V"]
    if V.ndim != 2 or V.shape[0] != V.shape[1]:
        raise ValueError(f"{fit_npz}: V must be square (D, D), got {V.shape}")
    meta = {key: z[key].item() if z[key].ndim == 0 else z[key]
            for key in z.files
            if key not in ("V", "eigvals", "eigvals_cls", "eigvals_mean", "tile_idx")}

    basis = str(meta.get("basis", "joint"))
    if basis == "split":
        H = int(meta["half_dim"])
        if 2 * H != V.shape[0]:
            raise ValueError(f"{fit_npz}: half_dim={H} does not halve V {V.shape}")
        if not 0 <= k <= H:
            raise ValueError(f"k={k} out of range for a split fit with half_dim={H}")
        rows = np.concatenate([V[:k], V[H:H + k]])
    elif basis == "joint":
        if not 0 <= k <= V.shape[0]:
            raise ValueError(f"k={k} out of range for a {V.shape[0]}-d fit")
        rows = V[:k]
    else:
        raise ValueError(f"{fit_npz}: unknown basis {basis!r}")
    meta["basis"] = basis
    meta["directions_removed"] = int(len(rows))
    return rows.astype(np.float32), meta


def project_out(feats: np.ndarray, Vk: np.ndarray) -> np.ndarray:
    """``f - (f @ Vk.T) @ Vk`` on an ``(N, D)`` block. ``Vk`` rows are orthonormal."""
    f = feats.astype(np.float32, copy=False)
    return f - (f @ Vk.T) @ Vk


def npz_width(path: Path) -> int:
    """Vector width of a PathoROB feature npz, read from its first entry."""
    z = np.load(path, allow_pickle=False)
    return int(z[z.files[0]].shape[-1])


def apply_to_npz(src: Path, dst: Path, Vk: np.ndarray) -> int:
    """Rewrite one center's npz, preserving the exact key -> vector mapping and dtype."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    z = np.load(src, allow_pickle=False)
    keys = list(z.files)
    if not keys:
        raise ValueError(f"{src} is empty")
    if not len(Vk):
        # k=0 control: copy the bytes rather than re-serialising them. Round-tripping
        # through np.stack/np.savez would almost certainly be value-identical, but
        # "almost certainly" is not what a control arm is for.
        shutil.copyfile(src, dst)
        return len(keys)
    dtypes = {z[key].dtype for key in keys}
    if len(dtypes) != 1:
        raise ValueError(f"{src} mixes dtypes {dtypes}; refusing to guess an output dtype")
    out_dtype = dtypes.pop()
    block = project_out(np.stack([z[key] for key in keys]), Vk).astype(out_dtype, copy=False)
    np.savez(dst, **{key: block[i] for i, key in enumerate(keys)})
    return len(keys)


def apply_fit(
    fit_npz: Path,
    k: int,
    src_model: str,
    dst_model: str,
    features_dir: Path,
    datasets=DATASETS,
) -> dict:
    """Materialise ``dst_model`` as ``src_model`` with the top-``k`` fit directions removed."""
    Vk, meta = load_basis(fit_npz, k)
    features_dir = Path(features_dir)
    dst_root = features_dir / dst_model
    if dst_root.exists():
        # See the module docstring: save_features merges, so "overwrite" is not a thing.
        shutil.rmtree(dst_root)

    counts: dict[str, int] = {}
    width: int | None = None
    for dataset in datasets:
        src_dir = features_dir / src_model / dataset
        if not src_dir.is_dir():
            raise SystemExit(f"no source features at {src_dir}")
        files = sorted(src_dir.glob("*.npz"))
        if not files:
            raise SystemExit(f"{src_dir} has no npz")
        for f in files:
            if width is None:
                width = npz_width(f)
                # Loud, up front: a 1024-d cls-only feature dir silently pairs with a
                # 2048-d clsmean fit if the only check is "the matmul ran".
                if width != Vk.shape[1]:
                    raise SystemExit(
                        f"fit {fit_npz} is {Vk.shape[1]}-d (pooling={meta.get('pooling')}, "
                        f"backbone={meta.get('backbone')}) but {f} is {width}-d; these are "
                        "different pooling protocols and the projection would be meaningless"
                    )
            counts[dataset] = counts.get(dataset, 0) + apply_to_npz(
                f, dst_root / dataset / f.name, Vk
            )
    return {"dst_model": dst_model, "k": k, "D": width, "vectors": counts,
            "basis": meta.get("basis"),
            # Under a split basis this is 2k, not k. Reported explicitly so a sweep table
            # never puts a split k next to a joint k as though they cost the same.
            "directions_removed": meta.get("directions_removed"),
            "fit_backbone": meta.get("backbone"), "fit_pooling": meta.get("pooling")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit", type=Path, required=True, help="npz from fit_svd_nuisance.py")
    ap.add_argument("--k", type=int, required=True, help="rank to remove; 0 = control")
    ap.add_argument("--src-model", default="phikonv2_clsmean_ours")
    ap.add_argument("--dst-model", required=True)
    ap.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    args = ap.parse_args()

    out = apply_fit(args.fit, args.k, args.src_model, args.dst_model,
                    args.features_dir, tuple(args.datasets))
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
