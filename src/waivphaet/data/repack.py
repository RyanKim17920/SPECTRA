"""Repack PLISM ``.h5`` slides into flat uint8 memmaps.

Why this exists
---------------
Each PLISM ``.h5`` is *flat*: 16,278 top-level datasets, one per tile, named
``tile_{level}_{x}_{y}``, each ``(224, 224, 3)`` uint8, uncompressed, no groups
(verified in Phase 1). Reading a tile therefore costs an HDF5 B-tree lookup plus a
seek, and the registered-pair sampler (PLAN.md 2) issues *random* reads scattered
across up to 91 open files. That is the classic many-tiny-reads pattern and it makes
training I/O-bound long before the GPU is.

The fix is boring and effective: because the key order is byte-identical across all 91
files (Phase 1, confirmed), a slide is exactly a ``(16278, 224, 224, 3)`` uint8 tensor
in a fixed row order. Dump it contiguously and tile ``i`` becomes a single
``pread`` at offset ``i * 150528`` -- no index structure at all. The key order is
stored **once**, in a shared ``keys.json``, not per slide.

Layout produced::

    <out_dir>/keys.json                        # 16,278 tile keys, canonical row order
    <out_dir>/GIVH_AT2_to_GMH_S60.npy          # (16278, 224, 224, 3) uint8, ~2.29 GiB
    ...

``.npy`` (not raw) so the shape/dtype are self-describing and ``np.load(mmap_mode="r")``
just works. Put ``out_dir`` on ``/data`` -- 91 slides is ~208 GiB (PLAN.md 5).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from waivphaet.data.conditions import NUM_TILES, parse_filename

TILE_SHAPE = (224, 224, 3)
TILE_NBYTES = int(np.prod(TILE_SHAPE))  # 150,528 bytes
KEYS_FILENAME = "keys.json"


# --- keys ------------------------------------------------------------------------------


def read_keys(h5_path: Path) -> list[str]:
    """Tile keys in native HDF5 order. This order *is* the canonical row order."""
    with h5py.File(h5_path, "r", libver="latest", swmr=True) as f:
        keys = list(f.keys())
    if len(keys) != NUM_TILES:
        raise ValueError(f"{h5_path.name}: expected {NUM_TILES} tiles, found {len(keys)}")
    return keys


def load_keys(out_dir: Path) -> list[str]:
    return json.loads((Path(out_dir) / KEYS_FILENAME).read_text())


def write_keys(out_dir: Path, keys: list[str]) -> Path:
    """Write the shared key list, or verify an existing one matches byte-for-byte.

    Phase 1 verified key order is identical across files; we re-assert it here rather
    than trust it, because a silent mismatch would train on *unregistered* pairs
    (PLAN.md 4, phase 1 item 3: "getting this wrong silently trains on unregistered
    pairs").
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / KEYS_FILENAME
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != keys:
            raise ValueError(
                f"tile key order differs from {path}; index alignment across slides is "
                "broken and positives would not be co-registered"
            )
        return path
    path.write_text(json.dumps(keys))
    return path


# --- repack ----------------------------------------------------------------------------


def npy_path(out_dir: Path, h5_path: Path) -> Path:
    """``/data/.../GIVH_AT2_to_GMH_S60.npy`` for a given source ``.h5``."""
    return Path(out_dir) / (parse_filename(h5_path.name).slide_id.replace(".tif", "") + ".npy")


def repack_slide(
    h5_path: Path,
    out_dir: Path,
    *,
    chunk: int = 512,
    overwrite: bool = False,
    progress: bool = True,
) -> tuple[Path, float]:
    """Repack one ``.h5`` into a contiguous ``.npy``. Returns (path, seconds)."""
    h5_path = Path(h5_path)
    out_dir = Path(out_dir)
    dst = npy_path(out_dir, h5_path)
    if dst.exists() and not overwrite:
        return dst, 0.0

    keys = read_keys(h5_path)
    write_keys(out_dir, keys)

    t0 = time.perf_counter()
    tmp = dst.with_suffix(".npy.tmp")
    arr = np.lib.format.open_memmap(
        tmp, mode="w+", dtype=np.uint8, shape=(NUM_TILES, *TILE_SHAPE)
    )
    with h5py.File(h5_path, "r", libver="latest", swmr=True) as f:
        it = range(0, NUM_TILES, chunk)
        for start in tqdm(it, disable=not progress, desc=dst.stem, unit="chunk"):
            stop = min(start + chunk, NUM_TILES)
            for i in range(start, stop):
                arr[i] = f[keys[i]][...]
    arr.flush()
    del arr
    tmp.replace(dst)  # atomic: a half-written slide never looks complete
    return dst, time.perf_counter() - t0


def open_slide(out_dir: Path, slide_stem: str) -> np.memmap:
    """Memory-map a repacked slide read-only. ``slide_stem`` e.g. ``GIVH_AT2_to_GMH_S60``."""
    return np.load(Path(out_dir) / f"{slide_stem}.npy", mmap_mode="r")


# --- verification ----------------------------------------------------------------------


def verify_slide(
    h5_path: Path, out_dir: Path, *, n_samples: int = 64, seed: int = 0
) -> dict[str, object]:
    """Byte-compare a random sample of tiles between the repack and its source ``.h5``.

    A repack bug is exactly as dangerous as a misregistration bug -- both silently feed
    the wrong pixels to a loss that cannot detect it -- so this is a required step, not
    an optional one.
    """
    h5_path = Path(h5_path)
    keys = load_keys(out_dir)
    packed = open_slide(out_dir, npy_path(out_dir, h5_path).stem)
    if packed.shape != (NUM_TILES, *TILE_SHAPE):
        raise ValueError(f"repacked shape {packed.shape} != {(NUM_TILES, *TILE_SHAPE)}")

    rng = np.random.default_rng(seed)
    idx = rng.choice(NUM_TILES, size=min(n_samples, NUM_TILES), replace=False)
    mismatches: list[int] = []
    with h5py.File(h5_path, "r", libver="latest", swmr=True) as f:
        # also re-assert this file's own key order against the shared canonical order
        if list(f.keys()) != keys:
            raise ValueError(f"{h5_path.name}: key order differs from {KEYS_FILENAME}")
        for i in sorted(int(x) for x in idx):
            if not np.array_equal(f[keys[i]][...], np.asarray(packed[i])):
                mismatches.append(i)
    return {
        "slide": h5_path.name,
        "checked": len(idx),
        "mismatches": mismatches,
        "ok": not mismatches,
    }


def benchmark(
    h5_path: Path, out_dir: Path, *, n_reads: int = 2000, seed: int = 0
) -> dict[str, float]:
    """Random-tile read throughput, source ``.h5`` vs repacked memmap.

    This is the number that justifies the 208 GiB of extra disk. Run it once per
    machine -- results are dominated by the page cache and the filesystem, not by us.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, NUM_TILES, size=n_reads)
    keys = load_keys(out_dir)

    with h5py.File(h5_path, "r", libver="latest", swmr=True) as f:
        t0 = time.perf_counter()
        for i in idx:
            _ = f[keys[int(i)]][...]
        h5_s = time.perf_counter() - t0

    packed = open_slide(out_dir, npy_path(out_dir, Path(h5_path)).stem)
    t0 = time.perf_counter()
    for i in idx:
        _ = np.asarray(packed[int(i)])
    npy_s = time.perf_counter() - t0

    return {
        "n_reads": float(n_reads),
        "h5_seconds": h5_s,
        "npy_seconds": npy_s,
        "h5_tiles_per_s": n_reads / h5_s,
        "npy_tiles_per_s": n_reads / npy_s,
        "speedup": h5_s / npy_s,
    }


# --- CLI -------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5-dir", type=Path, default=Path("/data/ryan.kim/plism"))
    p.add_argument("--out-dir", type=Path, default=Path("/data/ryan.kim/plism_packed"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verify", action="store_true", help="byte-check a random tile sample")
    p.add_argument("--verify-samples", type=int, default=64)
    p.add_argument("--benchmark", action="store_true", help="random-read speed h5 vs npy")
    p.add_argument("--bench-reads", type=int, default=2000)
    p.add_argument("--limit", type=int, default=None, help="only process the first N slides")
    args = p.parse_args(argv)

    h5_files = sorted(Path(args.h5_dir).glob("*.tif.h5"))[: args.limit]
    if not h5_files:
        raise SystemExit(f"no *.tif.h5 under {args.h5_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for h5 in h5_files:
        dst, secs = repack_slide(h5, args.out_dir, overwrite=args.overwrite)
        gib = dst.stat().st_size / 2**30
        print(f"repacked {h5.name} -> {dst.name}  {gib:.2f} GiB  {secs:.1f}s")
        if args.verify:
            print("  verify:", verify_slide(h5, args.out_dir, n_samples=args.verify_samples))
        if args.benchmark:
            b = benchmark(h5, args.out_dir, n_reads=args.bench_reads)
            print(
                f"  bench: h5 {b['h5_tiles_per_s']:.0f} tiles/s vs npy "
                f"{b['npy_tiles_per_s']:.0f} tiles/s -> {b['speedup']:.1f}x"
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
