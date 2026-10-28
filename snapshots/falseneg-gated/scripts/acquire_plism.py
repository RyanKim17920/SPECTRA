#!/usr/bin/env python
"""Stream-acquire all 91 PLISM slides: download -> repack -> verify -> delete source.

Why a streaming loop instead of "download everything, then repack everything"
-----------------------------------------------------------------------------
The raw ``.h5`` set is ~224 GB and the repacked memmaps are ~208 GiB. Holding both
at once costs ~432 GB and buys nothing: once a slide is repacked *and byte-verified*
against its source, the ``.h5`` is dead weight (PLAN.md 5 -- the repack is the training
input, the ``.h5`` never is). So we do one slide at a time and never let the two copies
coexist beyond a single slide (~2.3 GB of overlap).

The delete step is the load-bearing one. ``hf_hub_download`` normally materialises a
file *twice*: once as a blob under ``HF_HOME`` and once as a link in the destination.
Deleting only the destination silently keeps the blob and doubles storage. This script
downloads with ``local_dir=`` (which bypasses the hub blob store) *and* defensively
purges any ``datasets--owkin--plism-dataset`` cache tree after every slide.

Resumability
------------
State lives in ``manifest.json`` next to the repacked slides. A slide is skipped only
if the manifest records ``verified: true`` **and** the ``.npy`` is present at the exact
expected byte size. Anything weaker would let a truncated repack survive a crash, and a
bad repack is undetectable downstream -- it feeds wrong pixels to a loss that cannot
notice (same failure mode as misregistration, PLAN.md 4 phase 1 item 3).

Usage::

    export HF_HOME=/data/huggingface     # never let the default ~/.cache fill /admin
    python scripts/acquire_plism.py --dry-run
    python scripts/acquire_plism.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Make `src/` importable when run as a plain script from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from waivphaet.data.conditions import NUM_TILES, all_conditions  # noqa: E402
from waivphaet.data.repack import (  # noqa: E402
    TILE_SHAPE,
    npy_path,
    repack_slide,
    verify_slide,
)

REPO_ID = "owkin/plism-dataset"
GIB = 2**30
# .npy header is 128 bytes for this shape; assert the exact size so a truncated
# repack can never be mistaken for a finished one.
EXPECTED_NPY_BYTES = 128 + NUM_TILES * TILE_SHAPE[0] * TILE_SHAPE[1] * TILE_SHAPE[2]


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# --- disk accounting -------------------------------------------------------------------


def tree_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


# --- manifest --------------------------------------------------------------------------


def load_manifest(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            log(f"WARNING: {path} unreadable, starting a fresh manifest")
    return {}


def save_manifest(path: Path, manifest: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    tmp.replace(path)  # atomic: a crash mid-write never corrupts resume state


def already_done(cond, out_dir: Path, manifest: dict) -> bool:
    entry = manifest.get(cond.key)
    if not entry or not entry.get("verified"):
        return False
    npy = out_dir / f"{cond.slide_id.replace('.tif', '')}.npy"
    return npy.exists() and npy.stat().st_size == EXPECTED_NPY_BYTES


# --- HF cache hygiene ------------------------------------------------------------------


def purge_hf_cache(repo_id: str) -> int:
    """Delete the hub cache tree for ``repo_id``. Returns bytes reclaimed.

    We download with ``local_dir=``, so this tree *should* stay empty -- but if a
    huggingface_hub version ever falls back to the blob store, silently keeping it
    would double storage across 91 slides. Cheap to re-check, expensive to miss.
    """
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    slug = "datasets--" + repo_id.replace("/", "--")
    freed = 0
    for base in (hf_home / "hub" / slug, hf_home / slug):
        if base.exists():
            freed += tree_bytes(base)
            shutil.rmtree(base, ignore_errors=True)
    return freed


def purge_incomplete(h5_dir: Path) -> None:
    """Drop the per-file download metadata/partials ``local_dir`` leaves behind."""
    dl = h5_dir / ".cache" / "huggingface" / "download"
    if dl.exists():
        for p in dl.iterdir():
            if p.is_file():
                p.unlink(missing_ok=True)


# --- one slide -------------------------------------------------------------------------


def process(cond, args, manifest: dict) -> dict:
    from huggingface_hub import hf_hub_download

    h5_dir: Path = args.h5_dir
    out_dir: Path = args.out_dir

    t0 = time.perf_counter()
    local = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=cond.filename,
            repo_type="dataset",
            local_dir=str(h5_dir),
        )
    )
    dl_s = time.perf_counter() - t0
    src_gb = local.stat().st_size / GIB
    log(f"  downloaded {src_gb:.2f} GiB in {dl_s:.0f}s ({src_gb * 1024 / dl_s:.0f} MiB/s)")

    dst, pack_s = repack_slide(local, out_dir, overwrite=True, progress=False)
    log(f"  repacked -> {dst.name} in {pack_s:.0f}s")

    v = verify_slide(local, out_dir, n_samples=args.verify_samples)
    if not v["ok"]:
        raise RuntimeError(f"verification FAILED for {cond.filename}: {v}")
    log(f"  verified {v['checked']} random tiles byte-exact")

    # Only now is the source redundant.
    local.unlink()
    purge_incomplete(h5_dir)
    freed = purge_hf_cache(REPO_ID)
    if freed > 2**20:  # sub-MiB is just refs/ bookkeeping, not a duplicated blob
        log(f"  WARNING: purged {freed / GIB:.2f} GiB of unexpected HF blob cache")

    return {
        "stain": cond.stain,
        "scanner": cond.scanner,
        "filename": cond.filename,
        "path": str(dst),
        "bytes": dst.stat().st_size,
        "tiles": NUM_TILES,
        "verified": True,
        "verify_samples": int(v["checked"]),
        "download_seconds": round(dl_s, 1),
        "repack_seconds": round(pack_s, 1),
        "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# --- main ------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5-dir", type=Path, default=Path("/data/plism"))
    p.add_argument("--out-dir", type=Path, default=Path("/data/plism/repacked"))
    p.add_argument("--manifest", type=Path, default=None, help="default: <out-dir>/manifest.json")
    p.add_argument("--verify-samples", type=int, default=64)
    p.add_argument("--retries", type=int, default=2, help="extra attempts per slide after a failure")
    p.add_argument("--min-free-gb", type=float, default=400.0, help="abort if /data has less")
    p.add_argument("--max-resting-gb", type=float, default=230.0,
                   help="abort if the PLISM tree exceeds this between slides (duplication guard)")
    p.add_argument("--limit", type=int, default=None, help="only process the first N remaining slides")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    args.h5_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or (args.out_dir / "manifest.json")

    if "HF_HOME" not in os.environ:
        log("ERROR: HF_HOME is unset; refusing to fill the default cache on /admin")
        return 2
    log(f"HF_HOME={os.environ['HF_HOME']}")

    avail = free_gb(args.h5_dir)
    log(f"free on {args.h5_dir}: {avail:.0f} GB")
    if avail < args.min_free_gb:
        log(f"ERROR: need >= {args.min_free_gb:.0f} GB free, aborting")
        return 2

    manifest = load_manifest(manifest_path)
    conds = all_conditions()
    todo = [c for c in conds if not already_done(c, args.out_dir, manifest)]
    log(f"{len(conds)} slides total, {len(conds) - len(todo)} already repacked+verified, {len(todo)} to do")
    if args.limit is not None:
        todo = todo[: args.limit]

    if args.dry_run:
        for c in todo:
            log(f"  would acquire {c.filename}")
        log(f"dry run: {len(todo)} slides, ~{len(todo) * 2.29:.0f} GiB of repacked output")
        log(f"manifest would be written to {manifest_path}")
        return 0

    # A stray .h5 from a previous crash means two copies are alive; clean before starting.
    for stale in args.h5_dir.glob("*.tif.h5"):
        if already_done_name(stale.name, args.out_dir, manifest):
            log(f"removing redundant source {stale.name} (already repacked+verified)")
            stale.unlink()

    t_start = time.perf_counter()
    ok, failed = 0, []
    for n, cond in enumerate(todo, 1):
        log(f"[{n}/{len(todo)}] {cond.filename}")
        for attempt in range(args.retries + 1):
            try:
                manifest[cond.key] = process(cond, args, manifest)
                save_manifest(manifest_path, manifest)
                ok += 1
                break
            except Exception as e:  # noqa: BLE001 - one bad slide must not kill the run
                log(f"  attempt {attempt + 1}/{args.retries + 1} failed: {type(e).__name__}: {e}")
                stray = args.h5_dir / cond.filename
                stray.unlink(missing_ok=True)
                purge_incomplete(args.h5_dir)
                if attempt == args.retries:
                    failed.append(cond.key)
                    manifest[cond.key] = {
                        "stain": cond.stain, "scanner": cond.scanner,
                        "filename": cond.filename, "verified": False, "error": repr(e),
                    }
                    save_manifest(manifest_path, manifest)
                else:
                    time.sleep(5)

        resting = tree_bytes(args.h5_dir)
        # A .h5 for a slide we have not reached yet is a legitimate pre-existing source.
        # A .h5 for a slide already past is a copy we failed to delete -- that is the
        # duplication this loop exists to avoid.
        pending = {c.filename for c in todo[n:]}
        strays = [s for s in args.h5_dir.glob("*.tif.h5") if s.name not in pending]
        log(f"  resting usage {resting / GIB:.1f} GiB | free {free_gb(args.h5_dir):.0f} GB "
            f"| ok {ok} failed {len(failed)}")
        if strays:
            log(f"ERROR: leftover .h5 files {[s.name for s in strays]} -- stopping")
            return 3
        if resting / 1e9 > args.max_resting_gb:
            log(f"ERROR: resting usage {resting / 1e9:.0f} GB > {args.max_resting_gb:.0f} GB "
                "-- something is duplicating, stopping")
            return 3

    wall = time.perf_counter() - t_start
    log(f"done in {wall / 60:.1f} min: {ok} succeeded, {len(failed)} failed {failed}")
    log(f"manifest: {manifest_path}")
    verified_total = sum(1 for v in manifest.values() if v.get("verified"))
    log(f"manifest now records {verified_total}/91 verified slides")
    return 0 if not failed else 1


def already_done_name(filename: str, out_dir: Path, manifest: dict) -> bool:
    from waivphaet.data.conditions import parse_filename

    try:
        return already_done(parse_filename(filename), out_dir, manifest)
    except ValueError:
        return False


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
