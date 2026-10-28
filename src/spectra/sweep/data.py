"""Data for the Q8 trainer: condition pools, the grid sampler, the held-out split and the
fixed eval tile set.

The grid batch is ``spectra.data.grid``'s: C distinct conditions x ONE shared list of T
locations, flattened row-major to ``(C*T, 224, 224, 3)`` uint8. The sampler draws, per step,
``rng.choice(n_conditions, C)`` then ``rng.choice(train_locations, T)`` from
``default_rng(run.seed)``; it is resumable by step index.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler

from spectra.data.conditions import NUM_TILES, SCANNERS, STAINS, Condition, all_conditions
from spectra.data.grid import GridBatch, GridTileDataset, collate_grid_batch

from .config import REPO, SELECTION_HELDOUT_CORES, SweepConfig, cache_root, packed_dir

# ======================================================================================
# condition pools
# ======================================================================================


def cond_from_key(key: str) -> Condition:
    stain, scanner = key.split("_", 1)
    if stain not in STAINS or scanner not in SCANNERS:
        raise ValueError(f"unknown PLISM condition {key!r}")
    return Condition(stain=stain, scanner=scanner)


def slide_file(root: Path, c: Condition) -> Path:
    return Path(root) / f"{c.slide_id.replace('.tif', '')}.npy"


def present_conditions(root: str | Path) -> list[Condition]:
    """Conditions whose repacked memmap exists AND is marked verified in manifest.json."""
    root = Path(root)
    man_p = root / "manifest.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {}
    out = []
    for c in all_conditions():
        if not slide_file(root, c).exists():
            continue
        if man and not man.get(c.key, {}).get("verified", False):
            continue
        out.append(c)
    return out


def is_heldout_condition(cfg: SweepConfig, c: Condition) -> bool:
    return c.scanner in cfg.batch.heldout_scanners or c.stain in cfg.batch.heldout_stains


def resolve_conditions(cfg: SweepConfig) -> tuple[list[Condition], list[Condition]]:
    """(train, heldout) in the canonical all_conditions() order. Disjoint by assertion."""
    for sc in cfg.batch.heldout_scanners:
        if sc not in SCANNERS:
            raise ValueError(f"unknown held-out scanner {sc}")
    for st in cfg.batch.heldout_stains:
        if st not in STAINS:
            raise ValueError(f"unknown held-out stain {st}")
    present = present_conditions(packed_dir(cfg))
    train = [c for c in present if not is_heldout_condition(cfg, c)]
    held = [c for c in present if is_heldout_condition(cfg, c)]
    assert not ({c.key for c in train} & {c.key for c in held}), "train/heldout overlap"
    if len(train) < cfg.batch.n_cond:
        raise ValueError(f"only {len(train)} train conditions for C={cfg.batch.n_cond}")
    return train, held


def _resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def load_group_labels(path: str) -> np.ndarray:
    """(16278,) int64 tissue-core label per PLISM location, from a ``.json`` (``labels`` list,
    as packaged) or a ``.npy`` file."""
    p = _resolve_path(path)
    if p.suffix == ".json":
        d = json.loads(p.read_text())
        lab = np.asarray(d["labels"], dtype=np.int64)
        if "sha256_int64" in d and hashlib.sha256(lab.tobytes()).hexdigest() != d["sha256_int64"]:
            raise ValueError(f"{p}: labels do not match their recorded sha256")
    else:
        lab = np.load(p).astype(np.int64)
    if lab.shape != (NUM_TILES,):
        raise ValueError(f"group labels {path} have shape {lab.shape}, want ({NUM_TILES},)")
    return lab


def eval_locations(labels: np.ndarray, cores: Sequence[int], per_core: int, seed: int) -> np.ndarray:
    """Deterministic sample of ``per_core`` locations from each eval core (sorted)."""
    rng = np.random.default_rng(seed)
    picks = []
    for c in sorted(cores):
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            raise ValueError(f"eval core {c} has no locations")
        picks.append(rng.choice(idx, size=min(per_core, idx.size), replace=False))
    return np.sort(np.concatenate(picks))


def heldout_location_mask(cfg: SweepConfig, labels: np.ndarray) -> np.ndarray:
    """(16278,) bool: True = location of a held-out core (never trained on)."""
    cores = [int(c) for c in cfg.batch.heldout_cores]
    return np.isin(labels, cores) if cores else np.zeros(NUM_TILES, dtype=bool)


def train_location_pool(held_mask: np.ndarray) -> np.ndarray | None:
    """Locations the sampler may draw (None = all 16278)."""
    return None if not held_mask.any() else np.where(~held_mask)[0].astype(np.int64)


# ======================================================================================
# the grid sampler
# ======================================================================================


class SweepGridSampler(Sampler[GridBatch]):
    """C conditions x T shared locations per step; resumable by plan index.

    Reproduces ``spectra.data.grid.GridBatchSampler`` for the same seed (rng =
    default_rng(seed + epoch); cond = choice(n, C); tiles = choice(pool, T))."""

    def __init__(self, n_conditions: int, *, n_cond: int, n_tiles: int, steps: int,
                 seed: int, tiles: np.ndarray | None = None) -> None:
        self.n_conditions = n_conditions
        self.n_cond, self.n_tiles = n_cond, n_tiles
        self.batches_per_epoch = steps
        self.seed = seed
        self.epoch = 0
        self.start_index = 0
        self.tiles = np.arange(NUM_TILES, dtype=np.int64) if tiles is None else np.asarray(tiles, np.int64)
        if self.tiles.size < n_tiles:
            raise ValueError(f"only {self.tiles.size} locations for T={n_tiles}")

    def set_start_index(self, i: int) -> None:
        self.start_index = int(i)

    def __len__(self) -> int:
        return max(self.batches_per_epoch - self.start_index, 0)

    def _draw(self, rng: np.random.Generator) -> GridBatch:
        cond_idx = rng.choice(self.n_conditions, size=self.n_cond, replace=False)
        tile_idx = rng.choice(self.tiles, size=self.n_tiles, replace=False)
        return GridBatch(cond_idx=cond_idx.astype(np.int64), tile_idx=tile_idx.astype(np.int64))

    def __iter__(self) -> Iterator[GridBatch]:
        rng = np.random.default_rng(self.seed + self.epoch)
        skip, self.start_index = self.start_index, 0
        for _ in range(skip):
            self._draw(rng)
        for _ in range(self.batches_per_epoch - skip):
            b = self._draw(rng)
            b.validate(self.n_conditions)
            yield b


def build_train_loader(cfg: SweepConfig, train_conds: Sequence[Condition], tiles, start_step: int = 0):
    sampler = SweepGridSampler(len(train_conds), n_cond=cfg.batch.n_cond, n_tiles=cfg.batch.n_tiles,
                               steps=cfg.optim.steps, seed=cfg.run.seed, tiles=tiles)
    sampler.set_start_index(start_step)
    ds = GridTileDataset(str(packed_dir(cfg)), list(train_conds))
    return torch.utils.data.DataLoader(
        ds, sampler=sampler, batch_size=None, num_workers=cfg.batch.workers,
        collate_fn=collate_grid_batch, pin_memory=True,
        persistent_workers=False, prefetch_factor=2 if cfg.batch.workers > 0 else None,
    )


# ======================================================================================
# fixed eval tile set (cached as a contiguous .npy)
# ======================================================================================


def _gather(root: Path, conds: Sequence[Condition], locs: np.ndarray) -> np.ndarray:
    out = np.empty((len(conds), locs.size, 224, 224, 3), dtype=np.uint8)
    order = np.argsort(locs)
    for ci, c in enumerate(conds):
        mm = np.load(slide_file(root, c), mmap_mode="r")
        out[ci, order] = mm[locs[order]]
    return out


def cached_tiles(tag: str, builder) -> np.ndarray:
    root = cache_root()
    root.mkdir(parents=True, exist_ok=True)
    p = root / f"{tag}.npy"
    if p.exists():
        return np.load(p, mmap_mode="r")
    import fcntl

    with open(root / f"{tag}.lock", "w") as lk:  # one builder per tag across jobs
        fcntl.flock(lk, fcntl.LOCK_EX)
        if p.exists():
            return np.load(p, mmap_mode="r")
        arr = builder()
        tmp = root / f"{tag}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}.npy"
        np.save(tmp, arr)
        os.replace(tmp, p)
    return np.load(p, mmap_mode="r")


def eval_tile_set(cfg: SweepConfig, conds: Sequence[Condition], locs: np.ndarray) -> np.ndarray:
    """(K, N, 224, 224, 3) uint8 for K eval conditions x N locations."""
    key = json.dumps([[c.key for c in conds], locs.tolist()])
    tag = "eval_" + hashlib.sha1(key.encode()).hexdigest()[:12]
    return cached_tiles(tag, lambda: _gather(packed_dir(cfg), conds, locs))


def pick_eval_conditions(cfg: SweepConfig) -> list[Condition]:
    """Every present condition minus eval.exclude_conditions (train AND held-out axes)."""
    drop = {cond_from_key(k).key for k in cfg.eval.exclude_conditions}
    return [x for x in present_conditions(packed_dir(cfg)) if x.key not in drop]


def pick_eval_cores(cfg: SweepConfig) -> tuple[list[int], bool]:
    """(cores, in_train): eval.cores, else the held-out cores, else the selection cores."""
    held = [int(x) for x in cfg.batch.heldout_cores]
    cores = [int(x) for x in cfg.eval.cores] or held or list(SELECTION_HELDOUT_CORES)
    return cores, not set(cores) <= set(held)


# ======================================================================================
# tensors
# ======================================================================================


def to_float01(images_uint8: torch.Tensor) -> torch.Tensor:
    """(B,224,224,3) uint8 -> (B,3,224,224) float32 in [0,1]."""
    return images_uint8.permute(0, 3, 1, 2).float().div_(255.0)


def normalize(x01: torch.Tensor, mean, std) -> torch.Tensor:
    m = torch.tensor(mean, device=x01.device, dtype=x01.dtype).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x01.device, dtype=x01.dtype).view(1, 3, 1, 1)
    return (x01 - m) / s


def scanner_stain_labels(conds: Sequence[Condition]) -> dict[str, np.ndarray]:
    return {
        "scanner": np.array([SCANNERS.index(c.scanner) for c in conds]),
        "stain": np.array([STAINS.index(c.stain) for c in conds]),
        "condition": np.array([STAINS.index(c.stain) * len(SCANNERS) + SCANNERS.index(c.scanner)
                               for c in conds]),
    }
