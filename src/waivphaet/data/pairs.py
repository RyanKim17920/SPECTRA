"""Registered-pair sampler with the same-condition negative constraint.

PLAN.md 2 in code
-------------------

* **Positive** = same tile index ``i``, two *different* conditions ``c1 != c2``. PLISM's
  Elastix registration means tile ``i`` is the same tissue location in every slide, so
  the positive is a real photograph of the same tissue under different acquisition --
  not a synthetic augmentation. (Residual misregistration is ~5-50 px, so treat
  positives as near-identical *shifted crops*, never as pixel-exact.)

* **Negative** = different tile index ``j != i``, **same condition as the anchor**.
  This is the one load-bearing detail. If negatives spanned conditions, "different
  scanner" would be a partially-correct shortcut for "different tile", and InfoNCE
  would be rewarded for *retaining* acquisition signal -- the exact opposite of what we
  want. ScanGen encodes this as its "different specimen, same scanner" repulsion term.

Batch construction
------------------
InfoNCE takes its negatives from the other entries of the batch, so the constraint has
to be enforced *structurally*, in how the batch is laid out -- you cannot bolt it on in
the loss. We therefore emit **condition-homogeneous anchor groups**:

    group g:  anchors  a[g,0..G-1]  ALL from condition  c_anchor[g]
              positives p[g,0..G-1] each from its own   c_pos[g,k] != c_anchor[g]

Within a group every anchor shares a condition, so for anchor ``k`` the other ``G-1``
anchors are valid same-condition negatives. A batch is ``n_groups`` such groups; the
loss masks across groups (see :mod:`waivphaet.train.contrastive`). Tile indices are
unique within a group, so no "negative" is secretly the anchor's own tile.

Per-anchor positive conditions are drawn *independently* so a group does not degenerate
into a single (c_anchor -> c_pos) direction, which would let the model learn one
pairwise offset instead of general invariance (PLAN.md 0, Fig 3: the shift is a
near-linear offset per scanner -- easy to overfit one direction of).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from waivphaet.data.conditions import NUM_TILES, Condition, ConditionSplit, default_split
from waivphaet.data.repack import open_slide

__all__ = [
    "PairBatch",
    "PairBatchSampler",
    "RegisteredPairDataset",
    "collate_pair_batch",
    "build_pair_loader",
]


@dataclass(frozen=True)
class PairBatch:
    """One batch plan, produced by the sampler and consumed by the dataset.

    Shapes are ``(n_groups, group_size)`` throughout. ``anchor_cond`` is broadcast per
    group because the constraint *is* that it is constant within a group.
    """

    tile_idx: np.ndarray  # (n_groups, group_size) int64 -- tile index, unique within a group
    anchor_cond: np.ndarray  # (n_groups,) int64 -- condition index, shared by the whole group
    positive_cond: np.ndarray  # (n_groups, group_size) int64 -- != anchor_cond, per anchor

    @property
    def n_groups(self) -> int:
        return int(self.tile_idx.shape[0])

    @property
    def group_size(self) -> int:
        return int(self.tile_idx.shape[1])


class PairBatchSampler(Sampler[PairBatch]):
    """Yields :class:`PairBatch` plans honouring the same-condition negative rule.

    Args:
        conditions: the conditions to sample from. Pass ``split.train`` for training and
            ``split.heldout`` for the held-out-condition eval -- PLAN.md 3 risk 3 says
            held-out-*condition* splits are the only check against tile memorisation.
        n_groups: condition-homogeneous groups per batch.
        group_size: anchors per group. The in-group negative count is ``group_size - 1``,
            so this is the knob that controls negative hardness/volume.
        batches_per_epoch: this is an infinite sampling problem (16,278 tiles x 91
            conditions x pair choices); an "epoch" is just a checkpoint interval.
        tile_indices: restrict to a subset of tile locations (e.g. a held-out tile split).
        seed / epoch: ``seed + epoch`` seeds the RNG, so distributed ranks agree and runs
            replay exactly.
    """

    def __init__(
        self,
        conditions: Sequence[Condition],
        *,
        n_groups: int = 8,
        group_size: int = 32,
        batches_per_epoch: int = 1000,
        tile_indices: np.ndarray | None = None,
        seed: int = 0,
    ) -> None:
        if len(conditions) < 2:
            raise ValueError("need >=2 conditions: a positive requires a different condition")
        if group_size < 2:
            raise ValueError("group_size must be >=2 or a group has no in-group negatives")
        self.conditions = list(conditions)
        self.n_groups = n_groups
        self.group_size = group_size
        self.batches_per_epoch = batches_per_epoch
        self.tiles = (
            np.arange(NUM_TILES, dtype=np.int64) if tile_indices is None
            else np.asarray(tile_indices, dtype=np.int64)
        )
        if self.tiles.size < group_size:
            raise ValueError(f"need >= group_size ({group_size}) tiles, have {self.tiles.size}")
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[PairBatch]:
        rng = np.random.default_rng(self.seed + self.epoch)
        n_cond = len(self.conditions)
        for _ in range(self.batches_per_epoch):
            anchor_cond = rng.integers(0, n_cond, size=self.n_groups)
            # unique tiles within a group -> no in-group "negative" is the anchor's own tile
            tile_idx = np.stack(
                [rng.choice(self.tiles, size=self.group_size, replace=False)
                 for _ in range(self.n_groups)]
            )
            # positive condition != anchor condition, drawn per anchor.
            # Trick: draw in [0, n_cond-1) and shift past the anchor -> uniform over the
            # n_cond-1 valid conditions with no rejection loop.
            offs = rng.integers(0, n_cond - 1, size=(self.n_groups, self.group_size))
            positive_cond = offs + (offs >= anchor_cond[:, None])
            yield PairBatch(
                tile_idx=tile_idx.astype(np.int64),
                anchor_cond=anchor_cond.astype(np.int64),
                positive_cond=positive_cond.astype(np.int64),
            )


class RegisteredPairDataset(Dataset):
    """Materialises a :class:`PairBatch` plan into uint8 tile tensors.

    Reads from the contiguous memmaps written by :mod:`waivphaet.data.repack`. Slides are
    opened lazily and cached per worker process -- a memmap is not fork-safe to share, and
    with 91 slides we want one open handle per worker, not per item.

    ``__getitem__`` takes a :class:`PairBatch` (batch-level indexing), so wire this up
    with ``DataLoader(..., sampler=PairBatchSampler(...), batch_size=None)``.
    """

    def __init__(
        self,
        packed_dir: Path | str,
        conditions: Sequence[Condition],
        *,
        transform=None,
    ) -> None:
        self.packed_dir = Path(packed_dir)
        self.conditions = list(conditions)
        self.transform = transform
        self._slides: dict[int, np.memmap] = {}
        missing = [c.filename for c in self.conditions if not self._path(c).exists()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} condition(s) not repacked under {self.packed_dir}: "
                f"{missing[:3]}{'...' if len(missing) > 3 else ''}. Run `waiv-repack` first."
            )

    def _path(self, c: Condition) -> Path:
        return self.packed_dir / f"{c.slide_id.replace('.tif', '')}.npy"

    def _slide(self, cond_idx: int) -> np.memmap:
        mm = self._slides.get(cond_idx)
        if mm is None:
            c = self.conditions[cond_idx]
            mm = open_slide(self.packed_dir, c.slide_id.replace(".tif", ""))
            self._slides[cond_idx] = mm
        return mm

    def _gather(self, cond: np.ndarray, tiles: np.ndarray) -> np.ndarray:
        """Fetch tiles given equal-shaped condition / tile-index arrays -> (..., 224,224,3)."""
        flat_c, flat_t = cond.reshape(-1), tiles.reshape(-1)
        out = np.empty((flat_c.size, 224, 224, 3), dtype=np.uint8)
        # group by slide so each memmap is touched once, in ascending offset order
        order = np.argsort(flat_c, kind="stable")
        for ci in np.unique(flat_c):
            sel = order[flat_c[order] == ci]
            slide = self._slide(int(ci))
            for pos in sel[np.argsort(flat_t[sel], kind="stable")]:
                out[pos] = slide[int(flat_t[pos])]
        return out.reshape(*cond.shape, 224, 224, 3)

    def __getitem__(self, batch: PairBatch) -> dict[str, torch.Tensor]:
        anchor_cond = np.broadcast_to(
            batch.anchor_cond[:, None], batch.tile_idx.shape
        )
        anchors = self._gather(anchor_cond, batch.tile_idx)
        positives = self._gather(batch.positive_cond, batch.tile_idx)
        item = {
            "anchor": torch.from_numpy(anchors),  # (G, S, 224,224,3) uint8
            "positive": torch.from_numpy(positives),
            "tile_idx": torch.from_numpy(batch.tile_idx),
            "anchor_cond": torch.from_numpy(np.ascontiguousarray(anchor_cond)),
            "positive_cond": torch.from_numpy(batch.positive_cond),
            # group id -> which anchors may serve as each other's negatives
            "group_id": torch.arange(batch.n_groups).repeat_interleave(batch.group_size),
        }
        if self.transform is not None:
            item["anchor"] = self.transform(item["anchor"])
            item["positive"] = self.transform(item["positive"])
        return item

    def __len__(self) -> int:  # pragma: no cover - batch-indexed, length is nominal
        return NUM_TILES


def collate_pair_batch(item: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Flatten ``(G, S, ...)`` to ``(G*S, ...)``; ``group_id`` marks the negative blocks."""
    out = {}
    for k, v in item.items():
        out[k] = v if k == "group_id" else v.reshape(-1, *v.shape[2:])
    return out


def build_pair_loader(
    packed_dir: Path | str,
    *,
    split: ConditionSplit | None = None,
    subset: str = "train",
    n_groups: int = 8,
    group_size: int = 32,
    batches_per_epoch: int = 1000,
    num_workers: int = 8,
    seed: int = 0,
    transform=None,
    conditions: Sequence[Condition] | None = None,
) -> torch.utils.data.DataLoader:
    """Convenience wiring. ``subset`` is ``"train"`` or ``"heldout"`` (PLAN.md 3 phase 7)."""
    if conditions is None:
        split = split or default_split()
        conditions = split.train if subset == "train" else split.heldout
    ds = RegisteredPairDataset(packed_dir, conditions, transform=transform)
    sampler = PairBatchSampler(
        conditions,
        n_groups=n_groups,
        group_size=group_size,
        batches_per_epoch=batches_per_epoch,
        seed=seed,
    )
    return torch.utils.data.DataLoader(
        ds,
        sampler=sampler,
        batch_size=None,
        num_workers=num_workers,
        collate_fn=collate_pair_batch,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
