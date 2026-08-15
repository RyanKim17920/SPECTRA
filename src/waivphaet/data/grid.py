"""Shared-tile GRID sampler: every image is both an anchor and a query.

Why this exists (contrast with :mod:`waivphaet.data.pairs`)
-----------------------------------------------------------
The pair sampler draws ``n_groups`` condition-homogeneous ANCHOR groups, each over its
OWN independently-drawn tile set, and gives every anchor one positive from its own
independently-drawn different condition. The positives are therefore **query-only**: each
one sits in a condition nobody else in the batch shares, so it can never join a
condition-homogeneous candidate set. Half of every forward pass produces embeddings that
appear in exactly one row of the loss.

The grid sampler removes that waste by sharing ONE tile set across all condition groups.
PLISM is a complete 91 x 16,278 grid -- every tile index exists under every condition, and
Elastix registration means tile ``i`` is the same tissue location everywhere -- so a
``C x T`` block of (condition, tile) cells is fully populated with real photographs::

    conditions c_0 .. c_{C-1}   (DISTINCT)
    tiles      t_0 .. t_{T-1}   (unique, and THE SAME LIST for every condition)

    image[a, t]  =  tile t_t  photographed under condition c_a

Now every image plays both roles. Row ``(a, t)`` uses ``z[a, t]`` as the query and the
whole of condition group ``b != a`` as its candidate set::

    images/step        = C * T           (there is no separate positive tensor)
    negatives per row  = T - 1
    query rows R       = C * (C-1) * T

The candidate set for any row is one single condition group, so the same-condition-negative
invariant that :mod:`waivphaet.data.pairs` exists to protect is **preserved exactly**:
acquisition is constant down the candidate row and carries zero discriminative information.

The new load-bearing invariant
------------------------------
``pairs.py``'s load-bearing detail is "negatives share the anchor's condition". This module
inherits that one and adds a second, which is just as silent when broken:

    **every condition group must use the SAME tile list, in the SAME ORDER.**

The loss identifies the positive by POSITION -- ``z[a, t]`` matches ``z[b, t]``. If group
``a`` and group ``b`` were drawn over different tiles (or the same tiles in a different
order), position ``t`` would not be the same tissue, every "positive" would be a
mislabelled pair, and the model would be trained to match unrelated tissue. That failure
still produces a perfectly plausible falling loss curve, which is why
:func:`assert_grid_batch` runs on every step rather than being eyeballed once.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler

from waivphaet.data.conditions import NUM_TILES, Condition, ConditionSplit, default_split
from waivphaet.data.pairs import RegisteredPairDataset

__all__ = [
    "GridBatch",
    "GridBatchSampler",
    "GridTileDataset",
    "assert_grid_batch",
    "collate_grid_batch",
    "build_grid_loader",
]


def assert_grid_batch(
    batch: dict[str, torch.Tensor], *, allowed_conditions: set[int] | None = None
) -> dict[str, float]:
    """Assert the grid invariants on the *collated, flattened* ``(C*T, ...)`` batch.

    Same spirit as :func:`waivphaet.data.pairs.assert_same_condition_negatives`: this
    checks the thing the loss actually consumes, after the flatten, and returns observed
    statistics so the train loop can log evidence instead of trusting the sampler.

    Raises on, in order:

    1. duplicate condition indices in the batch's condition list -- two "different"
       condition groups that are secretly the same acquisition, so the cross-group
       "positive" is the same image twice and the row is trivially solvable;
    2. tile sets (or tile ORDER) that differ between condition groups -- **the new
       load-bearing invariant**; see the module docstring;
    3. duplicate tiles within the shared tile list -- two candidate columns hold the same
       tissue, so one true match is scored as a negative (a false negative);
    4. a condition index outside ``allowed_conditions`` -- a held-out condition leaking
       into a training batch;
    5. a candidate block that is not condition-homogeneous -- the original PLAN.md 2
       constraint: cross-condition candidates make "different acquisition" a
       partially-correct shortcut for "different tile".
    """
    gid = batch["group_id"]
    cond = batch["cond_idx"]
    tiles = batch["tile_idx"]
    groups = torch.unique(gid)
    n_cond = int(groups.numel())

    # --- 5. every candidate block is one single condition, and 1. those conditions differ
    per_group_cond: list[int] = []
    per_group_tiles: list[torch.Tensor] = []
    for g in groups.tolist():
        sel = gid == g
        c = torch.unique(cond[sel])
        if c.numel() != 1:
            raise AssertionError(
                f"grid group {g} mixes conditions {c.tolist()}: its members are NOT a "
                "condition-homogeneous candidate set, so acquisition becomes a "
                "partially-correct shortcut for tile identity"
            )
        per_group_cond.append(int(c.item()))
        per_group_tiles.append(tiles[sel])

    if len(set(per_group_cond)) != len(per_group_cond):
        dupes = sorted({c for c in per_group_cond if per_group_cond.count(c) > 1})
        raise AssertionError(
            f"duplicate condition index/indices {dupes} across grid groups: two groups are "
            "the same acquisition, so their cross-group 'positive' is the same image twice"
        )

    # --- 2. the tile list is SHARED, identically ordered, by every condition group
    reference = per_group_tiles[0]
    n_tiles = int(reference.numel())
    for g, t in zip(groups.tolist(), per_group_tiles):
        if t.numel() != n_tiles:
            raise AssertionError(
                f"grid group {g} has {t.numel()} tiles but group {groups[0].item()} has "
                f"{n_tiles}: the grid is ragged, so position t is not the same tissue "
                "in every group"
            )
        if not bool(torch.equal(t, reference)):
            raise AssertionError(
                f"grid group {g} does not use the shared tile list in the shared order "
                "(THE load-bearing grid invariant): position t is then a DIFFERENT tissue "
                "in group "
                f"{g} than in group {groups[0].item()}, so every positive the loss scores "
                "at that position is a mislabelled pair -- and the loss would still fall"
            )

    # --- 3. no tile repeats inside the shared list
    if int(torch.unique(reference).numel()) != n_tiles:
        raise AssertionError(
            "the shared tile list repeats a tile index: two candidate columns hold the "
            "same tissue, so a true match is scored as a negative"
        )

    # --- position bookkeeping must agree with the flatten the loss reshapes by
    if "tile_pos" in batch:
        expected = torch.arange(n_tiles, device=batch["tile_pos"].device).repeat(n_cond)
        if not bool(torch.equal(batch["tile_pos"], expected)):
            raise AssertionError(
                "tile_pos is not arange(T).repeat(C): the (C*T,) flatten does not reshape "
                "to (C, T, ...) row-major, so grid_info_nce would pair the wrong cells"
            )

    # --- 4. held-out leakage
    if allowed_conditions is not None:
        leaked = set(per_group_cond) - allowed_conditions
        if leaked:
            raise AssertionError(
                f"condition indices {sorted(leaked)} are outside the loader's condition "
                "list; a held-out condition has leaked into a training batch"
            )

    return {
        "n_cond": float(n_cond),
        "n_tiles": float(n_tiles),
        "negatives_per_anchor": float(n_tiles - 1),
        "n_rows": float(n_cond * (n_cond - 1) * n_tiles),
        "distinct_conditions": float(len(set(per_group_cond))),
    }


@dataclass(frozen=True)
class GridBatch:
    """One grid batch plan: a condition list and ONE tile list shared by all of them."""

    cond_idx: np.ndarray  # (C,) int64 -- DISTINCT condition indices
    tile_idx: np.ndarray  # (T,) int64 -- unique tile indices, shared by every condition

    @property
    def n_cond(self) -> int:
        return int(self.cond_idx.shape[0])

    @property
    def n_tiles(self) -> int:
        return int(self.tile_idx.shape[0])

    def validate(self, n_conditions: int | None = None) -> None:
        """Assert the plan-level invariants. Cheap; run on every batch.

        The collated-batch check :func:`assert_grid_batch` is the authoritative one (it
        sees what the loss consumes); this is the same three properties one step earlier,
        so a bad plan never reaches the pixel gather.
        """
        if self.cond_idx.ndim != 1 or self.tile_idx.ndim != 1:
            raise AssertionError("cond_idx and tile_idx must both be 1-D")
        if self.n_cond < 2:
            raise AssertionError(
                "need >=2 conditions in a grid batch: a query row's candidates come from a "
                "DIFFERENT condition group, so C=1 has no rows at all"
            )
        if self.n_tiles < 2:
            raise AssertionError("need >=2 tiles or a row has no negatives")
        if np.unique(self.cond_idx).size != self.n_cond:
            raise AssertionError(
                "duplicate condition index in the grid plan: two groups would be the same "
                "acquisition and their cross-group positive the same image twice"
            )
        if np.unique(self.tile_idx).size != self.n_tiles:
            raise AssertionError(
                "duplicate tile index in the shared tile list: a true match would be "
                "scored as a negative"
            )
        if n_conditions is not None:
            if int(self.cond_idx.min()) < 0 or int(self.cond_idx.max()) >= n_conditions:
                raise AssertionError(
                    f"condition index out of range [0,{n_conditions}); a condition outside "
                    "the sampler's list (e.g. a held-out one) has leaked into the batch"
                )


class GridBatchSampler(Sampler[GridBatch]):
    """Yields :class:`GridBatch` plans: C distinct conditions x T shared tiles.

    Args:
        conditions: conditions to sample from. ``split.train`` for training,
            ``split.heldout`` for the held-out-condition eval.
        n_cond: condition groups per batch, ``C``. Drawn WITHOUT replacement, so it may
            not exceed ``len(conditions)``. Query rows scale as ``C*(C-1)``.
        n_tiles: tiles per condition, ``T``. Negatives per row is ``T-1``.
        batches_per_epoch: an "epoch" is just a checkpoint interval here.
        tile_indices: restrict to a subset of tile locations.
        seed: ``seed + epoch`` seeds the RNG so runs replay exactly.
    """

    def __init__(
        self,
        conditions: Sequence[Condition],
        *,
        n_cond: int = 24,
        n_tiles: int = 100,
        batches_per_epoch: int = 1000,
        tile_indices: np.ndarray | None = None,
        seed: int = 0,
    ) -> None:
        if n_cond < 2:
            raise ValueError(
                "n_cond must be >=2: a grid row's candidates are a DIFFERENT condition "
                "group, so C=1 produces zero query rows"
            )
        if n_tiles < 2:
            raise ValueError("n_tiles must be >=2 or a row has no negatives")
        if len(conditions) < n_cond:
            raise ValueError(
                f"--grid-conditions {n_cond} exceeds the {len(conditions)} available "
                "conditions; conditions are drawn WITHOUT replacement (a duplicated "
                "condition group would make its cross-group positive the same image twice)"
            )
        self.conditions = list(conditions)
        self.n_cond = n_cond
        self.n_tiles = n_tiles
        self.batches_per_epoch = batches_per_epoch
        self.tiles = (
            np.arange(NUM_TILES, dtype=np.int64) if tile_indices is None
            else np.asarray(tile_indices, dtype=np.int64)
        )
        if self.tiles.size < n_tiles:
            raise ValueError(f"need >= n_tiles ({n_tiles}) tiles, have {self.tiles.size}")
        self.seed = seed
        self.epoch = 0
        #: Resume support -- see :meth:`set_start_index`. 0 means "replay from the top",
        #: which is what every non-resumed run does.
        self.start_index = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_start_index(self, start_index: int) -> None:
        """Skip the first *start_index* batch plans of the NEXT epoch only (resume).

        The plan sequence is a pure function of ``(seed, epoch, position)``: ``__iter__``
        reseeds from ``seed + epoch`` and draws position by position. So a run that is
        preempted at step ``s`` and restarted can reproduce the exact data stream it would
        have seen by advancing the RNG through the first ``s`` draws and discarding them --
        which costs microseconds of numpy, versus actually loading ``s`` batches of images.

        Self-clearing: it applies to the next ``__iter__`` and then resets to 0, so the
        following epochs replay in full. That is what reproduces a continuous run, whose
        step ``s`` sits at position ``s % batches_per_epoch`` of pass ``s // batches_per_epoch``.

        This ADVANCES the rng exactly as the yielding path does -- see ``_draw``, which is
        the single place the draws live precisely so skip and yield cannot drift apart.
        """
        if start_index < 0:
            raise ValueError(f"start_index must be >= 0, got {start_index}")
        if start_index > self.batches_per_epoch:
            raise ValueError(
                f"start_index {start_index} exceeds batches_per_epoch "
                f"{self.batches_per_epoch}: the caller should have taken it modulo the "
                "epoch length before handing it over"
            )
        self.start_index = start_index

    def __len__(self) -> int:
        return max(self.batches_per_epoch - self.start_index, 0)

    def _draw(self, rng: np.random.Generator, n_available: int) -> GridBatch:
        # WITHOUT replacement on both axes: duplicate conditions break invariant 1,
        # duplicate tiles break invariant 3.
        cond_idx = rng.choice(n_available, size=self.n_cond, replace=False)
        tile_idx = rng.choice(self.tiles, size=self.n_tiles, replace=False)
        return GridBatch(
            cond_idx=cond_idx.astype(np.int64),
            tile_idx=tile_idx.astype(np.int64),
        )

    def __iter__(self) -> Iterator[GridBatch]:
        rng = np.random.default_rng(self.seed + self.epoch)
        n_available = len(self.conditions)
        skip, self.start_index = self.start_index, 0
        for _ in range(skip):
            self._draw(rng, n_available)  # advance the rng, discard the plan
        for _ in range(self.batches_per_epoch - skip):
            batch = self._draw(rng, n_available)
            batch.validate(n_available)
            yield batch


class GridTileDataset(RegisteredPairDataset):
    """Materialises a :class:`GridBatch` plan into a ``(C, T, 224,224,3)`` uint8 block.

    Subclasses :class:`~waivphaet.data.pairs.RegisteredPairDataset` purely to reuse its
    lazily-opened per-worker memmap cache and its slide-ordered ``_gather``; the pair
    ``__getitem__`` is replaced, not extended, and the parent class is left untouched.

    Wire up with ``DataLoader(..., sampler=GridBatchSampler(...), batch_size=None)``.
    """

    def __getitem__(self, batch: GridBatch) -> dict[str, torch.Tensor]:
        # Re-assert in the worker process: last point before real pixels are gathered.
        batch.validate(len(self.conditions))
        c, t = batch.n_cond, batch.n_tiles
        # The whole point of the grid: ONE tile list, broadcast across every condition.
        cond = np.broadcast_to(batch.cond_idx[:, None], (c, t))
        tiles = np.broadcast_to(batch.tile_idx[None, :], (c, t))
        images = self._gather(cond, tiles)  # (C, T, 224,224,3) uint8
        item = {
            "image": torch.from_numpy(images),
            "cond_idx": torch.from_numpy(np.ascontiguousarray(cond)),
            "tile_idx": torch.from_numpy(np.ascontiguousarray(tiles)),
            # group_id -> which images form each other's candidate set (one condition)
            "group_id": torch.arange(c).repeat_interleave(t),
            # tile_pos -> the position the loss matches on, across groups
            "tile_pos": torch.arange(t).repeat(c),
            "n_cond": torch.tensor(c),
            "n_tiles": torch.tensor(t),
        }
        if self.transform is not None:
            item["image"] = self.transform(item["image"])
        return item


def collate_grid_batch(item: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Flatten ``(C, T, ...)`` to ``(C*T, ...)`` row-major; scalars and 1-D ids pass through.

    Row-major is load-bearing: :func:`waivphaet.train.contrastive.grid_info_nce` reshapes
    ``(C*T, D)`` straight back to ``(C, T, D)``, so index ``a*T + t`` must be condition
    ``a``, tile position ``t``. ``assert_grid_batch`` re-checks that via ``tile_pos``.
    """
    passthrough = {"group_id", "tile_pos", "n_cond", "n_tiles"}
    return {
        k: v if k in passthrough else v.reshape(-1, *v.shape[2:])
        for k, v in item.items()
    }


def build_grid_loader(
    packed_dir: Path | str,
    *,
    split: ConditionSplit | None = None,
    subset: str = "train",
    n_cond: int = 24,
    n_tiles: int = 100,
    batches_per_epoch: int = 1000,
    num_workers: int = 8,
    seed: int = 0,
    transform=None,
    conditions: Sequence[Condition] | None = None,
) -> torch.utils.data.DataLoader:
    """Convenience wiring, mirroring :func:`waivphaet.data.pairs.build_pair_loader`."""
    if conditions is None:
        split = split or default_split()
        conditions = split.train if subset == "train" else split.heldout
    ds = GridTileDataset(packed_dir, conditions, transform=transform)
    sampler = GridBatchSampler(
        conditions,
        n_cond=n_cond,
        n_tiles=n_tiles,
        batches_per_epoch=batches_per_epoch,
        seed=seed,
    )
    return torch.utils.data.DataLoader(
        ds,
        sampler=sampler,
        batch_size=None,
        num_workers=num_workers,
        collate_fn=collate_grid_batch,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
