"""PLISM acquisition-condition metadata.

PLISM ships 91 ``.h5`` files = 13 staining protocols x 7 scanners, every slide
Elastix-registered onto the same reference (``GMH_S60``). Filenames encode the
condition::

    {stain}_{scanner}_to_{reference}.tif.h5     e.g. GIVH_AT2_to_GMH_S60.tif.h5

A "condition" in PLAN.md 2 is the (scanner, stain) pair. It is the unit that:

* defines a **positive** -- same tile index, *different* condition;
* constrains a **negative** -- different tile index, *same* condition as the anchor
  (PLAN.md 2, the "one load-bearing detail": cross-condition negatives make
  "different scanner" a partially-correct shortcut for "different tile", which
  rewards *retaining* acquisition signal);
* defines the held-out split -- PLAN.md 3 phase 7 holds out 2 of 7 scanners and
  3-4 of 13 stains, because "held-out-*condition* splits are the only check"
  against tile-identity memorisation (PLAN.md 3, risk 3).

The 91 names are hard-coded here rather than globbed so that condition indices are
stable no matter how many ``.h5`` files happen to be present on a given machine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

# --- The PLISM design grid -------------------------------------------------------------
# Verified against `huggingface_hub.list_repo_files("owkin/plism-dataset")`: exactly
# 13 x 7 = 91 files, and every filename ends in `_to_GMH_S60.tif.h5`.

STAINS: tuple[str, ...] = (
    "GIV", "GIVH", "GM", "GMH", "GV", "GVH", "HR",
    "HRH", "KR", "KRH", "LM", "LMH", "MY",
)
SCANNERS: tuple[str, ...] = ("AT2", "GT450", "P", "S210", "S360", "S60", "SQ")

REGISTRATION_REFERENCE = "GMH_S60"
"""Every slide is warped onto this (stain, scanner). It is a constant, not a variable."""

NUM_TILES = 16_278
"""Tiles per slide. Byte-identical key order across all 91 files -- tile index i is the
same tissue location everywhere (verified in Phase 1). This is what makes positives free."""

_FILENAME_RE = re.compile(r"^(?P<stain>[A-Z]+)_(?P<scanner>[A-Z0-9]+)_to_(?P<ref>.+)\.tif\.h5$")


@dataclass(frozen=True, order=True)
class Condition:
    """One PLISM acquisition condition = one ``.h5`` file."""

    stain: str
    scanner: str

    @property
    def filename(self) -> str:
        return f"{self.stain}_{self.scanner}_to_{REGISTRATION_REFERENCE}.tif.h5"

    @property
    def slide_id(self) -> str:
        """Name plismbench uses for the per-slide feature directory (filename minus ``.h5``)."""
        return self.filename[: -len(".h5")]

    @property
    def key(self) -> str:
        return f"{self.stain}_{self.scanner}"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.key


def parse_filename(name: str) -> Condition:
    """Parse ``GIVH_AT2_to_GMH_S60.tif.h5`` (or a full path) into a :class:`Condition`."""
    base = name.rsplit("/", 1)[-1]
    m = _FILENAME_RE.match(base)
    if m is None:
        raise ValueError(f"not a PLISM slide filename: {name!r}")
    if m.group("ref") != REGISTRATION_REFERENCE:
        raise ValueError(
            f"unexpected registration reference {m.group('ref')!r} in {base!r}; "
            f"all PLISM slides register onto {REGISTRATION_REFERENCE}"
        )
    return Condition(stain=m.group("stain"), scanner=m.group("scanner"))


def all_conditions() -> list[Condition]:
    """All 91 conditions, in a deterministic order (stain-major, then scanner)."""
    return [Condition(s, sc) for s in STAINS for sc in SCANNERS]


# --- Held-out split --------------------------------------------------------------------


@dataclass(frozen=True)
class ConditionSplit:
    """A deterministic train / held-out partition over the 91 conditions.

    A condition is held out if *either* its scanner or its stain is held out. That makes
    the held-out set the union of two axis slices, which is what we want: it tests
    generalisation to an unseen scanner AND to an unseen stain, and (in the corner) to
    an unseen combination of both.
    """

    heldout_scanners: tuple[str, ...]
    heldout_stains: tuple[str, ...]

    @property
    def train(self) -> list[Condition]:
        return [c for c in all_conditions() if not self.is_heldout(c)]

    @property
    def heldout(self) -> list[Condition]:
        return [c for c in all_conditions() if self.is_heldout(c)]

    def is_heldout(self, c: Condition) -> bool:
        return c.scanner in self.heldout_scanners or c.stain in self.heldout_stains

    def summary(self) -> str:
        return (
            f"heldout scanners={list(self.heldout_scanners)} "
            f"stains={list(self.heldout_stains)} -> "
            f"{len(self.train)} train / {len(self.heldout)} heldout conditions"
        )


DEFAULT_HELDOUT_SCANNERS: tuple[str, ...] = ("GT450", "S210")
DEFAULT_HELDOUT_STAINS: tuple[str, ...] = ("HRH", "KR", "MY")
"""Default split: 2 of 7 scanners + 3 of 13 stains (PLAN.md 3, phase 7).

Leaves 5 training scanners, which is where ScanGen's ablation says cross-scanner gains
converge. ``GMH``/``S60`` are deliberately kept in training -- they are the registration
reference, so every other slide's geometry is defined relative to them.
``GT450`` (Leica) and ``S210``/``S60`` (Hamamatsu) are different vendors, so holding out
GT450 leaves no same-vendor twin to leak through.
"""


def default_split() -> ConditionSplit:
    return ConditionSplit(DEFAULT_HELDOUT_SCANNERS, DEFAULT_HELDOUT_STAINS)


def make_split(
    heldout_scanners: Iterable[str] = DEFAULT_HELDOUT_SCANNERS,
    heldout_stains: Iterable[str] = DEFAULT_HELDOUT_STAINS,
) -> ConditionSplit:
    """Build a split, validating that every named scanner/stain actually exists.

    Deterministic by construction -- the split is *named*, never sampled, so it is
    reproducible across machines and runs without carrying a seed.
    """
    sc = tuple(heldout_scanners)
    st = tuple(heldout_stains)
    unknown_sc = [s for s in sc if s not in SCANNERS]
    unknown_st = [s for s in st if s not in STAINS]
    if unknown_sc:
        raise ValueError(f"unknown scanner(s) {unknown_sc}; known: {list(SCANNERS)}")
    if unknown_st:
        raise ValueError(f"unknown stain(s) {unknown_st}; known: {list(STAINS)}")
    if len(sc) >= len(SCANNERS) or len(st) >= len(STAINS):
        raise ValueError("cannot hold out every scanner or every stain")
    split = ConditionSplit(sc, st)
    if len(split.train) < 2:
        raise ValueError("split leaves <2 training conditions; positives would be impossible")
    return split


def available_conditions(
    conditions: Sequence[Condition], present_filenames: Iterable[str]
) -> list[Condition]:
    """Intersect a condition list with what is actually on disk.

    Only 2 of 91 ``.h5`` files are local (PLAN.md 5 puts the full 224 GB on ``/data``),
    so smoke tests need to run on whatever subset exists.
    """
    present = {n.rsplit("/", 1)[-1] for n in present_filenames}
    return [c for c in conditions if c.filename in present]
