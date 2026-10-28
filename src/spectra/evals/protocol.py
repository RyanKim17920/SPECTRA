"""Per-benchmark, per-backbone pooling (the readout each benchmark is scored with).

* HEST    -- cls for most backbones, clsmean for the Virchow family.
* THUNDER -- the THUNDER leaderboard readout per backbone (cls or clsmean).
* PATHOROB -- ``clsmean`` for every backbone (PathoROB protocol).
* CPTAC   -- each backbone's official readout (cls, or clsmean for midnight / Virchow2 / Virchow).

Stdlib only.
"""

from __future__ import annotations

from .keys import resolve_repo

HEST_POOLING = {
    "owkin/phikon-v2": "cls",
    "SophontAI/OpenMidnight": "cls",
    "kaiko-ai/midnight": "cls",
    "paige-ai/Virchow": "clsmean",
    "paige-ai/Virchow2": "clsmean",
    "bioptimus/H-optimus-0": "cls",
    "MahmoodLab/UNI2-h": "cls",
}

THUNDER_POOLING = {
    "owkin/phikon-v2": "cls",
    "kaiko-ai/midnight": "clsmean",
    "paige-ai/Virchow2": "clsmean",
    "bioptimus/H-optimus-0": "cls",
    "MahmoodLab/UNI2-h": "cls",
    # not rows of the leaderboard's table: OpenMidnight uses the class token, Virchow v1 clsmean
    "SophontAI/OpenMidnight": "cls",
    "paige-ai/Virchow": "clsmean",
}

#: the backbones' OFFICIAL readouts: clsmean for the THUNDER-clsmean models + Virchow v1, cls
#: otherwise. NOTE midnight differs from the HEST table (cls there).
CPTAC_POOLING = {
    "bioptimus/H-optimus-0": "cls",
    "MahmoodLab/UNI2-h": "cls",
    "SophontAI/OpenMidnight": "cls",
    "owkin/phikon-v2": "cls",
    "kaiko-ai/midnight": "clsmean",
    "paige-ai/Virchow2": "clsmean",
    "paige-ai/Virchow": "clsmean",
}

TABLES = {"hest": HEST_POOLING, "thunder": THUNDER_POOLING, "cptac": CPTAC_POOLING}
BENCHES = ("hest", "thunder", "cptac", "pathorob")


def pooling_for(bench: str, backbone_repo: str, override: str | None = None) -> str:
    """-> "cls" | "clsmean" (| "mean" only via ``override``)."""
    if override:
        if override not in ("cls", "mean", "clsmean"):
            raise ValueError(override)
        return override
    b = bench.lower()
    if b not in BENCHES:
        raise KeyError(f"unknown bench {bench!r}; known {BENCHES}")
    if b == "pathorob":
        return "clsmean"
    repo = resolve_repo(backbone_repo)
    table = TABLES[b]
    if repo not in table:
        raise KeyError(f"no published {bench} pooling for {repo!r}; pass override= explicitly "
                       "(a guessed protocol yields a number that is not comparable)")
    return table[repo]
