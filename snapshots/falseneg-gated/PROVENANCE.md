# falseneg-gated — the code that produced the gated-backbone rows

Vendored 2026-09-07 from `/admin/home/ryan.kim/waiv-snapshots/falseneg-gated`
(an uncommitted snapshot dir on volatile storage), verified byte-identical to that source
by `diff -rq` modulo the exclusions below.

**Why this exists:** the H-optimus-0, UNI2-h, Virchow v1 and OpenMidnight results were
trained and evaluated by THIS code, and it was tracked nowhere. `docs/` RUNBOOK §1.2 named
the volatile path above as the pin those runs used. Neither HEAD nor the other vendored
snapshot (`snapshots/falseneg-pinned`) can load those four backbones at all, so without
this directory four of the seven backbone columns are unreproducible — and the failure is
not a clean one (see below).

## Relationship to `snapshots/falseneg-pinned`

`falseneg-gated` is `falseneg-pinned` **plus** local-directory backbone loading. Verified
at vendoring time:

    scripts/                 identical (diff -rq, no differences)
    src/                     ONE file differs: waivphaet/models/encoder.py (268 diff lines)
    tests/, pyproject.toml   present here, not carried in the falseneg-pinned copy

So both snapshots run the same training loop, the same `mask_same_core` loss and the same
pool-head delegation; they differ only in how a backbone's weights are found.

The added code in `encoder.py` is:

* `BACKBONE_LOCAL_DIRS` — a repo-id → directory table, plus the `WAIV_BACKBONE_LOCAL_DIRS`
  environment override, consulted **before** the hub by everything that would otherwise ask
  the hub (loader dispatch, FFN-shape probe, normalisation lookup, weight load).
* `BACKBONE_TIMM_KWARGS` — the architecture kwargs transcribed from each model card, which
  a gated repo cannot supply because `pretrained=True` is unavailable.
* `_build_local_timm` — a STRICT load: any missing or unexpected key is fatal.
* Published normalisation statistics for `bioptimus/H-optimus-0` (its own H&E-corpus stats,
  NOT ImageNet's) and `MahmoodLab/UNI2-h`.

Why each of those is fatal-by-design rather than best-effort: without the local binding,
`_hub_config` 403s, swallows the error, returns `None`, `is_timm_backbone` therefore
answers `False`, and a timm ViT is routed into `AutoModel`, which dies on "Unrecognized
model" — an error naming neither gating nor the real architecture. Without the kwargs,
UNI2-h's `config.json` architecture name (`vit_giant_patch14_224`) builds timm's DEFAULTS
— embed_dim 1408, depth 40, plain MLP — a different model entirely that builds without
complaint. And `strict=False`, the usual reflex for a noisy load, would leave those blocks
randomly initialised and still train, evaluate and print a plausible number.

**Backwards compatibility.** All of this is reached only via `BACKBONE_LOCAL_DIRS` /
`WAIV_BACKBONE_LOCAL_DIRS`, both empty for hub-served backbones, so on phikon-v2, midnight
and Virchow2 this snapshot is behaviourally identical to `falseneg-pinned`. The two are
kept separate anyway, and `scripts/gentle.sbatch` selects between them by arm, so each
published row is reproduced by the bytes that produced it rather than by an equivalent.

## Which arms need it

`scripts/gentle.sbatch` defaults `WAIV_ARM` ∈ {`hoptimus`, `uni2`, `openmidnight`,
`virchow`} to this snapshot. Two ways the binding is supplied:

* `hoptimus` / `uni2` — in the table in this snapshot's `encoder.py`, pointing at `/data`.
* `openmidnight` / `virchow` — NOT in the table; bound at submit time with
  `WAIV_BACKBONE_LOCAL_DIRS="repo=/dir"` against converted/pinned checkpoints under
  `$SPECTRA_INPUTS`. `/data` has been swept before, so expect to repoint these.

## Exclusions

Excluded from the copy: `__pycache__`, `*.pyc`, and the source dir's `.venv` and
`third_party` symlinks (both pointed back into the main checkout and are recreated there).
`snapshots/conftest.py` keeps this tree out of the repo's pytest collection — the copies in
`tests/` here share basenames with the real suite and would shadow it.

This is a verbatim code copy, NOT a merge. Do not treat it as the maintained source.
