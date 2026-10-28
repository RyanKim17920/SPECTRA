"""Model keys: the ONE name a checkpoint carries through every benchmark.

Rules (shared contract):

* ``<= 52`` chars, charset ``[A-Za-z0-9._-]`` (THUNDER appends ``_optimized`` -> ``<= 62 < 64``);
* deterministic from ``(backbone alias, run name, step, ema)`` -- never from wall-clock or
  weight bytes, so the same request always lands in the same directory;
* when the natural key is too long it is truncated and a 6-hex hash of the FULL natural key
  is appended, so two long keys that share a prefix never collide.

Forms::

    hopt0-base                         base backbone
    uni2-s1-9aa01da7-s400              sweep run s1-9aa01da7, ckpt step 400
    uni2-s1-9aa01da7-s400-ema          ... EMA weights

Pure stdlib: importable anywhere.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

MAX_KEY = 52
THUNDER_SUFFIX = "_optimized"
THUNDER_MAX = 64
KEY_RE = re.compile(r"^[A-Za-z0-9._-]+$")

#: repo id -> short alias used as the key prefix
SHORT = {
    "bioptimus/H-optimus-0": "hopt0",
    "MahmoodLab/UNI2-h": "uni2",
    "owkin/phikon-v2": "phikon2",
    "paige-ai/Virchow2": "virchow2",
    "paige-ai/Virchow": "virchow1",
    "kaiko-ai/midnight": "midnight",
    "SophontAI/OpenMidnight": "omid",
}

#: every spelling a user / run name may use for a backbone -> repo id
ALIASES = {
    "hoptimus": "bioptimus/H-optimus-0", "hoptimus0": "bioptimus/H-optimus-0",
    "hopt0": "bioptimus/H-optimus-0", "hopt": "bioptimus/H-optimus-0",
    "h-optimus-0": "bioptimus/H-optimus-0", "H-optimus-0": "bioptimus/H-optimus-0",
    "uni2": "MahmoodLab/UNI2-h", "uni2-h": "MahmoodLab/UNI2-h", "UNI2-h": "MahmoodLab/UNI2-h",
    "phikon": "owkin/phikon-v2", "phikon2": "owkin/phikon-v2", "phikon-v2": "owkin/phikon-v2",
    "virchow2": "paige-ai/Virchow2", "Virchow2": "paige-ai/Virchow2",
    "virchow": "paige-ai/Virchow", "virchow1": "paige-ai/Virchow", "Virchow": "paige-ai/Virchow",
    "midnight": "kaiko-ai/midnight",
    "openmidnight": "SophontAI/OpenMidnight", "omid": "SophontAI/OpenMidnight",
    "OpenMidnight": "SophontAI/OpenMidnight",
}

# tokens stripped from legacy run names (the alias prefix already says it); longest first
_BB_TOKENS = ("openmidnight", "hoptimus", "virchow2", "virchow", "midnight", "phikon", "uni2", "hopt")


def resolve_repo(backbone: str) -> str:
    """Alias or repo id -> repo id (unknown strings pass through as repo ids)."""
    if backbone in ALIASES:
        return ALIASES[backbone]
    if backbone.lower() in ALIASES:
        return ALIASES[backbone.lower()]
    return backbone


def alias_for(repo: str) -> str:
    if repo in SHORT:
        return SHORT[repo]
    return sanitize(repo.split("/")[-1].lower())[:12] or "model"


def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(s)).strip("-.")


def validate_key(key: str) -> str:
    if not key or len(key) > MAX_KEY or not KEY_RE.match(key) or key in (".", ".."):
        raise ValueError(f"invalid model key {key!r}: need 1..{MAX_KEY} chars of [A-Za-z0-9._-]")
    return key


def shorten(full: str, limit: int = MAX_KEY) -> str:
    """Deterministic truncation: prefix + '-' + sha1(full)[:6] when ``full`` is too long."""
    full = sanitize(full)
    if len(full) <= limit:
        return full
    h = hashlib.sha1(full.encode()).hexdigest()[:6]
    return full[: limit - 7].rstrip("-._") + "-" + h


def _strip_bb_tokens(name: str) -> str:
    parts = name.split("-")
    out = [p for p in parts if p.lower() not in _BB_TOKENS]
    return "-".join(out) or name


def ckpt_tag(kind: str, ckpt_path: str | None, run_name: str | None = None) -> str:
    """The checkpoint part of a key (no backbone alias, no step)."""
    if kind == "base" or not ckpt_path:
        return "base"
    if run_name:
        return _strip_bb_tokens(sanitize(run_name))
    p = Path(ckpt_path)
    return _strip_bb_tokens(sanitize(p.stem if p.is_file() or p.suffix else p.name))


def derive_key(repo: str, kind: str, ckpt_path: str | None = None, step: int | None = None,
               ema: bool = False, run_name: str | None = None) -> str:
    alias = alias_for(repo)
    head = f"{alias}-{ckpt_tag(kind, ckpt_path, run_name)}"
    tail = []
    if step is not None and kind != "base":
        tail.append(f"s{int(step)}")
    if ema:
        tail.append("ema")
    full = sanitize("-".join([head] + tail))
    if len(full) <= MAX_KEY:
        return validate_key(full)
    # truncate the checkpoint tag, never the step / ema suffix
    suffix = ("-" + "-".join(tail)) if tail else ""
    return validate_key(shorten(full, MAX_KEY - len(suffix)) + suffix)


def thunder_key(key: str) -> str:
    """THUNDER's model name (``<key>_optimized``; THUNDER caps names at 64 chars)."""
    k = validate_key(key) + THUNDER_SUFFIX
    assert len(k) <= THUNDER_MAX, k
    return k
