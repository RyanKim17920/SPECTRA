"""The v2 (Q8) trainer config: ONE nested config fully determines a run.

Loading order (later wins):

1. dataclass defaults below (= the Q8 recipe, held-out selection split),
2. ``--config file.yaml|json`` (may contain ``base: other.yaml`` to inherit, recursively),
3. ``key.sub=value`` overrides (values parsed as YAML, so ``1e-4``, ``[a, b]``, ``null``).

``run.smoke: true`` is applied LAST as a transform: 20 steps, tiny T, tiny eval -- see
:func:`apply_smoke`.

Every resolved config hashes (sha1 over canonical JSON of the fields that change what is
TRAINED -- not eval cadence, not checkpointing, not worker counts, not paths) into an 8-char
id, and the run name is ``<prefix>-<hash>`` with ``len <= 40``.

Paths are never hard-coded: the PLISM repack directory, the run root and the tile cache come
from the config or from environment variables (see :func:`packed_dir`, :func:`runs_root`,
:func:`cache_root`).
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[3]

#: short alias -> HF repo id. Anything not here is taken as a repo id verbatim.
BACKBONE_ALIASES: dict[str, str] = {
    "uni2": "MahmoodLab/UNI2-h",
    "hoptimus": "bioptimus/H-optimus-0",
    "phikon": "owkin/phikon-v2",
    "virchow2": "paige-ai/Virchow2",
    "virchow": "paige-ai/Virchow",
    "midnight": "kaiko-ai/midnight",
    "openmidnight": "SophontAI/OpenMidnight",
}
BACKBONE_SHORT = {v: k for k, v in BACKBONE_ALIASES.items()}

#: Selection-run held-out axes: every condition imaged on GT450 or stained MY (19 of 91).
SELECTION_HELDOUT_SCANNERS = ("GT450",)
SELECTION_HELDOUT_STAINS = ("MY",)
#: Selection-run held-out tissue cores (GT46 ids): one per difficulty bin of the frozen
#: cross-condition retrieval, 7 distinct organ groups, >= 2 tumour cores, fixed seed; drawn
#: before any training result existed. Their locations never reach a training batch, and the
#: in-run selection eval scores them under all conditions. Final (all-data) runs train on them.
SELECTION_HELDOUT_CORES = (5, 12, 13, 35, 36, 38, 45)

#: ground-truth tissue core per PLISM location (packaged with the code)
GT46_LABELS = str(Path(__file__).resolve().parent / "plism_gt46_labels.json")


def runs_root() -> Path:
    """Root of run directories: ``$SPECTRA_RUNS_DIR`` (default ``<repo>/runs/sweeps``)."""
    return Path(os.environ.get("SPECTRA_RUNS_DIR", str(REPO / "runs" / "sweeps")))


def cache_root() -> Path:
    """Tile caches (eval set, GTEx guard): ``$SPECTRA_CACHE_DIR`` (default ``<runs root>/_cache``)."""
    return Path(os.environ.get("SPECTRA_CACHE_DIR", str(runs_root() / "_cache")))


# ======================================================================================
# schema
# ======================================================================================


@dataclass
class BackboneCfg:
    #: alias (uni2, hoptimus, phikon, virchow2, midnight, openmidnight, virchow) or repo id
    name: str = "uni2"
    grad_checkpointing: bool = True


@dataclass
class AdaptCfg:
    #: only LoRA is supported (kept as a field so dumped configs are self-describing)
    mode: str = "lora"
    lora_rank: int = 16
    lora_alpha: float | None = 32.0  # None = 2 * rank
    lora_dropout: float = 0.0
    #: which in-block Linears get LoRA: attn | mlp | both
    lora_targets: str = "both"


@dataclass
class HeadCfg:
    """The single loss head: CLS token -> Linear -> [BN] -> GELU -> Linear projector."""
    proj_dim: int = 512
    proj_hidden: int = 1024
    proj_bn: bool = True


@dataclass
class ObjectiveCfg:
    """Grid InfoNCE (for every ordered condition pair (a, b), a != b: query (a, t), candidates
    = the T locations of condition b, target (b, t); same-core negatives kept) at several
    temperatures over the SAME similarity matrix: loss = sum_k tau_weights[k] x InfoNCE(sim /
    taus[k]). ``tau_weights`` empty = equal weights 1 / len(taus)."""
    taus: list = field(default_factory=lambda: [0.1, 0.6])
    tau_weights: list = field(default_factory=lambda: [0.5, 0.5])


@dataclass
class AdvCfg:
    """Queue adversary on the pooled CLS feature (before the projector).

    One linear head per nuisance target (stain / scanner), each with a FIFO memory bank of the
    last ``queue`` DETACHED features and their labels. Each step every head is trained on
    queue + current batch (a real multi-class classifier -- a batch alone shows only C
    conditions), while the reversed gradient (GRL, strength lambda_p = ``lam`` x DANN ramp
    2 / (1 + exp(-10 p)) - 1, p = step / optim.steps) flows ONLY through the current batch's
    features, against the head with its weights frozen for that term. Heads are their own AdamW
    group at the CONSTANT lr ``lr`` (no warmup / decay, weight decay ``weight_decay``) and are
    grad-clipped separately from the model. The total loss adds ``weight`` x (head CE + reversed
    CE) per target. ``lam`` (YAML key ``lambda`` accepted)."""
    targets: list = field(default_factory=lambda: ["stain", "scanner"])
    lam: float = 0.1
    weight: float = 1.0
    lr: float = 1e-3
    weight_decay: float = 0.0
    queue: int = 8192


@dataclass
class BatchCfg:
    #: C -- distinct conditions per step (drawn without replacement)
    n_cond: int = 2
    #: T -- shared locations per step (negatives per row = T-1)
    n_tiles: int = 256
    #: held-out AXES: every condition with one of these scanners / stains is never trained
    #: on. Selection runs: scanner GT450 + stain MY -> 72 train conditions. Finals: [] (91).
    heldout_scanners: list = field(default_factory=lambda: list(SELECTION_HELDOUT_SCANNERS))
    heldout_stains: list = field(default_factory=lambda: list(SELECTION_HELDOUT_STAINS))
    #: held-out tissue CORES (GT46 ids): their locations never reach a training batch
    #: (asserted every step). Selection runs: SELECTION_HELDOUT_CORES. Finals: [].
    heldout_cores: list = field(default_factory=lambda: list(SELECTION_HELDOUT_CORES))
    workers: int = 8


@dataclass
class OptimCfg:
    lr: float = 1e-4
    #: decoupled (AdamW) weight decay; never applied to biases / norms / other 1-d tensors
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.999
    #: linear warmup lr * (step + 1) / warmup_steps, then cosine to 0 at ``steps``
    warmup_steps: int = 200
    steps: int = 400
    #: stop (final eval + ckpt) at this step while the lr schedule keeps its ``steps`` horizon;
    #: 0 = steps. Selection runs: 250 (the stopping rule's last candidate).
    stop_step: int = 250
    grad_clip: float = 1.0


@dataclass
class CkptCfg:
    every: int = 25
    #: explicit checkpoint steps (non-empty overrides ``every``; never pruned by keep_last)
    schedule: list = field(default_factory=list)
    #: rewrite resume.pt (optimizer + rng + trainable params + adversary queue) every N steps
    resume_every: int = 25
    #: keep only the last N step checkpoints (0 = keep all)
    keep_last: int = 0
    dtype: str = "bfloat16"


@dataclass
class EvalCfg:
    """The in-run PLISM selection eval (see :mod:`spectra.sweep.evaluate`)."""
    every: int = 25
    at_start: bool = True
    #: steps at which to evaluate (non-empty overrides ``every``; the final step always is)
    schedule: list = field(default_factory=list)
    #: eval cores: [] = batch.heldout_cores, or SELECTION_HELDOUT_CORES when those are empty
    #: (a final run; then flagged cores_in_train = true and refused by the stopping rule)
    cores: list = field(default_factory=list)
    #: locations per eval core (deterministic, eval.seed); 7 x 64 x 90 = 40,320 tiles
    locations_per_core: int = 64
    #: core-level bootstrap replicates for the primary scalar's SE
    bootstrap: int = 200
    #: conditions dropped from the eval set (HRH_S60 is defocused)
    exclude_conditions: list = field(default_factory=lambda: ["HRH_S60"])
    #: exported readouts scored: any of cls | mean | clsmean
    readouts: list = field(default_factory=lambda: ["cls", "clsmean"])
    croma_m: int = 5
    croma_alpha: float = 0.1
    #: GTEx utility guard: "auto" = <cache root>/gtex_guard_v1.npy when present, "" = off,
    #: or a path to a build_guard_cache.py .npy (its "ret" split also feeds the retention metrics)
    guard: str = "auto"
    #: L2 weight of the guard's multinomial logistic probe (standardised features, LBFGS)
    guard_wd: float = 1e-4
    guard_knn_k: int = 20
    n_retention: int = 2048
    batch_size: int = 256
    seed: int = 1234
    #: per-location tissue-core labels (CRoMa typing, eval-location sampling, held-out cores)
    labels: str = GT46_LABELS


@dataclass
class RunCfg:
    #: human-readable prefix of the run name ("" = auto from backbone / C / T)
    name: str = ""
    seed: int = 0
    smoke: bool = False
    log_every: int = 10
    #: PLISM repack directory (spectra-repack output); "" = $SPECTRA_PLISM_DIR
    packed_dir: str = ""


@dataclass
class SweepConfig:
    backbone: BackboneCfg = field(default_factory=BackboneCfg)
    adapt: AdaptCfg = field(default_factory=AdaptCfg)
    head: HeadCfg = field(default_factory=HeadCfg)
    objective: ObjectiveCfg = field(default_factory=ObjectiveCfg)
    adv: AdvCfg = field(default_factory=AdvCfg)
    batch: BatchCfg = field(default_factory=BatchCfg)
    optim: OptimCfg = field(default_factory=OptimCfg)
    ckpt: CkptCfg = field(default_factory=CkptCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    run: RunCfg = field(default_factory=RunCfg)


#: sections / fields that do NOT change what is trained -- excluded from the hash
HASH_EXCLUDE = {"eval", "ckpt", "batch.workers", "run.name", "run.log_every", "run.packed_dir"}


def packed_dir(cfg: SweepConfig) -> Path:
    p = cfg.run.packed_dir or os.environ.get("SPECTRA_PLISM_DIR", "")
    if not p:
        raise RuntimeError("PLISM repack directory unknown: set run.packed_dir or $SPECTRA_PLISM_DIR")
    return Path(p)


# ======================================================================================
# dict <-> dataclass
# ======================================================================================

#: sections whose unknown keys could change the rebuilt MODEL; fatal even in tolerant loads
MODEL_SECTIONS = ("backbone", "adapt")


def _build(cls, data: dict | None, path: str = "", dropped: list | None = None):
    data = dict(data or {})
    if cls is AdvCfg and "lambda" in data:          # `lambda` is a python keyword; it wins over lam
        data["lam"] = data.pop("lambda")
    kwargs = {}
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown and dropped is not None:
        model = path.split(".")[0] in MODEL_SECTIONS
        # another schema's extra MODEL knob is harmless only while it is switched off
        off = {k for k in unknown if not model or data[k] in (0, False, None, "", [])}
        dropped += [f"{path}.{k}" if path else k for k in sorted(off)]
        for k in off:
            data.pop(k)
        unknown -= off
    if unknown:
        raise KeyError(f"unknown config key(s) under '{path or '<root>'}': {sorted(unknown)}; "
                       f"valid: {sorted(known)}")
    for name, f in known.items():
        if name not in data:
            continue
        v = data[name]
        sub = f"{path}.{name}" if path else name
        default = f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default
        if is_dataclass(default):
            kwargs[name] = _build(type(default), v, sub, dropped)
        else:
            kwargs[name] = _coerce(v, default, sub)
    return cls(**kwargs)


_NUM_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$|^[+-]?inf$")


def _coerce(v, default, path):
    """YAML 1.1 reads ``1e-4`` as a STRING; coerce to the field's numeric type."""
    if isinstance(default, bool):
        return v
    if isinstance(default, float) and isinstance(v, (int, str)) and not isinstance(v, bool):
        if isinstance(v, str) and not _NUM_RE.match(v.strip()):
            raise ValueError(f"{path}: {v!r} is not a number")
        return float(v)
    if isinstance(default, int) and isinstance(v, str) and _NUM_RE.match(v.strip()):
        f = float(v)
        if f != int(f):
            raise ValueError(f"{path}: {v!r} is not an integer")
        return int(f)
    return v


def to_dict(cfg) -> dict:
    return dataclasses.asdict(cfg)


def from_dict(d: dict) -> SweepConfig:
    cfg = _build(SweepConfig, d)
    validate(cfg)
    return cfg


def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_file(path: str | Path) -> dict:
    """Load YAML/JSON, resolving ``base:`` inheritance (relative to the file)."""
    path = Path(path)
    text = path.read_text()
    d = json.loads(text) if path.suffix == ".json" else (yaml.safe_load(text) or {})
    base = d.pop("base", None)
    if base:
        bp = Path(base)
        if not bp.is_absolute():
            cand = path.parent / bp
            bp = cand if cand.exists() else REPO / bp
        d = _deep_merge(load_file(bp), d)
    return d


def set_by_path(d: dict, key: str, value: Any) -> None:
    """``optim.lr=3e-5`` / ``batch.heldout_cores=[]`` style override into a plain dict."""
    parts = key.split(".")
    cur: Any = d
    for p in parts[:-1]:
        if p not in cur or cur[p] is None:
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def parse_override(s: str) -> tuple[str, Any]:
    if "=" not in s:
        raise ValueError(f"override {s!r} is not key=value")
    k, v = s.split("=", 1)
    val = yaml.safe_load(v)
    if isinstance(val, str) and _NUM_RE.match(val.strip()) and "inf" not in val:
        val = float(val) if any(ch in val for ch in ".eE") else int(val)
    return k.strip(), val


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> SweepConfig:
    d = to_dict(SweepConfig())
    if path:
        d = _deep_merge(d, load_file(path))
    for o in overrides or []:
        k, v = parse_override(o)
        set_by_path(d, k, v)
    cfg = from_dict(d)
    return apply_smoke(cfg) if cfg.run.smoke else cfg


def apply_smoke(cfg: SweepConfig) -> SweepConfig:
    """20 steps, tiny T, tiny eval. Idempotent; the smoke flag itself is in the hash."""
    cfg = copy.deepcopy(cfg)
    cfg.optim.steps = 20
    cfg.optim.stop_step = min(cfg.optim.stop_step, 20)
    cfg.optim.warmup_steps = min(cfg.optim.warmup_steps, 5)
    cfg.batch.n_tiles = min(cfg.batch.n_tiles, 64)
    cfg.adv.queue = min(cfg.adv.queue, 512)
    cfg.eval.every = 10
    cfg.eval.schedule = []
    cfg.eval.locations_per_core = min(cfg.eval.locations_per_core, 12)
    cfg.eval.n_retention = min(cfg.eval.n_retention, 256)
    cfg.eval.bootstrap = min(cfg.eval.bootstrap, 50)
    cfg.ckpt.every = 10
    cfg.ckpt.schedule = []
    cfg.ckpt.resume_every = 10
    cfg.run.log_every = 1
    return cfg


# ======================================================================================
# validation
# ======================================================================================

READOUTS = ("cls", "mean", "clsmean")
ADV_TARGETS = ("stain", "scanner", "condition")


def _need(cond, msg):
    if not cond:
        raise ValueError(f"config: {msg}")


def validate_model(cfg: SweepConfig) -> None:
    """The checks that matter for rebuilding the adapted backbone (all a tolerant load runs)."""
    _need(cfg.adapt.mode == "lora", f"adapt.mode {cfg.adapt.mode!r}: only lora is supported")
    _need(cfg.adapt.lora_targets in ("attn", "mlp", "both"), "adapt.lora_targets")
    _need(cfg.adapt.lora_rank > 0, "adapt.lora_rank > 0")


def validate(cfg: SweepConfig) -> None:
    need = _need

    validate_model(cfg)
    need(cfg.head.proj_dim > 0 and cfg.head.proj_hidden > 0, "head.proj_dim / proj_hidden > 0")
    obj = cfg.objective
    need(bool(obj.taus) and all(float(t) > 0 for t in obj.taus), "objective.taus: non-empty, > 0")
    need(not obj.tau_weights or (len(obj.tau_weights) == len(obj.taus)
                                 and all(float(w) > 0 for w in obj.tau_weights)),
         "objective.tau_weights: one positive weight per tau (or empty = equal weights)")
    av = cfg.adv
    need(bool(av.targets) and set(av.targets) <= set(ADV_TARGETS)
         and len(set(av.targets)) == len(av.targets), f"adv.targets {av.targets}")
    need(av.lam >= 0 and av.weight > 0 and av.lr > 0 and av.weight_decay >= 0, "adv lam / weight / lr / wd")
    need(int(av.queue) > 0, "adv.queue > 0")
    b = cfg.batch
    need(b.n_cond >= 2, "batch.n_cond >= 2")
    need(b.n_tiles >= 2, "batch.n_tiles >= 2")
    need(isinstance(b.heldout_cores, list), "batch.heldout_cores must be a list of GT46 core ids")
    o = cfg.optim
    need(o.steps > 0 and 0 <= o.warmup_steps < o.steps, "optim.steps / warmup_steps")
    need(0 <= o.stop_step <= o.steps, "optim.stop_step in [0, steps]")
    need(set(cfg.eval.readouts) <= set(READOUTS) and cfg.eval.readouts, "eval.readouts")
    need(cfg.ckpt.dtype in ("float32", "bfloat16"), "ckpt.dtype")


# ======================================================================================
# hashing / naming
# ======================================================================================


def _strip(d: dict, excl: set[str], prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if key in excl:
            continue
        out[k] = _strip(v, excl, key + ".") if isinstance(v, dict) else v
    return out


def _canon(v):
    if isinstance(v, float):
        if math.isinf(v) or math.isnan(v):
            return str(v)
        return float(f"{v:.10g}")
    if isinstance(v, dict):
        return {k: _canon(x) for k, x in sorted(v.items())}
    if isinstance(v, (list, tuple)):
        return [_canon(x) for x in v]
    return v


def config_hash(cfg: SweepConfig) -> str:
    d = _strip(to_dict(cfg), HASH_EXCLUDE)
    d["backbone"]["name"] = resolve_backbone(cfg.backbone.name)
    blob = json.dumps(_canon(d), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


def resolve_backbone(name: str) -> str:
    return BACKBONE_ALIASES.get(name, name)


def auto_prefix(cfg: SweepConfig) -> str:
    rid = resolve_backbone(cfg.backbone.name)
    bb = BACKBONE_SHORT.get(rid, re.sub(r"[^a-z0-9]", "", rid.split("/")[-1].lower())[:10])
    p = f"{bb}-q8-c{cfg.batch.n_cond}t{cfg.batch.n_tiles}"
    return "smoke-" + p if cfg.run.smoke else p


def run_name(cfg: SweepConfig) -> str:
    h = config_hash(cfg)
    prefix = cfg.run.name or auto_prefix(cfg)
    if cfg.run.smoke and not prefix.startswith("smoke"):
        prefix = "smoke-" + prefix
    prefix = re.sub(r"[^A-Za-z0-9_.-]", "-", prefix)[: 40 - 1 - len(h)].rstrip("-.")
    name = f"{prefix}-{h}"
    assert len(name) <= 40, name
    return name


def dump(cfg: SweepConfig, path: str | Path) -> None:
    d = to_dict(cfg)
    d["_meta"] = {"hash": config_hash(cfg), "run_name": run_name(cfg),
                  "backbone_repo": resolve_backbone(cfg.backbone.name)}
    Path(path).write_text(json.dumps(d, indent=2, default=str))


def load_dumped(path: str | Path, tolerant: bool = False) -> SweepConfig:
    """A run's stored config.json. ``tolerant``: keys this schema does not know (a config dumped
    by another trainer version) are dropped with a printed warning instead of raising -- except
    under ``backbone`` / ``adapt``, which shape the rebuilt model (there an unknown key is
    accepted only while it is switched off: 0 / false / null / empty). Only the model sections
    are validated (enough to rebuild the adapted backbone for inference)."""
    d = json.loads(Path(path).read_text())
    d.pop("_meta", None)
    if not tolerant:
        return from_dict(d)
    dropped: list[str] = []
    cfg = _build(SweepConfig, d, dropped=dropped)
    if dropped:
        print(f"[config] {path}: ignoring keys unknown to this schema: {dropped}", flush=True)
    validate_model(cfg)
    return cfg
