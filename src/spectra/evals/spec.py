"""Model SPEC resolution: (backbone, ckpt, step, ...) -> one JSON-serialisable dict.

The spec is resolved ONCE per model (``python -m spectra.evals.spec ... --out spec.json``) and
then every benchmark loads exactly that spec, so the four benches can never disagree about
which weights they scored.

``kind``::

    base          the frozen pretrained backbone
    sweep_lora    a spectra.sweep run (config.json + ckpt/step_XXXXXXX.safetensors) trained with
                  adapt.mode = lora; ``ema`` selects the ``ema_step_*`` weights

``identity_sha256`` = sha256 over canonical JSON of {kind, backbone repo + revision + weight
file sha256s, checkpoint file sha256s, step, ema, adapt cfg}. ``created`` and ``model_key`` are
NOT part of it. File hashes are cached under ``~/.cache/spectra-evals/sha256`` (override:
``SPECTRA_EVALS_SHA_CACHE``; keyed by realpath + size + mtime) and HF-cache LFS blobs use their
blob name (= sha256).

Gated backbones kept in a local directory: ``SPECTRA_BACKBONE_LOCAL_DIRS="repo=/dir,..."``.

Deps: stdlib (+ huggingface_hub, only to locate hub-cached weights).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import sys
from pathlib import Path

from . import keys as K

SCHEMA = "spectra-evals-spec/1"
SHA_CACHE = Path(os.environ.get("SPECTRA_EVALS_SHA_CACHE", str(Path.home() / ".cache" / "spectra-evals" / "sha256")))

#: immutable hub revision a gated local copy was taken from (where known)
LOCAL_REVISIONS = {
    "bioptimus/H-optimus-0": "b145cc1e6c6b30d3251aa8b1f844e6974188a743",
}
WEIGHT_NAMES = ("model.safetensors", "pytorch_model.bin")

_STEP_RE = re.compile(r"^(ema_)?step_(\d+)(?:\.safetensors)?$")


def local_dirs() -> dict[str, str]:
    """repo id -> local directory, from ``SPECTRA_BACKBONE_LOCAL_DIRS="repo=/dir,repo2=/dir2"``."""
    out: dict[str, str] = {}
    for item in os.environ.get("SPECTRA_BACKBONE_LOCAL_DIRS", "").split(","):
        if "=" in item:
            k, v = item.split("=", 1)
            out[k.strip()] = v.strip()
    return out


# --------------------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------------------


def file_sha256(path: str | Path) -> str:
    p = Path(path)
    real = p.resolve()
    if re.fullmatch(r"[0-9a-f]{64}", real.name) and real.parent.name == "blobs":
        return real.name                              # HF LFS blob: name IS the sha256
    st = real.stat()
    tag = hashlib.sha1(str(real).encode()).hexdigest()
    cp = SHA_CACHE / f"{tag}.json"
    try:
        c = json.loads(cp.read_text())
        if c["size"] == st.st_size and c["mtime_ns"] == st.st_mtime_ns and c["path"] == str(real):
            return c["sha256"]
    except Exception:  # noqa: BLE001 -- cache miss
        pass
    h = hashlib.sha256()
    with open(real, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    d = h.hexdigest()
    try:
        cp.parent.mkdir(parents=True, exist_ok=True)
        tmp = cp.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"path": str(real), "size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": d}))
        os.replace(tmp, cp)
    except OSError:
        pass
    return d


def backbone_identity(repo: str) -> dict:
    """{revision, source, files: {name: sha256}} for the backbone's weights + config."""
    ld = local_dirs().get(repo)
    if ld and (Path(ld) / "config.json").is_file():
        d = Path(ld)
        files = {n: file_sha256(d / n) for n in ("config.json",) + WEIGHT_NAMES if (d / n).is_file()}
        if len(files) < 2:
            raise FileNotFoundError(f"{repo}: no weight file in local dir {d}")
        return {"source": f"local:{d}", "revision": LOCAL_REVISIONS.get(repo, "local"), "files": files}
    from huggingface_hub import try_to_load_from_cache

    files, rev = {}, None
    for n in ("config.json",) + WEIGHT_NAMES:
        p = try_to_load_from_cache(repo, n)
        if isinstance(p, str) and Path(p).is_file():
            files[n] = file_sha256(p)
            parts = Path(p).parts
            if "snapshots" in parts:
                rev = parts[parts.index("snapshots") + 1]
            if n in WEIGHT_NAMES:
                break                              # the loader takes the first weight file
    if not any(n in files for n in WEIGHT_NAMES):
        raise FileNotFoundError(
            f"{repo}: weights not in the HF cache (HF_HOME={os.environ.get('HF_HOME')}); "
            "download on the login node first (compute nodes run HF_HUB_OFFLINE=1)")
    return {"source": "hf-cache", "revision": rev, "files": files}


def _canon_sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# --------------------------------------------------------------------------------------
# checkpoint resolution
# --------------------------------------------------------------------------------------


def _step_of(name: str) -> tuple[int | None, bool]:
    m = _STEP_RE.match(name)
    if not m:
        return None, False
    return int(m.group(2)), bool(m.group(1))


def resolve_ckpt(ckpt: str | None, step: int | None = None, ema: bool = False) -> dict:
    """-> {kind, ckpt_path, step, ema, run_dir?, run_name?, adapt?, files: {name: path}, backbone_hint}

    ``ckpt``: None / "base" (the frozen backbone), a sweep run dir (``step`` picks the ckpt,
    default the last one; ``ema`` -> ``ema_step_*``) or one ``ckpt/step_XXXXXXX.safetensors``."""
    if ckpt in (None, "", "base", "none"):
        return {"kind": "base", "ckpt_path": None, "step": None, "ema": False, "files": {}, "backbone_hint": None}
    p = Path(ckpt).expanduser().absolute()
    if not p.exists():
        raise FileNotFoundError(p)
    if p.is_dir():
        if not ((p / "config.json").is_file() and (p / "ckpt").is_dir()):
            raise ValueError(f"unrecognised checkpoint {p}: not a sweep run dir (config.json + ckpt/)")
        pre = "ema_step_" if ema else "step_"
        if step is not None:
            ck = p / "ckpt" / f"{pre}{int(step):07d}.safetensors"
            if not ck.is_file():
                have = sorted(c.name for c in (p / "ckpt").glob("*.safetensors"))
                raise FileNotFoundError(f"{ck} missing; available: {have}")
        else:
            cands = sorted(c for c in (p / "ckpt").glob(f"{pre}*.safetensors") if ema or not c.name.startswith("ema_"))
            if not cands:
                raise FileNotFoundError(f"no {pre}*.safetensors under {p / 'ckpt'}")
            ck = cands[-1]
        return resolve_ckpt(str(ck), None, ema)
    if p.suffix != ".safetensors" or p.parent.name != "ckpt" or not (p.parent.parent / "config.json").is_file():
        raise ValueError(f"{p}: expected <run>/ckpt/step_XXXXXXX.safetensors next to <run>/config.json")
    run_dir = p.parent.parent
    s, is_ema = _step_of(p.stem)
    if s is None:
        raise ValueError(f"{p.name}: not a step_XXXXXXX / ema_step_XXXXXXX checkpoint name")
    if step is not None and int(step) != s:
        raise ValueError(f"--step {step} disagrees with ckpt file {p.name}")
    if ema and not is_ema:
        raise ValueError(f"--ema given but {p.name} is not an ema_step_* file")
    cfg = json.loads((run_dir / "config.json").read_text())
    mode = cfg["adapt"]["mode"]
    if mode != "lora":
        raise ValueError(f"{run_dir}: adapt.mode={mode!r}; only LoRA runs are supported")
    return {"kind": "sweep_lora", "ckpt_path": str(p), "step": s, "ema": is_ema, "run_dir": str(run_dir),
            "run_name": run_dir.name, "adapt": cfg["adapt"], "files": {p.name: str(p)},
            "backbone_hint": cfg["backbone"]["name"]}


# --------------------------------------------------------------------------------------
# spec
# --------------------------------------------------------------------------------------


def _identity(spec: dict, ckpt_info: dict, bb_id: dict) -> str:
    return _canon_sha({
        "kind": spec["kind"],
        "backbone_repo": spec["backbone_repo"],
        "backbone_revision": bb_id.get("revision"),
        "backbone_files": bb_id["files"],
        "ckpt_files": {n: file_sha256(p) for n, p in ckpt_info.get("files", {}).items()},
        "step": spec.get("step"),
        "ema": bool(spec.get("ema")),
        "adapt": ckpt_info.get("adapt"),
    })


def resolve_spec(backbone: str | None = None, ckpt: str | None = None, step: int | None = None,
                 key: str | None = None, ema: bool = False) -> dict:
    """See module docstring. ``backbone`` may be omitted when the checkpoint names it (the sweep
    run's config.json)."""
    info = resolve_ckpt(ckpt, step, ema)
    hint = info.get("backbone_hint")
    if backbone is None and hint is None:
        raise ValueError("backbone is required (the checkpoint does not name its base model)")
    repo = K.resolve_repo(backbone) if backbone else K.resolve_repo(hint)
    if hint is not None and K.resolve_repo(hint) != repo:
        raise ValueError(f"backbone {repo!r} != checkpoint's base model {K.resolve_repo(hint)!r} ({info.get('ckpt_path')})")
    bb_id = backbone_identity(repo)
    spec = {
        "schema": SCHEMA,
        "model_key": None,
        "backbone_repo": repo,
        "backbone_alias": K.alias_for(repo),
        "backbone_revision": bb_id.get("revision"),
        "backbone_source": bb_id.get("source"),
        "kind": info["kind"],
        "ckpt_path": info.get("ckpt_path"),
        "step": info.get("step"),
        "ema": bool(info.get("ema")),
        "run_name": info.get("run_name"),
    }
    spec["identity_sha256"] = _identity(spec, info, bb_id)
    spec["model_key"] = K.validate_key(key) if key else K.derive_key(
        repo, info["kind"], info.get("ckpt_path"), info.get("step"), spec["ema"], info.get("run_name"))
    spec["created"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    return spec


def load_spec(spec_or_path) -> dict:
    if isinstance(spec_or_path, dict):
        return spec_or_path
    return json.loads(Path(spec_or_path).read_text())


def write_spec(spec: dict, out: str | Path, force: bool = False) -> tuple[Path, str]:
    """Write ``spec`` to ``out`` unless an existing file has the same identity (kept as is,
    status 'exists') or a different one (ValueError unless ``force``)."""
    out = Path(out)
    if out.is_file():
        old = json.loads(out.read_text())
        if old.get("identity_sha256") == spec["identity_sha256"] and old.get("model_key") == spec["model_key"]:
            return out, "exists"
        if not force:
            raise ValueError(f"{out} holds a different model (identity {old.get('identity_sha256', '?')[:12]} "
                             f"key {old.get('model_key')}) than requested ({spec['identity_sha256'][:12]} "
                             f"{spec['model_key']}); pass --force to overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, out)
    return out, "written"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m spectra.evals.spec", description=__doc__.split("\n")[0])
    ap.add_argument("--backbone")
    ap.add_argument("--ckpt", help="sweep run dir or <run>/ckpt/step_*.safetensors (default: the frozen base)")
    ap.add_argument("--step", type=int)
    ap.add_argument("--key")
    ap.add_argument("--ema", action="store_true")
    ap.add_argument("--out", help="write spec JSON here (refuses a different identity unless --force)")
    ap.add_argument("--evals-dir", action="store_true",
                    help="write to $SPECTRA_EVAL_OUT/<key>/spec.json (same refusal rule)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--print-key", action="store_true", help="print only the model key on stdout")
    a = ap.parse_args(argv)
    if not (a.backbone or a.ckpt):
        ap.error("need --backbone and/or --ckpt")
    spec = resolve_spec(a.backbone, a.ckpt, a.step, a.key, a.ema)
    outs = []
    if a.out:
        outs.append(Path(a.out))
    if a.evals_dir:
        from .results import out_dir

        outs.append(out_dir(spec["model_key"]) / "spec.json")
    for o in outs:
        try:
            path, status = write_spec(spec, o, a.force)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 3
        print(f"[spec] {status}: {path}", file=sys.stderr)
    if a.print_key:
        print(spec["model_key"])
    else:
        print(json.dumps(spec, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
