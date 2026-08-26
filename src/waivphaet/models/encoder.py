"""A *model-agnostic* ViT backbone with LoRA across **all** transformer blocks, plus a
wide projection head.

Design choices and where they come from
---------------------------------------
* **Base is a parameter, not a constant.** The default is ``owkin/phikon-v2`` (Dinov2,
  ViT-L/16, 24 blocks, 1024-d, ungated, 1.21 GB) -- PLAN.md 1: weakest starting point
  on PathoROB (Avg RI 0.469, Camelyon 0.019) and the largest published gain
  (-> 0.806 / 0.702), so it is the cheapest informative base. But the pipeline has to
  generalise, so ``hidden_size``, ``num_hidden_layers`` and ``patch_size`` are read off
  the *loaded* config and the LoRA target set is **discovered by module-name pattern**.
  Nothing about ViT-L/16 is baked in. Second validated backbone: ``kaiko-ai/midnight``
  (Dinov2 ViT-g/14, 40 blocks, 1536-d, SwiGLU FFN, MIT, 4.55 GB).

* **The loader is a branch, chosen by reading the repo's config.json.** Not every
  pathology foundation model is a ``transformers`` checkpoint. ``paige-ai/Virchow2`` is a
  *timm* one: its config.json carries ``architecture: vit_huge_patch14_224`` +
  ``model_args`` + ``pretrained_cfg`` and **no** ``model_type``, so ``AutoModel`` has
  nothing to dispatch on and cannot load it at all. ``is_timm_backbone`` reads that file
  and picks ``timm.create_model("hf-hub:...")`` instead -- by config *shape*, never by a
  model-name list, so the two published backbones cannot be re-routed by accident.
  Virchow2 is ViT-H/14, 32 blocks, 1280-d, packed-SwiGLU FFN, and carries **4 register
  tokens** on top of CLS -- see ``num_prefix_tokens`` and ``_pool``.

* **LoRA on every block, not head-only.** PLAN.md 2 + 0 (their Fig 4): base H-Optimus-0
  only develops cross-scanner matching in the last few blocks, and fine-tuning pushes
  that ~8 blocks earlier. Invariance has to build *across depth*, so head-only tuning is
  ruled out. LoRA rather than full FT is our deliberate anti-forgetting divergence
  (PLAN.md 2): it bounds drift on a backbone that saw 456M tiles, cuts memory, and
  merges back to full weights afterwards. Full FT is the escalation (PLAN.md 3 phase 9).

  **Why discovery rather than a fixed name list.** ``fc1``/``fc2`` is the HF Dinov2 MLP
  naming, but ``kaiko-ai/midnight`` sets ``use_swiglu_ffn=True`` and its FFN linears are
  ``mlp.weights_in`` / ``mlp.weights_out``. A fixed list would still have matched
  ``query/key/value/dense`` -- so the block-coverage assertion would have *passed* while
  silently adapting attention only and leaving 2/3 of the block parameters frozen. The
  failure mode is invisible in every log line and reads downstream as "LoRA had less
  effect on ViT-g". So we discover, and we assert the per-block match count is uniform
  and non-empty, and we log it.

* **Projection width >= 512.** PLAN.md 2: ScanGen used hidden 48/96 for binary MIL,
  far too narrow for retrieval among 16k tiles. Default 1024 hidden / 512 out. The
  projector's *input* width is ``embed_dim``, i.e. it is tied to the **training**
  pooling -- see ``build_model`` in ``scripts/extract_pathorob_features.py``.

* **Pooling defaults to ``clsmean``** (CLS token concatenated with the mean of patch
  tokens) because that is exactly what PathoROB's own ``phikonv2_clsmean`` entry uses --
  matching it is what makes our reproduced Avg RI 0.469 gate (PLAN.md 3 phase 5)
  meaningful. ``embed_dim`` is *derived*: ``hidden`` for cls/mean, ``2*hidden`` for
  clsmean -- 1024/2048 on phikon-v2, 1536/3072 on midnight.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from waivphaet.models.pooling import POOL_HEAD_NAMES, build_pool_head

DEFAULT_BACKBONE = "owkin/phikon-v2"

#: The two pools ``clsmean`` concatenates, in the order it concatenates them. Named here
#: so the split-head machinery and ``_pool`` cannot drift apart: ``pool_from_parts`` below
#: reconstructs ``_pool``'s output from these and is tested for bitwise equality with it.
POOL_PARTS: tuple[str, ...] = ("cls", "mean")

# ImageNet stats -- what phikon-v2's own BitImageProcessor uses, and what both PathoROB
# and plismbench feed it. Keep identical or the reproduced baseline drifts for free.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: Symmetric [-1, 1] normalisation.
HALF_MEAN = (0.5, 0.5, 0.5)
HALF_STD = (0.5, 0.5, 0.5)

#: **Normalisation is a property of the backbone, not of the pipeline.**
#:
#: phikon-v2 wants ImageNet stats (its own ``BitImageProcessor``), and PathoROB's
#: ``Phikonv2ModelWrapper.get_preprocess`` uses exactly those -- which is why our Avg RI
#: reproduces theirs to 6 decimals. ``kaiko-ai/midnight`` does **not**: its model card is
#: explicit -- "trained on 224x224 images normalized with a mean of (0.5, 0.5, 0.5) and a
#: standard deviation of (0.5, 0.5, 0.5). Please ensure you apply these exact
#: normalization parameters."
#:
#: Feeding midnight ImageNet stats does not crash and does not look wrong anywhere: it
#: just shifts and rescales every channel, quietly costing base accuracy. It would make
#: our base-midnight row disagree with Waiv's published 0.759 for a reason that has
#: nothing to do with the harness being faithful -- i.e. exactly the check we are running
#: it for. So it is table-driven and travels with the backbone id.
#:
#: This table is an **override**, not the only source: it wins over whatever the backbone's
#: own HF preprocessor says. These two entries are the ones our published numbers were
#: produced with, and they must never move because a hub config was re-uploaded.
BACKBONE_NORMALIZATION: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "owkin/phikon-v2": (IMAGENET_MEAN, IMAGENET_STD),
    "kaiko-ai/midnight": (HALF_MEAN, HALF_STD),
    # H-Optimus-0's model card publishes its own H&E-corpus statistics. They are NOT
    # ImageNet's, and feeding it ImageNet's does not crash, does not warn and does not
    # change a single shape -- it just quietly costs accuracy on every row we then
    # compare against the paper. Pinned so no lookup is ever attempted.
    "bioptimus/H-optimus-0": (
        (0.707223, 0.578729, 0.703617),
        (0.211883, 0.230117, 0.177517),
    ),
    # UNI2-h really is ImageNet -- its timm pretrained_cfg says so. It is pinned anyway
    # because "ImageNet because we read the card" and "ImageNet because the lookup fell
    # through" are indistinguishable at the call site, and only one of them is a decision.
    "MahmoodLab/UNI2-h": (IMAGENET_MEAN, IMAGENET_STD),
}


# ======================================================================================
# GATED backbones -- served off a local directory instead of the hub.
# ======================================================================================
#
# ``bioptimus/H-optimus-0`` and ``MahmoodLab/UNI2-h`` are gated repos that our token is
# not approved for: every hub call 403s, including the ``config.json`` fetch that decides
# timm-vs-AutoModel. The checkpoints exist on this machine, so the fix is a binding from
# repo id -> directory, consulted BEFORE the hub by everything that would otherwise ask
# the hub: the loader dispatch, the FFN-shape probe, the normalisation lookup, the weights.
#
# Without the binding the failure is not a clean 403. ``_hub_config`` swallows the error
# and returns ``None``, ``is_timm_backbone`` therefore answers ``False``, and the run takes
# the ``AutoModel`` path and dies with "Unrecognized model" -- an error whose text names
# nothing about gating, about the real architecture, or about what to do next.
#
# Overridable via ``WAIV_BACKBONE_LOCAL_DIRS="repo=/dir,repo2=/dir2"`` so a relocated
# checkpoint (``/data`` has been swept before) is a job-script edit, not a code change.

BACKBONE_LOCAL_DIRS: dict[str, str] = {
    "bioptimus/H-optimus-0": "/data/H-optimus-0",
    "MahmoodLab/UNI2-h": "/data/UNI2-h",
}

#: Weight file names we accept in a local backbone directory, in preference order.
_LOCAL_WEIGHT_NAMES: tuple[str, ...] = ("model.safetensors", "pytorch_model.bin")


def _local_dir_table() -> dict[str, str]:
    """``BACKBONE_LOCAL_DIRS`` with ``WAIV_BACKBONE_LOCAL_DIRS`` entries layered on top."""
    table = dict(BACKBONE_LOCAL_DIRS)
    raw = os.environ.get("WAIV_BACKBONE_LOCAL_DIRS", "")
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise RuntimeError(
                f"WAIV_BACKBONE_LOCAL_DIRS entry {item!r} is not 'repo_id=/path'; the "
                f"whole variable was {raw!r}"
            )
        repo, path = item.split("=", 1)
        table[repo.strip()] = path.strip()
    return table


def local_backbone_dir(backbone: str | None) -> Path | None:
    """Directory serving ``backbone``, or ``None`` when it is an ordinary hub backbone.

    ``None`` means "not bound, use the hub" and is the normal answer for phikon-v2,
    midnight and Virchow2. A binding that points at a directory with no ``config.json``
    RAISES: a swept ``/data`` must stop the run here, where the message can say which
    backbone and which path, rather than degrade into a 403 and a misrouted loader.
    """
    backbone = backbone or DEFAULT_BACKBONE
    raw = _local_dir_table().get(backbone)
    if raw is None:
        return None
    d = Path(raw)
    if not (d / "config.json").is_file():
        raise RuntimeError(
            f"backbone {backbone!r} is bound to local directory {str(d)!r} but there is "
            "no config.json in it. The hub repo is GATED (403), so there is no fallback: "
            "restore the checkpoint or repoint it with "
            f'WAIV_BACKBONE_LOCAL_DIRS="{backbone}=/new/path".'
        )
    return d


def local_weight_path(backbone: str | None) -> Path | None:
    """The weight file inside ``backbone``'s local directory, or ``None`` if hub-served."""
    d = local_backbone_dir(backbone)
    if d is None:
        return None
    for name in _LOCAL_WEIGHT_NAMES:
        p = d / name
        if p.is_file():
            return p
    raise RuntimeError(
        f"backbone {backbone!r} is bound to {str(d)!r} but it holds none of "
        f"{list(_LOCAL_WEIGHT_NAMES)}; found {sorted(p.name for p in d.iterdir())}"
    )


def local_state_dict(backbone: str | None) -> dict | None:
    """``backbone``'s raw tensor state dict off disk, or ``None`` if hub-served."""
    p = local_weight_path(backbone)
    if p is None:
        return None
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(p))
    return torch.load(str(p), map_location="cpu", weights_only=True)


# ======================================================================================
# Which loader? -- answered by the REPO'S OWN config.json, never by a model-name list.
# ======================================================================================
#
# ``paige-ai/Virchow2`` is a *timm* checkpoint published on the HF hub. Its config.json
# has no ``model_type``, so ``transformers`` has nothing to dispatch on and
# ``AutoModel.from_pretrained`` cannot load it at all. What it has instead is timm's
# shape: a top-level ``architecture`` ("vit_huge_patch14_224"), a ``model_args`` block
# and a ``pretrained_cfg``.
#
# So the loader is chosen by *reading that file*, not by matching the repo id against a
# hardcoded list. A name list would mean every new timm backbone silently takes the HF
# path and dies with an unrelated "Unrecognized model" error -- and, worse, that a repo
# re-tagged upstream keeps taking the wrong path forever. The config is the source of
# truth for what the checkpoint *is*.

_HUB_CONFIG_CACHE: dict[str, dict | None] = {}


def _hub_config(backbone: str) -> dict | None:
    """The backbone repo's raw ``config.json`` as a dict, or ``None`` if unreadable.

    Cached per process: it is consulted by both the loader and ``normalization_for``,
    and both may be called several times per run (scripts resolve normalisation before
    they build the model).
    """
    if backbone in _HUB_CONFIG_CACHE:
        return _HUB_CONFIG_CACHE[backbone]
    cfg: dict | None = None
    # A locally-served backbone must NEVER reach the hub here: its repo is gated, the
    # fetch 403s, and the ``except`` below would turn that into ``None`` -- which reads as
    # "not a timm checkpoint" and silently routes a timm ViT into ``AutoModel``.
    local_dir = local_backbone_dir(backbone)
    if local_dir is not None:
        with open(local_dir / "config.json", encoding="utf-8") as fh:
            loaded = json.load(fh)
        cfg = loaded if isinstance(loaded, dict) else None
        _HUB_CONFIG_CACHE[backbone] = cfg
        return cfg
    try:
        from huggingface_hub import hf_hub_download

        with open(hf_hub_download(backbone, "config.json"), encoding="utf-8") as fh:
            loaded = json.load(fh)
        cfg = loaded if isinstance(loaded, dict) else None
    except Exception as exc:  # noqa: BLE001 -- offline/gated/missing is just "unknown"
        print(f"[encoder] config.json lookup failed for {backbone!r}: {exc}", flush=True)
    _HUB_CONFIG_CACHE[backbone] = cfg
    return cfg


def _is_timm_config(cfg: dict | None) -> bool:
    """``True`` when a raw hub config.json describes a **timm** checkpoint.

    Pure function of the config so it is testable without the network.

    * ``model_type`` present  -> a ``transformers`` architecture. ``owkin/phikon-v2`` and
      ``owkin/phikon`` say ``"vit"``/``"dinov2"``, ``kaiko-ai/midnight`` says ``"dinov2"``.
      These MUST keep taking the ``AutoModel`` path -- every published number came from it.
    * no ``model_type`` but an ``architecture`` / ``model_args`` / ``pretrained_cfg``
      -> timm's own config format.
    """
    if not isinstance(cfg, dict):
        return False
    if cfg.get("model_type"):
        return False
    return any(k in cfg for k in ("architecture", "model_args", "pretrained_cfg"))


def is_timm_backbone(backbone: str) -> bool:
    """Does ``backbone`` need ``timm.create_model`` rather than ``AutoModel``?"""
    return _is_timm_config(_hub_config(backbone))


def _needs_packed_gated_mlp(fc1_out: int, fc2_in: int) -> bool:
    """Does this block's FFN pack a *gate* and a *value* projection into one ``fc1``?

    timm reads ``model_args`` out of config.json, but the MLP **class** is not expressible
    there -- Virchow2's model card constructs it by hand with
    ``mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU``. Keying that off the repo id would
    reintroduce exactly the per-model dispatch this module refuses, so we derive it from
    the checkpoint's own shapes instead:

    * plain MLP        -> ``fc1: [h, d]``, ``fc2: [d, h]``      i.e. ``fc1_out == fc2_in``
    * packed gated MLP -> ``fc1: [2h, d]``, ``fc2: [d, h]``     i.e. ``fc1_out == 2*fc2_in``

    Virchow2 measures 6832 and 3416 -> packed. Loading it without ``SwiGLUPacked`` raises
    a size mismatch on every one of its 32 blocks, so this is not a soft preference.
    """
    return fc2_in > 0 and fc1_out == 2 * fc2_in


def _ffn_shapes(backbone: str) -> tuple[int, int]:
    """``(blocks.0.mlp.fc1`` out-dim, ``blocks.0.mlp.fc2`` in-dim) for the checkpoint.

    Reads the LOCAL file when the backbone is bound to a directory -- the hub download the
    remote branch does would 403 on a gated repo, and the caller would then silently build
    a plain-MLP model.
    """
    local = local_weight_path(backbone)
    if local is not None:
        if local.suffix == ".safetensors":
            from safetensors import safe_open

            with safe_open(str(local), framework="pt") as f:
                keys = list(f.keys())
                k1 = next((k for k in keys if k.endswith("blocks.0.mlp.fc1.weight")), None)
                k2 = next((k for k in keys if k.endswith("blocks.0.mlp.fc2.weight")), None)
                if k1 is None or k2 is None:
                    return 0, 0
                return int(f.get_slice(k1).get_shape()[0]), int(f.get_slice(k2).get_shape()[1])
        sd = torch.load(str(local), map_location="meta", weights_only=True)
        k1 = next((k for k in sd if k.endswith("blocks.0.mlp.fc1.weight")), None)
        k2 = next((k for k in sd if k.endswith("blocks.0.mlp.fc2.weight")), None)
        if k1 is None or k2 is None:
            return 0, 0
        return int(sd[k1].shape[0]), int(sd[k2].shape[1])

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    path = hf_hub_download(backbone, "model.safetensors")
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        k1 = next((k for k in keys if k.endswith("blocks.0.mlp.fc1.weight")), None)
        k2 = next((k for k in keys if k.endswith("blocks.0.mlp.fc2.weight")), None)
        if k1 is None or k2 is None:
            return 0, 0
        return int(f.get_slice(k1).get_shape()[0]), int(f.get_slice(k2).get_shape()[1])


def _timm_extra_kwargs(backbone: str) -> dict:
    """``timm.create_model`` kwargs that config.json cannot express, derived from weights."""
    try:
        fc1_out, fc2_in = _ffn_shapes(backbone)
    except Exception as exc:  # noqa: BLE001 -- no safetensors / no such key => plain MLP
        print(f"[encoder] FFN-shape probe failed for {backbone!r}: {exc}", flush=True)
        return {}
    if not _needs_packed_gated_mlp(fc1_out, fc2_in):
        return {}
    from timm.layers import SwiGLUPacked

    print(
        f"[encoder] {backbone!r}: blocks.0.mlp fc1_out={fc1_out} == 2 x fc2_in={fc2_in} "
        "-> packed gated FFN; building with mlp_layer=SwiGLUPacked, act_layer=SiLU",
        flush=True,
    )
    return {"mlp_layer": SwiGLUPacked, "act_layer": nn.SiLU}


#: Architecture kwargs for locally-served backbones, transcribed from the model cards.
#:
#: These exist because ``pretrained=True`` on a hub id is what normally applies a repo's
#: ``pretrained_cfg`` and ``model_args``; a gated repo forces ``pretrained=False`` on a
#: bare architecture name, and then NOTHING applies them. UNI2-h is the sharp case: its
#: ``config.json`` names ``vit_giant_patch14_224``, whose timm defaults are embed_dim=1408,
#: depth=40, num_heads=16, plain MLP -- an entirely different model that builds without
#: complaint and then fails to load a single block. Values that a checkpoint can prove
#: (prefix tokens, depth, widths) are re-derived from the built model and asserted in
#: tests/test_new_backbones.py, so a wrong entry here surfaces as a number, not as drift.
#:
#: ``mlp_layer``/``act_layer`` are named as strings so importing this module does not drag
#: in timm; ``_timm_local_kwargs`` resolves them.
BACKBONE_TIMM_KWARGS: dict[str, dict] = {
    # https://huggingface.co/MahmoodLab/UNI2-h -- the card's ``timm_kwargs`` verbatim.
    "MahmoodLab/UNI2-h": {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "no_embed_class": True,
        "mlp_layer": "SwiGLUPacked",
        "act_layer": "SiLU",
        "reg_tokens": 8,
        "dynamic_img_size": True,
    },
    # https://huggingface.co/bioptimus/H-optimus-0 -- ``vit_giant_patch14_reg4_dinov2``
    # already carries depth/width/SwiGLU/reg_tokens=4, so only the card's ``init_values``
    # and ``dynamic_img_size`` are added. ``img_size`` is NOT optional: the architecture's
    # DINOv2 default is 518, whose pos_embed is (1, 1369, 1536) against this checkpoint's
    # (1, 256, 1536).
    "bioptimus/H-optimus-0": {
        "img_size": 224,
        "init_values": 1e-5,
        "dynamic_img_size": False,
    },
}

#: String -> timm/torch layer classes used inside ``BACKBONE_TIMM_KWARGS``.
_TIMM_LAYER_NAMES = ("mlp_layer", "act_layer")


def _resolve_layer(name: str):
    from timm.layers import SwiGLUPacked

    table = {"SwiGLUPacked": SwiGLUPacked, "SiLU": nn.SiLU, "GELU": nn.GELU}
    if name not in table:
        raise RuntimeError(f"unknown layer name {name!r} in BACKBONE_TIMM_KWARGS")
    return table[name]


def _timm_local_kwargs(backbone: str) -> dict:
    """Complete ``timm.create_model`` kwargs for a locally-served backbone.

    ``num_classes=0`` and ``global_pool=""`` are forced last and are not negotiable: this
    encoder consumes the full ``(B, T, D)`` token sequence, and timm's default
    ``global_pool="token"`` would hand back ``(B, D)`` -- already pooled over the CLS token
    only, with the register-token slice in ``_pool`` never reached.
    """
    kwargs: dict = {}
    pinned = BACKBONE_TIMM_KWARGS.get(backbone)
    if pinned is not None:
        kwargs.update(pinned)
        for key in _TIMM_LAYER_NAMES:
            if isinstance(kwargs.get(key), str):
                kwargs[key] = _resolve_layer(kwargs[key])
    else:
        # No transcribed entry: fall back to the same weight-shape probe the hub path
        # uses, which ``_ffn_shapes`` now answers off the local file.
        kwargs.update(_timm_extra_kwargs(backbone))
    kwargs["num_classes"] = 0
    kwargs["global_pool"] = ""
    return kwargs


def _timm_config_normalization(
    backbone: str,
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    """``(mean, std)`` off a timm repo's ``pretrained_cfg``, or ``None``.

    A timm repo has no ``preprocessor_config.json``, so ``AutoImageProcessor`` cannot
    answer for it -- but timm publishes the same information under ``pretrained_cfg``.
    Virchow2 carries ImageNet stats there, which is what its model card's
    ``resolve_data_config`` transform resolves to.
    """
    cfg = _hub_config(backbone)
    if not _is_timm_config(cfg):
        return None
    pc = cfg.get("pretrained_cfg") if isinstance(cfg, dict) else None
    if not isinstance(pc, dict):
        return None
    mean, std = pc.get("mean"), pc.get("std")
    if mean is None or std is None or len(mean) != 3 or len(std) != 3:
        return None
    return tuple(float(v) for v in mean), tuple(float(v) for v in std)


def _hf_preprocessor_normalization(
    backbone: str,
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    """``(mean, std)`` read off the backbone's own HF image processor, or ``None``.

    Most model repos publish ``image_mean``/``image_std`` in ``preprocessor_config.json``,
    which is the same thing the model card states -- so for a new backbone this is the
    authoritative value, not a guess. Any failure (no processor, offline hub, missing
    fields) returns ``None`` so the caller can still fall through to the override or to a
    hard error; a hub blip must not crash a run whose stats are already pinned above.
    """
    try:
        from transformers import AutoImageProcessor

        proc = AutoImageProcessor.from_pretrained(backbone)
        mean, std = getattr(proc, "image_mean", None), getattr(proc, "image_std", None)
    except Exception as exc:  # noqa: BLE001 -- any hub/config failure is just "not derivable"
        print(f"[encoder] AutoImageProcessor lookup failed for {backbone!r}: {exc}", flush=True)
        return None
    if mean is None or std is None or len(mean) != 3 or len(std) != 3:
        return None
    return tuple(float(v) for v in mean), tuple(float(v) for v in std)


def normalization_for(backbone: str | None) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Mean/std for ``backbone``: explicit override, else derived from HF, else refuse.

    A wrong-but-plausible normalisation is invisible in every log and every shape check --
    the old ImageNet fallback only printed a warning, which is unreadable in a 90k-line
    SLURM log. So there is no fallback any more: either we know the stats or we stop.
    """
    backbone = backbone or DEFAULT_BACKBONE
    # The override wins, unconditionally and FIRST. Both published backbones are in it,
    # so nothing added below can ever be reached for them -- not even a hub lookup.
    if backbone in BACKBONE_NORMALIZATION:
        return BACKBONE_NORMALIZATION[backbone]
    # timm repos have no preprocessor_config.json, so AutoImageProcessor cannot answer for
    # them; timm publishes the same fields under pretrained_cfg instead. Tried before the
    # HF processor because for a timm repo the HF lookup can only fail.
    derived = _timm_config_normalization(backbone)
    if derived is not None:
        print(
            f"[encoder] normalisation for {backbone!r} derived from its timm "
            f"pretrained_cfg: mean={derived[0]} std={derived[1]} "
            "(no BACKBONE_NORMALIZATION override)",
            flush=True,
        )
        return derived
    derived = _hf_preprocessor_normalization(backbone)
    if derived is not None:
        print(
            f"[encoder] normalisation for {backbone!r} derived from its HF image processor: "
            f"mean={derived[0]} std={derived[1]} (no BACKBONE_NORMALIZATION override)",
            flush=True,
        )
        return derived
    raise RuntimeError(
        f"no normalisation for backbone {backbone!r}: it has no BACKBONE_NORMALIZATION "
        "entry, no timm pretrained_cfg mean/std, and its HF image processor did not "
        "yield image_mean/image_std. Read the "
        "model card and add an explicit entry to BACKBONE_NORMALIZATION in "
        "src/waivphaet/models/encoder.py -- defaulting to ImageNet stats here would "
        "silently cost accuracy (e.g. kaiko-ai/midnight needs (0.5,0.5,0.5))."
    )

#: Superset of leaf module names that carry the bulk of a transformer block's parameters,
#: across the ViT namings we care about. This is a *candidate* set: the actual target set
#: is the intersection with what the loaded backbone really has, computed per block.
#:
#:   HF Dinov2 / BERT-style : query, key, value, dense, fc1, fc2
#:   SwiGLU FFN (ViT-g)     : weights_in, weights_out
#:   timm / fused-qkv ViTs  : qkv, proj
#:   HF CLIP-style attn     : q_proj, k_proj, v_proj, out_proj
#:   HF ViT, transformers>=5: q_proj, k_proj, v_proj, o_proj  (note o_proj, NOT out_proj)
LORA_CANDIDATE_MODULES: tuple[str, ...] = (
    "query", "key", "value", "dense", "fc1", "fc2",
    "weights_in", "weights_out",
    "qkv", "proj",
    "q_proj", "k_proj", "v_proj", "out_proj", "o_proj",
)

#: Backwards-compatible alias. The old fixed phikon-v2 list; kept so that an explicit
#: ``lora_target_modules=LORA_TARGET_MODULES`` still means what it used to.
LORA_TARGET_MODULES: tuple[str, ...] = (
    "query", "key", "value", "dense", "fc1", "fc2",
)

#: How a transformer block index appears in a module path. Covers HF (``encoder.layer.N``,
#: ``encoder.layers.N``), timm (``blocks.N``) and GPT-style (``h.N``).
_BLOCK_RE = re.compile(r"(?:^|\.)(?:layer|layers|blocks|block|h)\.(\d+)(?:\.|$)")


def _block_index(name: str) -> int | None:
    m = _BLOCK_RE.search(name)
    return int(m.group(1)) if m else None


@dataclass
class EncoderConfig:
    backbone: str = DEFAULT_BACKBONE
    pooling: str = "clsmean"  # "cls" | "mean" | "clsmean"
    # --- LoRA (PLAN.md 2: all blocks; rank is one of the unknown hyperparameters,
    # PLAN.md 3 risk 4 -> sweep it)
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    #: **Split loss heads** (``()`` = off, the legacy single concat projector).
    #:
    #: The training pool ``clsmean`` is ``cat([CLS, mean(patches)])`` -> ONE projector ->
    #: ONE InfoNCE. Nothing in that objective forces *both* halves to become invariant: it
    #: can satisfy itself through whichever half is easier. That matters because the eval
    #: pooling disagrees with the training pooling -- HEST and THUNDER-on-phikon-v2 read
    #: CLS only. Setting this to ``("cls", "mean")`` builds ONE ``ProjectionHead`` PER
    #: POOL, each taking ``hidden_size`` (not ``2*hidden_size``), and the trainer applies a
    #: separate InfoNCE to each.
    #:
    #: A single-element tuple is a genuinely single-head run: the other head is NOT built,
    #: so it burns no compute and -- the part that actually matters -- its
    #: ``nn.BatchNorm1d`` running stats are never updated. A zero-*weighted* dead head
    #: would still do both.
    #:
    #: ``embed_dim`` is deliberately UNTOUCHED by this: it is the eval-time pooled width
    #: (2048 at ``clsmean``), a protocol constant, and every eval path reads ``embed()``
    #: and never a projector.
    split_heads: tuple[str, ...] = ()
    #: **Which pooling the non-CLS loss head uses** (``"mean"`` = off, the incumbent).
    #:
    #: RESULTS 9: ``mean`` is LINEAR, so ``d(mean)/d(t_i) = (1/N) I`` -- the direct
    #: gradient reaching every patch token is the IDENTICAL vector, and a uniform
    #: translation of the token cloud is exactly what THUNDER's biased
    #: ``proj_dec = nn.Linear(d_encoder, d_model)`` absorbs into its bias. Only a pooling
    #: whose gradient is token-DEPENDENT can express a preference about the tokens'
    #: relative arrangement. See :mod:`waivphaet.models.pooling` for the variants and for
    #: why the GeM default is over ``softplus`` rather than the textbook ``clamp``.
    #:
    #: Requires ``split_heads`` to include ``"mean"``: with the single concat projector
    #: the pooled vector IS ``_pool``'s output, which is the eval protocol constant.
    pool_head: str = "mean"
    #: Apply the learned ``pool_head`` inside :meth:`WaivEncoder._pool`, i.e. in the
    #: EXPORTED embedding, not only in the training loss. OFF by default: with it off
    #: the eval representation stays the protocol constant (PathoROB's reference row is
    #: ``phikonv2_clsmean``) and every existing number remains comparable.
    #:
    #: Turning it ON is an ARCHITECTURAL change to what the run exports, and is the whole
    #: point of the "pooling at inference" arm: to date every pooling variant was scored
    #: through an identical ``clsmean`` head, so the learned pooling reached the readout
    #: only indirectly, via the LoRA weights it shaped. Dimensionality is UNCHANGED --
    #: every pool head maps ``(B, N, hidden) -> (B, hidden)``, so ``clsmean`` stays
    #: ``2 * hidden`` and downstream probes load exactly as before.
    #:
    #: Unlike ``pool_head`` itself this does NOT require ``split_heads``: at eval time the
    #: encoder is rebuilt with no split heads at all, and the head is restored from the
    #: checkpoint's ``pool_head.pt``.
    infer_pool_head: bool = False
    #: ``None`` (the default) = **discover** the target leaf names from the loaded
    #: backbone by intersecting ``LORA_CANDIDATE_MODULES`` with the block Linears it
    #: actually has. Pass an explicit tuple only to deliberately narrow the set.
    lora_target_modules: tuple[str, ...] | None = None
    lora_blocks: tuple[int, ...] | None = None  # None = ALL blocks (the default, on purpose)
    # --- projection head (PLAN.md 2: >= 512, NOT ScanGen's 48)
    proj_hidden_dim: int = 1024
    proj_out_dim: int = 512
    proj_use_bn: bool = True
    # --- memory/compute trade
    #: Recompute block activations in the backward pass instead of storing them. The
    #: same-condition constraint makes the *in-group* negative count ``group_size - 1``,
    #: so the only way to buy more negatives is a bigger forward batch -- and at
    #: ~0.21 GiB/image (measured, 128 img/step -> 27.35 GiB) a plain ViT-L/16 run caps
    #: out near 340 images on an 80 GiB H100. Checkpointing drops that to ~0.02 GiB/image
    #: for roughly +35% step time, which is the trade that makes 384-anchor groups
    #: possible at all. Off by default so the smoke-run numbers stay reproducible.
    grad_checkpointing: bool = False
    # --- misc
    freeze_backbone: bool = False  # True => frozen-feature probe (PLAN.md 3 phase 6)
    dtype: str = "float32"
    extra: dict = field(default_factory=dict)


def normalize_uint8(x: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    """``(B, 224, 224, 3)`` uint8 -> ``(B, 3, 224, 224)`` normalised float.

    The pair loader hands us raw uint8 NHWC straight off the memmap (no PIL, no resize:
    PLISM tiles are already exactly 224x224), so this is the whole preprocessing stack.
    ``mean``/``std`` default to ImageNet for backwards compatibility; callers inside the
    encoder pass the *backbone's* stats (``normalization_for``).
    """
    if x.dtype == torch.uint8:
        x = x.float().div_(255.0)
    if x.ndim == 4 and x.shape[-1] == 3:
        x = x.permute(0, 3, 1, 2)
    m = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - m) / s


class ProjectionHead(nn.Module):
    """2-layer MLP projector. InfoNCE is applied on its output, not on the backbone."""

    def __init__(self, in_dim: int, hidden_dim: int = 1024, out_dim: int = 512, use_bn: bool = True):
        super().__init__()
        if out_dim < 512:
            raise ValueError(
                f"proj_out_dim={out_dim} < 512; PLAN.md 2 rules out narrow heads "
                "(ScanGen's 48/96) for 16k-tile retrieval"
            )
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers += [nn.GELU(), nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _lora_target_names(model: nn.Module, cfg: EncoderConfig) -> tuple[list[str], dict[int, int], tuple[str, ...]]:
    """Resolve full module names so we can (a) hit every block and (b) *prove* we did.

    PEFT's ``target_modules`` accepts bare suffixes, but a bare suffix silently matches
    whatever happens to share the name. We enumerate explicit full names instead, then
    assert the per-block count -- head-only adaptation is the failure mode PLAN.md 2
    explicitly rules out, and it would otherwise be invisible.

    Architecture-agnostic by construction: we walk the backbone's ``nn.Linear`` modules,
    keep the ones that live inside a numbered transformer block, and select by leaf name
    against ``cfg.lora_target_modules`` -- or, when that is ``None``, against the
    *candidate* superset, which is what makes ``fc1/fc2`` (Dinov2 MLP) and
    ``weights_in/weights_out`` (SwiGLU FFN, ViT-g) both resolve without a per-model list.

    Returns ``(names, per_block_counts, resolved_leaf_names)``.
    """
    candidates = cfg.lora_target_modules or LORA_CANDIDATE_MODULES
    names: list[str] = []
    per_block: dict[int, int] = {}
    leaves: set[str] = set()
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in candidates:
            continue
        blk = _block_index(name)
        if blk is None:
            # A Linear with a matching leaf name outside any transformer block (a head,
            # a pooler). Never adapt it: LoRA-on-the-head is exactly what PLAN.md 2 rules
            # out, and it would inflate the "targets" count while adapting no depth.
            continue
        if cfg.lora_blocks is not None and blk not in cfg.lora_blocks:
            continue
        names.append(name)
        per_block[blk] = per_block.get(blk, 0) + 1
        leaves.add(leaf)
    return names, per_block, tuple(sorted(leaves))


class WaivEncoder(nn.Module):
    """Backbone (optionally LoRA-adapted) + projection head.

    ``forward`` returns ``(embedding, projection)``:

    * ``embedding`` -- pooled backbone output. This is what goes to PathoROB / plismbench.
    * ``projection`` -- L2-normalisable head output. This is what InfoNCE sees.
    """

    @staticmethod
    def _build_local_timm(backbone_id: str, local_dir: Path):
        """Build a GATED timm backbone from ``local_dir`` and load its weights STRICTLY.

        ``pretrained=True`` is unavailable here (the repo 403s), and ``pretrained=False``
        means the architecture kwargs and the weight load are both ours to get right. Both
        fail silently if we let them: a wrong ``mlp_layer`` or ``depth`` builds a clean
        model whose blocks then simply do not match, and ``load_state_dict(strict=False)``
        -- the usual reflex when a load is noisy -- reports that as a list nobody reads
        while leaving those blocks at their random init. The result trains, evaluates,
        and produces a plausible, warning-free, wrong number.

        So: any missing or unexpected key is fatal, and the counts are printed on success
        so the run log carries positive evidence rather than the absence of a complaint.
        """
        import timm

        cfg_json = _hub_config(backbone_id) or {}
        arch = cfg_json.get("architecture")
        if not arch:
            raise RuntimeError(
                f"{local_dir / 'config.json'} has no 'architecture' key, so there is no "
                f"way to know what to build for {backbone_id!r}"
            )
        kwargs = _timm_local_kwargs(backbone_id)
        model = timm.create_model(arch, pretrained=False, **kwargs)

        sd = local_state_dict(backbone_id)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"{backbone_id!r} did not load cleanly from {local_dir}: "
                f"{len(missing)} missing, {len(unexpected)} unexpected keys. "
                f"missing[:10]={list(missing)[:10]} unexpected[:10]={list(unexpected)[:10]}. "
                f"This means the architecture kwargs are wrong, not that the load is "
                f"'close enough' -- the unmatched parameters would stay randomly "
                f"initialised. Fix BACKBONE_TIMM_KWARGS[{backbone_id!r}]."
            )

        # The architecture's own default_cfg describes the HUB variant, not this file --
        # ``vit_giant_patch14_reg4_dinov2`` advertises input_size 518 while this checkpoint
        # is a 224 model. Prefer the local config's pretrained_cfg so ``config_image_size``
        # below reports the checkpoint we actually built.
        local_pc = cfg_json.get("pretrained_cfg")
        if isinstance(local_pc, dict):
            merged = dict(getattr(model, "pretrained_cfg", {}) or {})
            merged.update(local_pc)
            merged.setdefault("architecture", arch)
            model.pretrained_cfg = merged

        print(
            f"[encoder] {backbone_id!r}: built {arch} from {local_dir} "
            f"(GATED on the hub) -- 0 missing / 0 unexpected keys over "
            f"{len(sd)} tensors; kwargs={ {k: getattr(v, '__name__', v) for k, v in sorted(kwargs.items())} }",
            flush=True,
        )
        return model

    def __init__(self, cfg: EncoderConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or EncoderConfig()
        self.is_timm = is_timm_backbone(cfg.backbone)
        if self.is_timm:
            import timm

            local_dir = local_backbone_dir(cfg.backbone)
            if local_dir is None:
                backbone = timm.create_model(
                    f"hf-hub:{cfg.backbone}", pretrained=True, **_timm_extra_kwargs(cfg.backbone)
                )
            else:
                backbone = self._build_local_timm(cfg.backbone, local_dir)
            # Geometry off the BUILT timm model -- same rule as the HF branch, no literals.
            self.hidden_size = int(backbone.embed_dim)
            self.num_blocks = len(backbone.blocks)
            ps = backbone.patch_embed.patch_size
            self.patch_size = int(ps[0] if isinstance(ps, (tuple, list)) else ps) or None
            pc = getattr(backbone, "pretrained_cfg", {}) or {}
            insize = pc.get("input_size") or (0, 0, 0)
            self.config_image_size = int(insize[-1]) or None
            arch = (_hub_config(cfg.backbone) or {}).get("architecture") or pc.get(
                "architecture", "vit"
            )
            self.model_type = f"timm:{arch}"
            #: **Register tokens.** Virchow2 is ``reg_tokens=4``, so its 261-token output is
            #: ``[CLS] [reg x4] [patch x256]`` and positions 1..4 are NOT image content --
            #: the model card is explicit ("tokens 1-4 are register tokens so we ignore
            #: those"). Averaging them in is the classic silent-wrong-embedding bug: right
            #: shape, right dtype, no warning, just a worse number. timm tracks the count
            #: itself, so we read it rather than assuming.
            self.num_prefix_tokens = int(getattr(backbone, "num_prefix_tokens", 1))
        else:
            backbone = AutoModel.from_pretrained(cfg.backbone)
            # Everything geometric is READ OFF THE LOADED CONFIG. Nothing here may be a
            # literal: phikon-v2 is 1024/24/patch16, midnight is 1536/40/patch14.
            bc = backbone.config
            self.hidden_size = int(bc.hidden_size)
            self.num_blocks = int(bc.num_hidden_layers)
            self.patch_size = int(getattr(bc, "patch_size", 0)) or None
            #: Tokens the backbone emits at ``image_size``, per its own config. We feed
            #: 224px everywhere (PathoROB/HEST/THUNDER/PLISM all resize to 224), and Dinov2
            #: interpolates its position embeddings, so the *runtime* patch count is
            #: ``(224/patch_size)**2`` -- 196 on phikon-v2 (16), 256 on midnight (14).
            self.config_image_size = int(getattr(bc, "image_size", 0)) or None
            self.model_type = str(getattr(bc, "model_type", "unknown"))
            #: Every HF ViT/Dinov2 we run emits exactly one prefix token, the CLS. Fixed at
            #: 1 so ``_pool`` is bit-identical to what produced the published numbers.
            self.num_prefix_tokens = 1
        self.norm_mean, self.norm_std = normalization_for(cfg.backbone)

        if cfg.freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad_(False)

        self.lora_target_names: list[str] = []
        self.lora_target_leaves: tuple[str, ...] = ()
        self.lora_per_block: dict[int, int] = {}
        if cfg.use_lora:
            from peft import LoraConfig, get_peft_model

            self.lora_target_names, self.lora_per_block, self.lora_target_leaves = (
                _lora_target_names(backbone, cfg)
            )
            # A silently-empty target set trains NOTHING and reads downstream as "the
            # method had no effect on this backbone". Refuse, loudly, with the evidence.
            if not self.lora_target_names:
                sample = sorted({
                    n.rsplit(".", 1)[-1]
                    for n, m in backbone.named_modules()
                    if isinstance(m, nn.Linear) and _block_index(n) is not None
                })
                raise RuntimeError(
                    f"no LoRA targets matched on backbone {cfg.backbone!r} "
                    f"(model_type={self.model_type}): candidates="
                    f"{cfg.lora_target_modules or LORA_CANDIDATE_MODULES}, but the block "
                    f"Linears are named {sample}. Add the missing names to "
                    "LORA_CANDIDATE_MODULES -- an empty target set trains nothing."
                )
            covered = set(self.lora_per_block)
            expected = set(cfg.lora_blocks) if cfg.lora_blocks is not None else set(range(self.num_blocks))
            if covered != expected:
                raise RuntimeError(
                    f"LoRA covers blocks {sorted(covered)} but expected {sorted(expected)}; "
                    "PLAN.md 2 requires adaptation across the full depth, not head-only"
                )
            # Uniformity is the second half of the guard. A ragged count means the leaf
            # names differ between blocks, i.e. some blocks are only partly adapted --
            # which the block-coverage check above cannot see.
            counts = set(self.lora_per_block.values())
            if len(counts) != 1:
                ragged = {b: c for b, c in sorted(self.lora_per_block.items())}
                raise RuntimeError(
                    f"LoRA match count is not uniform across blocks: {ragged}"
                )
            # Third guard: an in-block Linear that matched NOTHING. Neither check above
            # can see this -- a leaf dropped uniformly from every block leaves the target
            # set non-empty and the per-block count uniform, so the adapter looks healthy
            # and is quietly missing a whole projection. owkin/phikon under transformers>=5
            # names its attention output o_proj while the candidate list had only out_proj,
            # which would have skipped all 12 attention outputs with no error at all.
            in_block_leaves = {
                n.rsplit(".", 1)[-1]
                for n, m in backbone.named_modules()
                if isinstance(m, nn.Linear) and _block_index(n) is not None
            }
            dropped = in_block_leaves - set(self.lora_target_leaves)
            if dropped:
                raise RuntimeError(
                    f"LoRA silently skipped in-block Linears on {cfg.backbone!r} "
                    f"(model_type={self.model_type}): {sorted(dropped)} matched no "
                    f"candidate. Targeted {sorted(self.lora_target_leaves)}. Add the "
                    "missing names to LORA_CANDIDATE_MODULES, or pass an explicit "
                    "lora_target_modules to state that skipping them is intended -- "
                    "a partly-adapted block reads downstream as a weak method, not a bug."
                )
            print(
                f"[encoder] backbone={cfg.backbone} type={self.model_type} "
                f"hidden={self.hidden_size} blocks={self.num_blocks} "
                f"patch={self.patch_size} | LoRA targets={len(self.lora_target_names)} "
                f"= {counts.pop()}/block x {self.num_blocks} blocks, "
                f"leaves={list(self.lora_target_leaves)}",
                flush=True,
            )
            backbone = get_peft_model(
                backbone,
                LoraConfig(
                    r=cfg.lora_rank,
                    lora_alpha=cfg.lora_alpha,
                    lora_dropout=cfg.lora_dropout,
                    target_modules=self.lora_target_names,
                    bias="none",
                ),
            )
        if cfg.grad_checkpointing:
            # use_reentrant=False: the reentrant autograd.Function variant needs at least
            # one input with requires_grad, and with LoRA the embedding output is frozen,
            # so the reentrant path silently produces *no* gradient for the early blocks.
            target = getattr(backbone, "base_model", backbone)
            target = getattr(target, "model", target)
            if self.is_timm:
                # timm has its own switch; it has no gradient_checkpointing_enable at all,
                # so the HF call below would AttributeError. timm's blocks use
                # torch.utils.checkpoint with use_reentrant=False internally.
                target.set_grad_checkpointing(True)
                took = bool(getattr(target, "grad_checkpointing", False))
            else:
                target.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                took = bool(getattr(target, "is_gradient_checkpointing", False))
            if not took:
                raise RuntimeError(
                    "gradient checkpointing did not take on the backbone; refusing to run "
                    "with the batch size it was requested for"
                )
        self.backbone = backbone

        self.embed_dim = self.hidden_size * (2 if cfg.pooling == "clsmean" else 1)
        #: Which pools get their own loss head. ``()`` = the legacy single concat head,
        #: which is the path every published number was produced on and is left
        #: bit-identical (same module, same construction order, same RNG draws).
        self.split_heads: tuple[str, ...] = tuple(cfg.split_heads or ())
        bad = [h for h in self.split_heads if h not in POOL_PARTS]
        if bad:
            raise ValueError(
                f"unknown split head(s) {bad}; valid names are {list(POOL_PARTS)}"
            )
        if len(set(self.split_heads)) != len(self.split_heads):
            raise ValueError(f"duplicate split head in {self.split_heads}")

        # --- pooling for the non-CLS loss head (RESULTS 9's "the fix") -----------------
        self.pool_head_name = str(cfg.pool_head or "mean")
        if self.pool_head_name not in POOL_HEAD_NAMES:
            raise ValueError(
                f"unknown pool_head {self.pool_head_name!r}; valid are "
                f"{list(POOL_HEAD_NAMES)}"
            )
        self.infer_pool_head = bool(getattr(cfg, "infer_pool_head", False))
        if self.infer_pool_head and self.pool_head_name == "mean":
            raise ValueError(
                "infer_pool_head=True with pool_head='mean' is a no-op: the mean arm's "
                "pool IS patches.mean(dim=1), which _pool already computes."
            )
        if self.cfg.pooling == "cls" and self.infer_pool_head:
            raise ValueError(
                "infer_pool_head=True with pooling='cls' is a no-op: the CLS pool never "
                "touches the patch tokens. Use pooling='clsmean' (or 'mean')."
            )
        # The 'mean' split head is what the pooling is TRAINED through. It is not needed to
        # APPLY a restored pool head at eval, where the encoder carries no split heads.
        if (
            self.pool_head_name != "mean"
            and "mean" not in self.split_heads
            and not self.infer_pool_head
        ):
            raise ValueError(
                f"pool_head={self.pool_head_name!r} requires split_heads to include "
                f"'mean' (got {list(self.split_heads) or 'the single concat projector'}). "
                "With the single concat head the pooled vector is _pool()'s output, which "
                "is the EVAL protocol constant (PathoROB's reference row is "
                "phikonv2_clsmean) -- repointing it would change what the run exports, not "
                "what it optimises. On a cls-only split arm there is no mean head to pool."
            )
        #: ``None`` on ``pool_head='mean'`` ON PURPOSE. That arm then keeps the literal
        #: ``patches.mean(dim=1)`` already inside :meth:`_pool_parts`, so it is the SAME
        #: CODE as the currently-running split-head arm rather than merely the same
        #: arithmetic -- no new module, no new parameters, no RNG draw, nothing to perturb.
        self.pool_head = (
            build_pool_head(self.pool_head_name, self.hidden_size)
            if self.pool_head_name != "mean"
            else None
        )

        if self.split_heads:
            # Each head sees ONE pool, so its input is `hidden_size` -- NOT `embed_dim`.
            self.projector = None
            self.projectors = nn.ModuleDict({
                name: ProjectionHead(
                    self.hidden_size, cfg.proj_hidden_dim, cfg.proj_out_dim, cfg.proj_use_bn
                )
                for name in self.split_heads
            })
        else:
            self.projectors = None
            self.projector = ProjectionHead(
                self.embed_dim, cfg.proj_hidden_dim, cfg.proj_out_dim, cfg.proj_use_bn
            )

    # --- pooling ------------------------------------------------------------------

    def _pool(self, tokens: torch.Tensor) -> torch.Tensor:
        # ``num_prefix_tokens`` is 1 on every HF backbone (CLS only), so this slice is
        # ``tokens[:, 1:, :]`` there -- bit-identical to what it always was. It is 5 on
        # Virchow2 ([CLS] + 4 registers), which is precisely the model card's
        # ``cat([output[:, 0], output[:, 5:].mean(1)])``.
        cls, patches = tokens[:, 0, :], tokens[:, self.num_prefix_tokens :, :]
        if self.cfg.pooling == "cls":
            return cls
        # OFF by default, so this is the literal `patches.mean(dim=1)` it always was. When
        # on, the mean SLOT is filled by the learned pool head instead -- same shape, same
        # dtype, so `embed_dim` and every downstream probe are untouched.
        if getattr(self, "infer_pool_head", False) and self.pool_head is not None:
            mean = self.pool_head(patches).to(patches.dtype)
        else:
            mean = patches.mean(dim=1)
        if self.cfg.pooling == "mean":
            return mean
        if self.cfg.pooling == "clsmean":
            return torch.cat([cls, mean], dim=1)
        raise ValueError(f"unknown pooling {self.cfg.pooling!r}")

    def _pool_parts(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """The two pools ``clsmean`` concatenates, kept SEPARATE. Each is ``(B, hidden)``.

        The slice is ``self.num_prefix_tokens``, read off the loaded backbone -- 1 on every
        HF ViT/Dinov2, **5 on Virchow2** ([CLS] + 4 registers). Hardcoding ``tokens[:, 1:]``
        here would silently average four register tokens into the "mean" head's input on
        Virchow2: right shape, right dtype, no warning, just a worse number. Same rule and
        the same source of truth as :meth:`_pool`, deliberately.
        """
        return {
            "cls": tokens[:, 0, :],
            "mean": tokens[:, self.num_prefix_tokens :, :].mean(dim=1),
        }

    def pool_from_parts(self, parts: dict[str, torch.Tensor]) -> torch.Tensor:
        """Reassemble :meth:`_pool`'s output from :meth:`_pool_parts`, per ``cfg.pooling``.

        Exists so the split-head trainer can still hand the retention term (and anything
        else that wants "the embedding this run exports") exactly the vector the
        single-head path would have produced, without a second backbone forward.
        """
        if self.cfg.pooling == "cls":
            return parts["cls"]
        if self.cfg.pooling == "mean":
            return parts["mean"]
        if self.cfg.pooling == "clsmean":
            return torch.cat([parts["cls"], parts["mean"]], dim=1)
        raise ValueError(f"unknown pooling {self.cfg.pooling!r}")

    # --- forward ------------------------------------------------------------------

    def tokens(self, images: torch.Tensor) -> torch.Tensor:
        """uint8 NHWC (or normalised float NCHW) -> raw ``(B, T, hidden)`` token sequence.

        Split out of :meth:`embed` (pure code motion -- the default path is bit-identical)
        so the split-head trainer can pool the SAME single forward two different ways
        instead of running the backbone twice.
        """
        if images.dtype == torch.uint8 or images.shape[-1] == 3:
            images = normalize_uint8(images, self.norm_mean, self.norm_std)
        if self.is_timm:
            # Built with global_pool="" and num_classes=0 (config.json model_args), so the
            # head is a no-op and forward() returns the raw token sequence. Going through
            # forward() rather than forward_features() keeps the PEFT wrapper in the path.
            tokens = self.backbone(images)
            if not isinstance(tokens, torch.Tensor) or tokens.ndim != 3:
                raise RuntimeError(
                    f"timm backbone {self.cfg.backbone!r} returned "
                    f"{type(tokens).__name__} shape "
                    f"{tuple(getattr(tokens, 'shape', ()))}, not a (B, T, D) token "
                    "sequence -- its config.json model_args must set global_pool='' and "
                    "num_classes=0, otherwise the head has already pooled and our "
                    "pooling/register-token handling is silently bypassed"
                )
        else:
            tokens = self.backbone(pixel_values=images).last_hidden_state
        return tokens

    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """uint8 NHWC (or normalised float NCHW) -> pooled embedding ``(B, embed_dim)``.

        Unaffected by ``split_heads``: this is the eval-time export, and the eval pooling
        is a protocol constant (PathoROB's reference row is ``phikonv2_clsmean``).
        """
        return self._pool(self.tokens(images))

    def embed_parts(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """``{"cls": (B, hidden), "mean": (B, hidden)}`` from ONE backbone forward."""
        return self._pool_parts(self.tokens(images))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.split_heads:
            raise RuntimeError(
                "this encoder was built with split loss heads "
                f"{list(self.split_heads)}, so there is no single `projector` to call. "
                "Use forward_split() (or embed() for the eval-time pooled embedding)."
            )
        emb = self.embed(images)
        return emb, self.projector(emb)

    def forward_split(
        self, images: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """``(parts, projections)`` for the split-head objective, one backbone forward.

        ``projections`` carries a key for each head that was BUILT -- a head omitted from
        ``cfg.split_heads`` does not exist, is not run, and cannot update BatchNorm stats.
        """
        if not self.split_heads:
            raise RuntimeError(
                "forward_split() requires an encoder built with cfg.split_heads; this one "
                "has the single concat projector. Use forward()."
            )
        tokens = self.tokens(images)
        # ``parts`` keeps its old meaning EXACTLY: the two halves of ``_pool``'s clsmean.
        # It is what ``pool_from_parts`` reassembles for the retention term, and that must
        # stay the eval-time pooled embedding no matter which pooling the LOSS head uses.
        parts = self._pool_parts(tokens)
        head_in = parts
        if self.pool_head is not None:
            pooled = self.pool_head(tokens[:, self.num_prefix_tokens :, :])
            head_in = {**parts, "mean": pooled}
            # Published under its own key so the diagnostics see what the head ACTUALLY
            # got, while ``pool_from_parts`` (which reads only cls/mean) is untouched.
            parts = {**parts, "pool": pooled}
        return parts, {name: self.projectors[name](head_in[name]) for name in self.split_heads}

    def pool_head_metrics(self) -> dict[str, float]:
        """Learned pooling state (GeM ``p``, LSE ``tau``, attention entropy, ...).

        Empty on the default path. Logged every ``log_every`` step so that "the pooling
        learned something" is an auditable number in ``history.json`` rather than an
        inference from the loss curve -- a GeM that decays to ``p=1`` or an attention that
        stays at uniform entropy 1.0 has quietly become the mean head again, and nothing
        else in the run would say so.
        """
        if self.pool_head is None:
            return {}
        fn = getattr(self.pool_head, "extra_metrics", None)
        return dict(fn()) if callable(fn) else {}

    @torch.no_grad()
    def encode(self, images: torch.Tensor, l2_normalize: bool = False) -> torch.Tensor:
        """Inference-time embedding for the eval adapters (PathoROB / plismbench)."""
        self.eval()
        emb = self.embed(images)
        return F.normalize(emb, dim=-1) if l2_normalize else emb

    # --- housekeeping -------------------------------------------------------------

    def trainable_parameter_summary(self) -> dict[str, int | float]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "backbone": self.cfg.backbone,
            "model_type": self.model_type,
            "hidden_size": self.hidden_size,
            "embed_dim": self.embed_dim,
            "patch_size": self.patch_size,
            "num_prefix_tokens": self.num_prefix_tokens,
            "loader": "timm" if self.is_timm else "transformers.AutoModel",
            "total": total,
            "trainable": trainable,
            "trainable_pct": 100.0 * trainable / max(total, 1),
            "lora_targets": len(self.lora_target_names),
            "lora_targets_per_block": (
                max(self.lora_per_block.values()) if self.lora_per_block else 0
            ),
            "lora_target_leaves": list(self.lora_target_leaves),
            "blocks": self.num_blocks,
            "split_heads": list(self.split_heads),
            "pool_head": self.pool_head_name,
            "pool_head_params": (
                sum(p.numel() for p in self.pool_head.parameters())
                if self.pool_head is not None else 0
            ),
        }

    def merge_lora(self) -> nn.Module:
        """Merge LoRA deltas into the base weights -> a plain Dinov2 checkpoint.

        PLAN.md 2: LoRA "merges to full weights afterwards", which is what lets the eval
        adapters and any downstream user load us as an ordinary ``owkin/phikon-v2``.
        """
        if not self.cfg.use_lora:
            return self.backbone
        return self.backbone.merge_and_unload()

    def set_lora_scale(self, scale: float) -> int:
        """Rescale every LoRA delta by ``scale`` in place; returns the layers touched.

        Interpolates between the base model (``scale=0``) and the trained one
        (``scale=1``) without rebuilding either. PEFT keeps the effective multiplier in
        ``LoraLayer.scaling[adapter]``, normally ``alpha / r``; we overwrite it rather
        than multiply so repeated calls are idempotent instead of compounding.

        Returns the LoRA layer count so callers can assert the adapter actually attached
        -- a silent 0 here would read as "interpolation is flat" rather than "no adapter".
        """
        from peft.tuners.lora import LoraLayer

        n = 0
        for module in self.backbone.modules():
            if isinstance(module, LoraLayer):
                for adapter in module.scaling:
                    base = module.lora_alpha[adapter] / module.r[adapter]
                    module.scaling[adapter] = base * scale
                n += 1
        if n == 0:
            raise RuntimeError(
                f"set_lora_scale({scale}) found no LoRA layers: the adapter did not "
                "attach, and scoring would silently return base-model features"
            )
        return n


def lora_scale_tag(scale: float) -> str:
    """Filename-safe token for a LoRA scale: 0.5 -> 'ls050', 0.75 -> 'ls075'.

    Both feature dirs and HEST exp_codes key on the model name ALONE, so two scales
    written under one name collide and the second silently scores the first's
    embeddings. Callers derive the required token from the value with this helper and
    hard-fail if it is absent from the name, which also catches a hand-typed 'ls050'
    while actually passing 0.75. Nothing is required at scale 1.0, so every name
    already on disk stays valid.
    """
    return f"ls{round(scale * 100):03d}"


#: The class was phikon-v2-specific when it was written; it no longer is. Alias kept so
#: saved checkpoints, the THUNDER entry point and any external caller keep importing.
PhikonEncoder = WaivEncoder


def build_encoder(**kwargs) -> WaivEncoder:
    return WaivEncoder(EncoderConfig(**kwargs))
