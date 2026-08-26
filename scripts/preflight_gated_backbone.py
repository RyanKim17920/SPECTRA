#!/usr/bin/env python3
"""Pre-flight for a GATED, locally-served backbone -- print the evidence, do not assert quietly.

Every check here guards something that fails SILENTLY: a non-strict weight load leaves
randomly-initialised blocks, a wrong prefix-token count averages register tokens into the
embedding, and a normalisation fall-through swaps H&E statistics for ImageNet's.  Each of
those produces a right-shaped, warning-free, WORSE number, so this script prints the
actual value it observed rather than only a pass/fail.

Run it before submitting a base evaluation for a backbone that is built from
``BACKBONE_LOCAL_DIRS``:

    python3 scripts/preflight_gated_backbone.py MahmoodLab/UNI2-h
    python3 scripts/preflight_gated_backbone.py --all --json out.json

Exit status is non-zero if any check fails, so an sbatch can gate on it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from waivphaet.models.encoder import (  # noqa: E402
    BACKBONE_LOCAL_DIRS,
    BACKBONE_NORMALIZATION,
    IMAGENET_MEAN,
    IMAGENET_STD,
    EncoderConfig,
    WaivEncoder,
    _hub_config,
    _timm_local_kwargs,
    is_timm_backbone,
    local_backbone_dir,
    local_state_dict,
    local_weight_path,
    normalization_for,
)


def _strict_load_counts(backbone: str) -> dict:
    """Rebuild the timm model and report the RAW missing/unexpected key counts.

    ``WaivEncoder`` already refuses a non-zero count, so this exists to print the number
    instead of inferring "it must have been zero because nothing raised".
    """
    import timm

    d = local_backbone_dir(backbone)
    cfg = _hub_config(backbone) or {}
    arch = cfg.get("architecture")
    kwargs = _timm_local_kwargs(backbone)
    model = timm.create_model(arch, pretrained=False, **kwargs)
    sd = local_state_dict(backbone)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    del model, sd
    return {
        "local_dir": str(d),
        "weight_file": str(local_weight_path(backbone)),
        "timm_arch": arch,
        "timm_kwargs": {k: repr(v) for k, v in sorted(kwargs.items())},
        "n_missing_keys": len(missing),
        "n_unexpected_keys": len(unexpected),
        "missing_keys": list(missing)[:10],
        "unexpected_keys": list(unexpected)[:10],
        "ok": not missing and not unexpected,
    }


def preflight(backbone: str) -> dict:
    out: dict = {"backbone": backbone, "checks": {}}
    ck = out["checks"]

    # 1. dispatch + strict load -------------------------------------------------------
    ck["dispatch_is_timm"] = {"value": is_timm_backbone(backbone), "ok": is_timm_backbone(backbone)}
    ck["strict_load"] = _strict_load_counts(backbone)

    # 2. normalisation: must be a STATEMENT, never a fall-through ---------------------
    mean, std = normalization_for(backbone)
    pinned = backbone in BACKBONE_NORMALIZATION
    is_imagenet = (tuple(mean), tuple(std)) == (tuple(IMAGENET_MEAN), tuple(IMAGENET_STD))
    ck["normalization"] = {
        "mean": list(mean),
        "std": list(std),
        "is_imagenet": is_imagenet,
        "pinned_in_BACKBONE_NORMALIZATION": pinned,
        # The mechanism, not the value, is what is being checked: "ImageNet by
        # coincidence" and "ImageNet by fall-through" are the same tuple at the call site.
        "ok": pinned,
    }

    # 3-5. the built encoder ----------------------------------------------------------
    enc = WaivEncoder(EncoderConfig(backbone=backbone, pooling="clsmean", use_lora=True))
    hidden = int(enc.hidden_size)
    npx = int(enc.num_prefix_tokens)
    ck["geometry"] = {
        "num_prefix_tokens": npx,
        "hidden_size": hidden,
        "num_blocks": int(enc.num_blocks),
        "patch_size": int(enc.patch_size),
        "ok": npx > 1,   # a register-carrying ViT reporting 1 means the probe failed
    }
    ck["lora"] = {
        "target_leaves": list(enc.lora_target_leaves),
        "n_targets": len(enc.lora_target_names),
        "blocks_covered": len(enc.lora_per_block),
        "per_block": sorted(set(enc.lora_per_block.values())),
        "ok": bool(enc.lora_target_names)
              and len(enc.lora_per_block) == int(enc.num_blocks)
              and set(enc.lora_per_block.values()) == {4},
    }

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        tokens = enc.tokens(x)
        enc.cfg.pooling = "clsmean"
        e_clsmean = enc.embed(x)
        enc.cfg.pooling = "cls"
        e_cls = enc.embed(x)
        enc.cfg.pooling = "clsmean"
        official = torch.cat([tokens[:, 0], tokens[:, npx:].mean(1)], dim=-1)
        naive = torch.cat([tokens[:, 0], tokens[:, 1:].mean(1)], dim=-1)

    ck["forward_widths"] = {
        "tokens": list(tokens.shape),
        "expected_tokens": [2, npx + 256, hidden],
        "cls_dim": list(e_cls.shape),
        "clsmean_dim": list(e_clsmean.shape),
        "ok": (list(tokens.shape) == [2, npx + 256, hidden]
               and list(e_cls.shape) == [2, hidden]
               and list(e_clsmean.shape) == [2, 2 * hidden]),
    }
    # The register-token test with teeth: equal to the card's slice, DIFFERENT from [:, 1:].
    max_abs_diff = float((official - naive).abs().max()) if npx > 1 else 0.0
    ck["pool_slice"] = {
        "slices_at": "[:, %d:, :]" % npx,
        "equals_official": bool(torch.equal(e_clsmean, official)),
        "differs_from_tokens_1_slice": bool(not torch.allclose(e_clsmean, naive)),
        "max_abs_diff_vs_naive": max_abs_diff,
        "ok": bool(torch.equal(e_clsmean, official))
              and (npx == 1 or not torch.allclose(e_clsmean, naive)),
    }

    out["ok"] = all(c.get("ok") for c in ck.values())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("backbone", nargs="*", help="backbone id(s); default --all")
    ap.add_argument("--all", action="store_true", help="every backbone in BACKBONE_LOCAL_DIRS")
    ap.add_argument("--json", type=Path, help="also write the full result here")
    a = ap.parse_args()

    names = list(a.backbone) or (list(BACKBONE_LOCAL_DIRS) if a.all else [])
    if not names:
        ap.error("give a backbone id or --all")

    results = []
    for b in names:
        print("=" * 78)
        print("PRE-FLIGHT  %s" % b)
        print("=" * 78)
        r = preflight(b)
        results.append(r)
        for name, c in r["checks"].items():
            print("  [%s] %s" % ("PASS" if c.get("ok") else "FAIL", name))
            for k, v in c.items():
                if k == "ok":
                    continue
                print("        %-34s %s" % (k, v))
        print("  OVERALL: %s" % ("PASS" if r["ok"] else "FAIL"))
        print()

    if a.json:
        a.json.write_text(json.dumps(results, indent=2))
        print("wrote %s" % a.json)
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
