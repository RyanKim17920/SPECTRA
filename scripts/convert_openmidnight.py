"""Convert the OpenMidnight DINOv2 *training* checkpoint into a timm-loadable backbone.

``/data/OpenMidnight_ckpts/openmidnight_checkpoint.pth`` is what DINOv2 writes during
training, not a released encoder: a single ``{"teacher": ...}`` dict holding the backbone
**plus** the DINO and iBOT projection heads, with ``block_chunks=4`` module nesting
(``backbone.blocks.<chunk>.<global_idx>.*``) and DINOv2's own SwiGLU names
(``mlp.w12`` / ``mlp.w3``). ``timm.create_model`` can load none of that.

Rather than teach ``encoder.py`` a per-model key-remap branch -- exactly the per-model
dispatch that module refuses -- the remap happens ONCE, here, and the result is written in
the same shape as the other locally-served backbones (``config.json`` +
``pytorch_model.bin``), so ``BACKBONE_LOCAL_DIRS`` picks it up with no loader change.

The remap itself is timm's own ``checkpoint_filter_fn`` (``_convert_dinov2``): it folds
``pos_embed[:, 0]`` into ``cls_token`` and drops it from ``pos_embed`` (timm's
``vit_giant_patch14_reg4_dinov2`` is ``no_embed_class=True``), renames w12/w3 -> fc1/fc2
and moves ``register_tokens`` -> ``reg_token``. Only the de-chunking is ours, and it is
asserted rather than assumed. The load is then verified STRICT (0 missing / 0 unexpected)
before anything is written.

Normalisation is NOT stored by the training checkpoint. The ``pretrained_cfg`` written
below carries ImageNet mean/std because the SophontAI/OpenMidnight model card states them
explicitly ("# ImageNet normalization"); see the note in ``BACKBONE_NORMALIZATION``.

    python scripts/convert_openmidnight.py
"""

import json
import os
import re

import timm
import torch
from timm.models.vision_transformer import checkpoint_filter_fn

SRC = "/data/OpenMidnight_ckpts/openmidnight_checkpoint.pth"
DST = "/data/OpenMidnight"
ARCH = "vit_giant_patch14_reg4_dinov2"
KW = dict(img_size=224, init_values=1e-5, dynamic_img_size=False, num_classes=0, global_pool="")

ck = torch.load(SRC, map_location="cpu", weights_only=False)
assert set(ck) == {"teacher"}, sorted(ck)
teacher = ck["teacher"]

# 1. keep only the backbone, drop the DINO/iBOT heads (training-only projectors).
# 2. de-chunk: DINOv2 was trained with block_chunks=4, so blocks are nested
#    blocks.<chunk>.<idx>; timm is flat blocks.<n>.
flat = {}
per_chunk = {}
for k, v in teacher.items():
    if not k.startswith("backbone."):
        continue
    k = k[len("backbone."):]
    m = re.match(r"blocks\.(\d+)\.(\d+)\.(.*)$", k)
    if m:
        c, b, rest = int(m.group(1)), int(m.group(2)), m.group(3)
        per_chunk.setdefault(c, set()).add(b)
        # DINOv2's block_chunks pads each chunk with i*chunksize nn.Identity placeholders,
        # so the SECOND index is already the GLOBAL block index (chunk 1 holds 10..19).
        k = f"blocks.{b}.{rest}"
    flat[k] = v
sizes = {c: len(v) for c, v in sorted(per_chunk.items())}
assert set(sizes.values()) == {10} and sorted(sizes) == [0, 1, 2, 3], sizes
assert sorted(i for v in per_chunk.values() for i in v) == list(range(40))
print("de-chunked", sizes, "->", len(flat), "backbone tensors")

model = timm.create_model(ARCH, pretrained=False, **KW)
sd = checkpoint_filter_fn(flat, model)
missing, unexpected = model.load_state_dict(sd, strict=False)
print("missing", len(missing), missing[:5])
print("unexpected", len(unexpected), unexpected[:5])
assert not missing and not unexpected

out = f"{DST}/pytorch_model.bin"
if os.path.exists(out):
    raise SystemExit(
        f"{out} already exists. Refusing to overwrite a checkpoint other runs may be "
        "pinned to -- move it aside first if you really mean to rebuild it."
    )
torch.save(sd, out)
cfg = {
    "architecture": ARCH,
    "num_classes": 0,
    "num_features": 1536,
    "global_pool": "token",
    "pretrained_cfg": {
        "custom_load": True,
        "input_size": [3, 224, 224],
        "fixed_input_size": True,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "num_classes": 0,
        "license": "Apache 2.0",
    },
}
with open(f"{DST}/config.json", "w") as fh:
    json.dump(cfg, fh, indent=2)
print("wrote", DST, len(sd), "tensors")
