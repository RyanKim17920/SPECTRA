"""CONCH (MahmoodLab/conch) backbone invariants.

The load-bearing one: the vector SpectraEncoder exports for CONCH must be EXACTLY
``coca.encode_image(x, proj_contrast=False, normalize=False)`` -- what SlideChat and the
SAE read -- and LoRA must cover all 12 trunk blocks and nothing outside the trunk.
Builds the real model on CPU, so it skips when the weights are not on this machine.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from spectra.models import backbones
from spectra.models.conch_visual import CONCH_MEAN, CONCH_STD, DEFAULT_CONCH_WEIGHTS, load_conch_full
from spectra.models.encoder import EncoderConfig, SpectraEncoder, normalization_for

W = os.environ.get("SPECTRA_CONCH_WEIGHTS", DEFAULT_CONCH_WEIGHTS)
needs_weights = pytest.mark.skipif(not Path(W).exists(), reason="CONCH weights not on this machine")
pytest.importorskip("conch")


def test_registry_entry():
    b = backbones.get("MahmoodLab/conch")
    assert b is not None and b.loader == "conch"
    assert b.embed_dim == 512 and b.num_blocks == 12 and b.patch_size == 16
    assert normalization_for("MahmoodLab/conch") == (CONCH_MEAN, CONCH_STD)


def test_non_cls_pooling_refused():
    with pytest.raises(ValueError, match="pooling='cls'"):
        SpectraEncoder(EncoderConfig(backbone="MahmoodLab/conch", pooling="clsmean", use_lora=False))


@needs_weights
def test_export_equals_encode_image_and_lora_targets():
    torch.manual_seed(0)
    enc = SpectraEncoder(EncoderConfig(backbone="MahmoodLab/conch", pooling="cls", use_lora=True,
                                       lora_rank=4, lora_alpha=8, proj_out_dim=512))
    s = enc.trainable_parameter_summary()
    assert s["embed_dim"] == 512 and s["blocks"] == 12 and s["loader"] == "conch"
    assert s["lora_targets"] == 48 and s["lora_targets_per_block"] == 4
    assert sorted(s["lora_target_leaves"]) == ["fc1", "fc2", "proj", "qkv"]
    assert all(n.startswith("trunk.blocks.") for n in enc.lora_target_names)
    # Only LoRA + projector train; pooler / ln / proj_contrast / trunk base weights frozen.
    trainable = [n for n, p in enc.named_parameters() if p.requires_grad]
    assert all(("lora_" in n) or n.startswith("projector.") for n in trainable), trainable[:5]

    full = load_conch_full(W)
    x = torch.randint(0, 256, (2, 448, 448, 3), dtype=torch.uint8)
    xn = (x.permute(0, 3, 1, 2).float() / 255 - torch.tensor(CONCH_MEAN).view(1, 3, 1, 1)) / torch.tensor(CONCH_STD).view(1, 3, 1, 1)
    with torch.no_grad():
        ref = full.encode_image(xn, proj_contrast=False, normalize=False)
        got = enc.encode(x)  # B (LoRA B is zero-initialised) => identical to base
        ref_p = full.encode_image(xn, proj_contrast=True, normalize=True)
        got_p = F.normalize(enc.backbone.base_model.model.project(got), dim=-1)
    assert got.shape == (2, 512)
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(got_p, ref_p, rtol=1e-4, atol=1e-4)
