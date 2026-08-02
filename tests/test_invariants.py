"""Tests for the invariants that fail *silently* if broken.

Every check here guards something that would not show up in the training loss:
a positive that isn't co-registered, or a negative that leaks acquisition signal.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from waivphaet.data.conditions import (
    SCANNERS, STAINS, all_conditions, make_split, parse_filename,
)
from waivphaet.data.pairs import (
    PairBatch, PairBatchSampler, assert_same_condition_negatives, collate_pair_batch,
)
from waivphaet.train.contrastive import masked_info_nce


def test_condition_grid_is_13x7():
    assert len(STAINS) == 13 and len(SCANNERS) == 7
    assert len(all_conditions()) == 91
    assert len({c.filename for c in all_conditions()}) == 91


def test_filename_roundtrip():
    for c in all_conditions():
        assert parse_filename(c.filename) == c


def test_split_is_deterministic_and_disjoint():
    a, b = make_split(), make_split()
    assert [c.key for c in a.train] == [c.key for c in b.train]
    assert set(a.train).isdisjoint(a.heldout)
    assert len(a.train) + len(a.heldout) == 91


def test_split_rejects_unknown_names():
    with pytest.raises(ValueError):
        make_split(["NOPE"], [])


def test_positive_condition_never_equals_anchor():
    """PLAN.md 2: a positive must be a *different* acquisition condition."""
    s = PairBatchSampler(all_conditions(), n_groups=32, group_size=16, batches_per_epoch=25)
    for b in s:
        assert not (b.positive_cond == b.anchor_cond[:, None]).any()


def test_group_is_condition_homogeneous_with_unique_tiles():
    """The structural guarantee the loss mask relies on."""
    s = PairBatchSampler(all_conditions(), n_groups=8, group_size=16, batches_per_epoch=10)
    for b in s:
        assert b.anchor_cond.shape == (8,)  # one condition per group, by construction
        for g in range(b.n_groups):
            assert len(np.unique(b.tile_idx[g])) == b.group_size


def test_positive_conditions_cover_the_grid():
    """The offset-shift trick must stay uniform over the other 90 conditions."""
    s = PairBatchSampler(all_conditions(), n_groups=64, group_size=16, batches_per_epoch=40)
    hist = np.zeros(91, dtype=int)
    for b in s:
        hist += np.bincount(b.positive_cond.ravel(), minlength=91)
    assert hist.min() > 0
    assert hist.max() < 2.0 * hist.mean()


def test_masked_infonce_negative_count_is_group_bounded():
    """Negatives must be the group (same condition), not the whole batch."""
    g = torch.arange(32) // 8
    z = torch.randn(32, 16)
    _, m = masked_info_nce(z, z.clone(), g)
    assert m["negatives_per_anchor"] == pytest.approx(7.0)  # group_size - 1, not 31


def test_masked_infonce_random_loss_is_log_group_size():
    g = torch.arange(64) // 8
    torch.manual_seed(0)
    _, m = masked_info_nce(torch.randn(64, 32), torch.randn(64, 32), g, temperature=1.0)
    assert m["loss"] == pytest.approx(np.log(8), abs=0.35)


def test_masked_infonce_ignores_cross_group_similarity():
    """A cross-group near-duplicate must not affect the loss -- it is not a valid negative."""
    g = torch.tensor([0, 0, 1, 1])
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    loss_a, _ = masked_info_nce(a, a.clone(), g, temperature=0.1)
    b = a.clone()
    b[2] = torch.tensor([1.0, 0.0])  # duplicate of row 0, but a different group
    loss_b, _ = masked_info_nce(b, b.clone(), g, temperature=0.1)
    assert torch.allclose(loss_a, loss_b, atol=1e-5)


# --- the negative constraint, asserted rather than eyeballed ---------------------------


def _fake_collated(n_groups=3, group_size=8, n_cond=91, seed=0):
    """A collated batch dict with the pixel tensors omitted (the checks never read them)."""
    s = PairBatchSampler(
        all_conditions()[:n_cond], n_groups=n_groups, group_size=group_size,
        batches_per_epoch=1, seed=seed,
    )
    b = next(iter(s))
    anchor_cond = np.broadcast_to(b.anchor_cond[:, None], b.tile_idx.shape)
    item = {
        "tile_idx": torch.from_numpy(b.tile_idx),
        "anchor_cond": torch.from_numpy(np.ascontiguousarray(anchor_cond)),
        "positive_cond": torch.from_numpy(b.positive_cond),
        "group_id": torch.arange(n_groups).repeat_interleave(group_size),
    }
    return b, collate_pair_batch(item)


def test_collated_batch_satisfies_the_negative_constraint():
    """Every group the loss will mask over must be condition-homogeneous."""
    _, batch = _fake_collated()
    stats = assert_same_condition_negatives(batch, allowed_conditions=set(range(91)))
    assert stats["negatives_per_anchor"] == 7.0
    assert stats["n_groups"] == 3.0


def test_assertion_catches_a_condition_mixed_group():
    """The check must FAIL on the bug it exists to catch, or it is decoration."""
    _, batch = _fake_collated()
    batch["anchor_cond"] = batch["anchor_cond"].clone()
    batch["anchor_cond"][0] = (batch["anchor_cond"][0] + 1) % 91  # one intruder
    with pytest.raises(AssertionError, match="mixes anchor conditions"):
        assert_same_condition_negatives(batch)


def test_assertion_catches_a_heldout_condition_leak():
    _, batch = _fake_collated()
    with pytest.raises(AssertionError, match="held-out condition has leaked"):
        assert_same_condition_negatives(batch, allowed_conditions={0})


def test_assertion_catches_a_repeated_tile_in_a_group():
    _, batch = _fake_collated()
    batch["tile_idx"] = batch["tile_idx"].clone()
    batch["tile_idx"][1] = batch["tile_idx"][0]  # anchor 1 is now anchor 0's own tile
    with pytest.raises(AssertionError, match="repeats a tile index"):
        assert_same_condition_negatives(batch)


def test_pairbatch_validate_rejects_a_same_condition_positive():
    b = PairBatch(
        tile_idx=np.array([[0, 1]]),
        anchor_cond=np.array([3]),
        positive_cond=np.array([[3, 5]]),  # first "positive" is the anchor's own condition
    )
    with pytest.raises(AssertionError, match="cross-acquisition"):
        b.validate(91)


def test_infonce_queries_positives_against_the_condition_homogeneous_anchors():
    """Orientation is load-bearing, not cosmetic.

    The candidate row must be the ANCHORS (one shared condition, so acquisition carries
    no signal), with the positives as queries. Running it the other way makes the
    candidate row span conditions and reintroduces the acquisition shortcut.
    """
    import torch.nn.functional as F

    torch.manual_seed(0)
    g = torch.arange(16) // 8
    a, p = torch.randn(16, 12), torch.randn(16, 12)
    an, pn = F.normalize(a, dim=-1), F.normalize(p, dim=-1)
    mask = g[:, None] == g[None, :]
    tgt = torch.arange(16)

    want = F.cross_entropy(((pn @ an.t()) / 0.07).masked_fill(~mask, float("-inf")), tgt)
    wrong = F.cross_entropy(((an @ pn.t()) / 0.07).masked_fill(~mask, float("-inf")), tgt)

    got, _ = masked_info_nce(a, p, g, temperature=0.07)
    assert torch.allclose(got, want, atol=1e-5)
    assert not torch.allclose(want, wrong, atol=1e-3)  # the two really are different


def test_symmetric_is_off_by_default():
    """PLAN.md 2: the anchor->positive direction has cross-condition candidates."""
    from waivphaet.train.contrastive import TrainConfig
    assert TrainConfig().symmetric is False


# --------------------------------------------------------------------------------------
# Backbone-agnostic encoder (PLAN.md §2: LoRA across the FULL depth, on any backbone).
#
# These are the guards for the failure that would otherwise be invisible: a LoRA target
# set that resolves to *fewer* modules on a new architecture. kaiko-ai/midnight sets
# use_swiglu_ffn=True, so its FFN linears are mlp.weights_in / mlp.weights_out rather
# than mlp.fc1 / mlp.fc2. The old fixed name list still matched query/key/value/dense in
# every block, so the block-coverage assertion PASSED while two thirds of each block's
# parameters stayed frozen -- and downstream that reads as "LoRA is weaker on ViT-g",
# not as a bug.


def test_block_index_parses_the_naming_schemes_we_target():
    from waivphaet.models.encoder import _block_index

    assert _block_index("encoder.layer.17.attention.attention.query") == 17
    assert _block_index("encoder.layers.3.mlp.fc1") == 3
    assert _block_index("blocks.11.attn.qkv") == 11
    # A Linear outside any numbered block must never be adapted: LoRA-on-the-head is
    # exactly what PLAN.md §2 rules out.
    assert _block_index("pooler.dense") is None
    assert _block_index("head.fc1") is None


def _fake_vit(n_blocks: int, ffn_names: tuple[str, str]):
    """Minimal module tree with HF-Dinov2 module *paths* and a swappable FFN naming."""
    import torch.nn as nn

    def block():
        b = nn.Module()
        b.attention = nn.Module()
        b.attention.attention = nn.Module()
        for n in ("query", "key", "value"):
            setattr(b.attention.attention, n, nn.Linear(8, 8))
        b.attention.output = nn.Module()
        b.attention.output.dense = nn.Linear(8, 8)
        b.mlp = nn.Module()
        for n in ffn_names:
            setattr(b.mlp, n, nn.Linear(8, 8))
        return b

    m = nn.Module()
    m.encoder = nn.Module()
    m.encoder.layer = nn.ModuleList([block() for _ in range(n_blocks)])
    m.pooler = nn.Module()
    m.pooler.dense = nn.Linear(8, 8)  # decoy: right leaf name, outside every block
    return m


@pytest.mark.parametrize(
    "ffn", [("fc1", "fc2"), ("weights_in", "weights_out")],
    ids=["dinov2-mlp", "swiglu-ffn"],
)
def test_lora_discovery_covers_every_block_under_both_ffn_namings(ffn):
    from waivphaet.models.encoder import EncoderConfig, _lora_target_names

    model = _fake_vit(24, ffn)
    names, per_block, leaves = _lora_target_names(model, EncoderConfig())

    assert len(names) == 144, "6 linears x 24 blocks"
    assert set(per_block) == set(range(24))
    assert set(per_block.values()) == {6}, per_block
    assert set(leaves) == {"query", "key", "value", "dense", *ffn}
    assert not any(n.startswith("pooler.") for n in names), "adapted a head-level Linear"


def test_lora_discovery_is_unchanged_on_the_phikon_v2_naming():
    """The regression half: discovery must reproduce the old FIXED list exactly, or the
    PathoROB 0.468611 gate is being re-run against a different model."""
    from waivphaet.models.encoder import (
        LORA_TARGET_MODULES,
        EncoderConfig,
        _lora_target_names,
    )

    model = _fake_vit(24, ("fc1", "fc2"))
    discovered, _, _ = _lora_target_names(model, EncoderConfig())
    fixed, _, _ = _lora_target_names(
        model, EncoderConfig(lora_target_modules=LORA_TARGET_MODULES)
    )
    assert discovered == fixed


def test_thunder_pooling_is_resolved_per_backbone_not_hardcoded():
    """arXiv:2607.22861 §3 line 106: in THUNDER, CLS+mean-pool concatenation is used only
    for Virchow2 / AquaViT / H0-mini / Midnight-12k. phikon-v2 is CLS there. Hardcoding
    either one makes the base-vs-fine-tuned rank sums non-comparable to their table."""
    import importlib.util
    import sys
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "waivphaet" / "eval" / "thunder_model.py"
    spec = importlib.util.spec_from_file_location("_waiv_thunder_model_test", src)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError:  # thunder lives in .venv-thunder, not the default venv
        pytest.skip("thunder not importable in this interpreter")
    sys.modules.pop("_waiv_thunder_model_test", None)

    assert mod._default_pooling(None) == "cls"                 # default = phikon-v2
    assert mod._default_pooling("owkin/phikon-v2") == "cls"
    assert mod._default_pooling("kaiko-ai/midnight") == "clsmean"


def test_thunder_auto_pooling_never_resolves_to_clsmean_for_segmentation():
    """clsmean advertises emb_dim = 2*hidden, but get_segmentation_embeddings returns raw
    hidden-d patch tokens; THUNDER sizes its seg decoder from emb_dim, so on Midnight
    (3072 vs 1536) the job dies at task_specific_models.py:121. The correction must be
    narrow: explicit pooling and every classification run are untouched."""
    import importlib.util
    import sys
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "waivphaet" / "eval" / "thunder_model.py"
    spec = importlib.util.spec_from_file_location("_waiv_thunder_seg_test", src)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError:  # thunder lives in .venv-thunder, not the default venv
        pytest.skip("thunder not importable in this interpreter")
    sys.modules.pop("_waiv_thunder_seg_test", None)

    mid, phi = "kaiko-ai/midnight", "owkin/phikon-v2"

    # The bug: auto + segmentation + a backbone whose cls dim != patch dim.
    assert mod.resolve_pooling(mid, None, True) == "cls"
    # ... and nothing else moves.
    assert mod.resolve_pooling(mid, None, False) == "clsmean"   # the 24 held cls jobs
    assert mod.resolve_pooling(phi, None, True) == "cls"
    assert mod.resolve_pooling(phi, None, False) == "cls"
    # Explicit pooling always wins, segmentation or not -- the 4 running jobs pass cls.
    for seg in (True, False):
        for backbone in (mid, phi, None):
            for explicit in ("cls", "clsmean", "mean"):
                assert mod.resolve_pooling(backbone, explicit, seg) == explicit

    # Task detection is an EXACT argv token match, so no dataset / task / loading mode /
    # model path used by the classification sweep can trip it.
    assert mod._is_segmentation_run(["benchmark", "custom:/x/thunder_model.py",
                                     "ocelot", "segmentation",
                                     "--loading-mode", "online_loading"])
    for task in ("knn", "linear_probing", "simple_shot", "pre_computing_embeddings"):
        for ds in ("bach", "bracs", "break_his", "ccrcc", "crc", "esca", "mhist",
                   "patch_camelyon", "tcga_crc_msi", "tcga_tils", "tcga_uniform", "wilds"):
            assert not mod._is_segmentation_run(
                ["benchmark", "custom:/admin/home/ryan.kim/waiv/src/waivphaet/eval/"
                 "thunder_model.py", ds, task,
                 "--loading-mode", "embedding_pre_loading"]
            )
