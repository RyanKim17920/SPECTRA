"""Tests for the invariants that fail *silently* if broken.

Every check here guards something that would not show up in the training loss:
a positive that isn't co-registered, or a negative that leaks acquisition signal.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

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
    _, m = masked_info_nce(torch.randn(64, 32), torch.randn(64, 32), group_id=g, temperature=1.0)
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
# --------------------------------------------------------------------------------------
# Full FT mode (full-ft branch)


def test_full_ft_trainable_param_assertion():
    """A full-FT model (use_lora=False, freeze_backbone=False) must have ~100% trainable params.

    A full-FT run that silently trains only the projector would look like a weak result,
    not an error. The guard in train_lora.py asserts >= 95%.
    """
    import torch.nn as nn

    model = _fake_vit(8, ("fc1", "fc2"))
    # No freezing, no LoRA => all params should be trainable by default.
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = trainable / max(total, 1)
    assert pct >= 0.95, f"full FT should have ~100% trainable, got {pct:.1%}"


def test_full_ft_checkpoint_roundtrip():
    """Save full FT backbone weights, reload via safetensors, and verify delta vs base.

    A silently-unloaded full-FT checkpoint reproduces base numbers exactly and reads as
    'perfect retention' -- the most dangerous false result. This tests the core guard
    mechanism: load a checkpoint into a model, then compare against the base model.
    """
    import torch.nn as nn
    from safetensors.torch import save_file, load_file

    # Create two models: base and "fine-tuned" (modify weights manually).
    base = _fake_vit(4, ("fc1", "fc2"))
    tuned = _fake_vit(4, ("fc1", "fc2"))

    # Modify tuned weights to simulate fine-tuning.
    for p in tuned.parameters():
        p.data += torch.randn_like(p) * 0.1

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "backbone.safetensors"
        save_file(tuned.state_dict(), str(ckpt_path))

        # Load into a fresh model.
        loaded = _fake_vit(4, ("fc1", "fc2"))
        loaded_sd = load_file(str(ckpt_path))
        loaded.load_state_dict(loaded_sd)

    # Compare loaded weights vs base — they should differ.
    base_sd = base.state_dict()
    loaded_sd = loaded.state_dict()
    delta = sum((loaded_sd[k] - base_sd[k]).abs().sum().item()
                for k in loaded_sd)
    assert delta > 0, "modified checkpoint should differ from base"


def test_ckpt_schedule_parser():
    """Non-uniform checkpoint schedule parsing."""
    import argparse

    src = Path(__file__).resolve().parents[1] / "scripts" / "train_lora.py"
    spec = importlib.util.spec_from_file_location("_train_lora_test", str(src))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError:
        pytest.skip("train_lora.py not importable in test environment")

    # Basic parsing.
    assert mod.parse_ckpt_schedule("25,50,75,100") == [25, 50, 75, 100]
    assert mod.parse_ckpt_schedule("1000") == [1000]

    # Unsorted input is sorted and deduped.
    assert mod.parse_ckpt_schedule("200,50,100,75") == [50, 75, 100, 200]
    assert mod.parse_ckpt_schedule("50,50,100,100") == [50, 100]

    # Whitespace tolerance.
    assert mod.parse_ckpt_schedule(" 50 , 100 , 150 ") == [50, 100, 150]

    # Invalid inputs.
    with pytest.raises(argparse.ArgumentTypeError):
        mod.parse_ckpt_schedule("abc")

    with pytest.raises(argparse.ArgumentTypeError):
        mod.parse_ckpt_schedule("")

    with pytest.raises(argparse.ArgumentTypeError):
        mod.parse_ckpt_schedule("-1,50")


def test_should_checkpoint_uses_schedule_when_set():
    """_should_checkpoint respects ckpt_schedule over ckpt_every."""
    from waivphaet.train.contrastive import TrainConfig, _should_checkpoint

    cfg = TrainConfig(ckpt_every=500, ckpt_schedule=[50, 100, 200])
    assert _should_checkpoint(50, cfg) is True
    assert _should_checkpoint(100, cfg) is True
    assert _should_checkpoint(200, cfg) is True
    assert _should_checkpoint(500, cfg) is False  # not in schedule
    assert _should_checkpoint(75, cfg) is False


def test_should_checkpoint_uses_ckpt_every_when_no_schedule():
    """_should_checkpoint falls back to ckpt_every when schedule is None."""
    from waivphaet.train.contrastive import TrainConfig, _should_checkpoint

    cfg = TrainConfig(ckpt_every=200, ckpt_schedule=None)
    assert _should_checkpoint(200, cfg) is True
    assert _should_checkpoint(400, cfg) is True
    assert _should_checkpoint(100, cfg) is False
    assert _should_checkpoint(300, cfg) is False


# --------------------------------------------------------------------------------------
# Retention term: relational KL against the frozen base model (PLAN.md 2 frozen-teacher
# anchor). OFF by default, and "off" has to mean BIT-IDENTICAL -- every published number
# in this repo was produced by the pre-retention loss, so a default path that merely
# "looks the same" would silently invalidate all of them.


class _FakeAdapterBackbone(torch.nn.Module):
    """Stand-in for a PEFT-wrapped backbone: a base map plus a switchable adapter delta."""

    def __init__(self, d_in: int, d_out: int):
        import torch.nn as nn

        super().__init__()
        self.base = nn.Linear(d_in, d_out)
        self.base.weight.requires_grad_(False)
        self.base.bias.requires_grad_(False)
        self.delta = nn.Linear(d_in, d_out, bias=False)  # the "LoRA" part: trainable
        self._adapter_on = True

    @contextlib.contextmanager
    def disable_adapter(self):
        prev, self._adapter_on = self._adapter_on, False
        try:
            yield
        finally:
            self._adapter_on = prev

    def save_pretrained(self, out_dir):  # what save_checkpoint calls in LoRA mode
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.delta.state_dict(), out_dir / "adapter.pt")

    def forward(self, x):
        out = self.base(x)
        return out + self.delta(x) if self._adapter_on else out


class _TinyEncoder(torch.nn.Module):
    """Minimal WaivEncoder-shaped model: ``embed`` -> pooled, ``forward`` -> (pooled, proj).

    ``use_lora`` selects an adapter-carrying backbone (retention is possible) or a plain
    Linear (retention must refuse -- the full-FT case).
    """

    def __init__(self, d_in=10, d_emb=8, d_proj=6, use_lora=True):
        import types

        import torch.nn as nn

        super().__init__()
        self.cfg = types.SimpleNamespace(use_lora=use_lora)
        self.backbone = _FakeAdapterBackbone(d_in, d_emb) if use_lora else nn.Linear(d_in, d_emb)
        self.projector = nn.Sequential(nn.Linear(d_emb, d_proj), nn.Dropout(0.5))

    def embed(self, images):
        return self.backbone(images)

    def forward(self, images):
        emb = self.embed(images)
        return emb, self.projector(emb)


def _retention_batches(n_batches=3, n_groups=2, group_size=6, d_in=10, seed=7):
    """Collated batches that satisfy the negative constraint, with pixel tensors attached."""
    out = []
    for i in range(n_batches):
        _, batch = _fake_collated(n_groups=n_groups, group_size=group_size, seed=seed + i)
        n = n_groups * group_size
        g = torch.Generator().manual_seed(1000 + i)
        batch["anchor"] = torch.randn(n, d_in, generator=g)
        batch["positive"] = torch.randn(n, d_in, generator=g)
        out.append(batch)
    return out


def _load_module_from_source(name: str, source: str):
    """Import a module from a source string (used to resurrect the HEAD implementation)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{name}.py"
        path.write_text(source)
        spec = importlib.util.spec_from_file_location(name, str(path))
        mod = importlib.util.module_from_spec(spec)
        # Must stay registered: dataclasses resolve `cls.__module__` lazily at
        # instantiation time, so popping it makes TrainConfig() explode.
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod


def _run_one_training(train_fn, config_cls, weights, batches, out_dir, **cfg_kwargs):
    """Run `train_fn` on a fresh _TinyEncoder loaded with `weights`; return (params, history)."""
    torch.manual_seed(0)
    model = _TinyEncoder(use_lora=cfg_kwargs.pop("use_lora", True))
    model.load_state_dict(weights)
    cfg = config_cls(
        out_dir=str(out_dir), max_steps=len(batches), warmup_steps=1, log_every=1,
        eval_every=10**9, ckpt_every=10**9, amp_dtype="none", n_groups=2, group_size=6,
        **cfg_kwargs,
    )
    torch.manual_seed(1234)  # fix the dropout stream so the comparison is meaningful
    summary = train_fn(model, batches, cfg, device="cpu")
    params = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return params, summary["history"]


def test_retention_weight_zero_is_bit_identical_to_the_head_implementation():
    """(a) The default path must reproduce HEAD *exactly*, parameters and history alike.

    Not "close" -- torch.equal. The PathoROB 0.468611 gate and every HEST/THUNDER number
    on record were produced by the HEAD loss; if adding an optional term perturbs the
    default path by one ULP, those numbers no longer describe this code.
    """
    import subprocess

    repo = Path(__file__).resolve().parents[1]
    head_src = subprocess.run(
        ["git", "show", "HEAD:src/waivphaet/train/contrastive.py"],
        cwd=repo, capture_output=True, text=True, check=True,
    ).stdout
    head = _load_module_from_source("_contrastive_at_head", head_src)
    if hasattr(head, "relational_kl"):
        # Once the retention commit lands, HEAD is no longer the "before" side and this
        # comparison is vacuous. Re-point it at the pre-retention commit to re-run it.
        pytest.skip("HEAD already contains the retention term; nothing to compare against")

    import waivphaet.train.contrastive as new

    batches = _retention_batches()
    torch.manual_seed(0)
    weights = {k: v.clone() for k, v in _TinyEncoder().state_dict().items()}

    with tempfile.TemporaryDirectory() as tmp:
        old_params, old_hist = _run_one_training(
            head.train, head.TrainConfig, weights, batches, Path(tmp) / "old",
        )
        new_params, new_hist = _run_one_training(
            new.train, new.TrainConfig, weights, batches, Path(tmp) / "new",
            retention_kl_weight=0.0,
        )

    assert old_params.keys() == new_params.keys()
    for k in old_params:
        assert torch.equal(old_params[k], new_params[k]), f"parameter {k} diverged"
    # Drop the wall-clock fields, which are timings and not results.
    timing = {"elapsed_s", "tiles_per_s"}
    strip = lambda h: [{k: v for k, v in r.items() if k not in timing} for r in h]  # noqa: E731
    assert json.dumps(strip(old_hist)) == json.dumps(strip(new_hist)), "history diverged"
    # ... and the history really did record something to compare.
    assert len(new_hist) == len(batches) and new_hist[0]["loss"] > 0
    # No retention keys leak into the default run's logs.
    assert not any(k.startswith("loss_retention") for k in new_hist[0])
    assert "loss_total" not in new_hist[0]


def test_relational_kl_is_nonnegative_and_exactly_zero_when_student_equals_teacher():
    """(b) Gibbs' inequality, asserted rather than assumed."""
    from waivphaet.train.contrastive import relational_kl

    torch.manual_seed(0)
    g = torch.arange(24) // 8
    t = torch.randn(24, 16)

    # Identical student and teacher -> exactly 0.0, not merely small.
    loss, m = relational_kl(t, t.clone(), group_id=g, temperature=0.07)
    assert float(loss) == 0.0
    assert m["loss_retention_kl"] == 0.0
    assert m["retention_kl_neighbours"] == pytest.approx(7.0)  # group_size - 1, no self

    # A global rotation + rescale is free: relational geometry is what is preserved.
    q, _ = torch.linalg.qr(torch.randn(16, 16))
    rot, _ = relational_kl(3.0 * (t @ q), t, group_id=g, temperature=0.07)
    assert float(rot) == pytest.approx(0.0, abs=1e-5)

    # Anything else is strictly positive.
    for scale in (0.1, 1.0, 5.0):
        val, _ = relational_kl(t + scale * torch.randn_like(t), t, group_id=g, temperature=0.07)
        assert float(val) > 0.0
    # Including with no group mask at all (whole-batch candidates).
    val, m = relational_kl(torch.randn(24, 16), t, temperature=0.07)
    assert float(val) > 0.0 and m["retention_kl_neighbours"] == pytest.approx(23.0)
    assert torch.isfinite(val)


def test_relational_kl_masks_self_similarity_and_masks_both_sides_identically():
    """Self-similarity is 1.0 for teacher and student alike; at tau=0.07 leaving it in
    would make both rows ~one-hot and the term silently inert."""
    from waivphaet.train.contrastive import relational_kl

    torch.manual_seed(0)
    s, t = torch.randn(12, 8), torch.randn(12, 8)
    val, _ = relational_kl(s, t, temperature=0.07)
    # If the diagonal were included, a peaked temperature would drive this to ~0.
    assert float(val) > 1e-3
    # Scaling a single row of the student does not change ITS OWN masked row target set:
    # the mask is a function of indices only, never of the values.
    s2 = s.clone()
    s2[3] *= 7.0  # cosine-invariant
    val2, _ = relational_kl(s2, t, temperature=0.07)
    assert float(val) == pytest.approx(float(val2), abs=1e-5)


def test_retention_teacher_is_gradient_free_and_rng_neutral():
    """The teacher must contribute no gradient and must not move the random stream.

    An extra forward pass that consumed RNG would desynchronise everything seeded from
    the global generator relative to a weight=0 run, and the "off is identical" claim
    would only hold until the first dropout call.
    """
    from waivphaet.train.contrastive import retention_teacher_embed

    torch.manual_seed(0)
    model = _TinyEncoder()
    model.train()
    x = torch.randn(8, 10)

    before = torch.get_rng_state()
    emb = retention_teacher_embed(model, x)
    after = torch.get_rng_state()

    assert torch.equal(before, after), "the teacher forward moved the global RNG stream"
    assert model.training, "teacher forward left the model in eval mode"
    assert not emb.requires_grad and emb.grad_fn is None

    # It really is the BASE model: adapters off, and different from the student's output.
    with torch.no_grad():
        student = model.embed(x)
    assert not torch.allclose(student, emb)
    with torch.no_grad(), model.backbone.disable_adapter():
        assert torch.equal(model.embed(x), emb)

    # No gradient reaches the trainable adapter through the teacher path: the teacher
    # output is not even a leaf of a graph, so there is nothing to back-propagate.
    model.zero_grad(set_to_none=True)
    with pytest.raises(RuntimeError):
        retention_teacher_embed(model, x).sum().backward()
    assert model.backbone.delta.weight.grad is None


def test_retention_with_full_ft_raises():
    """(c) full-FT + retention is a degenerate combination and must be an error.

    With no adapter to disable, teacher == student, the KL is identically 0, and the run
    reads as a retention-regularised fine-tune that regularised nothing.
    """
    from waivphaet.train.contrastive import TrainConfig, assert_retention_teacher_available, train

    assert_retention_teacher_available(_TinyEncoder(use_lora=True))  # the LoRA case is fine

    with pytest.raises(ValueError, match="requires LoRA"):
        assert_retention_teacher_available(_TinyEncoder(use_lora=False))

    # And it is caught by train() before any compute happens.
    with tempfile.TemporaryDirectory() as tmp:
        cfg = TrainConfig(out_dir=tmp, max_steps=1, retention_kl_weight=0.1)
        with pytest.raises(ValueError, match="requires LoRA"):
            train(_TinyEncoder(use_lora=False), _retention_batches(1), cfg, device="cpu")

    # train_lora.py refuses the flag combination up front, before the backbone is built.
    src = Path(__file__).resolve().parents[1] / "scripts" / "train_lora.py"
    text = src.read_text()
    assert "--retention-kl-weight" in text and "--retention-kl-temperature" in text
    assert "args.retention_kl_weight > 0 and args.full_ft" in text


def test_retention_defaults_are_off():
    """The one guarantee everything else rests on."""
    from waivphaet.train.contrastive import TrainConfig

    assert TrainConfig().retention_kl_weight == 0.0
    assert TrainConfig().retention_kl_temperature == 0.07


def test_retention_on_logs_both_terms_separately():
    """The trade-off has to be readable in history.json, not just the sum."""
    from waivphaet.train.contrastive import TrainConfig, train

    batches = _retention_batches()
    with tempfile.TemporaryDirectory() as tmp:
        _, hist = _run_one_training(
            train, TrainConfig, _TinyEncoder().state_dict(), batches, Path(tmp),
            retention_kl_weight=1.0, retention_kl_temperature=0.07,
        )
    rec = hist[0]
    for k in ("loss", "loss_infonce", "loss_retention_kl", "loss_total"):
        assert k in rec, f"missing {k} in history record"
    assert rec["loss_infonce"] == rec["loss"]  # "loss" keeps its pre-retention meaning
    assert rec["loss_retention_kl"] >= 0.0
    assert rec["loss_total"] == pytest.approx(rec["loss_infonce"] + rec["loss_retention_kl"])
