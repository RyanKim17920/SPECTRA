"""The v2 config schema, the shipped configs and the selection split (no PLISM data needed)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spectra.sweep import config as SC

REPO = Path(__file__).resolve().parents[1]
V2 = REPO / "configs" / "v2"


def test_v2_configs_load_and_encode_q8():
    sel = SC.load_config(V2 / "q8_selection.yaml")
    assert sel == SC.load_config(None)                                 # the defaults ARE the Q8 selection run
    assert sel.adapt.mode == "lora" and sel.adapt.lora_rank == 16 and sel.adapt.lora_alpha == 32
    assert sel.head.proj_dim == 512
    assert sel.objective.taus == [0.1, 0.6] and sel.objective.tau_weights == [0.5, 0.5]
    assert (sel.adv.targets, sel.adv.lam, sel.adv.queue, sel.adv.lr) == (["stain", "scanner"], 0.1, 8192, 1e-3)
    assert (sel.batch.n_cond, sel.batch.n_tiles) == (2, 256)
    assert (sel.optim.lr, sel.optim.warmup_steps, sel.optim.steps, sel.optim.stop_step) == (1e-4, 200, 400, 250)
    assert sel.batch.heldout_scanners == ["GT450"] and sel.batch.heldout_stains == ["MY"]
    assert sel.batch.heldout_cores == list(SC.SELECTION_HELDOUT_CORES)
    assert sel.eval.every == 25 and sel.eval.at_start and sel.ckpt.every == 25
    for bb, name in (("uni2", "q8_final_uni2"), ("hoptimus", "q8_final_hopt")):
        fin = SC.load_config(V2 / f"{name}.yaml")
        assert fin.backbone.name == bb
        assert fin.batch.heldout_scanners == [] and fin.batch.heldout_stains == [] and fin.batch.heldout_cores == []
        assert fin.optim.stop_step == 200 and fin.ckpt.schedule == list(range(25, 201, 25))
        assert not fin.eval.at_start
        # identical recipe: only the split, the stop and the bookkeeping differ
        a, b = SC.to_dict(fin), SC.to_dict(SC.load_config(V2 / "q8_selection.yaml", [f"backbone.name={bb}"]))
        for k in ("adapt", "head", "objective", "adv"):
            assert a[k] == b[k], k
        assert {k: v for k, v in a["optim"].items() if k != "stop_step"} == \
               {k: v for k, v in b["optim"].items() if k != "stop_step"}


def test_seed_override_changes_only_seed():
    f = V2 / "q8_final_uni2.yaml"
    s0, s2 = SC.load_config(f), SC.load_config(f, ["run.seed=2"])
    d0, d2 = SC.to_dict(s0), SC.to_dict(s2)
    assert d2["run"].pop("seed") == 2 and d0["run"].pop("seed") == 0 and d0 == d2
    assert SC.run_name(s0) != SC.run_name(s2)


def test_hash_name_and_overrides():
    a = SC.load_config(None, ["optim.lr=3e-5", "batch.n_tiles=512"])
    b = SC.load_config(None, ["batch.n_tiles=512", "optim.lr=0.00003"])
    assert SC.config_hash(a) == SC.config_hash(b)
    c = SC.load_config(None, ["optim.lr=3e-5", "batch.n_tiles=512", "eval.every=7", "batch.workers=2",
                              "run.packed_dir=/x", "ckpt.keep_last=3"])
    assert SC.config_hash(a) == SC.config_hash(c)                      # cadence / workers / paths not hashed
    assert SC.config_hash(a) != SC.config_hash(SC.load_config(None, ["optim.lr=1e-4", "batch.n_tiles=512"]))
    assert SC.config_hash(SC.load_config(None, ["backbone.name=uni2"])) == \
        SC.config_hash(SC.load_config(None, ["backbone.name=MahmoodLab/UNI2-h"]))
    for cfg in (a, SC.load_config(None, ["run.name=" + "x" * 80])):
        assert len(SC.run_name(cfg)) <= 40
    with pytest.raises(KeyError):
        SC.load_config(None, ["optim.lrr=1"])
    assert SC.load_config(None, ["adv.lambda=0.3"]).adv.lam == 0.3


@pytest.mark.parametrize("over", [
    "adapt.mode=full", "objective.taus=[]", "objective.taus=[0.1,-1]", "objective.tau_weights=[1.0]",
    "adv.queue=0", "adv.targets=[stain,stain]", "adv.targets=[tissue]", "batch.n_cond=1",
    "optim.stop_step=500", "eval.readouts=[gem]", "adapt.lora_targets=qkv",
])
def test_invalid_configs_rejected(over):
    with pytest.raises(ValueError):
        SC.load_config(None, [over])


@pytest.mark.parametrize("key", ["objective.temperature=0.07", "objective.proto={weight: 1.0}",
                                 "heads=[]", "aug={enabled: true}", "memory={mode: gradcache}"])
def test_stripped_features_are_unknown_keys(key):
    with pytest.raises(KeyError):
        SC.load_config(None, [key])


def test_smoke_transform():
    s = SC.load_config(V2 / "q8_selection.yaml", ["run.smoke=true"])
    assert s.optim.steps == 20 and s.optim.stop_step == 20 and s.batch.n_tiles == 64
    assert SC.run_name(s).startswith("smoke-")


def test_dump_roundtrip_and_tolerant_load(tmp_path):
    cfg = SC.load_config(V2 / "q8_final_hopt.yaml", ["run.seed=1"])
    p = tmp_path / "config.json"
    SC.dump(cfg, p)
    assert SC.load_dumped(p) == cfg
    d = json.loads(p.read_text())
    assert d["_meta"]["backbone_repo"] == "bioptimus/H-optimus-0"
    # a config dumped by a richer trainer schema: extra knobs are dropped when tolerant
    d["objective"]["proto"] = {"weight": 0.0}
    d["heads"] = [{"pool": "cls"}]
    d["adapt"].update(last_k=0, train_norm=False)
    p.write_text(json.dumps(d))
    with pytest.raises(KeyError):
        SC.load_dumped(p)
    got = SC.load_dumped(p, tolerant=True)
    assert got.adapt == cfg.adapt and got.backbone == cfg.backbone
    d["adapt"]["train_norm"] = True                                    # a model knob that is ON: still fatal
    p.write_text(json.dumps(d))
    with pytest.raises(KeyError):
        SC.load_dumped(p, tolerant=True)


def test_paths_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("SPECTRA_RUNS_DIR", str(tmp_path / "r"))
    monkeypatch.delenv("SPECTRA_CACHE_DIR", raising=False)
    assert SC.runs_root() == tmp_path / "r" and SC.cache_root() == tmp_path / "r" / "_cache"
    monkeypatch.setenv("SPECTRA_CACHE_DIR", str(tmp_path / "c"))
    assert SC.cache_root() == tmp_path / "c"
    cfg = SC.load_config(None)
    monkeypatch.delenv("SPECTRA_PLISM_DIR", raising=False)
    with pytest.raises(RuntimeError):
        SC.packed_dir(cfg)
    monkeypatch.setenv("SPECTRA_PLISM_DIR", str(tmp_path / "plism"))
    assert SC.packed_dir(cfg) == tmp_path / "plism"


def test_selection_split_and_heldout_cores():
    from spectra.data.conditions import all_conditions
    from spectra.sweep.data import (eval_locations, heldout_location_mask, is_heldout_condition,
                                    load_group_labels, pick_eval_cores, train_location_pool)

    cfg = SC.load_config(V2 / "q8_selection.yaml")
    train = [c for c in all_conditions() if not is_heldout_condition(cfg, c)]
    assert len(train) == 72 and all(c.scanner != "GT450" and c.stain != "MY" for c in train)
    lab = load_group_labels(cfg.eval.labels)
    assert lab.shape == (16278,) and lab.max() == 45 and len(np.unique(lab)) == 46
    cores = cfg.batch.heldout_cores
    held = heldout_location_mask(cfg, lab)
    pool = train_location_pool(held)
    assert held.sum() == 2492
    assert not np.isin(lab[pool], cores).any() and pool.size + held.sum() == lab.size
    assert pick_eval_cores(cfg) == (cores, False)
    locs = eval_locations(lab, cores, 64, 1234)
    assert locs.size == 7 * 64 and np.isin(lab[locs], cores).all()
    assert np.array_equal(locs, eval_locations(lab, cores, 64, 1234))
    fin = SC.load_config(V2 / "q8_final_uni2.yaml")
    assert train_location_pool(heldout_location_mask(fin, lab)) is None
    assert pick_eval_cores(fin) == (list(SC.SELECTION_HELDOUT_CORES), True)   # sanity eval, flagged
