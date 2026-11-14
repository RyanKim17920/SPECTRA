"""scripts/sweep/make_final.py: the all-data release configs are generated, not hand-written."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
V2 = REPO / "configs" / "v2"


@pytest.fixture(scope="module")
def mf():
    sys.path.insert(0, str(REPO / "scripts" / "sweep"))
    spec = importlib.util.spec_from_file_location("make_final", REPO / "scripts" / "sweep" / "make_final.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_release_steps(mf):
    assert mf.RELEASE_STEPS == {"virchow2": 100, "uni2": 175, "hoptimus": 100, "virchow": 225, "phikon": 225,
                                "midnight": 225, "om11": 225}
    assert mf.release_step("MahmoodLab/UNI2-h") == 175 and mf.release_step("om11") == 225
    with pytest.raises(KeyError):
        mf.release_step("some/other-model")
    with pytest.raises(ValueError):
        mf.final_overrides("uni2", 190)


def test_rendered_final_is_the_recipe_on_all_data(mf, tmp_path):
    from spectra.sweep import config as SC

    cfg = mf.load_rendered(mf.render("uni2", 175, tmp_path / "f.yaml"))
    assert cfg.optim.stop_step == 175 and cfg.ckpt.every == 25
    assert cfg.batch.heldout_cores == [] and cfg.batch.heldout_scanners == [] and cfg.batch.heldout_stains == []
    assert cfg.run.name == "v2-final-uni2" and not cfg.eval.at_start
    sel = SC.load_config(V2 / "spectra_v2.yaml")
    assert cfg.objective == sel.objective and cfg.heads == sel.heads and cfg.adapt == sel.adapt
    assert {k: v for k, v in SC.to_dict(cfg)["optim"].items() if k != "stop_step"} == \
           {k: v for k, v in SC.to_dict(sel)["optim"].items() if k != "stop_step"}   # same lr schedule


def test_main_writes_loadable_file(mf, tmp_path):
    from spectra.sweep import config as SC

    out = tmp_path / "hopt.yaml"
    assert mf.main(["--backbone", "hoptimus", "--out", str(out), "--selection", str(V2 / "spectra_v2.yaml")]) == 0
    cfg = SC.load_config(out)
    assert cfg.backbone.name == "hoptimus" and cfg.optim.stop_step == 100
