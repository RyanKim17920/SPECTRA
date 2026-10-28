"""scripts/sweep/make_final.py: the all-data final configs are generated, not hand-written."""

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


@pytest.mark.parametrize("bb,picks,name", [("uni2", [125], "q8_final_uni2"), ("hoptimus", [100], "q8_final_hopt")])
def test_committed_finals_are_make_final_output(mf, bb, picks, name):
    out = V2 / f"{name}.yaml"
    assert mf.render(bb, picks, out) == out.read_text()


def test_ckpt_steps_cover_every_pick(mf):
    assert mf.ckpt_steps([125]) == list(range(25, 201, 25))            # cap 200
    assert mf.ckpt_steps([125, 100, 200]) == list(range(25, 226, 25))  # max pick + 25
    with pytest.raises(ValueError):
        mf.ckpt_steps([190])


def test_rendered_final_loads_and_is_all_data(mf, tmp_path):
    from spectra.sweep import config as SC

    text = mf.render("uni2", [175, 125], tmp_path / "f.yaml")
    cfg = mf.load_rendered(text)
    assert cfg.optim.stop_step == 200 and cfg.ckpt.schedule[-1] == 200 and cfg.ckpt.keep_last == 1
    assert cfg.batch.heldout_cores == [] and cfg.run.name == "q8-final-uni2"
    sel = SC.load_config(V2 / "q8_selection.yaml")
    assert cfg.objective == sel.objective and cfg.adv == sel.adv and cfg.adapt == sel.adapt


def test_main_writes_loadable_file(mf, tmp_path):
    from spectra.sweep import config as SC

    out = tmp_path / "q8_final_x.yaml"
    assert mf.main(["--backbone", "hoptimus", "--picks", "100", "--out", str(out),
                    "--selection", str(V2 / "q8_selection.yaml")]) == 0
    assert SC.load_config(out).backbone.name == "hoptimus"
