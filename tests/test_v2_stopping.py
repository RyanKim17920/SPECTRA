"""The R1 stop rule (spectra.sweep.stopping)."""

from __future__ import annotations

import json

import pytest

from spectra.sweep import stopping as R


def test_r1_is_argmax_over_saved_steps_ties_earlier():
    P = {0: 0.9, 25: 0.60, 50: 0.65, 75: 0.70, 100: 0.74, 125: 0.73, 150: 0.74, 175: 0.72, 200: 0.71,
         225: 0.70, 250: 0.69}
    assert R.r1_pick(P) == 100                         # tie 100 / 150 -> earlier; step 0 is never a candidate
    assert R.r1_pick({25: 0.5}) == 25
    assert R.r1_pick({0: 0.5}) is None
    assert R.r1_pick({25: None, 50: 0.4}) == 50


def test_r1_reproduces_a_locked_pick():
    # Virchow2 selection run of the released model (P every 25 steps; locked pick 100)
    P = {0: 0.6301, 25: 0.6366, 50: 0.6777, 75: 0.7205, 100: 0.7404, 125: 0.7364, 150: 0.7368,
         175: 0.7354, 200: 0.7351, 225: 0.7384, 250: 0.7362}
    assert R.r1_pick(P) == 100


def test_candidates_and_complete():
    assert R.CANDIDATES == tuple(range(25, 251, 25))
    P = {s: 0.1 for s in range(0, 251, 25)}
    assert R.complete(P)
    assert R.complete({k: v for k, v in P.items() if k != 0})     # step 0 is not an input
    assert not R.complete({k: v for k, v in P.items() if k != 125})


def test_read_primary(tmp_path):
    recs = [{"kind": "info", "step": 0}, {"kind": "eval", "step": 0, "primary": 0.5},
            {"kind": "train", "step": 10, "loss": 1.0}, {"kind": "eval", "step": 25, "primary": 0.52}]
    (tmp_path / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs) + "not json\n")
    (tmp_path / "posthoc.jsonl").write_text(json.dumps({"kind": "eval", "step": 25, "primary": 0.53}) + "\n"
                                            + json.dumps({"kind": "eval", "step": 50, "primary": 0.55}) + "\n")
    assert R.read_primary(tmp_path) == {0: 0.5, 25: 0.53, 50: 0.55}  # posthoc wins
    (tmp_path / "metrics.jsonl").write_text(json.dumps({"kind": "eval", "step": 25, "primary": 0.5,
                                                        "cores_in_train": True}) + "\n")
    with pytest.raises(ValueError):
        R.read_primary(tmp_path)


def test_select_script(tmp_path, capsys):
    import importlib.util
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("select_v2", repo / "scripts" / "sweep" / "select.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    rd = tmp_path / "sel-uni2-abc"
    rd.mkdir()
    (rd / "config.json").write_text(json.dumps({"backbone": {"name": "uni2"}, "run": {"seed": 1}}))
    P = {s: 0.6 + 0.001 * s for s in range(0, 176, 25)} | {200: 0.70, 225: 0.71, 250: 0.70}
    (rd / "metrics.jsonl").write_text("".join(json.dumps({"kind": "eval", "step": s, "primary": v}) + "\n"
                                              for s, v in P.items()))
    row = m.pick_run(rd)
    assert row["complete"] and row["pick"] == 175
    assert m.main([str(rd), "--json", str(tmp_path / "p.json")]) == 0
    assert "--step 175" in capsys.readouterr().out
    (rd / "metrics.jsonl").write_text(json.dumps({"kind": "eval", "step": 25, "primary": 0.6}) + "\n")
    assert m.pick_run(rd)["pick"] is None
