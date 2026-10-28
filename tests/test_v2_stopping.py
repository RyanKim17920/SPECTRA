"""The slope-rule stopping (spectra.sweep.stopping)."""

from __future__ import annotations

import json

import pytest

from spectra.sweep import stopping as R


def test_slope_rule_semantics():
    H = [75, 100, 125, 150]
    P = {0: 0.0, 25: 0.02, 50: 0.06, 75: 0.10, 100: 0.13, 125: 0.15, 150: 0.155, 175: 0.16}
    pk, v = R.slope_pick(P, H, 0.016)
    assert v[75] == pytest.approx(0.04) and v[100] == pytest.approx(0.035) and v[125] == pytest.approx(0.025)
    assert v[150] == pytest.approx(0.0125) and pk == 125              # first failure at 150 -> keep previous
    flat = {s: 0.5 for s in range(0, 251, 25)}
    assert R.slope_pick(flat, H, 0.016)[0] == 75                     # never armed -> first candidate
    assert R.slope_pick({200: 0.5}, H, 0.016)[0] is None             # undefined everywhere
    rising = {s: s / 1000 for s in range(0, 251, 25)}                # 0.025 per 25 steps throughout
    assert R.slope_pick(rising)[0] == 250                            # armed to the end -> last candidate


def test_slope_rule_late_arming_and_dip():
    # a slow start then a burst: arms at 150, stops when the gain falls below theta at 200
    P = {0: 0.60, 25: 0.60, 50: 0.60, 75: 0.60, 100: 0.605, 125: 0.62, 150: 0.645, 175: 0.665, 200: 0.67,
         225: 0.672, 250: 0.673}
    pk, v = R.slope_pick(P)
    assert v[150] == pytest.approx(0.02) and v[175] == pytest.approx(0.0225) and v[200] < R.THETA
    assert pk == 175


def test_interpolation_and_gaps():
    assert R.interp({0: 0.0, 50: 1.0}, 25) == pytest.approx(0.5)
    assert R.interp({0: 0.0, 100: 1.0}, 50) is None                  # gap > 50 -> undefined
    assert R.slope({50: 0.1, 150: 0.2}, 100) is None
    s = {0: 0.0, 50: 0.06, 100: 0.13, 150: 0.155}                     # 50-step evals still work
    assert R.slope(s, 75) == pytest.approx(((0.06 + 0.13) / 2 - (0.0 + 0.06) / 2) / 2)


def test_defaults_are_the_q8_rule():
    assert R.THETA == 0.016
    assert R.CANDIDATES == (75, 100, 125, 150, 175, 200, 225, 250)


def test_complete():
    P = {s: 0.1 for s in range(0, 251, 25)}
    assert R.complete(P)
    assert not R.complete({k: v for k, v in P.items() if k != 0})
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
