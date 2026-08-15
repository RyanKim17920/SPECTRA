"""Resume-from-prior-attempt: the properties that make it safe to leave on.

Context. On 2026-08-14 arms a/b/c were preempted 10, 8 and 7 times. Every restart began at
step 0 and re-derived an identical curve, so ~25 attempts burned recomputing known data and
none of the three reached step 1500. Commit 42990ba made requeue SAFE (a requeued attempt
writes to a fresh ``.r<N>`` dir instead of clobbering its predecessor); this covers the code
that makes it CHEAP without giving that safety back.

The load-bearing claim is that a resumed run sees the same data in the same order as a
continuous one. That holds because the batch PLAN sequence is a pure function of
(seed, epoch, position) -- so the tests below pin exactly that, plus the guard that refuses
to stitch one curve out of two different configs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from waivphaet.data.grid import GridBatchSampler
from waivphaet.data.pairs import PairBatchSampler
from waivphaet.train.contrastive import (
    RESUME_CONFIG_EXEMPT,
    TrainConfig,
    assert_resume_config_matches,
    find_prior_attempt_checkpoint,
    find_resumable_checkpoint,
)

CONDS = [f"c{i}" for i in range(8)]
TILES = np.arange(64)


def grid_sampler(**kw):
    return GridBatchSampler(
        conditions=CONDS, tile_indices=TILES, n_cond=4, n_tiles=6,
        batches_per_epoch=kw.pop("batches_per_epoch", 20), seed=kw.pop("seed", 7), **kw,
    )


def pair_sampler(**kw):
    return PairBatchSampler(
        conditions=CONDS, tile_indices=TILES, n_groups=3, group_size=4,
        batches_per_epoch=kw.pop("batches_per_epoch", 20), seed=kw.pop("seed", 7), **kw,
    )


def grid_plans(s):
    return [(b.cond_idx.tolist(), b.tile_idx.tolist()) for b in s]


def pair_plans(s):
    return [(b.tile_idx.tolist(), b.anchor_cond.tolist(), b.positive_cond.tolist()) for b in s]


# ---------------------------------------------------------------------------------------
# The core property: skipping N plans lands on exactly the plans a continuous run would see.
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("make,plans", [(grid_sampler, grid_plans), (pair_sampler, pair_plans)])
@pytest.mark.parametrize("skip", [0, 1, 7, 19])
def test_skip_resumes_the_identical_plan_stream(make, plans, skip):
    """A rewound sampler yields the continuous run's tail, batch for batch."""
    full = plans(make())
    s = make()
    s.set_start_index(skip)
    assert plans(s) == full[skip:], (
        "resumed data stream diverged from the continuous one -- a resumed curve would be "
        "trained on different batches than the curve it claims to continue"
    )


@pytest.mark.parametrize("make,plans", [(grid_sampler, grid_plans), (pair_sampler, pair_plans)])
def test_skip_is_self_clearing(make, plans):
    """The offset applies to the next epoch only; later epochs replay in full.

    That is what matches a continuous run, whose step s sits at position
    ``s % batches_per_epoch`` of pass ``s // batches_per_epoch`` -- only the first pass
    after the restart is partial.
    """
    full = plans(make())
    s = make()
    s.set_start_index(5)
    first = plans(s)
    second = plans(s)
    assert first == full[5:]
    assert second == full, "the epoch after a resume must replay the whole plan sequence"


@pytest.mark.parametrize("make", [grid_sampler, pair_sampler])
def test_len_reflects_the_shortened_first_epoch(make):
    s = make(batches_per_epoch=20)
    assert len(s) == 20
    s.set_start_index(6)
    assert len(s) == 14, "tqdm/DataLoader length must not claim batches that were skipped"


@pytest.mark.parametrize("make", [grid_sampler, pair_sampler])
def test_skip_rejects_out_of_range(make):
    s = make(batches_per_epoch=20)
    with pytest.raises(ValueError):
        s.set_start_index(-1)
    with pytest.raises(ValueError):
        s.set_start_index(21)


@pytest.mark.parametrize("make,plans", [(grid_sampler, grid_plans), (pair_sampler, pair_plans)])
def test_default_behaviour_is_untouched(make, plans):
    """Resume is opt-in: an untouched sampler is byte-identical to the pre-change one."""
    s = make()
    assert s.start_index == 0
    assert len(plans(s)) == 20


# ---------------------------------------------------------------------------------------
# The guard: never stitch one curve out of two different experiments.
# ---------------------------------------------------------------------------------------

def base_cfg(**kw):
    kw.setdefault("max_steps", 1500)
    kw.setdefault("out_dir", "/o")
    return TrainConfig(packed_dir="/p", **kw)


def test_config_guard_accepts_an_identical_config():
    cfg = base_cfg()
    prior = json.loads(json.dumps(_asdict(cfg)))
    assert_resume_config_matches(prior, cfg)


def test_config_guard_allows_out_dir_and_resume_from_to_differ():
    """These two MUST differ between attempts -- that is the whole point of .r<N>."""
    cfg = base_cfg(out_dir="/runs/arm-123.r3", resume_from="/runs/arm-123.r2/step_0001000")
    prior = _asdict(base_cfg(out_dir="/runs/arm-123.r2", resume_from=None))
    assert_resume_config_matches(prior, cfg)


@pytest.mark.parametrize("field,value", [
    ("lr", 3e-5), ("temperature", 0.2), ("seed", 1), ("grad_accum", 2), ("max_steps", 3000),
])
def test_config_guard_refuses_a_changed_experiment(field, value):
    cfg = base_cfg()
    prior = _asdict(cfg)
    prior[field] = value
    with pytest.raises(ValueError, match="refusing to resume"):
        assert_resume_config_matches(prior, cfg)


def test_max_steps_is_checked_despite_being_exempt():
    """max_steps reshapes the cosine LR for every remaining step, so it cannot drift."""
    assert "max_steps" in RESUME_CONFIG_EXEMPT
    cfg = base_cfg(max_steps=1500)
    prior = _asdict(base_cfg(max_steps=2000))
    with pytest.raises(ValueError, match="max_steps"):
        assert_resume_config_matches(prior, cfg)


def _asdict(cfg):
    from dataclasses import asdict
    return asdict(cfg)


# ---------------------------------------------------------------------------------------
# Attempt discovery.
# ---------------------------------------------------------------------------------------

def mkckpt(d: Path, step: int, *, complete: bool = True) -> Path:
    c = d / f"step_{step:07d}"
    c.mkdir(parents=True)
    (c / "optim.pt").write_bytes(b"")
    if complete:
        (c / "metrics.json").write_text("{}")
    return c


def test_torn_checkpoints_are_invisible(tmp_path):
    """metrics.json is written last, so its absence means the job was killed mid-flush."""
    mkckpt(tmp_path, 500)
    mkckpt(tmp_path, 1000, complete=False)
    assert find_resumable_checkpoint(tmp_path).name == "step_0000500"


def test_no_checkpoints_is_none_not_an_error(tmp_path):
    assert find_resumable_checkpoint(tmp_path) is None
    assert find_resumable_checkpoint(tmp_path / "nope") is None


def test_discovery_picks_the_highest_step_across_attempts(tmp_path):
    mkckpt(tmp_path / "gridcmp2-a-380721", 500)
    mkckpt(tmp_path / "gridcmp2-a-380721.r1", 1000)
    mkckpt(tmp_path / "gridcmp2-a-380721.r2", 750)
    found = find_prior_attempt_checkpoint(tmp_path / "gridcmp2-a-380721.r3")
    assert found.name == "step_0001000"
    assert found.parent.name == "gridcmp2-a-380721.r1"


def test_discovery_never_returns_the_current_attempt(tmp_path):
    """Resume must READ a prior attempt and WRITE a fresh one -- 42990ba's property."""
    cur = tmp_path / "gridcmp2-a-380721.r1"
    mkckpt(cur, 900)
    assert find_prior_attempt_checkpoint(cur) is None


def test_discovery_is_scoped_to_this_job(tmp_path):
    """The base name embeds $SLURM_JOB_ID, which survives requeue -- so a different arm
    or a different job must never be picked up."""
    mkckpt(tmp_path / "gridcmp2-b-380722", 1400)
    mkckpt(tmp_path / "gridcmp2-a-380999", 1300)
    mkckpt(tmp_path / "gridcmp2-a-380721", 400)
    found = find_prior_attempt_checkpoint(tmp_path / "gridcmp2-a-380721.r1")
    assert found.parent.name == "gridcmp2-a-380721"


def test_resumed_optimizer_trajectory_equals_a_continuous_one(tmp_path):
    """The claim resume actually makes: same weights at step 2N whether or not it restarted.

    Surrogate model, but the three pieces under test are the real ones -- AdamW state
    round-tripped through the same ``optim.pt`` payload ``save_checkpoint`` writes, the real
    ``cosine_lr`` schedule, and the step counter. The data half of the claim is covered by
    the sampler tests above.

    Dropping AdamW's moments is the classic silent resume bug: training visibly continues,
    the loss looks plausible, and the first few hundred updates are quietly unconditioned.
    This is what catches that.
    """
    import torch

    from waivphaet.train.contrastive import cosine_lr

    cfg = base_cfg(max_steps=40, warmup_steps=5, lr=1e-2)
    torch.manual_seed(0)
    xs = [torch.randn(4, 3) for _ in range(40)]

    def run(steps, model=None, opt=None, start=0):
        if model is None:
            torch.manual_seed(1)
            model = torch.nn.Linear(3, 2)
            opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        for step in range(start, steps):
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, cfg)
            opt.zero_grad()
            model(xs[step]).pow(2).mean().backward()
            opt.step()
        return model, opt

    continuous, _ = run(40)

    # preempted at step 20 -> checkpoint -> fresh process -> resume
    partial, opt = run(20)
    blob = tmp_path / "optim.pt"
    torch.save({"optimizer": opt.state_dict(), "step": 20}, blob)

    torch.manual_seed(1)
    revived = torch.nn.Linear(3, 2)
    revived.load_state_dict(partial.state_dict())
    revived_opt = torch.optim.AdamW(revived.parameters(), lr=cfg.lr)
    loaded = torch.load(blob, map_location="cpu", weights_only=False)
    revived_opt.load_state_dict(loaded["optimizer"])
    resumed, _ = run(40, revived, revived_opt, start=int(loaded["step"]))

    for a, b in zip(continuous.parameters(), resumed.parameters()):
        assert torch.allclose(a, b, atol=1e-6), "resumed trajectory diverged from continuous"


def test_cosine_lr_needs_no_scheduler_state():
    """Why there is no scheduler to checkpoint: the LR is a pure function of the step."""
    from waivphaet.train.contrastive import cosine_lr

    cfg = base_cfg(max_steps=1500, warmup_steps=200, lr=1e-4)
    assert [cosine_lr(s, cfg) for s in range(0, 1500, 97)] == [
        cosine_lr(s, cfg) for s in range(0, 1500, 97)
    ]
    assert cosine_lr(0, cfg) < cosine_lr(200, cfg)          # warmup ramps up
    assert cosine_lr(1499, cfg) < cosine_lr(200, cfg)       # then decays


def test_attempt_zero_finds_nothing(tmp_path):
    """The ordinary first-launch case: no prior attempt, so train from step 0, not fail."""
    (tmp_path / "gridcmp2-a-380721").mkdir(parents=True)
    assert find_prior_attempt_checkpoint(tmp_path / "gridcmp2-a-380721") is None


def test_prior_attempts_are_ordered_and_never_include_a_later_one(tmp_path):
    from waivphaet.train.contrastive import prior_attempt_dirs

    for n in ["", ".r1", ".r2", ".r3"]:
        (tmp_path / f"gridcmp2-a-380721{n}").mkdir(parents=True)
    (tmp_path / "gridcmp2-a-380721-notanattempt").mkdir()
    got = [d.name for d in prior_attempt_dirs(tmp_path / "gridcmp2-a-380721.r2")]
    assert got == ["gridcmp2-a-380721", "gridcmp2-a-380721.r1"], (
        "a later attempt must never be carried forward into an earlier one, and a dir that "
        "merely shares the prefix is not an attempt"
    )


# ---------------------------------------------------------------------------------------
# RI-curve carry-forward: the follower must not re-score steps a prior attempt already has.
# ---------------------------------------------------------------------------------------

def write_curve_json(d: Path, points):
    d.mkdir(parents=True, exist_ok=True)
    (d / "ri_curve.json").write_text(json.dumps({"points": points}))


def load_carry_forward():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import importlib
    return importlib.import_module("eval_checkpoints").carry_forward_prior_curve


def test_carry_forward_fills_the_gap_before_the_resume_point(tmp_path):
    write_curve_json(tmp_path / "run-1", [{"step": 500, "ri": 0.1}, {"step": 1000, "ri": 0.2}])
    cur = tmp_path / "run-1.r1"
    cur.mkdir()
    curve = load_carry_forward()(cur, [{"step": 1500, "ri": 0.3}])
    assert [p["step"] for p in curve] == [500, 1000, 1500]
    assert curve[0]["carried_forward_from"] == "run-1"
    assert "carried_forward_from" not in curve[-1], "our own points must not be mislabelled"


def test_carry_forward_never_overwrites_a_point_scored_here(tmp_path):
    write_curve_json(tmp_path / "run-1", [{"step": 500, "ri": 0.1}])
    cur = tmp_path / "run-1.r1"
    cur.mkdir()
    curve = load_carry_forward()(cur, [{"step": 500, "ri": 0.9}])
    assert len(curve) == 1
    assert curve[0]["ri"] == 0.9, "a point measured against THIS attempt's checkpoint wins"


def test_carry_forward_prefers_the_later_attempt_on_a_duplicate_step(tmp_path):
    write_curve_json(tmp_path / "run-1", [{"step": 500, "ri": 0.1}])
    write_curve_json(tmp_path / "run-1.r1", [{"step": 500, "ri": 0.5}])
    cur = tmp_path / "run-1.r2"
    cur.mkdir()
    curve = load_carry_forward()(cur, [])
    assert len(curve) == 1 and curve[0]["ri"] == 0.5, (
        "the closer ancestor's measurement should win"
    )
    assert curve[0]["carried_forward_from"] == "run-1.r1"


def test_carry_forward_survives_a_corrupt_prior_curve(tmp_path):
    (tmp_path / "run-1").mkdir(parents=True)
    (tmp_path / "run-1" / "ri_curve.json").write_text("{ truncated")
    write_curve_json(tmp_path / "run-1.r1", [{"step": 500, "ri": 0.5}])
    cur = tmp_path / "run-1.r2"
    cur.mkdir()
    curve = load_carry_forward()(cur, [])
    assert [p["step"] for p in curve] == [500]
