"""Thin adapters onto existing eval harnesses.

We do NOT reimplement anyone's metrics. Both harnesses are cloned into ``third_party/``
(gitignored) and driven through their own entrypoints:

* :mod:`waivphaet.eval.pathorob_adapter` -- **primary** robustness index (PLAN.md 1).
* :mod:`waivphaet.eval.plism_adapter` -- PLISM retrieval, **diagnostic only**: we train
  on PLISM, so these numbers are never leaderboard-comparable (PLAN.md 1 + 6).

Retention (PLAN.md 3 risk 1 -- "forgetting is the default outcome, not a tail risk"):

* :mod:`waivphaet.eval.hest_adapter` -- **usable now.** Gene-expression regression,
  9 tasks, 42 GB ungated, no WSIs. Base reproduced exactly against their published
  phikon_v2 row, so a checkpoint delta on it is real.
* :mod:`waivphaet.eval.thunder_model` -- **harness ready, data not pulled.** 16 datasets
  / ~197 GB compressed from 8 different hosts; two of them (BRACS, MHIST) are behind
  registration walls whose downloader blocks on ``input()`` and will hang a SLURM job.
* Patho-Bench -- **not usable, and not worth making usable.** It is slide-level: the
  public precomputed features are UNI2-h patch embeddings, which are useless for scoring
  *our* patch encoder, so a number would cost ~7-8 TB of raw TCGA WSIs plus a full
  extraction pass. Waiv's quoted 54.1 -> 55.8 also has no traceable source -- the
  Patho-Bench paper publishes no results table and there is no leaderboard -- so there
  is nothing to reproduce even after paying that. Use HEST and THUNDER instead.
"""
