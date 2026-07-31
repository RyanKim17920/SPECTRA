"""Thin adapters onto existing eval harnesses.

We do NOT reimplement anyone's metrics. Both harnesses are cloned into ``third_party/``
(gitignored) and driven through their own entrypoints:

* :mod:`waivphaet.eval.pathorob_adapter` -- **primary** robustness index (PLAN.md 1).
* :mod:`waivphaet.eval.plism_adapter` -- PLISM retrieval, **diagnostic only**: we train
  on PLISM, so these numbers are never leaderboard-comparable (PLAN.md 1 + 6).
"""
