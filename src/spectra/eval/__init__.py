"""Thin adapters onto existing eval harnesses.

We do NOT reimplement anyone's metrics. Both harnesses are cloned into ``third_party/``
(gitignored) and driven through their own entrypoints:

* :mod:`spectra.eval.pathorob_adapter` -- **primary** robustness index (the design spec 1).
* :mod:`spectra.eval.plism_adapter` -- PLISM retrieval, **diagnostic only**: we train
  on PLISM, so these numbers are never leaderboard-comparable (the design spec 1 + 6).

Retention (the design spec 3 risk 1 -- "forgetting is the default outcome, not a tail risk"):

* :mod:`spectra.eval.hest_adapter` -- **usable now.** Gene-expression regression,
  9 tasks, 42 GB ungated, no WSIs. Base reproduced exactly against their published
  phikon_v2 row, so a checkpoint delta on it is real.
* :mod:`spectra.eval.thunder_model` -- **usable now, nothing downloaded.** The corpus
  already existed on this cluster in another user's scratch, *including*
  BRACS and MHIST, whose official downloaders are behind registration walls and block on
  ``input()``. 15 of the paper's 16 are present; ``segpath_epithelial`` is not, so the
  ``benchmark_segmentation`` aggregate row cannot be formed -- per-dataset F1 still can.
  That tree belongs to someone else and is read-only to us, so
  ``THUNDER_BASE_DATA_FOLDER=$SPECTRA_THUNDER`` and its ``datasets/`` is a symlink
  farm into it; splits, embeddings and outputs land on our side.
  Drivers: ``scripts/run_thunder.sbatch`` (one dataset per job, full sweep),
  ``scripts/run_thunder_retention.sbatch`` (four-dataset kNN tripwire, cheap enough for
  every checkpoint), ``scripts/collect_thunder.py`` (per-dataset F1 vs the published
  phikon-v2 row; never a rank sum).
* Patho-Bench -- **not usable, and not worth making usable.** It is slide-level: the
  public precomputed features are UNI2-h patch embeddings, which are useless for scoring
  *our* patch encoder, so a number would cost ~7-8 TB of raw TCGA WSIs plus a full
  extraction pass. Reference's quoted 54.1 -> 55.8 also has no traceable source -- the
  Patho-Bench paper publishes no results table and there is no leaderboard -- so there
  is nothing to reproduce even after paying that. Use HEST and THUNDER instead.
"""
