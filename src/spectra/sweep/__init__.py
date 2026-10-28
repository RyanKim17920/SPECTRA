"""The v2 (Q8) contrastive fine-tuning recipe on PLISM.

One config (:class:`spectra.sweep.config.SweepConfig`) fully determines a run; it is hashed
into a short run name and stored next to the run.

Modules
-------
config   -- the schema, YAML/JSON/CLI loading, hashing, run naming
data     -- condition pools, the held-out split, the grid sampler, the eval tile set
model    -- backbone loading (reuses ``spectra.models.encoder``), LoRA, projector, adversary heads
losses   -- multi-temperature grid InfoNCE, gradient reversal (pure tensor functions)
evaluate -- the in-run PLISM selection eval (held-out retrieval, CRoMa, primary) + GTEx guard
stopping -- the slope rule that picks the training step from the in-run primary
train    -- the trainer / entry point (``python -m spectra.sweep.train``)
build_guard_cache -- builds the GTEx utility-guard tile cache
"""
