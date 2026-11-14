"""The SPECTRA v2 contrastive fine-tuning recipe on PLISM.

One config (:class:`spectra.sweep.config.SweepConfig`) fully determines a run; it is hashed
into a short run name and stored next to the run.

Modules
-------
config   -- the schema, YAML/JSON/CLI loading, hashing, run naming
data     -- condition pools, the held-out split, the grid sampler, the eval tile set
model    -- backbone loading (reuses ``spectra.models.encoder``), LoRA, GeM pool, projectors
losses   -- grid InfoNCE (pure tensor function)
evaluate -- the in-run PLISM selection eval (held-out retrieval, CRoMa, primary) + GTEx guard
stopping -- the R1 stop rule (argmax of the in-run primary over the selection run's steps)
train    -- the trainer / entry point (``python -m spectra.sweep.train``)
build_guard_cache -- builds the GTEx utility-guard tile cache
"""
