# Held-out PLISM retrieval (top-1): base -> fine-tuned (c50)

**Generated file -- do not hand-edit.**  Regenerate with:

```
./.venv-hest/bin/python scripts/plism_retrieval.py
```

Top-1 retrieval of the SAME tile across a held-out scanner or stain switch
(GT450/S210 scanners, HRH/KR/MY stains -- never seen during training), CLS+
mean-patch (`clsmean`) embedding space, `n_tiles=256`, seed 1234. Base is the
untrained backbone's `probe_before.json` (adapter=None); tuned is the mean +/-
2 SD over the shipped c50 recipe's seeds at EACH seed's own 1-SE-selected
checkpoint (matching `docs/seed_stats.md`), not a fixed shared step.
See the script docstring for the base-value provenance per backbone.

| backbone | axis | base | tuned (mean +/- 2 SD) | gain | n |
|---|---|---|---|---|---|
| Phikon-v2 | cross-scanner | 0.8579 | 0.9946 +/- 0.0008 | +0.1367 | 3 |
| Phikon-v2 | cross-stain | 0.6958 | 0.8973 +/- 0.0027 | +0.2016 | 3 |
| Midnight-12k | cross-scanner | 0.7523 | 0.9914 +/- 0.0050 | +0.2391 | 3 |
| Midnight-12k | cross-stain | 0.5597 | 0.8828 +/- 0.0356 | +0.3231 | 3 |
| Virchow2 | cross-scanner | 0.8537 | 0.9914 +/- 0.0059 | +0.1377 | 3 |
| Virchow2 | cross-stain | 0.6961 | 0.9028 +/- 0.0250 | +0.2067 | 3 |
| H-optimus-0 | cross-scanner | 0.9781 | 0.9953 +/- 0.0006 | +0.0172 | 3 |
| H-optimus-0 | cross-stain | 0.8298 | 0.9152 +/- 0.0023 | +0.0853 | 3 |
| UNI2-h | cross-scanner | 0.9388 | 0.9966 +/- 0.0004 | +0.0578 | 3 |
| UNI2-h | cross-stain | 0.7609 | 0.9126 +/- 0.0308 | +0.1517 | 3 |
| Virchow | cross-scanner | 0.7582 | 0.9974 +/- 0.0004 | +0.2392 | 3 |
| Virchow | cross-stain | 0.5974 | 0.9310 +/- 0.0060 | +0.3335 | 3 |
| OpenMidnight | cross-scanner | 0.7034 | 0.9484 +/- 0.0155 | +0.2450 | 3 |
| OpenMidnight | cross-stain | 0.4860 | 0.7824 +/- 0.0293 | +0.2964 | 3 |

