# THUNDER 16-set panel -- seed-noise floor

**Generated file -- do not hand-edit.**  Regenerate with:

```
./.venv/bin/python scripts/thunder16_floor.py
```

One paired difference between two arms identical except for the training seed, per
(backbone, task), over the 16-dataset panel.  Read it as the scale below which a
delta in `docs/thunder16.md` is not interpretable -- **not** as a standard error:
n=2 supports a magnitude, not an interval.

Measured on the c3s generation, the only arms with a seed replicate on disk today.
The c50 replicates are queued; when they land, extend `SEED_PAIRS` and re-run.

| backbone | pair | knn | linear_probing | simple_shot | segmentation | calibration | adversarial_attack |
|---|---|---|---|---|---|---|---|
| phikon2 | c3s-s0-step250 vs c3s-s1-step250 | 1.10 | 0.60 | 0.40 | 1.10 | 0.10 | 0.60 |
| midnight | c3s-s0-step125 vs c3s-s1-step125 | 0.10 | 0.20 | 0.30 | 0.30 | 0.10 | 0.70 |
| virchow2 | c3s-s0-step125 vs c3s-s1-step125 | 0.00 | 0.30 | 0.30 | 0.20 | 0.10 | 0.10 |
| hoptimus0 | c3s-s0-step125 vs c3s-s1-step125 | 0.80 | 0.70 | 0.20 | 0.10 | 0.00 | 0.30 |
| uni2h | c3s-s0-step125 vs c3s-s1-step125 | 0.20 | 0.10 | 0.10 | 0.30 | 0.20 | 0.50 |

| task | max seed gap across backbones | n backbones |
|---|---|---|
| knn | 1.10 | 5 |
| linear_probing | 0.70 | 5 |
| simple_shot | 0.40 | 5 |
| segmentation | 1.10 | 5 |
| calibration | 0.20 | 5 |
| adversarial_attack | 0.70 | 5 |

## How to use this

A final-recipe delta in `docs/thunder16.md` smaller than its own (backbone, task)
gap above is inside seed noise and must not be reported as an effect.  The floor
varies by backbone -- it is not one constant -- so compare each cell against its own
row, never against the cross-backbone maximum.
