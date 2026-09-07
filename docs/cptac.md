# CPTAC (Patho-Bench subset): base -> fine-tuned

**Generated file -- do not hand-edit.**  Regenerate with:

```
./.venv/bin/python scripts/cptac_table.py
```

Metric: `mean_classification_macro_ovr_auc` from each cell's `aggregate.json`.
Seed cells are the 1-SE-selected checkpoints, matching `docs/seed_stats.md`.
There is no published per-model Patho-Bench leaderboard, so no rank column.

| backbone | base | fine-tuned (mean +/- SD) | gain | n | tasks |
|---|---|---|---|---|---|
| Phikon-v2 | 0.6491 | 0.6870 +/- 0.0012 | +0.0378 | 3 | 38 |
| Midnight-12k | 0.6643 | 0.6898 +/- 0.0008 | +0.0255 | 3 | 38 |
| Virchow2 | 0.6789 | 0.6879 +/- 0.0008 | +0.0090 | 3 | 38 |
| H-optimus-0 | 0.6728 | 0.6895 +/- 0.0002 | +0.0167 | 3 | 38 |
| UNI2-h | 0.6750 | 0.6980 +/- 0.0031 | +0.0229 | 3 | 38 |
| Virchow | 0.6608 | 0.6839 +/- 0.0013 | +0.0231 | 3 | 38 |
| OpenMidnight | 0.6561 | 0.6844 +/- 0.0011 | +0.0283 | 3 | 38 |

