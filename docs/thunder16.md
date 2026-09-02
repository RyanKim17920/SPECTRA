# Full THUNDER -- our base vs our fine-tuned

**Generated file -- do not hand-edit.**  Regenerate with:

```
./.venv/bin/python scripts/thunder16.py
```

Source: `/data/ryan.kim/pathfm-full-evals/thunder/outputs/res/results.csv`.  All six THUNDER tasks over all **16** classification
datasets and 4 segmentation datasets, read from the harness's own `benchmark_*`
roll-up rows.  No external target: every column below is ours, measured in one
harness, so the quantity of interest is the base-to-tuned delta.

Calibration is ECE and the adversarial column is the PGD f1 *drop*; for both,
**lower is better**, so a negative delta is an improvement.

## phikon2

| task | base | b00-s0-step200 | c3s-s0-step250 | c3s-s1-step250 | c50-s0-step200 | c50-s1-step200 | c50-s2-step200 | delta (c50-s0-step200) |
|---|---|---|---|---|---|---|---|---|
| knn | 73.9 | 77.2 | 77.4 | 76.3 | 77.3 | 77.7 | 77.3 | +3.4 (better) |
| linear_probing | 79.7 | 81.6 | 81.4 | 80.8 | 81.4 | 81.2 | 81.5 | +1.7 (better) |
| simple_shot | 71.8 | 73.0 | 72.3 | 71.9 | 72.5 | 72.5 | 72.5 | +0.7 (better) |
| segmentation | 67.3 | 67.2 | 66.2 | 67.3 | 67.4 | 67.0 | 67.4 | +0.1 (better) |
| calibration | 3.8 | 4.2 | 3.6 | 3.7 | 3.8 | 3.8 | 4.0 | +0.0 (flat) |
| adversarial_attack | 36.2 | 29.1 | 28.1 | 27.5 | -- | 29.5 | 29.3 | -- |

Classification datasets scored for this backbone: 16/16.

## midnight

| task | base | b00-s0-step150 | c3s-s0-step125 | c3s-s1-step125 | c50-s0-step150 | c50-s1-step100 | c50-s1-step150 | c50-s3-step100 | delta (c50-s0-step150) |
|---|---|---|---|---|---|---|---|---|---|
| knn | 80.0 | 82.0 | 81.9 | 81.8 | 81.4 | 82.1 | 80.5 | 81.8 | +1.4 (better) |
| linear_probing | 84.8 | 85.9 | 85.8 | 85.6 | 85.7 | 85.5 | 85.6 | 85.7 | +0.9 (better) |
| simple_shot | 71.5 | 77.4 | 77.1 | 77.4 | 77.0 | 77.1 | 76.8 | 76.9 | +5.5 (better) |
| segmentation | 68.0 | 68.3 | 68.6 | 68.3 | 68.1 | 69.0 | 68.4 | 69.0 | +0.1 (better) |
| calibration | 2.9 | 3.7 | 3.7 | 3.6 | 3.5 | 4.0 | 3.7 | 3.9 | +0.6 (worse) |
| adversarial_attack | 29.9 | 22.1 | 23.3 | 22.6 | 21.1 | 23.7 | 21.7 | 23.6 | -8.8 (better) |

Classification datasets scored for this backbone: 16/16.

## virchow2

| task | base | b00-s0-step100 | basectrl-fp32adv | c3s-s0-step125 | c3s-s1-step125 | c50-s0-step100 | c50-s1-step100 | c50-s1-step150 | c50-s3-step100 | delta (c50-s0-step100) |
|---|---|---|---|---|---|---|---|---|---|---|
| knn | 82.9 | 83.1 | 82.9 | 82.8 | 82.8 | 83.0 | 82.7 | 83.0 | 83.2 | +0.1 (better) |
| linear_probing | 84.7 | 85.4 | 84.7 | 85.3 | 85.6 | 85.2 | 85.3 | 85.6 | 85.6 | +0.5 (better) |
| simple_shot | 74.0 | 78.6 | 73.9 | 77.8 | 78.1 | 78.2 | 77.9 | 77.7 | 78.3 | +4.2 (better) |
| segmentation | 69.0 | 69.3 | 69.4 | 68.9 | 69.1 | 69.2 | 69.1 | 68.8 | 69.0 | +0.2 (better) |
| calibration | 4.0 | 4.0 | 4.0 | 4.3 | 4.2 | 4.7 | 4.1 | 4.3 | 4.4 | +0.7 (worse) |
| adversarial_attack | EXCLUDED | EXCLUDED | EXCLUDED | EXCLUDED | EXCLUDED | EXCLUDED | EXCLUDED | EXCLUDED | EXCLUDED | attack degenerate on this backbone (see note) |

Classification datasets scored for this backbone: 16/16.

## hoptimus0

| task | base | b00-s0-step100 | bm3-s0-step100 | c3s-s0-step125 | c3s-s1-step125 | c50-s0-step100 | c50-s0-step150 | c50-s0-step50 | c50-s1-step100 | c50-s3-step100 | delta (c50-s0-step100) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| knn | 81.4 | 82.3 | 80.6 | 80.9 | 81.7 | 81.6 | 81.2 | 81.8 | 81.2 | 81.7 | +0.2 (better) |
| linear_probing | 83.8 | 84.6 | 83.9 | 84.0 | 83.3 | 84.2 | 83.8 | 83.9 | 84.1 | 84.3 | +0.4 (better) |
| simple_shot | 76.1 | 77.8 | 76.7 | 77.0 | 77.2 | 77.0 | 77.0 | 76.8 | 77.0 | 77.1 | +0.9 (better) |
| segmentation | 64.6 | 64.8 | 65.7 | 65.2 | 65.3 | 64.8 | 65.1 | 64.3 | 64.6 | 65.5 | +0.2 (better) |
| calibration | 4.0 | 3.2 | 3.4 | 3.7 | 3.7 | 3.5 | 4.5 | 3.5 | 3.4 | 3.6 | -0.5 (better) |
| adversarial_attack | 32.4 | 28.0 | 25.7 | 26.4 | 26.1 | 27.5 | 27.6 | 32.0 | 26.2 | 26.5 | -4.9 (better) |

Classification datasets scored for this backbone: 16/16.

## uni2h

| task | base | b00-s0-step100 | base-ctrl-rep2 | bm3-s0-step100 | c3s-s0-step125 | c3s-s1-step125 | c50-s0-step100 | c50-s0-step150 | c50-s0-step50 | c50-s1-step150 | c50-s2-step100 | delta (c50-s0-step100) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| knn | 83.3 | 83.5 | 83.3 | 83.3 | 83.0 | 82.8 | 83.3 | 82.5 | 83.0 | 82.5 | 83.1 | +0.0 (flat) |
| linear_probing | 85.7 | 86.3 | 85.7 | 86.0 | 86.1 | 86.2 | 86.1 | 85.9 | 85.7 | 85.5 | 85.8 | +0.4 (better) |
| simple_shot | 79.8 | 79.7 | 79.8 | 79.1 | 78.8 | 78.7 | 79.2 | 78.7 | 80.0 | 78.2 | 78.9 | -0.6 (worse) |
| segmentation | 69.2 | 69.0 | 69.0 | 69.1 | 68.7 | 68.4 | 69.0 | 69.3 | 69.0 | 69.0 | 69.4 | -0.2 (worse) |
| calibration | 3.9 | 3.0 | 3.9 | 3.4 | 4.3 | 4.1 | 4.6 | 4.2 | 3.7 | 4.4 | 4.5 | +0.7 (worse) |
| adversarial_attack | 26.8 | 22.5 | 26.8 | 21.2 | 19.5 | 19.0 | 22.0 | -- | 27.8 | 20.5 | 20.5 | -4.8 (better) |

Classification datasets scored for this backbone: 16/16.

## openmidnightsq

TODO: no base-control on disk -- no delta can be expressed for this backbone.

## virchow1

| task | base | c50-s0-step150 | c50-s1-step150 | delta (c50-s0-step150) |
|---|---|---|---|---|
| knn | 77.4 | -- | -- | -- |
| linear_probing | 82.8 | -- | -- | -- |
| simple_shot | 71.8 | -- | -- | -- |
| segmentation | 69.2 | -- | -- | -- |
| calibration | 4.5 | -- | -- | -- |
| adversarial_attack | 29.9 | -- | -- | -- |

Classification datasets scored for this backbone: 1/16.

## virchow2f

| task | base |  | delta (c50-s0-step100) |
|---|---|---|
| knn | 82.9 |  | -- |
| linear_probing | 84.7 |  | -- |
| simple_shot | 73.9 |  | -- |
| segmentation | 69.4 |  | -- |
| calibration | 4.0 |  | -- |
| adversarial_attack | MISSING |  | MISSING |

Classification datasets scored for this backbone: 0/16.

## Notes

* **Virchow2 adversarial is excluded.** Its three models lose 0.1-0.3pp of f1
  under PGD where the other four backbones lose 19-32pp. That is an attack that
  failed to land, not a robust representation; reporting it would claim
  robustness we did not measure.
* **Seed floors do not carry over from the 12-set panel.** The floors in
  `docs/thunder_seed_floor_12ds.md` were measured over 12 datasets; a 16-set
  average has a different variance. TODO: re-measure the per-(backbone, task)
  floor on the 16-set panel before calling any delta above significant.
* `PENDING` means every per-dataset probe for that cell is on disk but THUNDER's
  summary stage has not written the roll-up yet; `--` means the task has not run.
* Single-seed cells are not marked here; see `docs/eval_matrix.md` for which
  (backbone, arm) pairs have a second seed on disk.
