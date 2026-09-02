# Across-seed mean +/- SD, shipped recipe

**Generated file -- do not hand-edit.**  Regenerate with:

```
./.venv/bin/python scripts/seed_stats.py
```

Significance is judged against the SD of this recipe's own seeds, not against a
floor imported from another arm or panel.  Each seed contributes its own 1-SE
selected checkpoint, so the SD includes selection variance -- part of the
procedure, therefore part of the error bar.

`n=2` gives a spread rather than an SD; cells are marked with their n and
anything below n=3 should be read as provisional.

## phikon2

seed cells: c50-s0-step200, c50-s1-step200, c50-s2-step200

| metric | base | tuned (mean +/- SD) | gain |
|---|---|---|---|
| PathoROB RI | 0.4701 | 0.8356 +/- 0.0069 (n=3) | +0.3654 = >10 SD |
| HEST | 0.3747 | 0.3906 +/- 0.0003 (n=3) | +0.0159 = >10 SD |
| THUNDER knn | 73.9 | 77.4333 +/- 0.2309 (n=3) | +3.5333 = >10 SD |
| THUNDER linear_probing | 79.7 | 81.3667 +/- 0.1528 (n=3) | +1.6667 = >10 SD |
| THUNDER simple_shot | 71.8 | 72.5000 +/- 0.0000 (n=3) | +0.7000 (SD=0 across seeds) |
| THUNDER segmentation | 67.3 | 67.2667 +/- 0.2309 (n=3) | -0.0333 = 0.1 SD |
| THUNDER calibration | 3.8 | 3.8667 +/- 0.1155 (n=3) | +0.0667 = 0.6 SD |
| THUNDER adversarial_attack | 36.2 | 29.4000 +/- 0.1414 (n=2) | -6.8000 = >10 SD |

## midnight

seed cells: c50-s0-step150, c50-s1-step100, c50-s3-step100

| metric | base | tuned (mean +/- SD) | gain |
|---|---|---|---|
| PathoROB RI | 0.7589 | 0.9082 +/- 0.0024 (n=3) | +0.1493 = >10 SD |
| HEST | 0.3952 | 0.4127 +/- 0.0007 (n=2) | +0.0175 = >10 SD |
| THUNDER knn | 80.0 | 81.7667 +/- 0.3512 (n=3) | +1.7667 = 5.0 SD |
| THUNDER linear_probing | 84.8 | 85.6333 +/- 0.1155 (n=3) | +0.8333 = 7.2 SD |
| THUNDER simple_shot | 71.5 | 77.0000 +/- 0.1000 (n=3) | +5.5000 = >10 SD |
| THUNDER segmentation | 68.0 | 68.7000 +/- 0.5196 (n=3) | +0.7000 = 1.3 SD |
| THUNDER calibration | 2.9 | 3.8000 +/- 0.2646 (n=3) | +0.9000 = 3.4 SD |
| THUNDER adversarial_attack | 29.9 | 22.8000 +/- 1.4731 (n=3) | -7.1000 = 4.8 SD |

## virchow2

seed cells: c50-s0-step100, c50-s1-step150, c50-s3-step100

| metric | base | tuned (mean +/- SD) | gain |
|---|---|---|---|
| PathoROB RI | 0.8610 | 0.9091 +/- 0.0037 (n=3) | +0.0481 = >10 SD |
| HEST | 0.4033 | 0.4082 +/- 0.0018 (n=2) | +0.0050 = 2.8 SD |
| THUNDER knn | 82.9 | 83.0667 +/- 0.1155 (n=3) | +0.1667 = 1.4 SD |
| THUNDER linear_probing | 84.7 | 85.4667 +/- 0.2309 (n=3) | +0.7667 = 3.3 SD |
| THUNDER simple_shot | 74.0 | 78.0667 +/- 0.3215 (n=3) | +4.0667 = >10 SD |
| THUNDER segmentation | 69.0 | 69.0000 +/- 0.2000 (n=3) | +0.0000 = 0.0 SD |
| THUNDER calibration | 4.0 | 4.4667 +/- 0.2082 (n=3) | +0.4667 = 2.2 SD |
| THUNDER adversarial_attack | 0.3 | 0.3000 +/- 0.1732 (n=3) | +0.0000 = 0.0 SD |

## hoptimus0

seed cells: c50-s0-step100, c50-s1-step100, c50-s3-step100

| metric | base | tuned (mean +/- SD) | gain |
|---|---|---|---|
| PathoROB RI | 0.7997 | 0.9055 +/- 0.0020 (n=3) | +0.1058 = >10 SD |
| HEST | 0.4150 | 0.4225 +/- 0.0009 (n=3) | +0.0075 = 8.2 SD |
| THUNDER knn | 81.4 | 81.5000 +/- 0.2646 (n=3) | +0.1000 = 0.4 SD |
| THUNDER linear_probing | 83.8 | 84.2000 +/- 0.1000 (n=3) | +0.4000 = 4.0 SD |
| THUNDER simple_shot | 76.1 | 77.0333 +/- 0.0577 (n=3) | +0.9333 = >10 SD |
| THUNDER segmentation | 64.6 | 64.9667 +/- 0.4726 (n=3) | +0.3667 = 0.8 SD |
| THUNDER calibration | 4.0 | 3.5000 +/- 0.1000 (n=3) | -0.5000 = 5.0 SD |
| THUNDER adversarial_attack | 32.4 | 26.7333 +/- 0.6807 (n=3) | -5.6667 = 8.3 SD |

## uni2h

seed cells: c50-s0-step100, c50-s1-step150, c50-s2-step100

| metric | base | tuned (mean +/- SD) | gain |
|---|---|---|---|
| PathoROB RI | 0.7566 | 0.9074 +/- 0.0035 (n=3) | +0.1509 = >10 SD |
| HEST | 0.4138 | 0.4238 +/- 0.0029 (n=3) | +0.0099 = 3.5 SD |
| THUNDER knn | 83.3 | 82.9667 +/- 0.4163 (n=3) | -0.3333 = 0.8 SD |
| THUNDER linear_probing | 85.7 | 85.8000 +/- 0.3000 (n=3) | +0.1000 = 0.3 SD |
| THUNDER simple_shot | 79.8 | 78.7667 +/- 0.5132 (n=3) | -1.0333 = 2.0 SD |
| THUNDER segmentation | 69.2 | 69.1333 +/- 0.2309 (n=3) | -0.0667 = 0.3 SD |
| THUNDER calibration | 3.9 | 4.5000 +/- 0.1000 (n=3) | +0.6000 = 6.0 SD |
| THUNDER adversarial_attack | 26.8 | 21.0000 +/- 0.8660 (n=3) | -5.8000 = 6.7 SD |

## openmidnightsq

seed cells: c50-s0-step150, c50-s1-step150, c50-s2-step150

| metric | base | tuned (mean +/- SD) | gain |
|---|---|---|---|
| PathoROB RI | -- | -- | -- |
| HEST | 0.3902 | 0.4048 +/- 0.0007 (n=3) | +0.0146 = >10 SD |
| THUNDER knn | -- | -- | -- |
| THUNDER linear_probing | -- | -- | -- |
| THUNDER simple_shot | -- | -- | -- |
| THUNDER segmentation | -- | -- | -- |
| THUNDER calibration | -- | -- | -- |
| THUNDER adversarial_attack | -- | -- | -- |

## virchow1

seed cells: c50-s0-step150, c50-s1-step150, c50-s2-step150

| metric | base | tuned (mean +/- SD) | gain |
|---|---|---|---|
| PathoROB RI | 0.8147 | -- | -- |
| HEST | 0.4061 | -- | -- |
| THUNDER knn | 77.4 | -- | -- |
| THUNDER linear_probing | 82.8 | -- | -- |
| THUNDER simple_shot | 71.8 | -- | -- |
| THUNDER segmentation | 69.2 | -- | -- |
| THUNDER calibration | 4.5 | -- | -- |
| THUNDER adversarial_attack | 29.9 | -- | -- |

## virchow2f

seed cells: c50-s0-step100, c50-s1-step150, c50-s3-step100

| metric | base | tuned (mean +/- SD) | gain |
|---|---|---|---|
| PathoROB RI | 0.8610 | -- | -- |
| HEST | 0.4033 | 0.4082 +/- 0.0018 (n=2) | +0.0050 = 2.8 SD |
| THUNDER knn | 82.9 | -- | -- |
| THUNDER linear_probing | 84.7 | -- | -- |
| THUNDER simple_shot | 73.9 | -- | -- |
| THUNDER segmentation | 69.4 | -- | -- |
| THUNDER calibration | 4.0 | -- | -- |
| THUNDER adversarial_attack | -- | -- | -- |
