# PathoROB rank and rank sum, base vs fine-tuned

**Generated file -- do not hand-edit.**  Regenerate with
`./.venv/bin/python scripts/pathorob_ranks.py`.

Field is the 23-model leaderboard from the PathoROB authors' repo
(github.com/bifold-pathomics/PathoROB); three of its rows are cited from external
publications rather than computed by them. PathoROB publishes no ranks, so dense
ranking is applied as for THUNDER. Midnight-12k is not in the field and cannot be
ranked.

## Check: our measured base RI vs their published row

| backbone | dataset | ours | theirs | diff |
|---|---|---|---|---|
| Phikon-v2 | tcga | 0.6172 | 0.619 | -0.002 |
| Phikon-v2 | camelyon | 0.0233 | 0.019 | +0.004 |
| Phikon-v2 | tolkach_esca | 0.7699 | 0.768 | +0.002 |
| Virchow2 | tcga | 0.8222 | 0.822 | +0.000 |
| Virchow2 | camelyon | 0.8063 | 0.806 | +0.000 |
| Virchow2 | tolkach_esca | 0.9545 | 0.955 | -0.000 |
| H-optimus-0 | tcga | 0.8025 | 0.812 | -0.010 **MISMATCH** |
| H-optimus-0 | camelyon | 0.6785 | 0.705 | -0.027 **MISMATCH** |
| H-optimus-0 | tolkach_esca | 0.9181 | 0.918 | +0.000 |
| UNI2-h | tcga | 0.8029 | 0.803 | -0.000 |
| UNI2-h | camelyon | 0.5441 | 0.544 | +0.000 |
| UNI2-h | tolkach_esca | 0.9226 | 0.923 | -0.000 |
| Virchow | tcga | 0.7612 | 0.761 | +0.000 |
| Virchow | camelyon | 0.7506 | 0.751 | -0.000 |
| Virchow | tolkach_esca | 0.9324 | 0.932 | +0.000 |

## Rank and rank sum

| backbone | dataset | base | rank | tuned | rank | move |
|---|---|---|---|---|---|---|
| Phikon-v2 | tcga | 0.6172 | 19 | 0.7926 | 11 | +8 |
| Phikon-v2 | camelyon | 0.0233 | 22 | 0.7753 | 5 | +17 |
| Phikon-v2 | tolkach_esca | 0.7699 | 17 | 0.9388 | 6 | +11 |
| Phikon-v2 | **rank sum** | | 58 | | 22 | **+36** |
| Midnight-12k | tcga | 0.8575 | 2* | 0.8709 | 2* | 0 (inserted into field) |
| Midnight-12k | camelyon | 0.4780 | 13* | 0.8843 | 2* | +11 (inserted into field) |
| Midnight-12k | tolkach_esca | 0.9411 | 6* | 0.9695 | 1* | +5 (inserted into field) |
| Midnight-12k | **rank sum** | | 21* | | 5* | **+16** (estimated) |
| Virchow2 | tcga | 0.8222 | 7 | 0.8415 | 3 | +4 |
| Virchow2 | camelyon | 0.8063 | 3 | 0.9165 | 2 | +1 |
| Virchow2 | tolkach_esca | 0.9545 | 3 | 0.9694 | 1 | +2 |
| Virchow2 | **rank sum** | | 13 | | 6 | **+7** |
| H-optimus-0 | tcga | 0.8025 | 9 | 0.8358 | 4 | +5 |
| H-optimus-0 | camelyon | 0.6785 | 8 | 0.9191 | 2 | +6 |
| H-optimus-0 | tolkach_esca | 0.9181 | 10 | 0.9617 | 2 | +8 |
| H-optimus-0 | **rank sum** | | 27 | | 8 | **+19** |
| UNI2-h | tcga | 0.8029 | 9 | 0.8584 | 2 | +7 |
| UNI2-h | camelyon | 0.5441 | 12 | 0.9009 | 2 | +10 |
| UNI2-h | tolkach_esca | 0.9226 | 9 | 0.9629 | 2 | +7 |
| UNI2-h | **rank sum** | | 30 | | 6 | **+24** |
| Virchow | tcga | 0.7612 | 12 | 0.8068 | 9 | +3 |
| Virchow | camelyon | 0.7506 | 6 | 0.9064 | 2 | +4 |
| Virchow | tolkach_esca | 0.9324 | 7 | 0.9580 | 3 | +4 |
| Virchow | **rank sum** | | 25 | | 14 | **+11** |
| OpenMidnight | tcga | 0.8049 | 9* | 0.8407 | 3* | +6 (inserted into field) |
| OpenMidnight | camelyon | 0.1886 | 16* | 0.8384 | 3* | +13 (inserted into field) |
| OpenMidnight | tolkach_esca | 0.8610 | 15* | 0.9590 | 3* | +12 (inserted into field) |
| OpenMidnight | **rank sum** | | 40* | | 9* | **+31** (estimated) |
