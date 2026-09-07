# THUNDER leaderboard rank, base vs fine-tuned

**Generated file -- do not hand-edit.**  Regenerate with
`./.venv/bin/python scripts/thunder_ranks.py`.

Rank is position in the 32-model 16-dataset panel (histopathology AND
natural-image models), using DENSE ranking, the leaderboard's own convention.
The rule is validated against all 32 published per-task ranks and rank sums
before use. To rank one of our models we replace that backbone's published row.
ECE and adversarial drop are lower-is-better; adversarial uses fp32-attack cells.

| backbone | task | base | rank | fine-tuned | rank | move |
|---|---|---|---|---|---|---|
| Phikon-v2 | knn | 73.9 | 20 | 77.4 | 18 | +2 |
| Phikon-v2 | lin | 79.7 | 19 | 81.4 | 16 | +3 |
| Phikon-v2 | few | 71.8 | 17 | 72.5 | 16 | +1 |
| Phikon-v2 | seg | 67.4 | 11 | 67.3 | 11 | 0 |
| Phikon-v2 | ece | 3.9 | 8 | 3.9 | 9 | -1 |
| Phikon-v2 | adv | 43.8 | 13 | 34.8 | 5 | +8 |
| Midnight-12k | knn | 79.9 | 8 | 81.8 | 5 | +3 |
| Midnight-12k | lin | 84.7 | 4 | 85.6 | 2 | +2 |
| Midnight-12k | few | 71.5 | 18 | 77.0 | 6 | +12 |
| Midnight-12k | seg | 68.8 | 9 | 68.7 | 6 | +3 |
| Midnight-12k | ece | 2.9 | 2 | 3.8 | 7 | -5 |
| Midnight-12k | adv | 37.0 | 6 | 29.1 | 1 | +5 |
| Virchow2 | knn | 82.9 | 3 | 83.1 | 3 | 0 |
| Virchow2 | lin | 84.8 | 4 | 85.5 | 2 | +2 |
| Virchow2 | few | 73.9 | 12 | 78.1 | 4 | +8 |
| Virchow2 | seg | 69.3 | 3 | 69.0 | 3 | 0 |
| Virchow2 | ece | 3.9 | 10 | 4.5 | 14 | -4 |
| Virchow2 | adv | 31.1 | 1 | 25.4 | 1 | 0 |
| H-optimus-0 | knn | 81.4 | 6 | 81.5 | 5 | +1 |
| H-optimus-0 | lin | 83.8 | 6 | 84.2 | 6 | 0 |
| H-optimus-0 | few | 76.2 | 8 | 77.0 | 6 | +2 |
| H-optimus-0 | seg | 65.2 | 14 | 65.0 | 14 | 0 |
| H-optimus-0 | ece | 4.0 | 10 | 3.5 | 6 | +4 |
| H-optimus-0 | adv | 43.9 | 15 | 37.7 | 7 | +8 |
| UNI2-h | knn | 83.3 | 2 | 83.0 | 2 | 0 |
| UNI2-h | lin | 85.7 | 1 | 85.8 | 1 | 0 |
| UNI2-h | few | 79.8 | 1 | 78.8 | 2 | -1 |
| UNI2-h | seg | 69.0 | 2 | 69.1 | 3 | -1 |
| UNI2-h | ece | 3.9 | 9 | 4.5 | 14 | -5 |
| UNI2-h | adv | 31.7 | 2 | 24.9 | 1 | +1 |
| Virchow | knn | 77.4 | 18 | 78.9 | 12 | +6 |
| Virchow | lin | 82.8 | 11 | 83.4 | 9 | +2 |
| Virchow | few | 71.8 | 17 | 73.7 | 14 | +3 |
| Virchow | seg | 69.2 | 2 | 68.9 | 4 | -2 |
| Virchow | ece | 4.5 | 14 | 4.0 | 10 | +4 |
| Virchow | adv | 38.3 | 8 | 30.3 | 1 | +7 |
| OpenMidnight | knn | 79.3 | 11 | 80.8 | 7 | +4 |
| OpenMidnight | lin | 84.7 | 4 | 85.0 | 4 | 0 |
| OpenMidnight | few | 43.7 | 30 | 65.9 | 19 | +11 |
| OpenMidnight | seg | 69.1 | 2 | 68.9 | 5 | -3 |
| OpenMidnight | ece | 5.4 | 19 | 4.1 | 11 | +8 |
| OpenMidnight | adv | 38.3 | 8 | 28.0 | 1 | +7 |

## Rank sum (lower is better)

`4 tasks` excludes ECE and adversarial, the subset with complete $n=3$ coverage on
every backbone. `5 tasks` adds adversarial where the fp32 cells have landed.

| backbone | 4 tasks base | tuned | move | +ECE base | tuned | move |
|---|---|---|---|---|---|---|
| Phikon-v2 | 67 | 61 | +6 | 76 | 70 | +6 |
| Midnight-12k | 36 | 19 | +17 | 38 | 26 | +12 |
| Virchow2 | 20 | 12 | +8 | 29 | 26 | +3 |
| H-optimus-0 | 33 | 31 | +2 | 43 | 37 | +6 |
| UNI2-h | 8 | 8 | 0 | 17 | 22 | -5 |
| Virchow | 48 | 39 | +9 | 62 | 49 | +13 |
| OpenMidnight | 49 | 35 | +14 | 68 | 46 | +22 |
