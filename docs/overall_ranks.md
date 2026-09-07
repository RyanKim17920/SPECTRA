# Overall leaderboard position, base -> fine-tuned

**Generated file -- do not hand-edit.**  Regenerate with
`./.venv/bin/python scripts/overall_ranks.py`.

Position is how each leaderboard orders models: THUNDER by rank sum over six tasks
(32 models), HEST by mean $r$ over nine tasks (26 models), PathoROB by mean RI over
three datasets (23 models). `*` marks a THUNDER row whose adversarial column is still
the fp16 value and is therefore provisional.

| backbone | PathoROB | HEST | THUNDER |
|---|---|---|---|
| Phikon-v2 | 21 -> 6 (+15) | 15 -> 10 (+5) | 18 -> 14 (+4) |
| Midnight-12k | 10 -> 2 (+8) (estimated) | 8 -> 5 (+3) | 5 -> 2 (+3) |
| Virchow2 | 3 -> 2 (+1) | 6 -> 5 (+1) | 3 -> 2 (+1) |
| H-optimus-0 | 9 -> 2 (+7) | 3 -> 2 (+1) | 8 -> 5 (+3) |
| UNI2-h | 10 -> 2 (+8) | 4 -> 1 (+3) | 1 -> 1 (0) |
| Virchow | 6 -> 2 (+4) | 5 -> 5 (0) | 11 -> 6 (+5) |
| OpenMidnight | 14 -> 3 (+11) (estimated) | 9 -> 6 (+3) | 12 -> 5 (+7) |
