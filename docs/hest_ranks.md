# HEST-Benchmark rank and rank sum, base vs fine-tuned

**Generated file -- do not hand-edit.**  Regenerate with
`./.venv/bin/python scripts/hest_ranks.py`.

HEST publishes per-task Pearson $r$ for 26 models but no rank column and no
rank sum, so both are computed here with dense ranking (the convention THUNDER uses
and which scripts/thunder_ranks.py validates against 32 published rows). The
transcribed table is validated by reproducing every published Average from its nine
task columns. Base and tuned are both from our harness, so the delta is
within-harness; our base reproduces the published row closely.

| backbone | our base avg | published avg | base rank sum | tuned rank sum | move | base avg-rank | tuned avg-rank |
|---|---|---|---|---|---|---|---|
| Phikon-v2 | 0.3747 | 0.3747 | 131 | 97 | +34 | 15 | 10 |
| Midnight-12k | 0.3952 | 0.3952 | 97 | 51 | +46 | 8 | 5 |
| Virchow2 | 0.4032 | 0.4034 | 65 | 48 | +17 | 6 | 5 |
| H-Optimus-0 | 0.4150 | 0.4150 | 37 | 29 | +8 | 3 | 2 |
| UNI2-h | 0.4138 | 0.4141 | 42 | 24 | +18 | 4 | 1 |
| Virchow | 0.4061 | 0.4061 | 67 | 56 | +11 | 5 | 5 |
| OpenMidnight | 0.3902 | 0.3912 | 94 | 64 | +30 | 9 | 6 |
