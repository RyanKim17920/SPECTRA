# Eval coverage matrix

**Generated file -- do not hand-edit.**  Regenerate with:

```
./.venv/bin/python scripts/eval_matrix.py
```

Every cell under `/admin/home/ryan.kim/pathfm-cells`, by the role it plays in the
paper.  `done` means the suite's terminal artifact is on disk; it is read from disk
at generation time and is never inferred from a queue state.

`op` marks a cell sitting at its backbone's 1-SE-selected operating point -- the
checkpoint the headline table in `docs/final_scoreboard.md` reads.  Cells at other
steps are sensitivity points, not substitutes for it.

`online` is THUNDER segmentation + PGD; it is a separate row from `thunder` because
`submit_partial.sh` deliberately skips it (4-4.5 of a ~7 GPU-hour path).

## c50 -- FINAL RECIPE (WAIV_BCLS=3.0 / WAIV_BMEAN=-inf)

| cell | backbone | step | thunder | online | pathorob | cptac | hest | complete |
|---|---|---|---|---|---|---|---|---|
| `hoptimus0-c50-s0-step100` | hoptimus0 | **100 (op)** | done | done | done | done | done | 5/5 |
| `hoptimus0-c50-s0-step150` | hoptimus0 | 150 | done | done | done | done | done | 5/5 |
| `hoptimus0-c50-s0-step50` | hoptimus0 | 50 | done | done | done | done | done | 5/5 |
| `hoptimus0-c50-s1-step100` | hoptimus0 | **100 (op)** | done | done | done | done | done | 5/5 |
| `hoptimus0-c50-s3-step100` | hoptimus0 | **100 (op)** | done | done | done | done | done | 5/5 |
| `midnight-c50-s0-step150` | midnight | **150 (op)** | done | done | done | done | done | 5/5 |
| `midnight-c50-s1-step100` | midnight | 100 | done | done | done | done | done | 5/5 |
| `midnight-c50-s1-step150` | midnight | **150 (op)** | done | done | done | done | done | 5/5 |
| `midnight-c50-s3-step100` | midnight | 100 | done | done | done | done | -- | 4/5 |
| `openmidnightsq-c50-s0-step150` | openmidnightsq | 150 | -- | -- | -- | -- | done | 1/5 |
| `openmidnightsq-c50-s1-step150` | openmidnightsq | 150 | -- | -- | -- | -- | done | 1/5 |
| `openmidnightsq-c50-s2-step150` | openmidnightsq | 150 | -- | -- | -- | -- | done | 1/5 |
| `phikon2-c50-s0-step200` | phikon2 | **200 (op)** | done | done | done | done | done | 5/5 |
| `phikon2-c50-s1-step200` | phikon2 | **200 (op)** | done | done | done | done | done | 5/5 |
| `phikon2-c50-s2-step200` | phikon2 | **200 (op)** | done | done | done | done | done | 5/5 |
| `uni2h-c50-s0-step100` | uni2h | **100 (op)** | done | done | done | done | done | 5/5 |
| `uni2h-c50-s0-step150` | uni2h | **150 (op)** | done | done | done | done | done | 5/5 |
| `uni2h-c50-s0-step50` | uni2h | 50 | done | done | done | done | done | 5/5 |
| `uni2h-c50-s1-step150` | uni2h | **150 (op)** | done | done | done | done | done | 5/5 |
| `uni2h-c50-s2-step100` | uni2h | **100 (op)** | done | done | done | done | done | 5/5 |
| `virchow1-c50-s0-step150` | virchow1 | 150 | -- | -- | -- | -- | done | 1/5 |
| `virchow1-c50-s1-step150` | virchow1 | 150 | -- | -- | -- | -- | done | 1/5 |
| `virchow1-c50-s2-step150` | virchow1 | 150 | -- | -- | -- | -- | -- | 0/5 |
| `virchow2-c50-s0-step100` | virchow2 | **100 (op)** | done | done | done | done | done | 5/5 |
| `virchow2-c50-s1-step100` | virchow2 | **100 (op)** | done | done | done | done | done | 5/5 |
| `virchow2-c50-s1-step150` | virchow2 | 150 | done | done | done | done | -- | 4/5 |
| `virchow2-c50-s3-step100` | virchow2 | **100 (op)** | done | done | done | done | done | 5/5 |
| `virchow2f-c50-s0-step100` | virchow2f | 100 | -- | -- | -- | -- | done | 1/5 |
| `virchow2f-c50-s1-step150` | virchow2f | 150 | -- | -- | -- | -- | -- | 0/5 |
| `virchow2f-c50-s3-step100` | virchow2f | 100 | -- | -- | -- | -- | done | 1/5 |

## b00 -- ABLATION: bias 0/0 -- arithmetically identical to no same-core masking

| cell | backbone | step | thunder | online | pathorob | cptac | hest | complete |
|---|---|---|---|---|---|---|---|---|
| `hoptimus0-b00-s0-step100` | hoptimus0 | **100 (op)** | done | done | done | done | done | 5/5 |
| `midnight-b00-s0-step150` | midnight | **150 (op)** | done | done | done | done | done | 5/5 |
| `phikon2-b00-s0-step200` | phikon2 | **200 (op)** | done | done | done | done | done | 5/5 |
| `uni2h-b00-s0-step100` | uni2h | **100 (op)** | done | done | done | done | done | 5/5 |
| `virchow2-b00-s0-step100` | virchow2 | **100 (op)** | done | done | done | done | done | 5/5 |

## bm3 -- ABLATION: symmetric bias +3/+3 on both heads

| cell | backbone | step | thunder | online | pathorob | cptac | hest | complete |
|---|---|---|---|---|---|---|---|---|
| `hoptimus0-bm3-s0-step100` | hoptimus0 | **100 (op)** | done | done | done | done | done | 5/5 |
| `uni2h-bm3-s0-step100` | uni2h | **100 (op)** | done | done | done | done | done | 5/5 |

## base-control -- BASE CONTROL: published backbone, no adapter

| cell | backbone | step | thunder | online | pathorob | cptac | hest | complete |
|---|---|---|---|---|---|---|---|---|
| `hoptimus0-base-control` | hoptimus0 | base | done | done | done | done | done | 5/5 |
| `midnight-base-control` | midnight | base | done | done | done | done | done | 5/5 |
| `openmidnightsq-base-control` | openmidnightsq | base | -- | -- | -- | -- | done | 1/5 |
| `phikon2-base-control` | phikon2 | base | done | done | done | done | done | 5/5 |
| `uni2h-base-control` | uni2h | base | done | done | done | done | done | 5/5 |
| `virchow1-base-control` | virchow1 | base | -- | -- | -- | -- | done | 1/5 |
| `virchow2-base-control` | virchow2 | base | done | done | done | done | done | 5/5 |

## c3s -- SUPERSEDED generation (kept for the three-generation comparison)

| cell | backbone | step | thunder | online | pathorob | cptac | hest | complete |
|---|---|---|---|---|---|---|---|---|
| `hoptimus0-c3s-s0-step125` | hoptimus0 | 125 | done | done | done | done | done | 5/5 |
| `hoptimus0-c3s-s1-step125` | hoptimus0 | 125 | done | done | done | done | done | 5/5 |
| `midnight-c3s-s0-step125` | midnight | 125 | done | done | done | done | done | 5/5 |
| `midnight-c3s-s1-step125` | midnight | 125 | done | done | done | done | done | 5/5 |
| `phikon2-c3s-s0-step250` | phikon2 | 250 | done | done | done | done | done | 5/5 |
| `phikon2-c3s-s1-step250` | phikon2 | 250 | done | done | done | done | done | 5/5 |
| `uni2h-c3s-s0-step125` | uni2h | 125 | done | done | done | done | done | 5/5 |
| `uni2h-c3s-s1-step125` | uni2h | 125 | done | done | done | done | done | 5/5 |
| `virchow2-c3s-s0-step125` | virchow2 | 125 | done | done | done | done | done | 5/5 |
| `virchow2-c3s-s1-step125` | virchow2 | 125 | done | done | done | done | done | 5/5 |

## other -- unclassified

| cell | backbone | step | thunder | online | pathorob | cptac | hest | complete |
|---|---|---|---|---|---|---|---|---|
| `uni2h-base-ctrl-rep2` | uni2h | base | done | done | done | done | done | 5/5 |
| `virchow2-basectrl-fp32adv` | virchow2 | base | done | done | done | -- | done | 4/5 |

## Totals

| suite | done | cells | remaining |
|---|---|---|---|
| thunder | 45 | 56 | 11 |
| online | 45 | 56 | 11 |
| pathorob | 45 | 56 | 11 |
| cptac | 44 | 56 | 12 |
| hest | 52 | 56 | 4 |

**231 of 280 cell x suite pairs complete (49 remaining).**
