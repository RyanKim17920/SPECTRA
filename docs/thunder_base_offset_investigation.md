# THUNDER base offset — why Waiv's base sits ~3 points above ours and above THUNDER's paper

Date: 2026-08-25. Read-only forensic pass; no code changed, no jobs launched.
Supersedes the "Unexplained, honest label" verdict in
[`baseline_comparability_audit.md`](baseline_comparability_audit.md) §2 and the
`(a) TWO-BASE FORMULA` note at `scripts/scoreboard.py:98-140`.

## Verdict

**CAUSE IDENTIFIED for classification (kNN / linear / few-shot).**
**NARROWED for segmentation** — a separate, smaller, Waiv-side re-run offset.

> Waiv's THUNDER classification numbers are means over **16** classification datasets —
> THUNDER's *current* roster, i.e. the paper's 12 plus the four SPIDER sets
> (`spider_breast`, `spider_colorectal`, `spider_skin`, `spider_thorax`), which postdate
> the paper. We average the paper's **12**. The four SPIDER sets are far easier than the
> other twelve (kNN F1 85–88, LP F1 89–91, against 12-set means of 70–83), so a 16-set mean
> sits mechanically ~+1.7 to +3.2 above a 12-set mean of the same embeddings.
>
> SPIDER has **no segmentation task**. So the roster change lifts classification and leaves
> segmentation untouched — which is exactly the "odd detail" (classification above the
> paper, segmentation below it) that defeated every single-cause story tried before.

Neither base is wrong. They are means over different rosters.

---

## 1. VERIFIED — the arithmetic reproduces Waiv on all three backbones

Our measured 12-dataset macro-F1, plus THUNDER's own published per-dataset SPIDER F1s
(pinned leaderboard, SPIDER table), recombined into a 16-dataset mean:

| backbone | task | our 12-ds | THUNDER SPIDER (Br, Co, Sk, Th) | **our implied 16-ds** | THUNDER up-to-date LB | **Waiv base** | Waiv − our16 |
|---|---|---|---|---|---|---|---|
| Phikon-v2 | kNN    | 70.28 | 80.2, 86.5, 83.3, 91.4 | **74.05** | 73.9 | **74.0** | **−0.05** |
| Phikon-v2 | linear | 76.54 | 86.0, 89.7, 87.2, 94.7 | **79.75** | 79.7 | **79.3** | −0.45 |
| Midnight-12k | kNN    | 78.25 | 77.1, 84.9, 85.7, 92.7 | **79.96** | 79.9 | **80.0** | **+0.04** |
| Midnight-12k | linear | 82.88 | 86.1, 89.6, 91.0, 94.4 | **84.73** | 84.7 | **84.4** | −0.33 |
| Virchow2 | kNN    | 80.87 | 82.3, 88.0, 89.1, 92.6 | **82.65** | 82.9 | **82.9** | +0.25 |
| Virchow2 | linear | 83.25 | 87.2, 90.8, 92.0, 93.9 | **85.18** | 84.8 | **84.8** | −0.38 |

kNN lands within **0.05 / 0.04 / 0.25** of Waiv's published base on the three backbones —
against an original discrepancy of **−3.72 / −1.75 / −2.03**. The gap does not shrink; it
disappears.

Reproduce: `scripts/collect_thunder.py --model base_cls` for the 12-ds column;
SPIDER columns from `/admin/home/ryan.kim/pathopress/source_data/pinned/thunder/leaderboards.md`,
"🏆 SPIDER Leaderboard" table.

### 1b. VERIFIED — Waiv's whole Table 2 base block *is* the up-to-date leaderboard

Independent of our runs. Every Waiv Table-2 `variant="base"` row
(`docs/waiv_published.json`) that has a leaderboard counterpart, differenced against both
THUNDER leaderboards:

| task | Waiv − **up-to-date** LB (16 cls ds) | Waiv − **original paper** LB (12 cls ds) |
|---|---|---|
| kNN | mean **+0.04**, max\|Δ\| 0.10, **4/7 exact** | mean +2.38, max\|Δ\| 3.90, 0/5 exact |
| linear | mean −0.19, max\|Δ\| 0.70, 1/7 exact | mean +2.12, max\|Δ\| 2.80, 0/5 exact |
| few-shot | mean **+0.000**, max\|Δ\| **0.00**, **7/7 EXACT** | mean +1.34, max\|Δ\| 1.70, 0/5 exact |
| segmentation | mean −1.33, max\|Δ\| 2.80, 0/7 exact | mean −1.38 (seg column is identical in both LBs) |
| calibration | mean −0.34 | mean −0.80 |
| adversarial | mean −1.14, 1/7 exact | mean −2.46 |

Models compared: UNI2-h, GenBio-PathFM, Virchow2, Midnight-12k, H0-mini, Phikon-v2, Phikon.
**Few-shot is exact on 7 of 7 models, to the printed decimal.** kNN is exact on 4 of 7 and
never off by more than 0.1 (one printed digit). That is not two labs agreeing; that is the
same roster and the same protocol.

Per-model detail for the three backbones we run (Waiv | up-to-date LB | Δ):

```
Virchow2      knn 82.9|82.9|+0.00   linear 84.8|84.8|+0.00   few_shot 73.9|73.9|+0.00   seg 68.2|69.3|-1.10
Midnight-12k  knn 80.0|79.9|+0.10   linear 84.4|84.7|-0.30   few_shot 71.5|71.5|+0.00   seg 66.0|68.8|-2.80
Phikon-v2     knn 74.0|73.9|+0.10   linear 79.3|79.7|-0.40   few_shot 71.8|71.8|+0.00   seg 66.5|67.4|-0.90
```

### 1c. VERIFIED — the 16-vs-12 split is visible inside THUNDER's own linear-probing table

THUNDER's "Per-dataset linear probing performance" table lists all 16 datasets and its own
`Avg`. Taking the first 12 columns versus all 16:

| model | 12-ds mean | 16-ds mean | THUNDER's printed Avg | Waiv |
|---|---|---|---|---|
| PHIKON2 | **76.46** | 79.69 | 79.70 | 79.3 |
| MIDNIGHT | 82.90 | 84.74 | 84.70 | 84.4 |
| VIRCHOW2 | 82.68 | 84.76 | 84.80 | **84.80** |
| UNI2-H | 83.91 | 85.73 | 85.70 | 86.3 |
| GenBio-PFM | 83.37 | 85.32 | 85.30 | 85.1 |
| PHIKON | 78.45 | 80.94 | 80.90 | 80.2 |

The 12-ds column reproduces the paper (`collect_thunder.PUBLISHED`, 76.46 for phikon-v2) and
our own measurement (76.54). The published `Avg` is the 16-ds column. Waiv tracks the 16-ds
column. The +3.2 is the SPIDER quartet, arithmetically.

---

## 2. Hypotheses eliminated

### H1 — metric mismatch. **ELIMINATED.**

Every metric key in every base `outputs.json` was enumerated and averaged over the 12
PAPER_CLS datasets for all three backbones. Available keys: `f1`, `accuracy`, `jaccard`,
`balanced_accuracy`, `roc_auc` (+ `ECE/MCE/SCE/ACE/TACE` on linear probing only).
Additionally, `per_sample_pred` and `label` are stored per sample, so every sklearn
averaging mode was recomputed from raw predictions.

First, a confirmation: **THUNDER's stored `f1` is exactly `f1_score(average="macro")`** —
max abs difference over all 108 (backbone × task × dataset) cells is **0.0**. So
`collect_thunder._score` is not mis-reading anything.

12-dataset means (×100), all recomputed from raw predictions:

| backbone | task | macro | weighted | micro/acc | bal-acc | binary-F1 on 2-class sets | **Waiv** |
|---|---|---|---|---|---|---|---|
| Phikon-v2 | knn | 70.28 | 75.05 | 75.01 | 70.70 | 66.14 | 74.00 |
| Phikon-v2 | linear | 76.54 | 80.57 | 80.57 | 77.47 | 73.76 | 79.30 |
| Phikon-v2 | simple_shot | 69.33 | 73.62 | 73.54 | 73.89 | 67.29 | 71.80 |
| Midnight | knn | 78.25 | 82.40 | 82.57 | 78.63 | 74.93 | 80.00 |
| Midnight | linear | 82.88 | 86.56 | 86.68 | 83.03 | 80.19 | 84.40 |
| Midnight | simple_shot | 70.64 | 76.03 | 75.61 | 72.61 | 66.79 | 71.50 |
| Virchow2 | knn | 80.87 | 84.28 | 84.48 | 81.36 | 77.92 | 82.90 |
| Virchow2 | linear | 83.25 | 86.42 | 86.46 | 83.48 | 80.99 | 84.80 |
| Virchow2 | simple_shot | 72.75 | 77.11 | 76.46 | 75.84 | 70.16 | 73.90 |

Waiv lies strictly *between* macro and weighted in all nine cells, but at a fraction of the
interval that ranges from 0.16 (Midnight few-shot) to 0.78 (phikon kNN). No metric
substitution is consistent. Weighted/micro overshoot on every cell; balanced-accuracy
undershoots on six of nine. **Not a metric.**

### H2 — kNN `k` / few-shot shot-count selection. **ELIMINATED as the cause.**

THUNDER persists only the val-selected `k` (varies per dataset: 1, 3, 5, 20, 30, 40, 50 all
appear), so there is no max-over-k to be had from the files. `simple_shot` keeps all of
{1,2,4,8,16} and we read 16, THUNDER's published shot count. Since Waiv's few-shot equals
the leaderboard **exactly on 7/7 models**, their shot protocol is THUNDER's default; this
axis carries no offset.

### H3 — adaptation setting (frozen vs LoRA). **ELIMINATED.**

Only `frozen/` exists under every task directory for all three base runs. Waiv §3.3 states
frozen. No LoRA results exist to have been confused with.

### H4 — dataset subset within the 12. **ELIMINATED.**

Exhaustive search over all 4,017 subsets of size 4–12, under macro / weighted /
balanced-accuracy, minimising max\|error\| against Waiv:

- Best subset shared by all 3 backbones **and** all 3 tasks: max\|err\| **0.84** (macro),
  0.73 (weighted), 2.27 (bal-acc). No subset reconciles the nine cells.
- Best subset per task, shared across backbones: max\|err\| 0.19–0.28, but the winning
  subsets are different, arbitrary, and mutually inconsistent across tasks
  (knn wants `{bracs, ccrcc, crc, esca, mhist, wilds}`; linear wants an 8-set;
  few-shot a different 8-set).
- Per (backbone, task) independently, errors of <0.01 are reachable — but that is 4,017
  candidates fitted to a target printed to 0.1, i.e. pure overfitting, and is reported here
  only so it is not mistaken for evidence.

**Size-weighted** (by `nb_test_samples`) means overshoot badly in the wrong direction
(phikon kNN 76.51, linear 82.86 vs Waiv 74.0 / 79.3), so weighting is not it either.

The *correct* subset answer turned out to be a **superset**, not a subset — see §1.

### H5 — scale / rounding / transcription. **ELIMINATED.**

`docs/waiv_published.json` stores THUNDER values on the 0–100 scale
(`"knn": 74.0`, `"segmentation": 66.5`), matching Waiv's Table 2 as printed; our collectors
carry 0–1 and `scripts/scoreboard.py:WAIV_THUNDER` re-states them on the 0–100 scale
consistently. No factor-of-100 confusion exists anywhere in the chain. The transcription is
faithful: all 7 comparable base rows agree with THUNDER's published leaderboard to ≤0.7,
which would be impossible under a transcription error.

---

## 3. Segmentation — NARROWED, not identified

**VERIFIED:** our 4-dataset segmentation means reproduce THUNDER's leaderboard closely on
all three backbones, and Waiv is the outlier.

| backbone | ocelot | pannuke | segpath_epi | segpath_lymph | **our 4-ds** | THUNDER LB | **Waiv** | Waiv − ours |
|---|---|---|---|---|---|---|---|---|
| Phikon-v2 (`base_cls`) | 80.01 | 60.79 | 69.46 | 60.65 | **67.73** | 67.4 | 66.5 | −1.23 |
| Midnight (`mbase_cls`) | 78.42 | 61.81 | 70.95 | 63.75 | **68.73** | 68.8 | 66.0 | −2.73 |
| Virchow2 (`vbase_cls`) | 79.45 | 62.78 | 70.64 | 63.17 | **69.01** | 69.3 | 68.2 | −0.81 |

Two things follow.

1. **The `support_2v4` flag is now stale for the base runs.** `collect_final5.py` averages
   only `{ocelot, pannuke}` (giving 70.40 for phikon, the "+3.9 above Waiv" cell in the
   comparability audit), but **all four** segmentation datasets have results on disk for all
   three backbones under the `*_cls` run names. Recomputed on the matched 4-dataset support,
   our base is 67.73 / 68.73 / 69.01 — i.e. we sit **1.2 / 2.7 / 0.8 above** Waiv, not 3.9 /
   4.1 / 2.9. Most of the apparent segmentation excess was the 2-vs-4 support, not a real
   difference.
2. **The residual is a Waiv-side re-run offset, cause not established.** The segmentation
   column is *identical* in the original and up-to-date leaderboards, so the roster change
   cannot touch it. Waiv sit 0.9–2.8 below THUNDER's published segmentation on all 7
   comparable models (mean −1.33, 0/7 exact) — the only task where they never reproduce the
   leaderboard exactly.

**INFERRED (consistent, not proven): mixed precision.** `collect_thunder.py:66` records that
Waiv's own mixed-precision re-run of phikon-v2 moves linear probing ≈−0.4 and segmentation
≈−0.9 against THUNDER's numbers. Observed for PHIKON2: linear **−0.40**, segmentation
**−0.90** — both to the digit. This also explains the residual pattern in §1b, where the
*deterministic* readouts (kNN, few-shot: nearest-neighbour and prototype rules on frozen
embeddings) reproduce the leaderboard exactly while the *trained* heads (linear probing
−0.19 mean, segmentation −1.33 mean) drift consistently downward. Midnight's segmentation
(−2.8) is larger than a precision effect alone comfortably explains, so this is offered as
the leading candidate, not a conclusion.

Under this reading the full story is additive and self-consistent:

```
classification  =  +1.7…+3.2 (SPIDER roster)  −0.0…−0.4 (AMP on trained heads)  → NET UP
segmentation    =   0        (no SPIDER seg)  −0.9…−2.8 (AMP / re-run)          → NET DOWN
```

---

## 4. What the prior docs said, and whether it holds

| claim | source | status |
|---|---|---|
| "Waiv's THUNDER base is on a different scale from both us and the THUNDER authors; a LEVEL comparison is invalid" | `scripts/scoreboard.py:98-107` | **HOLDS**, and is now explained: different roster. The two-base gain-ratio formula was the right defensive call. |
| "Not a dataset subset. Both use the same 12 PAPER_CLS datasets." | `baseline_comparability_audit.md` §2 | **REFUTED.** Ours is 12; Waiv's is 16. The audit tested subsets of our 12 and never tested a superset. |
| "Unexplained is the honest label." | `baseline_comparability_audit.md` §2 | **SUPERSEDED** for classification. |
| "different metric or adaptation setting (INFERRED, not established)" | `baseline_comparability_audit.md` §2 | **REFUTED** — see H1/H3. |
| "our base reproduces THUNDER's own paper to ~0.1" | `baseline_comparability_audit.md` §2 | **HOLDS**, reconfirmed (kNN 70.28 vs 70.14; LP 76.54 vs 76.46; 4-ds seg 67.73 vs 67.42). |
| "THUNDER: 16 of Waiv's 16" | `docs/CAVEATS.md:35` | **WRONG as stated.** We run 12 classification + 4 segmentation = the *paper's* 16. Waiv run 16 classification + 4 segmentation = 20. The coincidence of the number 16 masked the mismatch. |
| segmentation `support_2v4` mismatch | `scoreboard.py` (b), `collect_final5.PAPER_SEG` | **HOLDS as a defect, but is now fixable** — all four seg datasets exist on disk for all three base runs (§3). |

---

## 5. Consequences and what closes this

**Actionable, and cheap: the four SPIDER datasets are already on disk** at
`/data/ryan.kim/thunder/datasets/{spider_breast,spider_colorectal,spider_skin,spider_thorax}`.
Running kNN / linear / few-shot on them for `base_cls`, `mbase_clsmean`, `vbase_clsmean`
would convert §1's *reconstruction* (our 12 + THUNDER's SPIDER) into a *direct* measurement,
and would let every fine-tuned run be scored on Waiv's actual roster instead of a two-base
gain ratio. That is the single experiment that closes this file. (Not launched — this pass
is read-only.)

Until then:

- **Do not "fix" our base upward.** Our 12-dataset numbers are correct and match THUNDER's
  paper. The published-vs-ours delta is a roster difference, not a harness defect.
- **The two-base gain-ratio formula in `scoreboard.py` remains necessary**, and now has a
  stated reason rather than an unexplained one.
- **Retention claims are affected in a known direction.** SPIDER sets are easy (85–91 F1) and
  compress the dynamic range; a recipe's damage measured on the 12 hard sets will look
  *smaller* on Waiv's 16-set mean by roughly a factor of 12/16 = 0.75, before any
  SPIDER-specific effect. Any `pct_of_waiv` on THUNDER classification inherits this.
- **`docs/CAVEATS.md:35` should be corrected** from "16 of Waiv's 16" to "12 of Waiv's 16
  classification sets; 4 of their 4 segmentation sets".
- **`collect_final5.PAPER_SEG` should move from 2 datasets to 4**, since the results exist.

---

## Provenance

| artifact | path |
|---|---|
| raw THUNDER outputs | `/data/ryan.kim/thunder/outputs/res/<dataset>/<run>/<task>/frozen/outputs.json` |
| base runs read | `base_cls` (phikon-v2), `mbase_clsmean` + `mbase_cls` (Midnight), `vbase_clsmean` + `vbase_cls` (Virchow2) |
| THUNDER leaderboards (pinned) | `/admin/home/ryan.kim/pathopress/source_data/pinned/thunder/leaderboards.md` — sections "Up-to-date Rank-sum", "Original (paper) Rank-sum", "SPIDER", "Per-dataset linear probing" |
| Waiv Table 2 transcription | `/admin/home/ryan.kim/waiv/docs/waiv_published.json` |
| collector under audit | `/admin/home/ryan.kim/waiv/scripts/collect_thunder.py` (`_score`, `PAPER_CLS`, `PUBLISHED`, `PUBLISHED_TASKMEAN`) |
| prior verdict superseded | `/admin/home/ryan.kim/waiv/docs/baseline_comparability_audit.md` §2 |
| SPIDER data (unrun) | `/data/ryan.kim/thunder/datasets/spider_{breast,colorectal,skin,thorax}` |
