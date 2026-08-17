# Our improvement system vs. published Waiv — full comparison

Date: 2026-08-17. Companion to `FINDINGS_2026-08-16.md` (instrument analysis) and `RESULTS.md` (raw results).

Evidence tags: **MEASURED** (reproduced from disk) · **INFERRED** (derived from measured numbers) · **UNVERIFIED** (asserted but not reproducible) · **NOT COMPARABLE** (protocols differ).

---

## 0. The framing fact

**Published Waiv has no method section.** arXiv:2607.22861 (Filiot/Thaeter/Schmauch/Guillou, 2026-07-24) released gated weights and results but no loss, algorithm, corpus, hyperparameters, or code (`README.md:1-10`). Their two named releases are **Phaet** (from `owkin/phikon-v2`) and **Mascaret** (from `kaiko-ai/midnight`), plus fine-tuned variants of 8 other backbones.

**Our system is therefore not a reproduction of their method — it is a reconstruction of their results.** Ours: masked InfoNCE over PLISM co-registered scanner pairs (positives = same tile index, different acquisition condition; negatives from the anchor's own condition), applied via LoRA on all transformer blocks with the backbone otherwise frozen. PathoROB is never seen in training.

**Consequence for every comparison below: compare Δ to Δ, not absolute to absolute.** Where base models differ, absolute scores are not the question; the size of the improvement is.

Reference data: `docs/waiv_published.json` (34 KB, transcribed 2026-08-13 from the published PDF) — 20 models × {RI per dataset, HEST per cancer type, 6 THUNDER tasks, Patho-Bench 63-task matrix, Figure-1 composite rank}.

---

## 1. PathoROB robustness index — fully comparable

Same harness, same three datasets, base reproduced independently by us.

| Backbone | our base | Waiv base | our FT | Waiv FT | **our Δ** | **Waiv Δ** | gap closed |
|---|---|---|---|---|---|---|---|
| phikon-v2 | 0.4686 | 0.469 | **0.8080** (s1000) | 0.806 | **+0.3394** | +0.3370 | **~101%** |
| Midnight-12k | 0.7589 | 0.759 | **0.9080** (s500) | 0.924 | **+0.1491** | +0.1650 | **90.4%** |
| Virchow2 | 0.8582 | 0.858 | **0.9035** (s250) | 0.918 | **+0.0453** | +0.0600 | **75.5%** |

**MEASURED** from `runs/waiv-real-369043/`, `runs/waiv-midnight-369159/`, `runs/waiv-virchow2-375367/` `ri_curve.json`.

### 1.1 The base reproduction is genuine, not circular — verified

This matters because "base reproduces to ±0.0004" is what licenses every Δ-vs-Δ above. `ri_curve.json`'s `targets` block is a verbatim dump of the hardcoded `TARGETS` in `pathorob_adapter.py:53-67`, i.e. the *published* reference — so the curve file alone would be circular. **But the base measurement exists independently**, in standalone trees under `third_party/PathoROB/results/robustness_index/` (untracked by git, and not named `*_base*` for two of three backbones), each with a `results_summary.json` carrying 11 RI variants, ID/OOD AUCs, and per-center balanced accuracies over 229 MB of measured `.npz` features.

Three independent proofs it is real measurement:

1. **Two separate phikon-v2 base runs disagree at the ~7th decimal, not the 4th** (tcga 0.6187718 vs 0.6187714) with byte-identical camelyon — the signature of two float-summation orderings over real embeddings. A copied constant would be bit-identical everywhere.
2. **Virchow2's per-dataset base RIs cannot have been copied, because they were never published.** `pathorob_adapter.py:65` holds only `{"virchow2_base": {"avg": 0.858}}`; the per-dataset keys are deliberately absent. Yet our tree contains three measured per-dataset RIs (0.7989 / 0.8218 / 0.9541) that average 0.85824. **This is the strongest datapoint.**
3. A fourth backbone, `uni2h_clsmean`, was measured at 0.75667 vs published 0.757 and is quoted nowhere in our narrative — an echo-the-targets pipeline would have no reason to get it right.

**Corrected phrasing:** say "average RI agrees within 0.0005, and per-dataset agreement is within 0.0005 wherever published per-dataset values exist." The blanket "within 0.0004" flatters the evidence, since the average benefits from per-dataset errors cancelling.

### 1.2 Verdict

**TIE on phikon-v2, LOSE on Midnight and Virchow2.** We land on Phaet's number (difference 0.3× the 0.0070 RI floor). We fall genuinely short on the two stronger backbones by 2.1–2.3 floor units.

The shortfall is **entirely camelyon**: 0.8844 vs their 0.907 (Midnight), 0.9006 vs 0.935 (Virchow2). On tcga and tolkach_esca we are within 0.004–0.022 everywhere. Gap-closed falls monotonically with base strength (101% → 90% → 76%), tracking remaining headroom rather than architecture.

---

## 2. THUNDER — comparable on 4 of Waiv's 6 tasks

| Task | phikon Δ ours / Waiv | Midnight Δ ours / Waiv | Virchow2 Δ ours / Waiv |
|---|---|---|---|
| kNN | **+4.92** / +3.7 | **+2.19** / +1.7 | **+0.03** / −0.30 |
| linear probing | **+2.69** / +1.4 | **+1.24** / +0.2 | **+1.34** / +0.30 |
| few-shot | **+1.66** / +1.5 | **+5.74** / +3.7 | **+4.41** / +2.70 |
| segmentation | **−0.12** / −1.2 | **−0.33** / **+1.6** | **−0.12** / −0.20 |
| **mean over 4** | **+2.29** / +1.35 | **+2.21** / +1.80 | **+1.42** / +0.625 |

Our Δ ≥ Waiv's on **11 of 12** shared model×task pairs.

**But quote the sign test, not the means.** The means are graded against floors that are wide (§4): 32 of 36 classification model×dataset pairs improve, **p ≈ 2×10⁻⁶** (`RESULTS.md:165`). That is the load-bearing evidence, and it is strong. The +2.29/+2.21/+1.42 figures are not individually resolvable against per-task seed noise.

### 2.1 The two remaining tasks

**`calibration` was never actually missing.** It is not a separate THUNDER task — the task list at `third_party/thunder/src/thunder/benchmark.py:25` has no such entry. Calibration is computed *inside* `linear_probing`: `tasks/train_eval_probe.py:716` calls `compute_calibration_metrics` and merges ECE/MCE/SCE/ACE/TACE into the same `outputs.json` as f1. All six of our checkpoints already carry ECE for all 12 classification datasets; `collect_thunder.py` was simply dropping the keys.

**MEASURED.** Mean ECE ×100 over 12 classification datasets (lower is better):

| backbone | our base | our FT | **our Δ** | Waiv base | Waiv FT | **Waiv Δ** |
|---|---|---|---|---|---|---|
| phikon-v2 | 4.73 | 5.10 | **+0.37 (worse)** | 4.5 | 3.0 | **−1.5 (better)** |
| Midnight | 3.19 | 3.36 | **+0.17 (worse)** | 2.4 | 2.3 | **−0.1 (better)** |
| Virchow2 | 5.16 | 4.34 | **−0.82 (better)** | 3.6 | 4.2 | **+0.6 (worse)** |

**Calibration goes 1–2 against us.** Our fine-tuning slightly degrades calibration on phikon-v2 and Midnight while Waiv's improves it; we win only on Virchow2. Magnitudes align with the published column closely enough that mean-ECE×100-over-12-datasets is almost certainly their definition, but our base rows do not exactly reproduce theirs, so the honest comparison is Δ-vs-own-base.

**This corrects an earlier optimistic framing.** The hypothesis that the two unmeasured tasks would favour us is now half-refuted: calibration does not.

**`adversarial_attack` — MEASURED, and the metric's direction was being read backwards.**

**CORRECTION.** Earlier drafts of this document described Waiv's adversarial column as a catastrophic collapse. That was wrong. The column is **`drop/accuracy`** — the *drop* in accuracy under PGD attack, so **lower is better**. Identified by sweeping all 12 candidate metrics in `adversarial_attack/frozen/outputs.json` against Waiv's published base triple: `drop/accuracy` matches at mean absolute error **1.71** (Virchow2 31.5 vs 31.1, Midnight 35.5 vs 35.7), and decisively, only the drop metrics reproduce the published **rank ordering** — phikon-v2 is highest (41.9) and is also the weakest backbone on clean accuracy (80.6 vs 86–87), so it cannot top a post-attack score but naturally tops a drop. Verified in-file: `drop = clean − adversarial` in absolute points.

**So Waiv's "Virchow2 31.1 → 7.7" is their single strongest result** — a 75% reduction in adversarial accuracy drop — not a collapse.

PGD-linf, eps 1.5e-3, 5 steps (`tasks/adversarial_attack.py`). All 72 jobs COMPLETED.

**Base → FT, `drop/accuracy` ×100, 12-dataset mean (lower = better):**

| backbone | ours base | ours FT | **our Δ** | Waiv base | Waiv FT | **Waiv Δ** |
|---|---|---|---|---|---|---|
| Virchow2 | 31.5 | 28.7 | −2.9 (−9%) | 31.1 | **7.7** | **−23.4 (−75%)** |
| Midnight | 35.5 | 25.6 | −9.9 (−28%) | 35.7 | 23.2 | −12.5 (−35%) |
| phikon-v2 | 46.4 | **34.0** | **−12.5 (−27%)** | 41.9 | 38.8 | −3.1 (−7%) |

Clean accuracy is not traded away — it *rises* under our FT on all three (Virchow2 86.2→87.4, Midnight 87.2→87.4, phikon-v2 80.6→82.7).

**Verdict: our fine-tuning improves adversarial robustness on all three backbones — but this is not a place we win.**
- **Virchow2 — we lose badly.** 2.9 points of drop reduction vs their 23.4. Their 7.7 is in a different league from our 28.7.
- **Midnight — draw at best.** We end at 25.6 vs their 23.2; our relative gain (−28%) is comparable to theirs (−35%) but we remain worse in absolute terms.
- **phikon-v2 — a genuine but modest win on Δ.** We cut 12.5 points where they cut 3.1, though our worse base (46.4 vs 41.9) means our endpoint (34.0) only edges theirs (38.8).

Superseded (retained to show what was corrected) — the published column, previously misread as a collapse:

| backbone | Waiv base → FT adversarial |
|---|---|
| Virchow2 | 31.1 → **7.7** |
| Midnight | 35.7 → **23.2** |
| phikon-v2 | 41.9 → 38.8 |

72 jobs (6 checkpoints × 12 datasets, IDs 384466–384537) are running. This remains the one place a full 6-task comparison could still move in our favour, at zero training cost — but it is now the *only* one, not two.

---

## 3. HEST — no result on our side

| Backbone | protocol | our base | Waiv base | our Δ | Waiv Δ | comparable? |
|---|---|---|---|---|---|---|
| phikon-v2 | 9-task, `cls` | 0.37470 | 0.3747 | +0.0047 (s1000), +0.0098 (s3500 best) | **+0.0196** | **YES** |
| Virchow2 | 9-task, `clsmean` | 0.40327 | 0.4034 | +0.0050 (s250) | **+0.0101** | Δ-vs-Δ only |
| Midnight | 9-task, `clsmean` | 0.41210 | 0.3952 | +0.0011 (s500), +0.0035 (s250 best) | **+0.0215** | **NO on absolutes** |

**Midnight absolutes are not comparable.** Our base is **+0.0169 above** Waiv's — already near Mascaret's 0.4167 — because Waiv never state a pooling protocol for Midnight and 0.3952 is plausibly CLS-only against our CLS+mean. A test of exactly this is running (job 384456, Midnight base under `cls`, fresh exp_code `midnight_base_cls_9task_v1`).

### 3.1 Verdict — say this precisely

All three of our deltas sit **inside** the 0.0075 seed bar. All three of Waiv's sit **outside** it.

**The correct statement is: "our fine-tuning preserves HEST; Waiv's improves it."** Not "we improve HEST by less" — we have no measured improvement to compare.

Four independent experiments say this is a property of our recipe, not a tuning failure: it survives every checkpoint (best-in-range +0.0098 phikon, +0.0035 Midnight), LoRA ranks 8–128, full fine-tuning, and an added frozen-teacher retention-KL term in which every λ>0 was strictly dominated by λ=0.

**Caveat that cuts both ways:** the 0.0075 bar uses the centred construction shown defective in `FINDINGS_2026-08-16.md:§1.4` and has not been recomputed under `offset_2se`. It will likely widen — which would weaken Waiv's HEST claims too, since theirs are graded against the same bar.

---

## 4. Statistical grading

Floors: RI 0.0070 (n=2, a **bound** not an estimate); HEST 2SE 0.0075 (centred, likely understated); THUNDER floors now persisted in `docs/thunder_seed_floor.md`, and **under the correct `offset_2se` construction every THUNDER protocol is 0.2–0.6 noise units — as blind as HEST.**

| Comparison | value | floor | ×floor | verdict |
|---|---|---|---|---|
| RI Δ, phikon | +0.3394 | 0.0070 | **48×** | **REAL** |
| RI Δ, Midnight | +0.1491 | 0.0070 | **21×** | **REAL** |
| RI Δ, Virchow2 | +0.0453 | 0.0070 | **6.5×** | **REAL** |
| RI shortfall vs Waiv, phikon | +0.0024 | 0.0070 | 0.3× | **TIE** |
| RI shortfall vs Waiv, Midnight | −0.0159 | 0.0070 | 2.3× | real shortfall |
| RI shortfall vs Waiv, Virchow2 | −0.0147 | 0.0070 | 2.1× | real shortfall |
| HEST Δ, ours (all three) | +0.0047 / +0.0011 / +0.0050 | 0.0075 | 0.15–0.68× | **INSIDE NOISE** |
| HEST Δ, Waiv (all three) | +0.0196 / +0.0215 / +0.0101 | 0.0075 | 1.3–2.9× | theirs clear it |
| THUNDER LP Δ, ours | +2.69 / +1.24 / +1.34 pp | 1.56 pp (cls, offset) | 0.8–1.7× | weak |
| THUNDER kNN Δ, Virchow2 | +0.03 pp | 2.97 pp | ~0 | **null** |
| THUNDER classification sign test | 32/36 improve | — | p ≈ 2×10⁻⁶ | **REAL** |

**PathoROB RI is the only axis on which our effect is unambiguously large.** It is also the only readout with usable dynamic range (~2–3 nu). Everything else is graded against instruments that cannot separate arms.

---

## 5. Win / lose / tie

- **PathoROB — TIE on phikon-v2, LOSE on Midnight and Virchow2.** Shortfall is real (2.1–2.3× floor) and localised entirely in camelyon.
- **THUNDER classification — WIN.** 11/12 shared pairs, sign test p ≈ 2×10⁻⁶. Caveat: 4 of their 6 tasks, and our absolute levels stay below theirs even where our deltas are larger (fp32 vs their mixed precision).
- **THUNDER segmentation — LOSE on Midnight** (−0.33 vs their +1.6), tie elsewhere. Honest characterisation: our segmentation is flat-to-negative on **all three** backbones (3 of 12 dataset pairs improve, sign test p ≈ 0.15), so this is **"classification yes, segmentation no, on every backbone tested"** — a recipe property, not a Midnight-specific defect. *(Note: Waiv's own fp32 Table-5 re-run gives Mascaret segmentation −1.2, so their +1.6 may itself be a mixed-precision artifact — we deliberately do not lean on this.)*
- **HEST — NO RESULT, 3 of 3.** Ours preserve; theirs improve.
- **THUNDER calibration — LOSE, 1–2.** Our fine-tuning degrades ECE on phikon-v2 (+0.37) and Midnight (+0.17) where Waiv's improves it; we win only on Virchow2 (−0.82). Data already existed on disk (§2.1).
- **THUNDER adversarial — MEASURED. We improve on all three backbones, but LOSE overall.** The metric is `drop/accuracy` (lower better), not a post-attack score, so Waiv's 31.1 → 7.7 on Virchow2 is their *best* result, not a collapse. Our Δ: Virchow2 −2.9 vs their −23.4 (clear loss), Midnight −9.9 vs −12.5 (draw), phikon-v2 −12.5 vs −3.1 (modest win). See §2.1.
- **Patho-Bench — NOT PLAYED.** Waiv's Mascaret is rank 1 of 20 (58.0). We have no number.

---

## 6. What is missing, and what it costs

**Free (eval-only, no training budget):**

1. ~~THUNDER calibration + adversarial~~ **DONE.** Calibration already existed on disk (it is computed inside `linear_probing`) and goes 1–2 against us. Adversarial: all 72 jobs completed; we improve on all three backbones but lose the comparison, decisively on Virchow2. Neither was the hoped-for opening.
2. **Midnight HEST under `cls`.** Converts the Midnight HEST row from "absolutes incomparable" to a real head-to-head. *In flight (job 384456).*
3. **Recompute the HEST 0.0075 bar under `offset_2se`.** Zero GPU. Regrades every HEST verdict on both sides.
4. **Locate/confirm base PathoROB artifacts.** *Done — §1.1, claim survives.*

**Costs training budget (~5 runs remaining):**

5. **Seed replicates of the three headline arms.** Every reconstruction result is n=1; every floor here is a 2-run bound imported from a different arm family. **Highest-value spend: 2 extra seeds on Midnight** (~3.4 h each) — the one backbone where we lose on both RI and segmentation — converting "−0.0159 = 2.3× a bound" into an actual interval.
6. **A segmentation-targeted arm on Midnight.** 1 run. Speculative: segmentation is negative on all three backbones, so this tests the recipe rather than the checkpoint.

**Do not spend budget on HEST.** Four independent experiments say the gap is a recipe property, and the instrument is blind at ~0.5 nu. No training run can resolve a +0.0011 delta against a 0.0075 (probably wider) bar.

**Drop or relabel the Figure-1 rank claim.** README's "within 1–2 rank points (45 vs 44, 12 vs 11, 20 vs 18)" is **INFERRED**: it assumes our Patho-Bench rank equals theirs, and Patho-Bench is one of three components in that composite. We have never run it.

---

## 7. One-paragraph summary

Against a system whose method was never published, we reconstruct **essentially all** of Waiv's robustness gain on phikon-v2 (101% of their Δ), **90%** on Midnight, and **76%** on Virchow2 — and PathoROB RI is the only benchmark here with the dynamic range to say that with confidence (6.5–48× its noise floor). On downstream classification we improve *more* than they do on 11 of 12 shared task×backbone pairs, with a sign test at p ≈ 2×10⁻⁶. Against that, we lose on three axes: we do not reproduce their HEST gain (ours preserves where theirs improves, and four independent experiments locate that in the recipe rather than in tuning); we are flat-to-negative on segmentation across all three backbones; and our fine-tuning slightly degrades calibration on two of three backbones where theirs improves it. Adversarial robustness, once the hoped-for opening, closed against us: the column is a *drop* metric (lower better), so their Virchow2 31.1 → 7.7 is their strongest result rather than a collapse, and while our fine-tuning does improve adversarial robustness on all three backbones without trading clean accuracy, we lose that comparison decisively on Virchow2. **The defensible headline is narrow and strong: we match or nearly match a closed system's robustness gains and beat it on downstream classification, while trailing it on retention, segmentation, calibration, and adversarial robustness.** There is no remaining unmeasured axis that could reverse this.
