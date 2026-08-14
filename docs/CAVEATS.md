# Caveats and reporting discipline

Every claim in [`RESULTS.md`](RESULTS.md) is conditioned on this file. These are not
formalities: each entry below records a trap that was actually hit or actually misled during
this reproduction.

---

## 1. Caveats

- **n=1 seed** throughout, no error bars — but this is **protocol parity with Waiv, not a
  shortfall against them**. Per their §3.3 (arXiv:2607.22861), THUNDER is "evaluated on frozen
  features following the default protocol" with no seed repetition, HEST follows "the default
  protocol and implementation", and PathoROB states no repetition at all. Patho-Bench is the
  only benchmark where they repeat — "Each task is run three times, and we report the mean
  over all data folds and random seeds" — and that is precisely the benchmark we dropped. So
  n=1 matches their protocol on all three benchmarks we run. The real asymmetry is **model
  count, not seed count**: their significance claims (e.g. p=7.5e-3, one-sided Wilcoxon
  signed-rank on THUNDER rank sums) aggregate across 10 model pairs, which we cannot replicate
  at 2 backbones no matter how many seeds we ran. We make no significance claim.
- **fp32 vs their mixed precision.** Each Δ is internally precision-consistent, so Δ-vs-Δ
  is valid; **absolute levels are not comparable** across the two studies.
- **Checkpoint selection is not neutral.** THUNDER and HEST use step 1000 (phikon-v2) and
  step 500 (Midnight), chosen because they were the best *PathoROB* checkpoints. A full HEST
  sweep over every checkpoint was run on **both** backbones ([`RESULTS.md`](RESULTS.md) §2): phikon-v2's HEST optimum is
  step 3500 (+0.0098, still climbing at the end of the trained range), Midnight's is step 250
  (+0.0035, decaying thereafter and negative by 1250). Robustness and retention peak at
  different steps and in *opposite* orderings on the two backbones, so any single-checkpoint
  headline understates one axis — but even the best checkpoint stays far short of Waiv, so
  the gap is not a selection artefact.
- **Coverage.** All three benchmarks now cover both backbones. THUNDER: 16 of Waiv's 16
  datasets (12/12 classification, 4/4 segmentation), so the [`RESULTS.md`](RESULTS.md) §2 segmentation average is
  like-for-like against theirs. HEST: phikon-v2 (`cls`) and Midnight (`clsmean`), base and
  fine-tuned. PathoROB: both. The only benchmark dropped is Patho-Bench (~8 TB of WSIs, no
  traceable target number).
- **The Tier-1 probe tripwire is not an early-stopping signal.** `scripts/probe_follow.py`
  detects collapse; it does not predict PathoROB RI, in either direction. On the full-FT run
  it fired its scanner-regression signal from step 300 while RI was still climbing — acting on
  it would have discarded the best checkpoint (0.8007 @ 500). On the LoRA run the opposite
  held: heldout cross-scanner separation kept rising (base 0.376 → 0.394 @ 500 → 0.439 @ 2500)
  while RI fell after step 1000. Its thresholds are diagnostics, not actionable. Relatedly, **matched cosine and
  top-1 mislead under collapse**: full FT drove heldout cross-scanner matched cosine
  0.741 → 0.963, but the random-pair baseline climbed just as fast, 0.365 → 0.597. Report
  rank-based top-1 and separation, never matched cosine alone.
- **Regressions.** `tcga_uniform` regresses on both backbones and on every probe
  (phikon-v2 −8.2 kNN / −5.7 lin / −7.3 few-shot; Midnight −2.6 / −2.9). Consistency
  across independent probes rules out a local-neighbourhood artefact — information is
  genuinely lost. `bach` regresses on Midnight only (−2.1 / −5.6 / −8.6) while it *improves*
  on phikon-v2. Both are unexplained and should be understood before any publication claim.
- **PathoROB curves are plateaus, not climbs.** Both backbones reach their peak by the
  first or second checkpoint and then flatten (phikon-v2 0.803–0.808 across steps
  500–3000; Midnight 0.898–0.908 across 250–1500). Training longer buys nothing on the
  headline metric.
- **camelyon's base RI of 0.019** is pathologically low, so its headline jump is the least
  surprising of the three PathoROB datasets.
- **camelyon is saturated on any probe other than RI, and it dominates avg RI.** Untuned
  phikon-v2 already scores **1.0000** slide-level LOCO on camelyon and 0.9833 tile-linear LOCO
  ([`RESULTS.md`](RESULTS.md) §12.6) — no quality headroom. Meanwhile a covariance decomposition
  over the six §12.2 arms attributes **86%** of the between-arm variance in *avg* RI to camelyon
  (8% tcga, 5% tolkach_esca). Avg RI is therefore largely a readout on the one dataset with
  nothing left to measure. **Prefer the per-dataset RI table to the average for selection.** This
  is a concern about avg-RI's *composition*; RI itself is a valid transfer metric (§12.5).
- **Never quote a tile-level kNN number without "same-slide excluded".** On tcga, excluding only
  the query tile gives base kNN@1 = 0.9983 on a 4-class problem against 0.7777 with same-slide
  neighbours removed — a **22.1-point** artifact, because tcga has exactly 30 tiles per slide.
  Camelyon shows the same defect at −5.7 points. **PathoROB's own RI is clear of this** — it
  filters neighbours by `slide_id` before the SS/SO/OS/OO counts are built
  (`robustness_index_utils.py:511-529` → `robustness_index.py:172`), verified in §12.4. The
  leak was in our own probes, not in the benchmark.
- **A seed floor from two runs is a BOUND, not an estimate, and can be off by an order of
  magnitude.** The tcga slide-LOCO floor has moved **0.0016 → 0.0104 → 0.0214** as better
  measurements arrived, and it is still only a bound. It is not a constant: it varies by step and
  by C (at step 500, `ctrlseed` differs from `ctrl` by **0.0214** at C=1e-2 — two runs identical
  apart from the random seed). **Directions are established in this doc; magnitudes are not**,
  pending a 3–4 seed study that would give an SD instead of a two-point bound. Treat every
  "×floor" as a scale reference, never a p-value.
  This is the **third** methodology failure of the same shape, and all three shrank a denominator
  and made effects look realer than they were: (1) `max` over a hyperparameter grid, (2) a
  difference of averages where per-task errors cancel, (3) a two-run floor. **When a result
  depends on a ×floor ratio, interrogate the floor first.**
- **Do not apply the avg-RI floor (0.0070) to the other PathoROB metrics** — `ID_performance` (0.0017), `OOD_performance` (0.0022), `prediction_performance`
  (0.0020) and `balanced_accuracy` (0.0014) are 3.5–5× tighter, because camelyon's own RI floor
  (0.0143) dominates the average. Judging those metrics against 0.0070 understates real
  differences by ~3.5×. Per-metric floors: §12.3.
- **NEVER report a linear probe as `max` over the regularisation grid.** This is the most
  important methodology note in this file. The `readout_*.json` probes were reported as the max
  over C ∈ {1e-4 … 1}, selected on the metric being reported. That is not merely optimistically
  biased — **different arms and checkpoints peak at different C, so it manufactures effects that
  exist at no fixed C.** A "max-over-C seed floor" compares two differently-selected models and is
  not a floor at all; on slide-LOCO it read 0.0016 against a true fixed-C floor of **0.0104**
  (6.7× too tight), and on tile-linear LOCO 0.0004 against 0.0111 (**26.7×**). Ratios built on
  those denominators were inflated by the same factors and have been retracted
  ([`RESULTS.md`](RESULTS.md) §12.0). Rules: read probes **at fixed C**; compute the seed floor
  **at the same fixed C** as the comparison it scales; prefer **sign agreement across the whole C
  grid** ("worse at 5 of 5 C") to any single ×floor number; and scale against the **widest**
  fixed-C floor, since per-C floors go degenerate (one is 0.0000, yielding a nonsense +227×).
  Unaffected: kNN (k=1 is a fixed choice, not a max), silhouette, the RI/PathoROB metrics, HEST.
- **NEVER take a HEST seed floor from the difference of 5-task averages.** The naive
  |ctrl − ctrlseed| on the average is **0.0003**, and it is that small only because per-task
  errors **cancel** (SKCM +0.0061, COAD −0.0022, READ +0.0101, PAAD −0.0011, LUNG −0.0116). The
  honest floor comes from the per-task spread: SD 0.0084 → **SE 0.0037, 2 SE = 0.0075**, 25× the
  naive figure. This generalises beyond HEST: **an average-of-averages difference can look tiny
  purely through cancellation.** Always derive the floor from per-task dispersion.
- **On HEST, only the two ends of the table separate.** Against 2 SE ([`RESULTS.md`](RESULTS.md)
  §12.2): SPLIT +2.4 SE and MEANONLY −2.1 SE are **real**; every arm beats untuned base at
  −3.5 SE. GeM at +1.9 SE is **marginal**. **CTRL / CTRLSEED / GRID2 / GRID49 span 0.0019, a
  quarter of one SE — do not rank within that group.**
- **HEST protocol, exact wording: "2–3 coarse, slide-disjoint folds per task."** 11 folds total
  (SKCM 2/2 samples, COAD 2/4, READ 2/4, PAAD 3/3, LUNG 2/2), all verified train/test disjoint on
  `sample_id`. It is **not** leave-one-patient-out — the `hest_adapter.py:23` docstring is wrong.
  **Patient-level disjointness cannot be verified from the local data; slide-level disjointness is
  verified.** There is no donor identifier anywhere under `/data/ryan.kim/hest_bench/`, so the
  question is closed rather than pending — say **slide**-disjoint and do not overclaim, and do not
  file it as something to go and check. Nothing rests on it: the held-out-slide mechanism below
  needs only slide-disjointness. The folds are coarse (COAD fold 0 trains on one slide, tests on
  three); only PAAD is a genuine leave-one-out.
- **HEST magnitudes are upper bounds; the ranks are not affected.** The 5-task subset was chosen
  because those tasks carry the largest |Δ| vs base in a **prior, different arm family**
  (ft500–ft3500, CLS pooling) — `hest_arms.sbatch:28-31`. That selection was blind to the current
  arms' ordering and to RI, so it inflates absolute deltas without touching rank correlations.
  Also read the per-task table, not the average: **READ regresses below base for every tuned arm**
  and LUNG is flat for ctrl; the gains concentrate in SKCM and COAD. And never compare any of it
  to the published phikon-v2 **0.3747** — that row is phikon-v2 *CLS*, ours is `clsmean`
  (`scripts/run_hest.py:138-142`).
- **`confounder_insensitivity` predicts external performance better than avg RI.** Over n=7 arms,
  HEST correlates +0.873 with CI and +0.713 with avg RI (Pearson); `ID_performance` −0.919,
  `OOD_performance` −0.915, `prediction_performance` −0.900 all run the *wrong* way. The pattern
  survives dropping the MEANONLY outlier. Mechanism: HEST predicts on **held-out whole slides**,
  where slide-identity features cannot transfer, while the PathoROB kNN metrics are computed
  inside a neighbour table where they help. Prefer CI to avg RI as a single-scalar proxy. n=7 —
  the p-values are fragile.
- **RI selects a configuration; whether it selects a checkpoint is untested.** The earlier
  "ctrl peaks early then degrades" reading is **retracted** — ctrl@250 = 0.8792 was a max-over-C
  artifact, and at fixed C the direction flips with C (C=1e-2/1e-1 peak at 250, C=1e-4 reverses,
  C=1.0 peaks at 500), with trajectory spread 0.25×–1.4× the floor. **The ctrl trajectory is flat
  within noise.** Do not write "training over-specialises". On the HEST side, step 500 is *worse*
  than step 1500 for **both** arms whose RI peaks at 500 (SPLIT −0.0063, GeM −0.0023) — neither
  clears 2 SE, so it is not a finding, but **two independently trained arms reproduce the same
  divergence in the same direction**, which is more than an anecdote. **RI curves should not be
  used as checkpoint selectors** (§12.6, §12.2).
- **`k_opt` from PathoROB is not "the optimal k".** `filter_out_query_case_from_neighbors:527`
  truncates every query's neighbour list to the **global minimum** survivor count across all
  queries, so the effective neighbour count is capped by the single worst-case query. It affects
  all arms identically and biases no comparison, but the reported k (11 camelyon, 61 tcga) should
  not be quoted as a tuned value.


---

## 2. Reporting discipline (`PLAN.md` §6)

- Cross-stain and cross-scanner **separately** — the composite hides the hard axis.
- Never cosine similarity alone (PLIP: 0.878 cosine at 0.054 top-10).
- Any PLISM number is a **training diagnostic**. Label it; never print it next to a
  leaderboard number.
- Retention (HEST / THUNDER) alongside **every** robustness claim, as a pair. Forgetting is
  the default outcome here, not a tail risk.
