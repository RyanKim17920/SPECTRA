# Robustifying Phikon-v2 via registered-pair contrastive fine-tuning
### A PHAET-class reproduction attempt

---

## 0. Situation

Waiv released PHAET (fine-tuned `owkin/phikon-v2`) and MASCARET (fine-tuned `kaiko-ai/midnight`) with a paper — **arXiv:2607.22861**, *Robustifying pathology foundation models via fine-tuning* (Filiot, Thaeter, Schmauch, Guillou; Waiv + TUM; 24 Jul 2026).

**Verified by reading the paper directly:** it has *no method section*. Structure is Intro / Related work / Experimental setup (Backbones, Robustness eval, Performance eval) / Results / Understanding the robustness gains / Conclusion. There is no loss equation, no algorithm box, no training corpus, no hyperparameters, no methods appendix. No code released; weights only, gated, non-commercial.

The **only** disclosed constraints on the recipe:
- "lightweight, **label-free** fine-tuning"
- deep adaptation: it "does not merely re-tune the output layer... instills scanner invariance progressively throughout the network"
- not teacher-distillation (dismissed in related work as "requires a robust teacher model and is computationally expensive")
- compute: IDRIS (GENCI 2026-A0201012519) + EuroHPC MareNostrum 5

So this is **not** a reproduction of a published recipe — there is no published recipe. We are filling a gap Waiv left open deliberately, and measuring against their released numbers.

### Where our approach sits in their own related-work taxonomy
They cite "contrastive losses over co-registered scanner pairs [5, 24, 45]" — including **ScanGen** (Carloni et al., MICCAI MedAGI 2025, = arXiv:2507.22092) — and dismiss the whole family with one sentence: *"These downstream methods all keep the backbone frozen."*

**Our bet is that the loss family is right and frozen-backbone was the limitation.** We apply registered-pair contrastive learning *to the backbone*. That is exactly the gap their sentence describes.

Two paper findings support this:
- **Fig 3:** on PLISM, scanner shift is "a simple, near-linear offset in feature space" — per-scanner clouds keep shape, largely translated. Structured, low-dimensional nuisance is what contrastive collapse handles well.
- **Fig 4:** but feature-level correction is "not deep enough to remove it entirely" — invariance must build across depth. Base H-Optimus-0 develops cross-scanner matching only in the last few blocks; fine-tuned reaches equal quality ~8 blocks earlier, mAP **0.91 → 0.99**.

---

## 1. Experimental design

The key structural insight: **PLISM is not Waiv's benchmark.** It appears only in Section 5 as a qualitative PCA figure. Their robustness axis is PathoROB; SCORPION provides a retrieval curve. That frees PLISM to be our *training* set with no contamination of the headline metric.

| Role | Dataset | Why |
|---|---|---|
| **Train** | PLISM — 7 scanners × 13 stains × 16,278 Elastix-aligned tiles, 46 TMA organs | Registered pairs = exact positive supervision. CC-BY-4.0, ungated. |
| **Eval — robustness (primary)** | PathoROB (TCGA / Camelyon / Tolkach) | Directly comparable to their Table 1. Never touched in training. |
| **Eval — robustness (retrieval)** | SCORPION, 480 samples × 5 scanners | Metric closest to our training objective; target mAP 0.91→0.99 shape. |
| **Eval — retention** | HEST, THUNDER, Patho-Bench | Detects the forgetting failure mode. Non-negotiable. |
| **Diagnostic only** | PLISM top-k retrieval | We train on it → **NOT leaderboard-comparable.** Never report next to H0-mini's 0.541. |

### Targets (Waiv Table 1, Phikon-v2 → Phaet)

| | TCGA RI↑ | Camelyon RI↑ | Tolkach RI↑ | **Avg RI↑** | HEST Pearson↑ | THUNDER ranksum↓ | Patho-Bench↑ |
|---|---|---|---|---|---|---|---|
| Phikon-v2 (base) | 0.619 | 0.019 | 0.768 | **0.469** | 0.3747 | 97 | 54.1 |
| Phaet (their FT) | 0.785 | 0.702 | 0.932 | **0.806** | 0.3943 | 83 | 55.8 |

Camelyon 0.019 → 0.702 is the headline. Their 5 Dutch centers were **unseen during their fine-tuning** (stated in §4.1), so it is a fair held-out target for us too.

Phikon-v2 is the right base: weakest starting point, largest published gain, cheapest (ViT-L/16, 1.21 GB). Their own framing — "fine-tuning acts as an equalizer... gains are largest for the least robust base models."

---

## 2. Method to implement

**Objective:** InfoNCE over PLISM registered pairs, full-depth backbone adaptation, label-free.

- **Positives:** same tile location, different (scanner, stain) condition. Real images, not synthetic augmentation — this is what PLISM's registration buys.
- **Negatives — the one load-bearing detail.** Draw negatives from the *same* (scanner, stain) as the anchor. If negatives span conditions, "different scanner" becomes a partially-correct shortcut for "different tile" and the objective rewards *retaining* acquisition signal. This is what ScanGen's "different specimen, same scanner" repulsion term encodes; it carries over.
- **Projection width:** ScanGen used hidden 48/96 for binary MIL. Far too narrow for retrieval among 16k tiles — use 512+, or train on backbone output directly.
- **Anti-forgetting (ours, a deliberate divergence — Waiv implies they don't distill):** LoRA across *all* blocks first, not full FT. Bounds drift, cuts memory, merges to full weights afterward. Full FT is the escalation if LoRA underfits. Optional: TCGA replay tiles, frozen-teacher anchor.

Head-only tuning is ruled out by their Fig 4.

---

## 3. Risks, honestly

1. **Forgetting is the default outcome, not a tail risk.** Phikon-v2 saw 456M tiles / 60K WSIs (PANCAN-XL). PLISM is 46 TMA organs from **one institution** (Univ. Tokyo). Pure contrastive FT on that will buy invariance and shed general biology. HEST/THUNDER/Patho-Bench are the only detectors — a robustness win that costs retention is a failed reproduction. Waiv report regressions themselves (H0-mini on THUNDER, GenBio-PathFM on HEST).
2. **Transfer of invariance is the whole bet.** PLISM's acquisition manifold (7 scanners, 13 stains, one lab) must generalize to Camelyon's (5 Dutch centers, 3 scanner types, different tissue). Waiv demonstrated theirs did; their corpus is undisclosed and plausibly far broader.
3. **16,278 tile locations is a small instance-discrimination set** for backbone training. Memorization of tile identity won't show in training loss. Held-out-*condition* splits are the only check.
4. **No recipe means hyperparameter search**, not a single run. LR / steps / LoRA rank / temperature all unknown.
5. **PathoROB's own limitation**, per their conclusion: the index "does not directly measure downstream robustness under domain shift for specific tasks."

---

## 4. Phases

**Phase 1 — Setup & verification (day 1)**
1. **[USER ACTION]** Request HF access to `wearewaiv/phaet` — gated, manual, needs institutional email + ORCID. Not blocking, but it's the ceiling reference and approval takes days. Skip MASCARET.
2. Pull ungated bases: `owkin/phikon-v2` (1.21 GB), optionally `kaiko-ai/midnight` (4.55 GB).
3. **Verify PLISM registration indexing against a real `.h5`.** The card reports both "16,278 tiles/slide" and "3,417 aligned groups" — these don't reconcile, and the entire pair sampler depends on which is true. If tile index *i* corresponds across all 91 files, sampling is trivial. Getting this wrong silently trains on unregistered pairs. **Blocking.**
4. Stand up PathoROB + SCORPION eval harnesses.

**Phase 2 — Reproduce the baselines (day 2–3)**
5. Run base phikon-v2 through our PathoROB harness. **Gate: reproduce Avg RI 0.469 (Camelyon 0.019).** If we can't, the harness is wrong and nothing downstream means anything.
6. Frozen-feature probe as headroom lower bound + pipeline validation: extract once (~5 GB), fit a linear/MLP correction. Their Fig 3 predicts this gets partway; Fig 4 predicts it plateaus below backbone FT. Cheap, and it de-risks the expensive run.

**Phase 3 — Backbone fine-tuning (week 2)**
7. LoRA-all-blocks + InfoNCE on PLISM registered pairs. Hold out 2 of 7 scanners and 3–4 of 13 stains (ScanGen's ablation: gains converge at 5 training scanners; 3 still gives −27% CoV).
8. Sweep LR / rank / temperature / steps. Checkpoint often; evaluate retention at every checkpoint, not just at the end.
9. Escalate to full FT if LoRA plateaus below target.

**Phase 4 — Evaluation (week 3)**
10. PathoROB (primary) + SCORPION retrieval, vs. the Table-1 targets.
11. HEST / THUNDER / Patho-Bench retention.
12. If PHAET access lands: run it through the identical harness. Gives the ceiling on this base and tells us what fraction of achievable gain we captured.

---

## 5. Resources

**Compute:** SLURM, 8 nodes × 8 H100 80GB (`main`/`n` partitions). ViT-L/16 @ 224px with large-batch InfoNCE fits comfortably; LoRA more so.

**Storage** (verified live via HF API):

| Artifact | Size | Gated |
|---|---|---|
| `owkin/phikon-v2` | 1.21 GB | no |
| `kaiko-ai/midnight` | 4.55 GB | no |
| `wearewaiv/phaet` | 1.22 GB | **manual** |
| `wearewaiv/mascaret` | 4.55 GB | **manual** |
| `owkin/plism-dataset` (91 × .h5) | ~224 GB | no, CC-BY-4.0 |
| `owkin/plism-dataset-tiles` (293 parquet) | ~146 GB | no, CC-BY-4.0 |

Models total ~11 GB — non-issue. Backbone training needs raw tiles resident: put the 224 GB on **`/data`** (7.6 TB free), never `/admin` (1.4 TB free).

---

## 6. Reporting discipline

- Report cross-stain and cross-scanner **separately**; the composite hides the hard axis (cross-stain is ~4× harder).
- Never report cosine similarity alone — PLIP scores 0.878 cosine at 0.054 top-10.
- Any PLISM number is a training diagnostic and must be labelled as such.
- Report retention alongside every robustness claim, always as a pair.
