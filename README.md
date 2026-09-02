# waivphaet

A reproduction attempt of Waiv's robustness fine-tuning (arXiv:2607.22861), which released
**Phaet** (from `owkin/phikon-v2`) and **Mascaret** (from `kaiko-ai/midnight`) as gated
weights with **no method section** — no loss, no algorithm, no corpus, no hyperparameters,
no code. The recipe here is reconstructed from the one gap their related-work section
leaves open: they dismiss "contrastive losses over co-registered scanner pairs" because
"these downstream methods all keep the backbone frozen", so we apply that loss family *to
the backbone* — masked InfoNCE over PLISM co-registered pairs (positives = same tile index,
different acquisition condition; negatives drawn from the anchor's own condition), LoRA on
all transformer blocks. PathoROB is the primary metric and is never seen in training.

---

## Results are not in this file

The tables that used to live here were written 2026-08-19, against three backbones, a
250/500/1000 checkpoint grid, the since-retired `confounder_insensitivity >= 0.75` stopping
rule, and a 4-task THUNDER roster that has since been shown to be the wrong denominator.
`docs/README.md` classifies every one of those numbers as uncitable. Rather than leave a
plausible-looking stale table at the front door, it is gone.

The current state is five backbones (`phikon-v2`, `midnight`, `Virchow2`, `H-Optimus-0`,
`UNI2-h`), a 50-step checkpoint grid, a parameter-free 1-SE stopping rule on PathoROB
avg RI, and THUNDER on Waiv's full 16-dataset roster. Several cells are still filling in.

**For any number, read the generated scoreboard, not prose:**

```
./.venv/bin/python scripts/final_scoreboard.py    # writes docs/final_scoreboard.md
```

Every cell in it is read from disk at generation time; `MISSING` means the metric is not on
disk for that cell and is never substituted from another checkpoint, step, or arm. Runs that
have not plateaued under the stopping rule are reported `NOT SELECTED` rather than silently
graded at their last checkpoint.

## Start here

| file | contents |
|---|---|
| [`docs/README.md`](docs/README.md) | **The doc map.** 24 documents accumulated here; this says which three are current and which numbers must never be pasted into the paper. Read before opening anything else in `docs/`. |
| [`docs/RUNBOOK.md`](docs/RUNBOOK.md) | Current recipe, commands and config — the single source of truth for how to run this. |
| [`docs/final_scoreboard.md`](docs/final_scoreboard.md) | Generated scoreboard: the graded RI / HEST / THUNDER criterion, per run and per cell. |
| [`docs/CAVEATS.md`](docs/CAVEATS.md) | Reporting discipline the project holds itself to (its own example numbers are historical). |
| [`PLAN.md`](PLAN.md) | The spec the recipe was reconstructed against; every module docstring cites its section. |

Waiv's published targets are transcribed once, in
[`docs/waiv_published.json`](docs/waiv_published.json) (Tables 1–4), and loaded from there by
every comparison script.

## Layout

| path | contents |
|---|---|
| `src/waivphaet/` | Library: `data/` (PLISM repack + pair sampling), `models/` (backbone + LoRA), `train/` (masked InfoNCE), `eval/` (per-benchmark adapters) |
| `scripts/` | Training entrypoints, SLURM submitters, per-benchmark runners and collectors, verification tools |
| `tests/` | Unit tests |
| `third_party/` | Cloned harnesses (PathoROB, plism-benchmark), gitignored |
| `runs/` | Run directories: checkpoints, `ri_curve.json`, per-benchmark summaries |

## Standing caveats

- fp32 here against Waiv's mixed precision: deltas compare, absolute levels do not.
- Every benchmark has a measured seed floor and several are near it — a single-seed
  HEST or THUNDER delta is not gradeable. See `docs/thunder_seed_floor_12ds.md`.
- Patho-Bench is not run.
