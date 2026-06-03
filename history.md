# PhD_LANL — Project History

> **Exhaustive development history of the PhD_LANL repository**, reconstructed
> from the complete git commit record (929 commits, 2026-05-08 → 2026-06-03)
> spanning the human-authored commits by Guillermo Reyes-Carmenaty and the
> automated agent ("Claude") commits produced across many Claude Code sessions.

## About this document

This history is reconstructed **from git commit messages and the diffs they
reference**. The live chat transcripts of individual Claude Code web sessions
(e.g. `session_015ja563pMfBzfW3fUCURwaA`, `session_01PJh21SX7ZvQHXkrYcnqaA7`,
`session_015b788LhAVFN1jkfD6L3AiQ`) live in the web app and are not readable
from inside a working container, so the commit record — which is the durable,
authoritative log of *what was actually changed* — is used as the spine of
this narrative. Where commit messages report quantitative outcomes (e.g. SCI
scores, case counts), those numbers are quoted verbatim from the commits.

The document is organised into chronological **phases**. Each phase lists the
goal, the key commits, the technical decisions made, and the outcome.

### Commit activity at a glance

| Date | Commits | Focus |
|------|--------:|-------|
| 2026-05-08 | 28 | 3SBB calibration overhaul (SCI 0.46 → 0.95) |
| 2026-05-09 | 5  | Calibration finalisation + PR #1 |
| 2026-05-10 | 9  | MODEL.md technical reference |
| 2026-05-11 | 22 | First ML pipeline + CFDAC CNN + REPORT.md |
| 2026-05-12 | 73 | CFDAC variants / HPO build-out |
| 2026-05-13 | 104 | Dataset + feature-engineering expansion |
| 2026-05-14 | 132 | Model-correlation optimisation |
| 2026-05-15 | 104 | Training-pipeline improvements |
| 2026-05-16 | 205 | Largest sweep day — feature/model sweeps |
| 2026-05-17 | 24 | Consolidation |
| 2026-05-18 | 31 | Continued sweeps |
| 2026-05-19 → 05-20 | 20 | Wind-down / tidy |
| 2026-05-21 | 58 | Dataset bridge to pyMODAL |
| 2026-05-22 → 06-01 | ~44 | Intermittent refinement |
| 2026-06-02 | 54 | Vision-model sweep begins |
| 2026-06-03 | 22 | High-resolution CFDAC pipeline (last activity) |

---

## Phase 1 — 3SBB Reduced-Model Calibration Overhaul (2026-05-08 → 2026-05-09)

**Goal.** Take the 3-Storey Bookshelf Building (3SBB) reduced-order finite-element
model and calibrate it against experimental FRF data so that the model-vs-experiment
agreement, measured by the **SCI** (Spectral Correlation Index) metric, rises from a
poor baseline (~0.46) to as close to the physical ceiling as possible across all
**61 IQS damage scenarios** (9 sensor channels each, 1601 frequency bins).

### 1.1 Starting point and data assembly

- `44a35b2` **Update 3SBB reduced model: per-mode damping, recalibration, damage
  scenarios fix** — opening commit; reworks per-mode damping and fixes the damage
  scenario definitions.
- `afb3c0b`, `faabf6d` — `.gitignore` updates and an initial file upload.
- `1a15b37` **Optimise 3SBB calibration: per-storey JSR + median dataset workflow** —
  introduces per-storey Joint Stiffness Ratio (JSR) calibration and a median-dataset
  workflow.
- `b2224d9` / `f59a542` **Add `median_frfs.h5`** — the complex **median FRF per
  scenario per sensor** (61 scenarios × 9 channels × 1601 frequencies) becomes the
  calibration target dataset.

### 1.2 Notebook-driven exploration and the Colab bootstrap

A 3SBB exploration notebook was built up to drive calibration interactively, with a
Colab bootstrap cell so it could run in Google Colab against the latest branch data:

- `4a34072` Run 3SBB exploration notebook against `median_frfs.h5`.
- `7818cb4` Add Colab bootstrap cell to the exploration notebook.
- `09c67d5` Point Colab bootstrap at the current branch.
- `32a5959` Make the bootstrap pull the latest commit; **fail fast on stale data files**.
- `2f7e3fe` Harden bootstrap: drop `--depth`, add fallback, surface git stderr.
- `cd82be2` Bootstrap: `chdir` to `/` before git when the cwd has been deleted —
  defensive handling of Colab's ephemeral filesystem.

### 1.3 Fixing the amplitude mismatch and building the comparison view

- `92fc9ad` **Fix model/experiment amplitude mismatch and add sensor-by-sensor
  comparison** — a foundational correctness fix aligning model and experimental FRF
  amplitudes, plus per-sensor comparison tooling.
- `2195a1a` Restructure the notebook around a **per-case (3D model + 9-sensor) view**.
- `894b662` Cover **every IQS scenario**, refit damping, and add **CFDAC + SCI**
  computations to the notebook (CFDAC = Complex Frequency Domain Assurance Criterion).

### 1.4 Physics enrichment of the reduced model

A sequence of commits progressively enriched the reduced model's physics to capture
features the experimental FRFs showed but the model missed:

- `54e64b6` **Joint calibration**; the CFDAC view is made to mirror the FRF view
  (3D + experimental + synthetic + SCI).
- `f3fb842` Include **rigid-body accelerance**; SCI-direct calibration lifts the
  **mean SCI from 0.49 → 0.89**.
- `0961d5b` **SCI calibration v2 with frequency anchoring**; fixes damping indexing.
- `010dedf` Add **per-plate flexural DOF** to capture the 75–100 Hz rise on floor 3.
- `3dc6a40` Wire in a **second-flexural framework**; document architectural limits found.
- `aee31b5` Expand `compute_frf_matrix` with a **direct frequency-domain inversion path**.

### 1.5 Per-case calibration refinement — climbing toward the ceiling

The final push used increasingly fine-grained, per-case parameter overrides plus a
focused random search:

- `277102b` **Per-column-end JSR** with an asymmetric semi-rigid formula.
- `304ba32` Per-case overrides for D(85%) 1BD+2BD and AD+BD: mean SCI **0.922 → 0.925**.
- `70ff0d3` **Plate discretisation** via `sensors_on_flex`: mean SCI **0.925 → 0.934**.
- `8fdfe75` Tune flex config (90 Hz / 4 kg / 4 kg) + wider per-case override grid.
- `564a8b6` Iterative per-case fit; mean SCI 0.934 → 0.933, **47 cases above 0.95**.
- `62c0a75` **Per-mode damping** in case overrides; more cases above 0.95.
- `e9f7b44` **Per-corner JSR overrides**; mean SCI **0.943**, 54/61 above 0.90.
- `e62d8f9` Focused random search: mean SCI **0.943 → 0.950**, 55/61 above 0.90.
- `b945ed6` More cases lifted via focused search; mean SCI **0.951**.
- `a98bdca` **Final: mean SCI 0.951, 49/61 above 0.95, intrinsic ceiling explained.**

### 1.6 Outcome

- `656c8b6` **Calibration overhaul: mean SCI 0.46 → 0.95 across 61 IQS cases (#1)** —
  merged as PR #1.

**Result:** the 3SBB reduced model was calibrated from a mean SCI of ~0.46 to **0.951**,
with 49 of 61 cases above 0.95 and 55 of 61 above 0.90, and the residual gap explained
as an intrinsic modelling ceiling. Supporting experimental data (`experimental_frfs.h5`,
stored as 15×20 MB chunks for GitHub — commit `7cc08c5`) was added alongside the median
FRF dataset.

---

## Phase 2 — MODEL.md: Exhaustive Technical Reference (2026-05-10 → 2026-05-11)

**Goal.** Capture the calibrated model's theory and per-case behaviour in a single
authoritative document, `MODEL.md`, suitable as a technical reference for the thesis.

The day was dominated by writing the document and then fighting Markdown/LaTeX
rendering quirks so the maths displayed correctly on GitHub. Each agent commit was
mirrored by a human merge commit (PRs #2–#6), showing a tight review-and-merge loop:

- `99aff52` → `f40236c` (#2) **Add MODEL.md** — exhaustive theory + per-case explanation.
- `3891bd3` → `3016dfc` (#3) **MODEL.md v2** — fix LaTeX, add a **JSR section**, embed
  **11 figures**, expand depth.
- `5bdeb04` → `aa262d6` (#4) **Protect every underscore from LaTeX subscript-mode
  parsers** — GitHub's math renderer was misinterpreting `_` in identifiers.
- `193a723` → `797c1be` (#5) Avoid `\text{}` with underscores; fix CFDAC/SCI braces.
- `b8ba759` → `aad2d34` (#6) Replace `^{*}` with `^{\ast}` in the CFDAC formula.

**Outcome.** A complete, figure-rich technical reference (`MODEL.md`) covering the
reduced-model theory, the JSR formulation, the CFDAC and SCI metrics, and per-case
explanations — rendered cleanly on GitHub after the LaTeX-escaping fixes. The
recurring theme of this phase is the friction of GitHub-flavoured Markdown math:
underscores, `\text{}`, and `*`-vs-`\ast` all required workarounds.

---

## Phase 3 — First ML Pipeline, CFDAC CNN, and REPORT.md (2026-05-11)

**Goal.** Pivot from physics calibration to **machine-learning-based damage
identification**: build a synthetic training dataset from the calibrated model, train
classifiers/regressors, and evaluate them on the real IQS experimental data.

### 3.1 Synthetic dataset and the first model fleet

- `5d837db` **Add a 10,000-sample synthetic damage dataset and ML pipeline** — the
  calibrated model is used as a generator to produce labelled synthetic damage cases.
- `ecb266d` **Train 50 models and evaluate on IQS experimental data** — the first
  model fleet; synthetic-trained models are tested against real experimental FRFs to
  measure the **sim-to-real** gap.

### 3.2 CFDAC + 2-D CNN and hyperparameter optimisation

- `87401c5` **Add CFDAC + 2-D CNN, full-grid HPO, comprehensive plots & theory** —
  CFDAC images (the complex frequency-domain assurance matrix) are fed to a 2-D CNN;
  a full-grid hyperparameter optimisation (HPO) is introduced.
- `00a8f8f` gitignore the HPO run log (regenerated by `hpo.py`).

### 3.3 Reporting infrastructure

A substantial reporting effort built up the documentation that would track ML results,
fighting the same inline-figure issues seen in Phase 2:

- `88cdefe` Make RESULTS.md / THEORY.md links relative + embed key plots inline.
- `1c2198a` **Fix a scaler bug** in the plot/eval predictors + add per-plot commentary.
- `25d16bb` Embed every plot inline in `PLOTS.md` (`![]()` instead of `[]()`).
- `e554e27` Add a task-by-task narrative + plot-reading guide + protocol doc.
- `7c76769` **Consolidate everything into a single `REPORT.md`** — the canonical
  results document going forward.

### 3.4 CFDAC variants, full-experimental eval, and model-by-model triage

The evening pushed into a much larger experiment matrix — **CFDAC variants**, a **3D
CNN**, and **22 damage indicators** — evaluated against the *full* experimental set:

- `cef8c74` WIP: add the **CFDAC-variants pipeline** + **3D CNN** + full experimental
  features.
- `f0c9348` WIP: CFDAC-variants HPO results + the first 9 **indicator regressors**.
- `3a1522b` **Full-experimental eval (2,638 cases)** + remaining indicator regressors
  + report.
- `e35cc78` **Rebalanced datasets** + drop "indicator-as-input" + variant-table report.
- `7256425` Variant + indicator figures, distribution plots, inline trial dump.
- `374299e` §2 distribution plot + drop the 22-indicator vector + a **tabular CFDAC
  HPO script**.
- `164cbc4` `hpo_cfdac_allmodels`: RF + MLP, switch **XGB off**, in-flight checkpointing.
- `2fb6ef2` §7.1.18 references the tabular CFDAC results; §12 trial dump refreshed.
- `967ca5d` Variant tables + 22 indicator subsections + indicator-vs-damage table.

### 3.5 Pragmatic model-pruning decisions

Two notable "negative results" were recorded as deliberate engineering decisions:

- `04465fa` **Drop XGB for multi-class CFDAC variants** — intractable runtime.
- `a83220d` **Skip the Transformer for severity** — an unbounded regression head
  explodes on the flattened CFDAC reshape.
- `7bfc8c0` §7.X.19 tables + §7.6 subsections refreshed with the new variant cells.

**Outcome.** By end of May 11 the project had a working end-to-end ML pipeline:
synthetic-dataset generation → CFDAC feature extraction (plus variants) → a fleet of
models (RF, MLP, 2-D/3-D CNN; XGB and Transformer triaged out for specific tasks) →
full-grid HPO with checkpointing → evaluation on 2,638 experimental cases → a single
consolidated `REPORT.md` with inline figures. This set the stage for the large
multi-day sweeps that follow.

---
