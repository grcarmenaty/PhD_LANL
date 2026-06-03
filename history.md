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
