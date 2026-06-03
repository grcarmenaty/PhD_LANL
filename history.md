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

## Phase 4 — Transfer Learning, Resolution & Noise Sweeps; the Watchdog (2026-05-12)

**Goal.** Systematically probe how the ML pipeline behaves under three axes of
variation — **transfer learning** (synth→exp), **frequency resolution**, and
**measurement noise** — and build the automation needed to run these long sweeps
unattended across ephemeral web-session containers. (73 commits this day.)

### 4.1 Transfer-learning and resolution sweeps

- `7219aa7` Drop unbalanced refs + add **transfer-learning** and **resolution-sweep**
  scaffolding.
- `dd1d31c` **Transfer-learning sweep (850 rows)** + resolution-sweep scaffolding.
- `544e14f` → `27dbd29` **Resolution sweep** runs as a background job, writing §11 of
  the report incrementally via a string of checkpoints (25 → 34 → 268 → 293 → 331 →
  360 → 382 → 410 cells). **Final: resolution sweep complete, 410/495 cells.**
- `44dab45` Plots + detailed analysis for §10 (transfer learning) and §11 (resolution).

### 4.2 Noise sweep — Gaussian noise on the timeseries

- `ca80d5c` **Noise-sweep framework** — add Gaussian noise to the timeseries and do a
  **full re-extraction** of features/CFDAC from the noised signals.
- `94e9bb3` → `1461def` The **20 dB** case runs first: features + CFDAC complete, then
  HPO checkpoints climb 22 → 38 → 52 → 66 → … → **189 HPO cells complete**.
- `c2ea360` **OOM event recorded:** `hpo_cfdac_allmodels` killed (out-of-memory after
  2 h 14 m), but **189 cells were saved** thanks to checkpointing; indicators continued
  at 52/66 — an early sign that the all-models CFDAC HPO is memory-hungry.
- `3b6dcb9` → `0fab7ea` 20 dB indicators + balanced eval done; transfer learning runs.
- `93c2182` **20 dB pipeline complete; launched a 35 / 25 / 15 / 10 dB sweep** — the
  noise study is generalised to a multi-SNR ladder.

### 4.3 The watchdog and SessionStart auto-resume — surviving ephemeral containers

This is the pivotal infrastructure of the day: the work was being run in ephemeral
web-session containers that get reclaimed, so a **watchdog** and a **SessionStart hook**
were built to make the long sweeps self-resuming and self-checkpointing:

- `99cb4ab` **`watchdog` auto-checkpoint** commits begin (committed under a `watchdog`
  author identity), tagging each with the sweep PID and a UTC timestamp.
- `f83a8f8` Add the **watchdog script** + ignore local sweep-launcher logs.
- `621ace9` **`SessionStart` hook: install deps + auto-resume the noise sweep on web
  sessions** — so a freshly-provisioned container automatically picks the sweep back up.
- `75b3675` HPO / indicator regression now **skip cells whose JSON already exists** —
  idempotent resume, so a restarted sweep doesn't redo finished work.

### 4.4 35 dB sweep grinds through the night

From `b2f03fa` onward the 35 dB stage runs overnight, with the watchdog committing an
auto-checkpoint roughly every ~10 minutes and Claude committing per-cell progress in
between. The HPO cells advance through the task families that recur throughout the
project — **severity**, **col_location**, **mass_location**, the CFDAC variants, and
the all-models CFDAC HPO (89 → 109 → 127 cells by end of day).

**Outcome.** May 12 established both the **scientific sweep matrix** (transfer learning,
resolution 410/495, and a multi-SNR noise ladder starting at 20 dB and 35 dB) and,
critically, the **resilience infrastructure** — watchdog auto-checkpointing + a
SessionStart auto-resume hook + idempotent skip-if-exists logic — that let multi-hour
sweeps survive container reclamation. The recorded OOM kill also flagged the
`hpo_cfdac_allmodels` memory pressure that shapes later decisions.

---

## Phase 5 — OOM Engineering, Lazy/Streaming Datasets & the Mixed-SNR Pivot (2026-05-13)

**Goal.** Finish the multi-SNR noise ladder, but more importantly **fix the
memory-exhaustion problems** that were killing CFDAC HPO and resolution sweeps, and
make every long job resilient to VM suspend/reboot. Mid-day the strategy pivots from
running one sweep per noise level to a single **mixed-training** dataset. (104 commits.)

### 5.1 Finishing the SNR ladder (overnight 35 → 25 dB)

- `04f1e24` 35 dB `hpo_cfdac_allmodels` **DONE (189 cells**; "OOM at end is harmless").
- `46fb80c` 35 dB indicators DONE; **eval/transfer FAILED** — missing
  `experimental_features_balanced.h5` (a data-dependency gap, fixed later as "B.4").
- `4187336` 35 dB done (resolution **OOM at trial 81/~143**, partial); **25 dB HPO
  started** and progresses to ~83 all-models cells.

### 5.2 The OOM fix — `LazyCFDACDataset` and true streaming

The repeated OOMs were traced to loading entire CFDAC tensors into memory. The fix was
a lazy, streaming data layer:

- `91cf1b4` **OOM fix: switch CFDAC HPO + resolution_sweep to `LazyCFDACDataset`.**
- `a7d4543` / `8dcaf35` Patched code runs `cfdac_all/imag/mag/magphase` **without OOM**,
  pushing **past the prior trial-81 OOM** to trial 92/945 and beyond.
- `38efbc0` `lazy_datasets`: add **reshape modes** (`conv2d`/`conv3d`/`seq`/`flat`) for
  true streaming via `DataLoader`.
- `1eda38b` / `41eefed` `hpo_cfdac_allmodels` and `hpo_cfdac_variants` now **stream**
  CNN/Transformer/MLP via `DataLoader` over `LazyCFDACDataset`.
- `acafff1` / `3066b80` `hpo.py`: **lazy-load features per cell** (pymodal-style
  `Dataset`) **to survive VM suspend**.
- `33e95b2` `hpo.py`: **sort the plan by feature cost** (`modal < frf_mag < timeseries
  < cfdac`) so cheap cells finish first — "early wins" before expensive CFDAC work.

### 5.3 The mixed-SNR training pivot

- `66ca53e` **Halt the per-level sweep** (set a `PAUSE_SWEEP` flag) and **pivot to a
  mixed-training design**: clean + 5 noisy levels = **60k samples**.
- `7fcbb35` Mixed-training infrastructure: a **VDS (virtual dataset) builder** + a
  per-SNR build helper.
- `77035c5` / `4c56910` Drop the per-level results dirs; restore the tracked
  `experimental_frfs_chunks`; gitignore the large `features_mixed.h5` VDS.

### 5.4 Heartbeat, auto-relaunch, and auto-commit cadence

To keep the `noisy_mixed` HPO grinding unattended across reboots, the monitoring layer
matured into a heartbeat + auto-commit system:

- `dd3d485` **`hpo_ping`: a 10-minute heartbeat + auto-relaunch helper.**
- `57b8de3` `session-start`: nudge Claude to **re-arm the 10-min ping after VM reboot**.
- `ceb809f` / `f43b5b5` Verbose per-tick status block; fix the cell counter (the script
  writes `task__model__feat.json`, not `best.json`).
- `5ed2ad4` `hpo_ping.sh`: **auto-commit new `noisy_mixed` artefacts each tick.**
- `803a49a` Split into **two cadences** — a 60 s auto-commit and a 600 s verbose ping.
- `347c9a5` Show **Barcelona local time** (Europe/Madrid) in the ping instead of UTC.

### 5.5 noisy_mixed grinds through the day

From `bf15418` onward the `noisy_mixed` HPO works through the task/model/feature grid,
auto-committing "+2 cell artefacts" every ~10–20 minutes for the rest of the day
(modal cells first, then severity/type across sklearn/MLP/RF/XGB). The day closes with
an **orchestrator script** (`495afab`) wiring the hook + ping together.

**Outcome.** The two defining changes of the project landed here: (1) a **lazy,
streaming dataset layer** (`LazyCFDACDataset` + reshape modes + per-cell lazy loading)
that eliminated the OOM kills and let jobs survive VM suspend; and (2) the strategic
**pivot to a single mixed-SNR (60k-sample) training set** instead of one sweep per
noise level. The monitoring layer also reached its mature form: a self-relaunching
heartbeat that auto-commits results every minute — maximising survivable progress in
ephemeral containers.

---

## Phase 6 — The `noisy_mixed` HPO Grind & Checkpointing Hardening (2026-05-14)

**Goal.** Drive the mixed-SNR (60k-sample) HPO to completion across the full
task × model × feature grid, while hardening the checkpoint/ETA machinery so the
unattended run survives VM reboots. (132 commits — the large majority are automated
`noisy_mixed: auto-commit N cell artefact change(s)` ticks.)

### 6.1 Substantive engineering commits

Buried among the auto-commits, a handful of commits did the real work of the day:

- `b042959` `hpo_cfdac_variants`: fix a **`NameError(X_tr)`** in the model-save path.
- `c4d2e87` **Trial-level checkpointing** for the CFDAC HPO — VM-reboot resilience at
  finer granularity than per-cell (a resumed run no longer loses an in-progress cell's
  completed trials).
- `b9c8aac` `hpo_ping`: add an **ETA line per ping** (remaining trials in the active step).
- `74d5b69` `hpo_ping`: **auto-commit *modified* artefacts**, not just new files — so
  updated `best.json` cells are captured, not only first-write ones.
- `521c160` Milestone: `severity__cnn2d__cfdac_magphase` cell complete.
- `1647e90` **Extend the ETA** estimator to cover `hpo_cfdac_allmodels` (step 3, 140 cells).

### 6.2 The unattended grind

The remainder of the day is the heartbeat doing its job: ~120 automated commits, each
capturing 1–4 newly-finished HPO cells roughly every few minutes, occasionally pausing
for longer (e.g. the 10:35→12:25 and 13:24→16:04 gaps, consistent with container
suspensions that the SessionStart hook later auto-resumed). This is precisely the
pattern the watchdog/ping system was designed to produce — **continuous, durable,
self-healing progress** that does not depend on a human or a live session being present.

**Outcome.** The `noisy_mixed` HPO advanced steadily through its grid with no data loss
across multiple suspend/resume cycles, validating the resilience design from Phase 5.
The only hand-written changes were bug fixes (`X_tr` NameError), a finer **trial-level
checkpoint** granularity, and **ETA reporting** — incremental hardening rather than new
science. This day is the clearest demonstration in the whole history of the
checkpoint-and-auto-commit strategy working as intended.

---

## Phase 7 — CFDAC All-Models Completion & the Flossgraben Bridge Dataset (2026-05-15)

**Goal.** Finish the `noisy_mixed` `cfdac_allmodels` HPO step, contain the remaining
multi-channel CFDAC OOMs, and — a new strand — begin bringing in a **second, real-world
structure**: the **Flossgraben bridge**, packaged via pyMODAL. (104 commits, again
mostly automated cell auto-commits.)

### 7.1 Multi-channel CFDAC OOM containment

The lazy/streaming layer from Phase 5 handled most cases, but a few model × feature
combinations still exhausted memory and were explicitly skipped:

- `3460d23` `hpo_cfdac_allmodels`: **skip RF/XGB on multi-channel CFDAC** (OOM) — the
  tree models can't stream and blow up on the wide multi-channel CFDAC tensors.
- `84fd392` `resolution_sweep`: **skip multi-channel CFDAC variants** (OOM).
- `c574dac` `resolution_sweep`: also skip the **legacy `cfdac` alias** (still OOMing).

### 7.2 `noisy_mixed` cfdac_allmodels step complete

- `ce23fde` **`noisy_mixed`: cfdac_allmodels step complete** — a major milestone, the
  most expensive HPO step of the mixed-SNR study finished.

### 7.3 The Flossgraben bridge enters the project

- `a026a25` **`flossgraben_bridge`: add pyMODAL build script + catalogue assets** — the
  first appearance of a real bridge structure, set up to be processed with the
  **pyMODAL** toolkit. This seeds the "bridge dataset → pyMODAL" work that becomes a
  dedicated branch later in the project, broadening the scope from the 3SBB lab
  structure to field bridge data.

### 7.4 The continuing grind

As on May 14, the bulk of the day's commits are heartbeat auto-commits of finished HPO
cells; the substantive commits above are the signal within that noise.

**Outcome.** The headline mixed-SNR `cfdac_allmodels` step reached completion, the
last multi-channel CFDAC OOM holdouts were contained by explicit skips (RF/XGB and the
legacy alias can't stream, so they're excluded rather than crashing the sweep), and the
project's scope expanded with the **Flossgraben bridge** + pyMODAL build pipeline — the
beginning of generalising the methodology beyond the 3SBB laboratory model.

---
