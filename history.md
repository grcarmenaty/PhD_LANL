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

## Phase 8 — Largest Day: `noisy_mixed` Completes + Flossgraben FEM Calibration v2→v6 (2026-05-16)

**Goal.** Finish the entire `noisy_mixed` pipeline and ship its report, while in
parallel building and calibrating a finite-element model of the **Flossgraben bridge**.
At 205 commits this is the single busiest day in the project. It also includes **PR #7**,
the full feature-branch squash that merged the whole ML pipeline into `main`.

### 8.1 Final resolution-sweep OOM mitigations

The resolution sweep needed several more defensive patches to run cleanly:

- `625427b` `resolution_sweep`: **`n_jobs=1` + free per-ratio buffers** (OOM).
- `6e791ad` Skip a cell **if every ratio is already persisted** (idempotent resume).
- `c292ce9` **Skip RF/XGB on CFDAC variants** (can't fit the cycle).
- `285e224` / `c9bc78c` **Skip CNN- and Transformer-on-CFDAC** in the plan — a
  pre-existing Conv1D failure mode, excluded rather than repeatedly failed.

### 8.2 `noisy_mixed` reaches the finish line

- `ee49024` / `931d9b9` / `e1d62ea` Sweep-progress snapshots: **69 → 191 → 251 / 375 rows**.
- `9e674c0` **`noisy_mixed` pipeline complete — final `REPORT_noisy_mixed.md`** — the
  culmination of the mixed-SNR study begun on May 13.

### 8.3 Flossgraben bridge: from spec to a calibrated beam FEM

A complete second-structure modelling effort happened in parallel, mirroring the 3SBB
calibration arc but for a real bridge:

- `94aea4c` Drop the noisy variant from the Flossgraben pyMODAL build.
- `778eb81` Add a **Salome / Code_Aster model specification** for the bridge (a
  high-fidelity FE route).
- `88876f1` Add a **Python beam-FEM** + initial run results in `model.md §15`.
- `bb2b731` Add a **SciPy `differential_evolution` calibration loop.**
- `ca7a008` **v2 beam FEM** with torsion + per-band damping.
- `daf7334` / `1dc49b0` Refactor the DE loss to module level and **load the experimental
  cache at import** so parallel DE workers can see it.
- `ce0ea94` / `311b603` `flush=True` + per-generation logging; **force single-thread
  BLAS** in the DE calibrator (avoid oversubscription with DE's own parallelism).
- `60b1ec8` **v3 FEM:** finite pier compliance + sensor-x offset.
- `38900eb` **v4 calibrator:** smoothed-CFDAC SCI — an **OMA-appropriate metric**
  (operational modal analysis, where only output data is available).
- `0a8008b` **v5:** optimise smoothed-CFDAC at σ = 6 bins (1.5 Hz).
- `185c88f` **v6 calibrator:** balanced min + mean SCI, seeded near v5.
- `4b4d55a` **Final v6 calibration: smooth-SCI 0.82+ across all scenarios.**

### 8.4 The big merge

- `2db3ea4` **ML pipeline + datasets + reports: full feature-branch squash (#7)** — the
  accumulated ML work is squash-merged into `main`.
- `32c5a27` Merge `origin/main` back into `claude/create-ml-dataset-rMOoj` to re-sync
  the working branch after the squash.

**Outcome.** Two arcs concluded the same day: the **mixed-SNR ML study** finished with
its consolidated `REPORT_noisy_mixed.md`, and the **Flossgraben bridge** went from a
bare spec to a calibrated beam FEM achieving **smoothed-CFDAC SCI ≥ 0.82 across all
scenarios** — using an OMA-appropriate metric suited to a field structure where only
operational (output-only) data exists. PR #7 consolidated the ML pipeline into `main`.
The day also shows the now-standard pattern: every model/feature combination that can't
stream within memory is explicitly skipped rather than allowed to crash the sweep.

---

## Phase 9 — HBTA Bridge, Report Twins & Channel-Normalised Calibration (2026-05-17)

**Goal.** Replace the Flossgraben bridge with a better-suited structure — **HBTA** —
and calibrate its FEM; enrich the ML reports with experimental-test "visual twins";
and deepen the report analysis. (Merged via PRs #8, #9, #10.)

### 9.1 Report consolidation (PRs #8–#10)

- `758763e` (#8) **`noisy_mixed`: resolution sweep complete + `REPORT_noisy_mixed.md`.**
- `0c7795f` → `89da31f` (#9) Generate a **full-depth `REPORT_noisy_mixed.md`** by
  cloning `REPORT.md` and integrating the noisy results.
- `ee745f4` / `a15fd07` / `59ecbb6` → `6439b29` (#10) **Experimental-test visual twins**:
  `plots_experimental.py` produces exp-test counterparts of the test-time visuals, and
  `integrate_report` embeds each twin with a **Δ caption per cell** — so every reported
  metric is shown alongside its real-experimental analogue.
- `689831c` Add **§2.7 exp-dataset deep-dive** + **§9.5 clean-vs-noisy transfer comparison**.

### 9.2 HBTA bridge replaces Flossgraben

- `76511f2` **Add HBTA bridge scaffold to replace Flossgraben.**
- `6eeb259` Add the **HBTA pyMODAL loader** (`build_hbta_pymodal.py`).
- `d77b35a` HBTA diagnostic: **median H1 FRF + output spectrum per class**.
- `7761ca8` Remove `flossgraben_bridge`; add **HBTA `model.md`**.
- `2a82705` HBTA **stage-1 FEM + UDS comparison**; `model.md §15` populated.

### 9.3 HBTA calibration rounds

- `2a9dac6` Round 2: per-sensor axis fix, valid-band restriction, DE calibration setup.
- `e517b7a` Round 2: add **deck stringer beams** + widen DE bounds.
- `86ce6dc` Round 2 calibration: DE over 6 knobs → **smooth-SCI 0.626**.
- `c73bda7` DE loss: switch to **per-channel-normalised CFDAC**.
- `3cba5cd` **Round 3 calibration: channel-normalised CFDAC → smooth-SCI 0.783.**

### 9.4 Heartbeat control

- `d805797` `session-start`: honour a **`PAUSE_PINGS` flag** to mute the 10-min heartbeat
  — manual override for when the auto-relaunch loop should stay quiet.

**Outcome.** The bridge strand switched to **HBTA** (Flossgraben removed), calibrated to
**smooth-SCI 0.783** via channel-normalised CFDAC over a DE search. The ML reports
gained experimental-test visual twins (with per-cell Δ captions) and two new analytical
sections, all merged through PRs #8–#10.

---

## Phase 10 — Sim-to-Real Improvements (P0–P2) & Vision-Model Backbones (2026-05-18)

**Goal.** A structured, numbered campaign (P0 → P2) to close the **sim-to-real gap**,
followed by the introduction of modern **vision-model backbones** for CFDAC images.

### 10.1 P0 — correctness and domain fixes

- `c9864ea` **P0.1:** an **experimental-pristine reference** for CFDAC and pyMODAL
  indicators (a clean real-data anchor).
- `546ec60` **P0.2:** **bounded sigmoid heads** on severity; opt-out for unbounded
  indicators (prevents the regression-head blow-ups seen earlier).
- `83f0f61` **P0.3:** **per-domain scaler refit** — `exp_pristine` becomes the default.
- `0aa7a2f` **P0.4 + P0.5:** drop timeseries from training; guard an FRF divide-by-zero.

### 10.2 P1 — normalisation, augmentation, joint fine-tuning

- `ca5ea8f` **P1.1–P1.4:** per-sample normalisation, widened damage-ratio ranges,
  augmented chunks, and **joint synth+exp fine-tuning**.
- `0718cd7` **P1.1 bugfix:** thread normalisation through `LazyCFDACDataset`.
- `eb7e2c1` **P1.1 ablation:** per-sample feature normalisation, 30 cells retrained.
- `b1fa398` / `b0ffdd5` Thread the `normalize` flag through `transfer_learn`; add
  `--tasks` / `--unfreezes` filters for focused runs.
- `86b5296` `transfer_learn`: **incremental JSON save after every cell**.
- **P1.4 joint synth+exp fine-tune ablation**, task by task with snapshots between each:
  - `4dc2ae6` severity (partial): **+0.7 R²**.
  - `4f5b421` type (completed): **+0.22 accuracy**.
  - `0a4b08a` **P1.4 ablation COMPLETE** across all 4 tasks (severity, type,
    col_location, mass_location).

### 10.3 P2 — asymmetric damage and self-supervised pretraining

- `0403fb1` **P2.2 + P2.3 scaffolding:** asymmetric crack/hole damage, **SSL
  pretraining**, and an `--init-from` flag for warm starts.

### 10.4 New canonical reports

- `1777ff0` Add **`REPORT_simtoreal.md`** — companion to `REPORT.md` documenting the sweep.
- `327f127` Add 11 figures embedded in it (REPORT.md-style).
- `e49158c` Add **`REPORT_final.md`** — a standalone canonical report with proper
  diagnostic plots.

### 10.5 Vision-model backbones enter the project

- `8fe8ef2` **Add vision-model backbones — ResNet50, EfficientNet-B0, ConvNeXt-T,
  Swin-T, ViT-B/16 — for CFDAC**, plus a synth-only training driver. This is the seed of
  the vision-sweep work that dominates early June.
- `21ec8c3` `train_vision`: **batch the exp-eval forward pass** + checkpoint partial runs.
- `c9df92d` / `09c9f50` First vision-sweep results (ResNet50 + EfficientNet-B0 on
  type/cfdac_all) + plots.
- `62fe28e` gitignore `models_vision/*.pt`; drop stale under-trained artefacts.
- `eead988` / `2b6af8e` Snapshot `vision_eval.json` as cfdac_all → cfdac_mag →
  cfdac_realimag rows land (7/10 cells).

**Outcome.** The sim-to-real gap was attacked methodically: bounded heads, per-domain
scalers, per-sample normalisation, and **joint synth+exp fine-tuning** (measurably
**+0.7 R²** on severity, **+0.22 accuracy** on type). The reporting set grew to include
`REPORT_simtoreal.md` and a canonical `REPORT_final.md`. Crucially, **five modern
vision backbones** (CNN and Transformer families) were introduced for CFDAC-image
classification — the technical foundation for the large vision sweep that becomes the
project's focus in June.

---

## Phase 11 — Vision Sweep, "Trenchcoat" Decomposition & Report Consolidation (2026-05-19 → 2026-05-20)

**Goal.** Complete the first full vision sweep, explore a binary-decomposition trick to
improve classification, run severity-stratified analysis, and consolidate the sprawling
report set — while tightening reproducibility.

### 11.1 First vision sweep and its report

- `d38f303` **Vision sweep complete: 5 models × 3 features × type (15 cells).**
- `58c73dd` Add **`REPORT_vision.md`** — 5 vision backbones × 3 CFDAC features, synth-only.

### 11.2 The "trenchcoat" binary decomposition

- `25635dd` **Vision sweep v2: Tier-1 fixes + binary-"trenchcoat" decomposition** —
  decomposing a multi-class problem into a stack of binary classifiers ("a trenchcoat
  full of binaries").
- `9203899` Trenchcoat: **three aggregator strategies** + a diagnostic plot suite.
- `219854a` Add **`REPORT_vision_v2.md`** — Tier-1 + trenchcoat results, 5 plots.

### 11.3 Severity-stratified analysis and report set consolidation

- `1de53e7` Severity-stratified analysis: **accuracy vs severity-threshold curves**.
- `348b8d3` v2: widen to 12 cells (non-vision + vision + trenchcoat) and **correct the
  prior conclusion** — an explicit self-correction.
- `06eb330` Add **`REPORT_full.md`** (comprehensive catalogue) + **`REPORT_definitive.md`**
  (executive summary).
- `30f0aa0` `REPORT_definitive.md`: add **physics-aware augmentation**; drop the joint
  synth+exp fine-tune from the headline recommendation.

### 11.4 Housekeeping and reproducibility (May 20)

- `9c87a0a` **Discard the abandoned per-SNR noise sweep** (superseded by mixed-SNR).
- `ac1542d` / `c7f8e88` Bring **all report figures under the 2000px image limit** (so
  they render/are readable in tooling that caps image dimensions).
- `a40ed6d` `hpo.py`: **deterministic torch seeding**; eval adds **macro-F1 / balanced
  accuracy** (fairer metrics for imbalanced classes).
- `87815a3` / `c1f114e` / `39e391d` WIP: a **controlled A/B sweep** — plain vs augmented
  features — with the plain arm seeded and completed, the augmented arm resuming.

**Outcome.** The first vision sweep produced `REPORT_vision.md`; the trenchcoat
binary-decomposition idea was implemented and reported (v2); severity-stratified
analysis prompted an honest correction of an earlier conclusion; the report set was
consolidated into a comprehensive `REPORT_full.md` plus an executive `REPORT_definitive.md`.
Reproducibility hardened with **deterministic seeding** and **macro-F1 / balanced-accuracy**
metrics, and a **controlled A/B (plain vs augmented)** experiment was launched.

---

## Phase 12 — Seeded A/B Evidence & the Council-Reviewed Definitive Report (2026-05-21)

**Goal.** Turn the A/B experiment into rigorous, seeded, reproducible evidence; restructure
the reports around four clear diagnosis goals; and harden everything through several rounds
of "council" review. (58 commits — most are the ~15-min auto-checkpoints of the seeded
`hpo_cfdac` re-run.)

### 12.1 Controlled A/B evidence

- `bfa102b` **Controlled A/B evidence:** seeded plain vs augmented + baseline **macro-F1
  re-score** — a like-for-like comparison with fixed seeds.
- `f3ceeaf` `hpo_cfdac_*`: **deterministic torch seeding** (consistency with `hpo.py`).

### 12.2 The "council review" loop on `REPORT_definitive.md`

A multi-round internal review/verification process polished the definitive report:

- `52f43a5` Methodology-corrected edition.
- `d04b83f` Address **council review (round 1)**.
- `c075b36` Address **council verification (round 2)**.
- `05cf49b` **Final-gate polish (round 3)**.

### 12.3 Restructure around the four diagnosis goals

- `f4f505f` / `f0417f8` / `22d61e7` Restructure `REPORT_definitive.md` and `REPORT_full.md`,
  then **consolidate the whole report set around the four diagnosis goals**.
- `b43d449` Reports: cell grid, deployment-data assumption, **honest high-severity eval**.
- `b8b0593` Iteration 1: characterise **mass-plate location** (recommendation 2).
- `8535ef6` Iteration 2: consistency fixes + reproducible artefacts.

### 12.4 The seeded full re-run

- `f30a7b8` → `c8b9fe0` A long seeded `hpo_cfdac` re-run, auto-checkpointed every ~15 min
  through the day (with `hpo_cfdac_variants` done mid-run).
- `9861967` `hpo_cfdac_allmodels` partial (**326 cells**, pre-relaunch).
- `ca970fe` / `6b13b4b` **Seeded `hpo_cfdac_allmodels` done; full seeded sweep +
  evaluation complete.**

### 12.5 Folding the seeded results back into the reports

- `0c06d8f` / `78d9fc1` Fold the seeded `hpo_cfdac` sweep into `REPORT_definitive.md`
  and `REPORT_full.md`.
- `f4cbbd6` / `d02cc47` Iteration 3: surface **one-vs-rest type detection**, fix the
  detection cell, and propagate the findings.

**Outcome.** The project reached a **rigorously reproducible** state: a fully **seeded**
HPO sweep re-run end to end, a controlled **A/B (plain vs augmented)** comparison on fixed
seeds, fairer **macro-F1 / balanced-accuracy** metrics, and a `REPORT_definitive.md`
hardened through three rounds of council review and restructured (with `REPORT_full.md`)
around four explicit diagnosis goals. This is the project's "publishable-rigor" milestone.

---

## Phase 13 — Multi-Seed Variance, Pre-Registered Ablations & Negative Results (2026-05-22 → 2026-06-01)

**Goal.** Replace single-run claims with **multi-seed variance estimates** (seeds 42 /
101 / 202), and test several dataset-improvement hypotheses under **pre-registered
success criteria** — accepting the verdicts even when negative. This ~11-day stretch is
sparse in commit count (long runs, one checkpoint per completed seed) but high in
scientific rigour.

### 13.1 Multi-seed variance and a reversed headline

- `3d0c7a6` `hpo.py` / `hpo_cfdac_*`: add **`--seed`** for multi-seed variance runs.
- `0c7894c` → `3391e17` Seeds **101** and **202** complete (over several days).
- `9bfca5f` **Multi-seed (3 seeds) complete — reverses the iteration-3 `is_bolt`
  headline**: a conclusion that held for one seed did not survive replication.
- `5f226c7` / `43fac83` Iteration 4: propagate the reversal to `REPORT_full`, add a
  severity SD column, and **re-frame one-vs-rest around robust modal-MLP cells**.

### 13.2 The V2 dataset experiment — pre-registered and REJECTED

- `7083615` `generate_dataset`: a **V2 schema bridge** (scalar means + array fields).
- `d88ad6c` / `9fe3329` **Pre-register v2 success criteria** + a `compare_v1_v2.py`
  harness that judges against those criteria.
- `e5a3b0c` / `3bbee35` **v2 seed-42 REJECT:** the widened damage-ratio (DR) range
  **collapses the modal-MLP one-vs-rest cells**.
- `ee3f3f1` / `f70935c` Report the rejection; recommend a **v2a/v2b ablation** to
  disentangle *DR widening* from *asymmetric damage*.

### 13.3 The V2a ablation — also REJECTED, but it isolates the cause

- `be70907` **v2a ablation:** v1 DR + v2 **asymmetric Crack/Hole damage**; pre-register
  criteria. `781b32a` parameterises the comparison harness via `--label`.
- `f0483f5` Seed 42 only → **INCONCLUSIVE**, awaiting seeds 101 + 202.
- `a7d3ae3` 2-seed decision: **REJECT** (`is_hole` floor regressed 0.661 → 0.642, miss
  by 0.011 — a quantified, narrow miss).
- `affe029` **v2a REJECTED across 3 seeds:** asymmetric damage is *net-harmful*; the
  **widened DR was the dominant (harmful) v2 driver** — the ablation successfully
  attributes the v2 failure.

### 13.4 The modal-gap diagnostic and paper-style sensitivity eval

- `c637248` **Modal-gap diagnostic:** the synth-real gap is an **absolute-magnitude
  covariate shift that inverts the discriminant** — a precise mechanistic explanation
  of why synth-trained models mis-transfer to experimental data.
- `4cff853` Add a **DT/IT-swept evaluation:** accuracy vs minimum stiffness reduction,
  following the **paper's sensitivity methodology** — aligning the project's evaluation
  with the reference literature.

### 13.5 Disk hygiene and the v1/v2 re-evaluation

- `b278810` **Remove stale single-seed v1 model weights (2.7 GB)** — predictions
  retained in `_seeded.json`; v1 regenerated fresh for the DT re-eval. (A deliberate
  large-artefact cleanup — directly relevant to keeping containers lean.)
- `0d9a71a` → `05ef475` A multi-day **re-evaluation** under the new DT-swept methodology:
  v1 seeds 42/101/202 and v2 seeds 42/101 complete, one checkpoint per seed.

**Outcome.** This phase is the project at its most scientifically disciplined: a
single-seed headline (`is_bolt`) was **overturned by replication**; two pre-registered
dataset hypotheses (**v2** widened-DR and **v2a** asymmetric-damage) were **rejected on
their own stated criteria**, with the ablation cleanly attributing v2's failure to DR
widening; and the sim-to-real gap got a precise mechanistic account (**absolute-magnitude
covariate shift inverting the discriminant**). Evaluation was realigned to the reference
paper's DT/IT stiffness-sensitivity methodology, and a 2.7 GB stale-weights cleanup kept
the repository manageable.

---

## Phase 14 — DT-Stratified 3-Way Verdict & the timm Vision Sweep (2026-06-02)

**Goal.** Close out the v1/v2/v2a comparison with a DT-stratified verdict, then launch a
much larger, disk-light **vision sweep over `timm` backbones across all 10 tasks**.
(54 commits.)

### 14.1 The DT-stratified three-way comparison

- `e8ba437` v2 seed 202 complete — finishing the re-eval matrix from Phase 13.
- `38c6455` **DT-stratified 3-way comparison: v1 vs v2 vs v2a.**
- `e3b3d90` Feature-dimensional DT sweep + a multi-axis tier filter.
- `e84e237` Reports: **DT-stratified 3-way verdict** + **supersession banners** (older
  reports explicitly marked as superseded).
- `885b64a` / `178b514` DT-stratified **vision-vs-bespoke** check on the v1 type task;
  full **DT-3way report with 7 plots**.

### 14.2 The timm vision-sweep pipeline and its supervisor

- `2a50e52` **Vision-sweep pipeline: `timm` backbones, all 10 tasks, disk-light
  streaming** — a scaled-up, memory-frugal redesign of the vision sweep.
- `c2ea528` Add a **vision-sweep supervisor**: deps + **resumable relaunch** + periodic
  commit/push (the resilience pattern, applied to the vision sweep).
- `0f288b2` **Retry per (variant, seed) until 90/90 — survive OOM kills**: the sweep
  self-heals around the recurring OOM problem rather than aborting.
- `c6c8a7e` / `701a1c2` **Tighten the supervisor commit cadence** 600 s → 120 s → **30 s**
  to "bound the untracked window" — i.e. minimise how much work a container reclamation
  can lose. (A direct, late-project response to exactly the lost-work risk this whole
  history is about.)

### 14.3 New synth report and report-tree tidy

- `ae683bf` / `a8e6019` Add **`REPORT_synth.md`** — synthetic-domain (pre-transfer)
  training results, fleshed out with 5 plots, per-task explanations, and the sim-to-real
  gap.
- `6234c63` Move non-canonical reports to **`results/legacy/`**.
- `88916e2` / `bcb19d0` Ignore regenerable derived feature files (`features_v2a.h5` ~3 GB,
  v2/v2a per-seed dirs) — disk hygiene.

### 14.4 The vision sweep grinds

- `83c97ab` → `14e2fb3` The vision sweep auto-commits per-case progress, climbing **3 →
  5 → 8 → 13 → 23 → 30 → 42 → 46 per-case files**.
- `07668a8` **Refresh synth report — ConvNeXt v1/seed42 complete (30 cells)**.

**Outcome.** The v1/v2/v2a question was settled with a **DT-stratified 3-way verdict**
(7 plots, supersession banners on the old reports), and a new **`timm`-based vision
sweep across all 10 tasks** was launched with a self-healing, OOM-surviving supervisor.
Notably, the supervisor's commit cadence was tightened all the way to **every 30 seconds**
specifically to bound how much progress an ephemeral-container reclamation could destroy.

---

## Phase 15 — High-Resolution CFDAC Pipeline (2026-06-03, final activity)

**Goal.** Push the CFDAC features to a **higher frequency resolution (1601 bins)** and
train the top-performing model per task on these hi-res features. This is where the
most recent session was working when its container was reclaimed.

### 15.1 Finishing the standard-res vision sweep

- `bf6233c` / `14e2fb3` Vision sweep reaches **42 → 46 per-case files** (continuing the
  Phase 14 sweep into the new day).

### 15.2 High-resolution CFDAC scaffolding and build scripts

- `df56b76` **WIP: high-res CFDAC scaffolding** — `cfdac_runtime` + `train_vision_hires`.
- `eff5621` `generate_dataset`: add **`--n-t` / `--fs`** args to override the module's
  `N_T` / `FS` for hi-res regeneration.
- `e395212` **Hi-res CFDAC pipeline: build scripts for synth + exp features at 1601 bins.**
- `646d239` **Hi-res top-cell-per-task training driver** (lazy CFDAC, 1601 resolution) —
  trains only the best-performing model/feature per task, at high resolution, using the
  lazy/streaming dataset layer from Phase 5.

### 15.3 Preliminary hi-res results (last commits)

- `acca875` Hi-res preliminary: **in-progress per-case results**.
- `bb4a91f` / `fd17cf1` Hi-res preliminary: **cell results** (the final commit, touching
  `results_hires/synth_test.json`).

**Status at last activity.** The synthetic hi-res cell results were being generated and
committed; the natural next step — visible in the working session's own notes — was
**running experimental-data inference at hi-res** ("synth done, exp inference next"). All
of this work is committed and pushed to `claude/rescue-failing-session-xjHZb`; only the
live session context was lost when the container was reclaimed.

---

## Cross-cutting Themes

A few threads run through the entire history and are worth stating explicitly:

1. **Resilience against ephemeral containers.** From the watchdog (May 12) through the
   `hpo_ping` heartbeat, the SessionStart auto-resume hook, idempotent skip-if-exists
   logic, trial-level checkpointing, and finally a **30-second** supervisor commit
   cadence (June 2), an enormous amount of engineering went into ensuring that
   multi-hour, multi-day runs survive container reclamation and VM suspend. This is the
   same infrastructure relevant to the "stuck generating cloud container" problem.

2. **OOM is the recurring adversary.** The single most common failure mode was
   out-of-memory on wide CFDAC tensors. The durable fix was the **lazy/streaming dataset
   layer** (`LazyCFDACDataset`); where streaming was impossible (RF/XGB, some
   CNN/Transformer-on-CFDAC cells) those combinations were **explicitly skipped** rather
   than allowed to crash.

3. **Disk hygiene vs. committed results.** Large derived artefacts (`features_*.h5`,
   `models_*/*.pt`, per-seed dirs, 2.7 GB stale weights) were repeatedly gitignored or
   purged. At the same time, *results JSON* were committed continuously for durability —
   which is what makes the per-case `results_*/` directories large, and what drives the
   big branch diffs.

4. **Scientific discipline.** Pre-registered success criteria, multi-seed replication
   that overturned single-seed headlines, accepted negative results (v2, v2a, the
   mass-plate ensemble), multi-round "council review", deterministic seeding, and
   fairer metrics (macro-F1 / balanced accuracy) characterise the later phases.

5. **Two structures, one methodology.** The CFDAC/SCI calibration-then-ML approach was
   developed on the **3SBB** laboratory model and then generalised to a real bridge —
   first **Flossgraben**, then **HBTA** — via pyMODAL and differential-evolution
   calibration.

---

*Part I above was reconstructed from the git commit record on 2026-06-03 and is
authoritative for what was changed and when. Quantitative figures are quoted directly
from commit messages. **Part II below** adds the human-driven narrative recovered from
the actual Claude Code web-session transcripts (exported via the session event API),
capturing the reasoning, instructions, and operational reality the commits don't show.*

---

# Part II — Session Narratives (recovered from transcripts)

> Reconstructed from the full event streams of the Claude Code web sessions
> (exported through the `/v1/sessions/<id>/events` API). These consolidate **what was
> asked and why**, complementing the commit-based phases above.

## Session `015b788` — The Master Rescue & Autonomous-Improvement Session (2026-05-20 → 2026-06-03)

**Scale.** 33,934 events over ~14 days (≈10,200 assistant turns, ≈4,200 user events,
≈2,800 result events). This single session is the spine of the entire final fortnight —
it began as a rescue of a *different* failed session and grew into a continuous,
council-driven, compute-heavy improvement loop that produced almost all of the May 20 →
June 3 commits (Phases 11–15).

### Origin — why it exists

The session opened (05-20 08:57) with:
> *"https://claude.ai/code/session_015ja563… This session started failing. Assess why,
> rescue everything from there, push into main, be prepared to continue."*

The root cause of the original failure, in the user's words (09:10):
> *"An image in the conversation exceeds the dimension limit for many-image requests
> (2000px). Start a new session with fewer images."*

So `session_015ja563` (the original ML workhorse) **died on the 2000px many-image
limit** — which is exactly why the late-May commits repeatedly forced every report
figure under 2000px. `015b788` was spun up to rescue that work and carry it forward.

### Arc 1 — the two reports (05-20)

The first mandate (12:16): a comprehensive **`REPORT_full.md`** (a graph for every
experiment, all figures inline) plus a polished **`REPORT_definitive.md`** (problem →
improvements → limitations → solution → implementation → results, with the **best
synth-trained / real-tested** models). Two corrections the user demanded immediately:
- **Physics-aware augmentation** (`variation_v2.py` widened domain randomisation +
  `build_augmented_chunks.py`) had been *underplayed* — it got its own section.
- The **joint synth+exp fine-tune** story was **cut** from the definitive report — the
  canonical claim became strictly *synth-only training, real-data testing*. The headline
  synth-only numbers settled around: binary 0.825 (= class-prior floor), type macro
  ≈0.51, type@severity≥0.7 ≈0.66, severity R² ≈0.18, col/mass-location ≈0.51–0.53.

### Arc 2 — the 5-agent council and the methodology reckoning (05-20 → 05-21)

The defining instruction (05-20 16:02):
> *"have a council of 5 agents, one very critical, one very adulatory, the remaining
> three in between, evaluate it. Give them the real data and tell them to stick to it.
> After that, assess and continue until nothing else can be improved."*

This established the recurring **5-reviewer council pattern** (harsh critic, appreciative,
scientific-rigor, clarity, completeness — each forced to *verify every number against the
result JSONs*). Successive council rounds drove the project's biggest corrections:
- The HPO pipeline was **unseeded** → seeding added across `hpo.py` / `hpo_cfdac_*`.
- Accuracy headlines were **class-prior collapse** → switched to **macro-F1 / balanced
  accuracy**; a "best" cell was exposed as a non-reproducible fluke.
- The augmentation A/B was **single-seed and confounded** (20k aug vs 10k plain) →
  reframed from "negative result" to "predicted lift not observed; inconclusive."

The user then set the autonomous loop (05-21 08:28):
> *"Follow recommendations. Once all recommendations are done, summon the council and
> draft new recommendations. Follow those and repeat. Stop once you're not able to come
> up with recommendations."* — plus *"Perform even heavy compute ones, save regular
> checkpoints."*

### Arc 3 — seeded sweep & multi-seed validation (05-21 → 05-24)

A **seeded 244-cell sweep** (`experimental_full_evaluation_seeded.json`) became canonical,
then a **3-seed validation** (seeds 42/101/202, `multiseed_summary.json`). The textbook
result: the iteration-3 headline **`is_bolt` 0.71 was a single-seed fluke** (seeds 101/202
gave 0.49/0.51); but the multi-seed view *also* found the genuinely **robust survivors** —
modal-MLP one-vs-rest cells (`is_hole`/mlp/modal ≈0.661 ± 0.004). Measured noise band:
median sd 0.011, p90 ≈0.071.

### Arc 4 — the v2 / v2a physics ablations (05-24 → 06-02)

To test whether richer synthetic physics helps, three variants were compared on equal
footing (3 seeds each): **v1** (baseline), **v2** (widened DR + asymmetric crack/hole
damage), and **v2a** (v1 DR + asymmetric damage — the disentangling ablation). Verdict:
**v2 hurt**, and v2a isolated the cause — the **widened domain randomisation**, not the
asymmetric geometry, was the harmful driver. (This required a `SampleParamsV2` schema
bridge so `generate_dataset.py --variation v2` could dump scalar params from per-end/
per-mode arrays.)

### Arc 5 — DT-stratified analysis (the user's core hypothesis) (05-28 → 06-02)

Prompted by a reference paper, the user pushed a **damage-threshold (DT) stratification**:
> *"apply metrics on ALL, not only is_bolt and mass_location. My hunch is that once DT
> sweeps are done, we will see learning across more models and use cases, only they only
> work for high damage scenarios."*

and a feature-dimensional version:
> *"DT sweeps based on feature dimensions… mass, tightness, column section, crack depth,
> hole size… For is_pristine use a multi-axis threshold to see when it gets interesting
> based on what damaged spectra you're excluding."*

This produced `dt_compare_variants.py`, `dt_feature_sweep.py` (tiers all/med+/severe), a
**DT-stratified 3-way verdict** (`REPORT_dt_3way_*.md`, 7 plots), and confirmed the
hypothesis shape: detection (binary/is_pristine) flat-drops under restriction (class
collapse — the user caught a suspicious 0.825 binary as all-damaged), while signal
concentrates in high-damage regimes. The user also caught that **0.825 binary was class
collapse**, not signal.

### Arc 6 — vision-model transfer learning (06-02)

The user repeatedly flagged the biggest omission:
> *"I see vision model transfer learning wasn't used, that's where my paper found
> strongest gains."*

→ the `timm` vision sweep (ResNet50, EfficientNet-B0, ConvNeXt-T, Swin-T, ViT-B/16),
extended to the **top-3 backbones across all applicable cells**, plus `REPORT_synth.md`
(pre-transfer synthetic-domain results) and a move of non-canonical reports to
`results/legacy/`.

### Arc 7 — high-resolution CFDAC (06-03, the final push)

A late pivot on resolution:
> *"what's the CFDAC resolution you've been using?" → "Why not full? Experiments can have
> CFDACs of up to 1601×1601" → "Pause the sweep, I want it now" → "regenerate synth, more
> simulation time, whatever is needed. I want for each task, top CFDAC model, trained on
> full 1601 resolution synth, tested on both synth and non-synth."*

This is the **hi-res CFDAC pipeline** (`cfdac_runtime`, `train_vision_hires`, 1601-bin
features, top-cell-per-task driver) — the work that was mid-flight when the container was
reclaimed (Phase 15). Final exchanges clarified channel counts (experimental has many
channels) and asked *"why no GPU?"* — the recurring constraint below.

### The operational saga — fighting container reclamation

The transcript's single most pervasive theme isn't ML at all; it's **keeping the session
alive on an ephemeral, idle-suspending, GPU-less 4-core container.** The user typed
*"keep going"* / *"status report"* / *"How is it going?"* **dozens of times**, plus
escalating nudges: *"4h ago, re-think how not to let this die"*, *"what do you recommend
to keep this alive? Diagnose, solve"*, *"Hey I'm going to go sleep. Please stay running
for the night."* The engineering response evolved through:
- a **heartbeat protocol** — *"read nothing and say hi, nothing else, every 15 minutes,
  until the task finishes"* (later realised **only tool calls reset the idle timer**,
  not text);
- **Monitor-based keep-alive** + **auto-restart** drivers with PID files;
- bug fixes for **`pgrep` self-matching** (switched to `kill -0 $(cat …pid)` and
  `awk '$2==1'` init-parent matching);
- a driver `rm -rf models` **cleanup bug** that overwrote a good seed result with a
  2-byte empty JSON (recovered from git);
- finally a **`/loop`** self-pacing supervisor and a **30-second** commit cadence.

This is the real-world counterpart to the watchdog/checkpoint infrastructure documented
in Phases 4–6 — and the direct backstory to the "stuck generating cloud container"
problem that started this whole investigation. The user also twice asked to *"clean your
context by removing pings and status reports"* — which can't be done to a fixed
transcript, and is precisely why this `history.md` exists.

### Context compactions

The session ran out of context and **auto-compacted at least twice** (05-24 and 06-02),
each time regenerating a detailed state summary and resuming — itself a testament to the
session's length.

## Session `01PJh21S` — Genesis: Dataset, First ML Pipeline & CFDAC Variants (2026-05-10 → 2026-05-13)

**Scale.** 7,206 events over ~3 days (≈3,200 assistant turns). This is the **earliest**
session — where the machine-learning side of the project was born. It maps onto Phases
3–5 and ran on branch `claude/create-ml-dataset-rMOoj`.

### The founding request

The very first instruction (05-10 23:21):
> *"create a dataset of 10000 samples, with all cases having an equal amount of samples,
> distributed in files of less than 20MB. Each sample slightly different… variable but
> bounded variation, holes from 1 to 6mm… holes of 3.234574mm in one case and 1.348285mm
> in another. The purpose is to train ML models. Once you're done, train models and test
> them against experimental data."*

…immediately amended (23:23): *"noise should not be included in this."* This produced
the 10,000-sample synthetic damage dataset and the first model fleet (Phase 3).

### What the user demanded — depth of reporting

A defining trait of this session (and the whole project) appears here: an exacting
standard for **explanatory reporting**. The user repeatedly pushed for more:
- *"I want plots and graphs… confusion matrices… the theory behind all this, why the
  models were constructed as they were, the exact parameters, what libraries and
  implementations were used. Have all models gone through HPO? I want response surfaces."*
- *"For each plot I want indications on what the plot is, how to interpret it, what can
  be seen, and what conclusions can be taken."*
- *"I get file not found for all these links" / "I see links not figures"* → the push to
  **embed every figure inline** (the `![]()` fixes in the commits).
- *"I want a single REPORT.md file with everything neatly organized"* → the consolidation
  into one canonical `REPORT.md`.

### CFDAC variants and the indicator pivot

The major technical expansion (05-11 16:50):
> *"For CFDAC you're only using magnitudes, train models using real part, imaginary part,
> phase part, 3d-cnns with real + imaginary, magnitude + phase, and 5d-cnns with all
> parts of the CFDAC. There are close to 3000 experimental FRFs, so use those to test.
> Also disregard the damage indicators as input features, but give me models to predict
> all of them… (regression model to get SCI independent of damage typology or location).
> Explain location distribution in the data."*

This created the **CFDAC-variant catalogue** (real/imag/mag/phase, 3D real+imag and
mag+phase, and an "all-parts" stack), the move to evaluate on the **full 2,638-case
experimental set**, the **22 indicator regressors** (RF/modal hit R² 0.85–0.999 on most
indicators), and the dropping of indicators as *input* features.

### The pymodal / OOM lesson

A pivotal user correction (05-11 17:55):
> *"pymodal is literally made to make models while loading from disks without worrying
> about memory."*

This — after CFDAC HPO was **OOM-killed at trial 80** on the `cfdac_all` variant —
seeded `lazy_datasets.py` / `LazyCFDACDataset`, the lazy/streaming layer that became the
project's durable fix for memory exhaustion (Phase 5).

### Transfer, resolution, and noise — the sweep trilogy is born

- (05-12 05:42) *"take the synth models and retrain them (only last layers unfrozen) with
  experimental data. 10% 20% 30% 40% 50%… test against the remaining experimental… Also
  map for everything the effect of feature resolution (from full to half resolution, with
  5 divisions total)."* → the **transfer-learning** and **resolution sweeps** (Phase 4).
- (05-12 12:21) *"now let's go for a new file… exactly the same study but with gaussian
  noise added to the synth dataset."* → the **noise sweep** (Phase 4).

The session also shows the first **context compaction** (05-11 18:24) and the first
"Report every now and then" cadence request — early seeds of the heartbeat behaviour.

---

## Session `015ja563` — The Sim-to-Real Diagnosis & Fix Workhorse (2026-05-18 → 2026-06-02)

**Scale.** 2,898 events (≈1,100 assistant turns). This is **the session that "started
failing"** and was rescued by `015b788`. Its substantive work (05-18 → 05-20) maps onto
Phase 10 and the front of Phase 11; the later sparse events are the failing tail.

### Framing the problem

It opened (05-18 06:59) with the project's thesis stated plainly:
> *"The main gist of this repo is finding a way to train on FE models and work with
> experimental, real, data. The current approach clearly under-performs. What can be
> done?"* → *"Make a comprehensive plan to fix all of this."*

### Three-front diagnostic investigation

Rather than guess, the session launched **three deep, parallel investigations** (visible
as long structured subagent prompts) into the exact mechanics of the sim-to-real gap
(type accuracy **0.88 synth → 0.25 experimental**):
1. **Feature pipeline** — found the **reference-FRF bug**: experimental CFDAC and the 22
   indicators were computed against the *synthetic* pristine mean
   (`build_experimental_features.py:77`), never an experimental pristine reference;
   unbounded severity heads; inconsistent scaling; fabricated experimental timeseries.
2. **Physics knobs** — found domain randomisation far too narrow (`JITTER_JSR` ±5% vs
   per-case overrides spanning 0.3–3.0×), a purely linear ROM, and **symmetric crack/hole
   damage** making BD vs AD degenerate (the ~0.67 `col_location` ceiling).
3. **Training/eval wiring** — mapped where to slot fixes (joint training in
   `transfer_learn.py`, richer corruption in `build_noisy_chunks.py`, bounded heads in
   `models.py`).

### The phased fix plan (P0 → P2)

These fed a single **execution-ready phased plan** — the P0/P1/P2 structure that became
Phase 10's commits: P0 (reference-FRF fix, per-sample normalisation, bounded sigmoid
heads, per-domain scaler, drop timeseries, div-zero guard), P1 (widened DR, augmented
chunks, joint synth+exp fine-tune), P2 (asymmetric crack/hole, SSL pretraining,
nonlinear bolt — scaffolded).

### Synth-only focus, vision models, and the trenchcoat

- (05-18 16:27) *"I'm most interested in what happens when you train without using
  experimental data… Add general-purpose vision models and retraining them to use CFDAC
  (all possible features). Pick top 5 vision models."* → the **vision backbones** (Phase
  10/11).
- (05-19 14:56) *"the results are underwhelming. analyze why and improve."*
- (05-19 15:41) *"reframe the typology detector as a bunch of binaries in a trenchcoat:
  is it or is it not this kind of damage? Same as pristine."* → the **trenchcoat
  decomposition** (Phase 11).
- (05-19 19:07) *"Does accuracy improve if I judge the models only against the most
  extreme cases? Beyond what point does the accuracy start to improve"* → the seed of the
  **severity-stratified / DT-threshold** analysis that `015b788` later generalised.

### The failure

The last stretch is the session **breaking**: the council instruction
(*"have a council of 5 agents, one very critical, one very adulatory…"*) appears **eight
times in a row** on 05-20 (08:55 → 15:43) as the user re-sent it against a stuck session —
the **2000px many-image limit** had wedged it. That is the exact failure `015b788` was
created to assess and rescue, closing the loop with the master session above.

---

*All three target sessions are now documented in Part II. Together they trace the full
ML arc: `01PJh21S` (genesis: dataset + pipeline + CFDAC + sweeps, May 10–13) →
`015ja563` (sim-to-real diagnosis + P0–P2 fixes + vision + trenchcoat, May 18–20, until
it hit the image-limit failure) → `015b788` (rescue + autonomous council-driven
improvement through hi-res CFDAC, May 20 → June 3). Quoted user text is verbatim from the
recovered event streams.*
