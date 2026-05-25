# Variation v2 chunk regeneration — results report (stub)

**Status:** stub — populated by `ml_pipeline/compare_v1_v2.py` after
all 3 v2 seeds (42 / 101 / 202) complete. The numerical fields below
will be filled in then; the pre-registered framework is locked.

**Pre-registration:** [`chunk_regen_preregistered.md`](chunk_regen_preregistered.md).

## 1. Hypothesis

Variation v2 (`ml_pipeline/variation_v2.py`) introduces:

* **Widened domain randomisation** — per-end JSR factor (24 slots),
  per-mode damping (9 slots), per-channel sensor gain + phase (9 slots),
  input chirp gain and low-shelf gain.
* **Asymmetric Crack and Hole damage** — v1 applied the stiffness
  reduction symmetrically; v2 applies it to a single end, matching the
  LANL bookcase saw-cut / drilled-hole conditions.

We hypothesised v2 should improve two cells:

* **is_crack/mlp/modal** balanced accuracy (asymmetric Crack should
  produce a directional fingerprint matching real Crack).
* **col_location/mlp/modal** macro-F1 (asymmetric damage should also
  resolve the column-end ambiguity).

## 2. v1 baseline (3-seed mean ± sd, from `multiseed_summary.json`)

| Task | Best cell | Metric | v1 |
|---|---|---|---|
| is_crack | mlp / modal | balanced acc | 0.615 ± 0.026 |
| col_location | mlp / modal | macro-F1 | 0.157 ± 0.025 |
| is_hole | mlp / modal | balanced acc | 0.661 ± 0.004 |
| binary | best cell | macro-F1 | (computed from v1 sweep) |

## 3. v2 results (TBD — fill from `chunk_regen_v2_decision.json`)

| Task | Best cell | Metric | v2 (n seeds) | v2 mean ± sd |
|---|---|---|---|---|
| is_crack | mlp / modal | balanced acc | TBD | TBD |
| col_location | mlp / modal | macro-F1 | TBD | TBD |
| is_hole | mlp / modal | balanced acc | TBD | TBD |
| binary | (search v2) | macro-F1 | TBD | TBD |

## 4. Decision table (pre-registered)

| # | Criterion | v1 | v2 | threshold | pass / fail |
|---|---|---|---|---|---|
| C1 | is_crack/mlp/modal BA | 0.615 | TBD | 0.787 | TBD |
| C2 | col_location/mlp/modal macro-F1 | 0.157 | TBD | 0.329 | TBD |
| C3 | is_hole/mlp/modal BA improves | 0.661 | TBD | 0.833 | TBD |
| C4 | is_hole/mlp/modal BA no regression (floor) | 0.661 | TBD | 0.653 | TBD |
| C5 | binary best macro-F1 floor | TBD | TBD | TBD | TBD |

Decision rule: **ADOPT v2** iff (C1 OR C2 pass) AND C4 AND C5 pass.
**REJECT v2** iff C4 OR C5 fail. **INCONCLUSIVE** otherwise.

## 5. Verdict (TBD)

(One paragraph after the numbers land. State whether v2 is adopted,
rejected, or inconclusive, and what changes that triggers in the
reports.)

## 6. Cost / wall-clock log

| stage | start | end | notes |
|---|---|---|---|
| generate v2 chunks | 2026-05-24 08:48 | 08:48:46 | 10 000 samples in 41 s |
| extract v2 features | 08:48:46 | 08:49:39 | ~ 1 min |
| CFDAC (real / imag) | 08:51:24 | 08:52:30 | ~ 1 min |
| CFDAC variants (mag / phase) | 08:52:30 | 08:53:20 | < 1 min |
| seed 42 hpo | 08:53:20 | 09:47 | ~ 55 min (410 cells) |
| seed 42 cfdac variants | 09:47 | ~12:11 | ~ 2 h (interrupted; resumed next day) |
| seed 42 cfdac variants (resume) | 2026-05-25 08:13 | 08:22 | last 8 cells |
| seed 42 allmodels | 08:22 | TBD | 410 cells, ~6 h estimated |
| seed 42 eval | TBD | TBD | |
| seed 101 (full pipeline) | TBD | TBD | |
| seed 202 (full pipeline) | TBD | TBD | |

(The early-seed container suspensions and supervisor restarts are why
the wall-clock log is interleaved with restarts.)

## 7. Artefacts

* `ml_pipeline/variation_v2.py` — the new variation module.
* `ml_pipeline/generate_dataset.py` — `--variation v2` flag wires it in.
* `dataset_v2/` — 20 chunks, 10 000 samples (regenerable).
* `dataset/features_v2.h5` — full feature set incl. cfdac variants
  (regenerable).
* `results/experimental_full_evaluation_v2_seed{42,101,202}.json` —
  per-seed eval outputs.
* `results/chunk_regen_v2_decision.json` — pass/fail by criterion +
  ADOPT / REJECT / INCONCLUSIVE decision (`compare_v1_v2.py` output).
