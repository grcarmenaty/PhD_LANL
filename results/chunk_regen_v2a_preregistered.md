# V2a chunk-regeneration pre-registration

**Registered:** 2026-05-25 (before running the v2a generate→features→hpo→eval pipeline).
**Purpose:** ablate the v2 chunk regeneration's two bundled changes.
**Predecessor:** v2 was REJECTED (see
[`REPORT_v2_chunk_regen.md`](REPORT_v2_chunk_regen.md)) — `is_hole/mlp/modal`
BA crashed from 0.661 (v1) to 0.500 (chance).

## What changed (v1 → v2a)

`ml_pipeline/variation_v2a.py`. Only one change vs v1:

* **Asymmetric Crack and Hole damage** (P2.1 / P2.2 from v2) — the
  stiffness reduction is applied to a single end pair (2 corners,
  exponent ^0.5) instead of all 4 corners (exponent ^0.25 in v1).
  Total stiffness reduction is comparable; the geometric asymmetry
  matches the LANL bookcase saw-cut / drilled-hole.

**v1's domain randomisation is unchanged.** v2's P1.2 widened DR
(per-end JSR ×24, per-mode damping ×9, per-channel gain/phase ×9,
input gain & low-shelf) is *deliberately omitted*.

## Hypothesis

The v2 regression was driven by P1.2 (widened DR) drowning the
discriminative modal signal. v2a — same widened-DR-free baseline as
v1, plus only the asymmetric-damage geometry — should either:

* **Improve** the column-end / Crack-aware cells without losing the
  modal-MLP cells (P2.1/P2.2 was the right physics fix); or
* **Match v1** without improving (the synth-real Crack anti-correlation
  has causes beyond geometric symmetry); or
* **Regress** (the asymmetric-damage geometry itself is incorrect).

## Baseline (v1, 3-seed mean ± sd, from `multiseed_summary.json`)

| Task | Best cell | Metric | v1 |
|---|---|---|---|
| is_crack | mlp / modal | balanced acc | 0.615 ± 0.026 |
| col_location | mlp / modal | macro-F1 | 0.157 ± 0.025 |
| is_hole | mlp / modal | balanced acc | 0.661 ± 0.004 |
| binary | mlp / cfdac_magphase | macro-F1 | 0.482 ± 0.052 |

## Pre-registered v2a success criteria

A criterion **passes** iff the v2a 3-seed mean exceeds v1 mean by at
least 2 × the measured noise band (p90 = 0.086, torch family).

| # | Criterion | Threshold |
|---|---|---|
| C1 | **Primary — asymmetric Crack helps:** is_crack/mlp/modal BA improves | ≥ 0.615 + 2·0.086 = 0.787 |
| C2 | **Primary — column-location improves:** col_location/mlp/modal macro-F1 | ≥ 0.157 + 2·0.086 = 0.329 |
| C3 | **Secondary — Hole asymmetry helps:** is_hole/mlp/modal BA improves | ≥ 0.661 + 2·0.086 = 0.833 |
| C4 | **Floor — no regression on robust cell:** is_hole/mlp/modal BA | ≥ 0.661 − 2·0.004 = 0.653 |
| C5 | **Floor — no regression on detection:** binary best-cell macro-F1 | ≥ 0.482 − 2·0.086 = 0.310 |

## Decision rule (same as v2)

* **ADOPT v2a** if (C1 OR C2) AND C4 AND C5 pass.
* **REJECT v2a** if C4 OR C5 fails.
* **INCONCLUSIVE** otherwise.

## Pre-registered interpretive priors

Given v2 fully regressed (C4 fail by 0.15), four v2a outcomes are
possible, with the matching interpretation:

| v2a outcome | Interpretation |
|---|---|
| C1/C2 pass + C4/C5 pass — **ADOPT** | The asymmetric-damage fix was the right physics; v2's DR widening was the regression driver. **Future**: rerun with v2a as the new baseline. |
| C4 passes, C1/C2/C3 fail — **INCONCLUSIVE** | Asymmetric damage is harmless but doesn't help on its own. The synth-real gap on Crack/col-location is not purely geometric — needs another fix. |
| C4 fails — **REJECT** | Asymmetric damage is itself wrong (or interacts badly with v1's symmetric JSR jitter). Stick with v1. |
| (C1 pass + C4 fail is implausible.) | — |

## Sample counts & compute

* v2a chunks: 10,000 samples (matching v1 baseline statistical power).
* Seeds: 42 / 101 / 202 (matching the v1 multiseed run).
* Per-seed compute: ~6h hpo + ~2h variants + ~6h allmodels + ~1h eval (post-cache).
  Total ~45h wall-clock for 3 seeds. Resumable via the supervisor.

## Disk budget

Available: ~4.7 GB. v2a chunks ~170 MB; features_v2a.h5 ~2.8 GB. Tight.
Plan to remove `dataset/features_v2.h5` (2.8 GB, the rejected v2 features)
once v2a features build cleanly, freeing 2.8 GB.

— pre-registration ends here. Results in `REPORT_v2a_chunk_regen.md`.
