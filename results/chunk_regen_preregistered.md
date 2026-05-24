# Chunk-regeneration pre-registration (variation_v2)

**Registered:** 2026-05-24 (before running the v2 generate→features→hpo→eval pipeline).
**Purpose:** lock in success criteria so the v2 result is judged against
fixed thresholds rather than narrative re-fit after the fact.

## What changed (v1 → v2)

`ml_pipeline/variation_v2.py` (P1.2 + P2.1 + P2.2):

- **Widened domain randomisation.** Per-end JSR factor (24 slots), per-mode
  damping (9 slots), per-channel sensor gain & phase (9 slots), input chirp
  gain and low-shelf gain — all randomised per sample.
- **Asymmetric Crack and Hole damage.** v1 applied the stiffness reduction
  symmetrically; v2 applies it to a single end (matching the LANL bookcase
  saw-cut / drilled-hole conditions). This is the physically faithful
  change motivated by the iteration-3 council's "is_Crack ≈ chance" finding.

## Baseline (v1, 3-seed mean ± sd, from `results/multiseed_summary.json`)

| Task            | Best cell                | Metric    | v1 value      |
|-----------------|--------------------------|-----------|---------------|
| is_bolt         | cnn2d / cfdac            | bal.acc.  | 0.636 ± 0.035 |
| is_crack        | mlp / modal              | bal.acc.  | 0.615 ± 0.026 |
| is_hole         | mlp / modal              | bal.acc.  | 0.661 ± 0.004 |
| is_mass         | mlp / modal              | bal.acc.  | 0.628 ± 0.011 |
| is_pristine     | cnn2d / cfdac_real       | bal.acc.  | 0.531 ± 0.053 |
| col_location    | mlp / modal              | macro-F1  | 0.157 ± 0.025 |
| mass_location   | mlp / cfdac_imag         | macro-F1  | 0.421 ± 0.065 |

Measured run-to-run noise band (3 seeds, torch-only cells): median sd
0.016, p90 sd 0.086.

## Pre-registered v2 success criteria

A criterion **passes** iff the v2 3-seed mean is at least 2×p90 above v1
mean — i.e. a real signal beyond the measured noise band.

| # | Criterion (v2 vs. v1, same cell)                                          | Threshold                |
|---|---------------------------------------------------------------------------|--------------------------|
| C1 | **Primary — asymmetric Crack helps:** is_crack/mlp/modal BA improves      | ≥ 0.615 + 2·0.086 = 0.787 |
| C2 | **Primary — column-location improves:** col_location/mlp/modal macro-F1   | ≥ 0.157 + 2·0.086 = 0.329 |
| C3 | **Secondary — Hole asymmetry helps:** is_hole/mlp/modal BA                | ≥ 0.661 + 2·0.086 = 0.833 |
| C4 | **Floor — no regression on robust cell:** is_hole/mlp/modal BA            | ≥ 0.661 − 2·0.004 = 0.653 |
| C5 | **Floor — no regression on detection:** binary best-cell macro-F1          | ≥ v1 best − 2·0.086       |

C1 and C2 are the **headline tests** of the physics hypothesis. The
council's rigor reviewer originally listed weaker thresholds ("AUC ≥ 0.5",
"macro-F1 > chance + 2·torch-p90"); those are too weak — chance is already
the v1 baseline-floor for `is_crack` (0.5 BA = chance) but `is_crack`
already exceeds chance in v1, so the relevant test is improvement-over-v1,
not improvement-over-chance.

C4 / C5 are floor tests: if v2 regresses on the robust cells, the
widened DR has destabilised something and the v2 pipeline should not be
adopted even if C1/C2 pass.

## Decision rule

- **Adopt v2** if C1 OR C2 passes AND C4 AND C5 pass.
- **Reject v2** if C4 OR C5 fails (regression on a stable baseline).
- **Inconclusive** otherwise — report as a non-result; do not re-frame.

## Sample counts

- v2 chunks: 10,000 samples (same per-type counts as v1) to match the
  v1 baseline statistical power.
- Seeds: 42 (default), 101, 202 — same triple as the v1 multiseed run.

## Compute estimate

Generation ~6h (CPU FE simulation); feature extraction ~1h; seeded HPO
(3 cells × 3 seeds) ~12h; eval ~1h. **Total ~20h wall-clock.**

## Disk budget

Free: 8.1 GB. v2 chunks ~170 MB, v2 features.h5 ~2.8 GB, models cache
(deleted after each seed by the driver) ~500 MB peak. Should fit; if
disk pressure hits, fall back to streaming the chunks dir to /tmp.

## What this run does NOT decide

- Whether the v1 hyperparameter grid is still optimal for v2 — we reuse
  the same grid. A v2 win is necessary but not sufficient evidence that
  the physics is "right"; it may be that the new DR distribution just
  matches the experimental noise better.
- Anything about additional damage classes not in {Pristine, Bolt, Crack,
  Hole, Mass}.

— pre-registration ends here. Results to be appended in
`results/REPORT_v2_chunk_regen.md` after the run completes.
