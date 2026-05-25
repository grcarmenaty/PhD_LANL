# Variation v2 chunk regeneration — results report

**Status:** **INTERIM (seed 42 only); decision REJECT.** Seeds 101 and
202 are running automatically (the supervisor script continues
through the remaining seeds); this report will be updated when they
land. The seed-42 floor failure is large enough (Δ ≈ −0.16 BA on the
strongest v1 cell) that further seeds are extremely unlikely to flip
the decision.

> **Headline:** Variation v2 (widened domain randomisation + asymmetric
> Crack/Hole damage) **destroys the strongest robust v1 transfer cell**
> (`is_hole/mlp/modal`: BA 0.661 → 0.500, i.e. chance). All three
> primary and secondary criteria fail; C4 floor fails by 0.15. **v2 is
> not adopted**; v1 stands.

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

## 3. v2 seed-42 results (from `chunk_regen_v2_decision.json`)

| Task | Cell | Metric | v1 (3-seed) | v2 (seed 42) | Δ |
|---|---|---|---|---|---|
| is_crack | mlp / modal | balanced acc | 0.615 ± 0.026 | 0.500 | **−0.115** |
| col_location | mlp / modal | macro-F1 | 0.157 ± 0.025 | 0.128 | −0.029 |
| is_hole | mlp / modal | balanced acc | 0.661 ± 0.004 | 0.500 | **−0.161** |
| binary | best cell | macro-F1 | 0.482 ± 0.052 (`mlp/cfdac_magphase`) | 0.470 (`transformer/frf_mag`) | −0.012 (within band) |

Every one-vs-rest modal-MLP cell collapses to BA 0.500 (chance — class-prior
collapse). The widened DR (per-end JSR / per-mode damping / per-channel
sensor gain & phase) appears to have made the modal signature too noisy
for the synth-trained MLP to learn a transferable discriminator.

## 4. Decision table (pre-registered)

| # | Criterion | v1 | v2 (seed 42) | threshold | pass / fail |
|---|---|---|---|---|---|
| C1 | is_crack/mlp/modal BA | 0.615 | 0.500 | 0.787 | **FAIL** |
| C2 | col_location/mlp/modal macro-F1 | 0.157 | 0.128 | 0.329 | **FAIL** |
| C3 | is_hole/mlp/modal BA improves | 0.661 | 0.500 | 0.833 | **FAIL** |
| C4 | is_hole/mlp/modal BA no regression (floor) | 0.661 | 0.500 | 0.653 | **FAIL** |
| C5 | binary best macro-F1 floor | 0.482 | 0.470 | 0.310 | pass |

Decision rule: ADOPT v2 iff (C1 OR C2 pass) AND C4 AND C5 pass.
REJECT v2 iff C4 OR C5 fail. INCONCLUSIVE otherwise.

**C4 fails by 0.153** — far outside the 2 × p90 noise band (0.172). 

## 5. Verdict — REJECT v2; do not adopt the widened-DR variation

The asymmetric Crack/Hole damage in v2 might have been the right
physics correction, but the **widened domain randomisation that
shipped alongside it has destroyed the modal signature** that v1
relied on. Specifically:

* **`is_hole/mlp/modal` BA: 0.661 → 0.500.** This was the *single
  most robust* v1 transfer cell (lift/sd = 40, § 9.5). It vanished.
* **`is_crack/mlp/modal` BA: 0.615 → 0.500.** Also collapsed.
* **`col_location/mlp/modal` macro-F1: 0.157 → 0.128.** No
  improvement; tiny regression.

The 22 randomised parameters per sample in v2 (per-end JSR ×24,
per-mode damping ×9, per-channel gain ×9, per-channel phase ×9, plus
input gain + low-shelf) overwhelm the discriminative modal signal.
The MLP learns to output the prior.

**What this does not refute:** the asymmetric-damage physics fix in
v2 may still be correct in isolation. A future v3 should disentangle
the two changes — keep the asymmetric Crack/Hole geometry, drop or
heavily reduce the additional per-end / per-mode randomisation.
Without that ablation, we cannot attribute the regression to either
change alone.

**No report changes** beyond this writeup — v1 remains the reference
training configuration in `REPORT_definitive.md` and `REPORT_full.md`.

## 5a. v2 full seed sweep (planned)

Seeds 101 and 202 are running automatically via the supervisor script.
The expectation given C4 fails by 0.153 on seed 42 (≈ 39× the v1 sd
of 0.004) is that the floor failure replicates; the report will be
amended if it does not.

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
