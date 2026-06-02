# DT-stratified 3-way verdict: v1 vs v2 vs v2a

**Supersedes** the v2 rejection in [`REPORT_definitive.md` §10 rec 3](REPORT_definitive.md)
and the single-seed [`REPORT_v2_chunk_regen.md`](REPORT_v2_chunk_regen.md).
This is the multi-seed (42 / 101 / 202), damage-threshold-stratified
re-evaluation of all three synthetic-physics variants, scored on the
2 638-case real LANL 3SBB experimental set.

- Source data: 9 per-case prediction files (`results/experimental_full_per_case_v{1,2}_seed{42,101,202}.json`
  and `results_v2a_seed*/experimental_full_per_case.json`).
- Analysis: [`ml_pipeline/dt_compare_variants.py`](../ml_pipeline/dt_compare_variants.py)
  (stiffness-reduction sweep) and
  [`ml_pipeline/dt_feature_sweep.py`](../ml_pipeline/dt_feature_sweep.py)
  (per-damage-type physical-axis sweep + multi-axis tier filter).
- Raw output: [`dt_compare_v1_v2_v2a.json`](dt_compare_v1_v2_v2a.json),
  [`dt_feature_sweep.json`](dt_feature_sweep.json).
- No retraining: pure post-hoc re-scoring of predictions already on disk.

## The three variants

| variant | domain randomisation | damage geometry |
|---|---|---|
| **v1** | baseline (`variation.py`) | symmetric Crack/Hole |
| **v2** | **widened** (P1.2, `variation_v2.py`) | asymmetric Crack/Hole (P2.1/P2.2) |
| **v2a** | baseline (= v1) | asymmetric Crack/Hole only |

v2a is the **disentangling ablation** the old report asked for: it carries
v2's asymmetric-damage geometry on top of v1's domain randomisation, so the
only difference between v2 and v2a is the widened DR.

---

## 1. The headline finding: the old v2 rejection was right, but for the wrong reason

The definitive report rejected v2 because `is_hole/mlp/modal` — v1's most
robust cell — crashed from 0.661 to 0.500 on seed 42. The multi-seed re-run
**confirms the collapse is real, not a seed fluke**, and the v2a ablation
**identifies the cause**:

| cell | v1 (BA, 3-seed) | v2 (BA, 3-seed) | v2a (BA, 3-seed) |
|---|---|---|---|
| `is_hole/mlp/modal` | 0.651 ± 0.010 | **0.500 ± 0.000** | 0.621 ± 0.032 |
| `is_crack/mlp/modal` | 0.596 ± 0.041 | **0.500 ± 0.000** | 0.626 ± 0.029 |

v2 collapses the modal-MLP cells to exact chance across **all three seeds**.
v2a — same asymmetric damage, but v1's domain randomisation — **does not
collapse** (0.621, 0.626). The only thing v2 adds over v2a is the **widened
domain randomisation (P1.2)**.

> **Conclusion: v2's regression was caused by the widened DR drowning the
> discriminative modal signal — not by the asymmetric-damage physics.**
> This is exactly the prior registered in
> [`chunk_regen_v2a_preregistered.md`](chunk_regen_v2a_preregistered.md),
> now confirmed.

---

## 2. v2a against its own pre-registered criteria — REJECTED (marginal)

The v2a pre-registration set a fixed-cell decision rule
([`chunk_regen_v2a_preregistered.md`](chunk_regen_v2a_preregistered.md)).
Evaluated on the 3-seed means:

| # | criterion | threshold | v2a result | pass? |
|---|---|---|---|---|
| C1 | `is_crack/mlp/modal` BA improves (2σ) | ≥ 0.787 | 0.626 | ✗ (improves vs v1 0.596, but not by 2σ) |
| C2 | `col_location/mlp/modal` macro-F1 (2σ) | ≥ 0.329 | 0.115 | ✗ (worse than v1 0.160) |
| C3 | `is_hole/mlp/modal` BA improves (2σ) | ≥ 0.833 | 0.621 | ✗ |
| **C4** | **Floor:** `is_hole/mlp/modal` BA ≥ v1−2σ | ≥ 0.653 | 0.621 ± 0.032 | ✗ (marginal, within ~1σ) |
| C5 | Floor: `binary` best-cell macro-F1 | ≥ 0.310 | 0.491 | ✓ |

Decision rule = ADOPT iff (C1 ∨ C2) ∧ C4 ∧ C5. **C4 fails → v2a is
REJECTED** by its own rule.

Two honest caveats on that rejection:
1. **It is marginal.** C4's floor (0.653) was calibrated against the older
   v1 number (0.661 ± 0.004). The fresh 3-seed v1 re-run gives 0.651 ± 0.010;
   v2a's 0.621 is ~1 noise band below it — a small regression on this one
   cell, nothing like v2's hard collapse.
2. **The asymmetric-damage geometry is benign-to-helpful, not wrong.** On
   the crack cell it *improves* (0.596 → 0.626); it only fails the demanding
   2σ bar. The geometry is not the problem; it simply isn't a decisive win
   on the designated modal cells.

**Net:** neither v2 nor v2a clears its pre-registered bar. **v1 remains the
baseline.** But the *reason* is now understood (widened DR), which makes a
**v2b (widened-DR-only)** unnecessary and points the next physics iteration
at the DR ranges, not the damage model.

---

## 3. DT-stratified picture (best-cell-per-task, exploratory)

The pre-registered rule fixes a single cell per task. Letting the best cell
per task be chosen *on the severity-restricted test set* — and sweeping the
Damage Threshold — shows what each variant's models can do **within their
domain of competence**. These are post-hoc selections (hypothesis-generating,
not confirmatory), but they reveal structure the pooled gate hid.

### 3.1 Tasks where severity restriction reveals hidden learning

Macro-F1 of the best cell per task, by minimum damage severity:

| task | axis | v1 | v2 | v2a |
|---|---|---|---|---|
| **is_bolt** | bolt %loosening | 0.626 → 0.720 → **0.820** (@85 %) | 0.542 → 0.682 → 0.807 | **0.676** → 0.773 → **0.823** |
| **is_hole** | hole Ø mm | 0.619 → **0.659** (@6 mm) | 0.555 → 0.654 | 0.598 → **0.659** |
| **col_location** | tier | 0.179 → 0.309 (severe) | 0.131 → 0.188 | 0.169 → **0.328** |
| **type** (5-cls) | tier | 0.278 → **0.306** | 0.178 → 0.289 | 0.269 → 0.296 |
| **severity** (MAE↓) | bolt %loosening | 0.230 → 0.173 | 0.239 → **0.164** | 0.225 → 0.166 |

Every one of these climbs as low-severity positives are removed — the models
*have* learned the damage signal; the pooled metric is diluted by sub-
threshold damage the spectra barely encode. **`is_bolt` at 85 % loosening
reaches ~0.82 macro-F1 for all three variants** — the most learnable signal
in the study.

### 3.2 Tasks where restriction does NOT help (the real failure mode)

| task | v1 | v2 | v2a | reading |
|---|---|---|---|---|
| **is_pristine** (tier→negatives) | 0.494 → 0.480 → 0.451 | 0.470 → 0.456 → 0.438 | 0.493 → 0.479 → 0.451 | flat-to-falling |
| **binary** (tier→positives) | 0.480 → 0.466 → 0.438 | 0.470 → 0.456 → 0.423 | 0.491 → 0.477 → 0.447 | flat-to-falling |

Restricting the negative set to *unambiguously severe* damage does **not**
improve pristine recognition. The models can name a damage type when it is
severe, but cannot reliably answer "is this damaged at all?" — pristine and
damaged spectra are not separable in the learned representation regardless of
severity. **This, not v2, is the central synth-to-real failure.** It also
explains why the old report's pooled `binary` gate was the worst possible
acceptance metric: it keys on the one question the models cannot answer.

### 3.3 Localization: v2 specifically degrades

| task | v1 | v2 | v2a |
|---|---|---|---|
| `col_location` (severe) | 0.309 | **0.188** | 0.328 |
| `mass_location` (DT-invariant) | 0.429 | **0.259** | 0.451 |

v2's asymmetric placement + widened DR degrades spatial localization
relative to both v1 and v2a. v2a (asymmetric damage, baseline DR) is the
**best** localizer — consistent with §1's "the DR, not the geometry, is the
culprit."

---

## 4. Per-variant summary

- **v1** — the baseline. Best or tied on localization; robust modal-MLP bank
  intact. Remains the reference.
- **v2** — **REJECTED** (confirmed, 3-seed). Modal cells collapse to chance;
  localization degrades. Cause identified: **widened domain randomisation
  (P1.2)**, not the asymmetric-damage geometry. Retains competitive
  damage-*type* and severity-regression signal at high severity, but not
  enough to offset the collapses.
- **v2a** — **REJECTED by pre-registered rule** (C4 floor marginally fails),
  but the failure is within ~1σ and the asymmetric-damage geometry is
  benign-to-helpful (best localizer; improves crack/bolt cells directionally).
  Not adopted, but it served its purpose: it **isolated the cause of v2's
  failure**.

## 5. What changed vs the definitive report

1. The v2 rejection is **upheld and explained** — widened DR, not damage
   physics.
2. A recommended-but-unrun ablation (v2a) was **run**; it answers the
   disentangling question and makes v2b unnecessary.
3. The DT-stratified analysis shows the **binary/is_pristine "detection"
   goal is the true bottleneck**, not any single variant — and that the
   pooled detection gate used to reject v2 was itself the wrong metric.
4. Type, bolt-severity, hole-size and bolt-detection all **transfer at high
   severity** for all three variants — the study's positive results are
   robust to the physics variant.

## 6. Reproduce

```bash
python -m ml_pipeline.dt_compare_variants   # stiffness-reduction sweep
python -m ml_pipeline.dt_feature_sweep      # per-axis + tier sweep
```

Both read the 9 per-case JSONs and the experimental HDF5; no GPU, ~1 min each.
