# V2a chunk-regeneration — RESULTS (REJECT)

**Run window:** 2026-05-25 16:45 → 2026-05-28 12:15 UTC (3 seeds, ~12 reaper
interruptions, all resumed from cache).
**Pre-registration:** [`chunk_regen_v2a_preregistered.md`](chunk_regen_v2a_preregistered.md).
**Decision artifact:** [`chunk_regen_v2a_decision.json`](chunk_regen_v2a_decision.json).

## What v2a tested

v2a isolates **one** of the two changes that the rejected v2 run bundled:

* v2 = widened domain randomisation (P1.2) **+** asymmetric Crack/Hole
  damage geometry (P2.1/P2.2).
* **v2a = v1's domain randomisation + only the asymmetric damage geometry.**

The goal was to attribute v2's catastrophic regression (is_hole/mlp/modal
balanced-acc 0.661 → 0.500, chance) to one of the two changes.

## Result: REJECT (3-seed gate)

Noise band = 2 × p90(torch-family seed scatter) = 0.172.

| # | Criterion | v1 | v2a (3-seed) | threshold | pass |
|---|---|---|---|---|---|
| C1 | is_crack/mlp/modal BA | 0.615 | 0.626 | ≥0.787 | ✗ |
| C2 | col_location/mlp/modal macro-F1 | 0.157 | 0.115 | ≥0.329 | ✗ |
| C3 | is_hole/mlp/modal BA improves | 0.661 | 0.621 | ≥0.833 | ✗ |
| **C4** | **is_hole/mlp/modal BA — no regression (floor)** | **0.661** | **0.621** | **≥0.653** | **✗** |
| C5 | binary best macro-F1 (floor) | 0.482 | 0.491 | ≥0.310 | ✓ |

**Decision rule:** REJECT if C4 or C5 fails. **C4 fails → REJECT.**

### Per-seed detail of the floor cell

| seed | is_hole/mlp/modal BA | is_crack/mlp/modal BA | col_location/mlp/modal F1 |
|---|---|---|---|
| 42 | 0.658 | 0.652 | 0.112 |
| 101 | 0.625 | 0.641 | 0.162 |
| 202 | 0.580 | 0.585 | 0.072 |
| **mean** | **0.621 (sd 0.032)** | 0.626 | 0.115 |

The is_hole cell degrades monotonically across the seed sweep
(0.658 → 0.625 → 0.580); the regression is consistent, not a single-seed
artefact.

## Scientific conclusion

Comparing the three runs on the critical is_hole/mlp/modal cell:

| run | is_hole/mlp/modal BA | what changed vs v1 |
|---|---|---|
| v1 | 0.661 | baseline |
| **v2a** | **0.621** | + asymmetric damage only |
| v2 | 0.500 | + asymmetric damage + widened DR |

1. **Widened DR (P1.2) was the dominant regression driver.** It alone
   accounts for the 0.621 → 0.500 collapse — the bulk of v2's damage.
   This vindicates the original suspicion in the v2 post-mortem.

2. **But the asymmetric Crack/Hole geometry (P2.1/P2.2) is also net-harmful
   on its own.** Even with v1's unchanged DR, it costs the is_hole modal-MLP
   cell ~0.04 BA (0.661 → 0.621, past the 2σ floor) and provides no
   compensating gain: is_crack improves only +0.011 (far below the 0.172
   significance band) and col_location actually regresses −0.042.

The asymmetric-damage hypothesis — that mapping damage to a single
end-pair of columns would break the col_location degeneracy — is **not
supported**. In-domain synth val scores rose (is_crack cnn2d val ~0.86 vs
v1's 0.80 ceiling), but that extra in-domain separability did **not**
transfer to the experimental bookcase; if anything it widened the
synth-real gap on the modal features.

## Recommendation

**Keep v1 as the generation baseline.** Do not promote either v2 or v2a.
The col_location / Crack synth-real gap is not a geometric-symmetry
artefact and needs a different intervention (candidate directions: revisit
the modal feature extractor's sensitivity to the per-column stiffness
pattern, or the FRF→modal projection, rather than the damage geometry).

## Compute / robustness notes

* The remote container reaped the detached supervisor ~12 times during the
  run (every ~30 min of session inactivity). Every restart resumed from the
  per-cell JSON cache with zero recomputation; the only cost was the dead
  cell's partial work. Per-seed eval JSONs were committed immediately on
  completion, so no completed seed was ever lost.
* `ml_pipeline/variation_v2a.py`, the `--variation v2a` flag in
  `generate_dataset.py`, and the `--label` parameterisation of
  `compare_v1_v2.py` are retained for future ablations.
