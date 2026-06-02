# DT/IT-swept evaluation — reproducing the paper's sensitivity methodology

> **Note.** Extended to a 3-way variant comparison (v1/v2/v2a) with a
> per-damage-type physical-axis sweep in
> [`REPORT_dt_3way_verdict.md`](REPORT_dt_3way_verdict.md).

**Date:** 2026-05-28. **Trigger:** the question *"are you accounting for what
happens with ever-more-restrictive minimum damage?"* — i.e. the Damage
Threshold / Improvement Threshold sensitivity analysis from the CFDAC
transfer-learning paper (Figs 9, 12, 13). Until now every metric in this
project (the v1/v2/v2a gate, the modal-gap diagnostic) was a single pooled
balanced-accuracy over the whole severity range — the paper's pessimistic
"no threshold" case.

**Tool:** [`ml_pipeline/dt_sweep.py`](../ml_pipeline/dt_sweep.py). Pure
post-processing on the existing `experimental_full_per_case*.json`
predictions (no retraining). A damaged test sample is kept only if its
stiffness reduction ≥ DT; the threshold is swept. Stiffness reduction is
derived from the calibrated `variation.py` ratio functions:

| type | experimental severities | stiffness reduction |
|---|---|---|
| bolt | 11 / 20 / 50 / 85 % | 0.15 / 0.30 / 0.45 / 0.61 |
| crack | 5 / 8 mm | 0.04 / 0.064 |
| hole | 4 / 6 mm | 0.02 / 0.03 |
| mass | 1.2 kg | n/a (inertia, not stiffness) |

## Headline result — is_bolt reproduces the paper's effect

`is_bolt/mlp/modal`, balanced accuracy vs minimum stiffness reduction
(v2a = 3-seed mean, v1 = single seed):

| min Δstiffness | n_bolt | v2a BA | v1 BA | dropped→non-damage |
|---|---|---|---|---|
| ≥ 0 (all) | 1338 | 0.589 | 0.554 | — |
| ≥ 20 % | 938 | 0.631 | 0.593 | 0.91 |
| ≥ 45 % | 538 | **0.765** | **0.707** | 0.95 |

Exactly the paper's finding: **accuracy climbs monotonically as the minimum
required alteration increases** (0.59 → 0.63 → 0.77), and the desirable
failure mode holds — **91–95 % of the excluded sub-threshold bolts were
being predicted as non-damage / pristine**, not misclassified. The pooled
0.589 understates the operationally-relevant 0.765 by ~0.18.

## Reframing the is_hole / is_crack "failure"

The experimental crack/hole damage is physically near-pristine: holes change
stiffness by only **2–3 %**, cracks by **4–6 %**. There is essentially no DT
range to climb — at any DT ≥ 0.05 the positive class empties out:

| cell | DT=0 (pooled) | DT≥0.05 | dropped→pristine |
|---|---|---|---|
| is_hole/mlp/modal | 0.621 (v2a) / 0.665 (v1) | undefined (0 holes ≥5%) | 0.63 |
| is_crack/mlp/modal | 0.626 / 0.603 | 0.594 (8 mm only) | 0.61 |

So the low is_hole/is_crack scores are **not a model failure** — they are
the accuracy *at the detection floor*, measured on alterations that barely
perturb the structure. 63 % of holes are (desirably) read as pristine. Any
fair report of these tasks must state the severity regime; a pooled scalar
implies a discrimination failure that the physics does not support.

## Impact on the v2a verdict

Under DT stratification v2a is **not uniformly worse** than v1: it is
slightly *better* on is_bolt at every threshold (+0.035 → +0.058) and on
is_crack at DT=0 (+0.023), and worse only on is_hole pooled (−0.044). The
asymmetric-damage geometry's effects are small and consistent across the DT
axis — this does not overturn the v2a REJECT (the is_hole floor still
fails), but it shows the rejection was driven by a near-undetectable damage
regime, and v2a is closer to a wash than the single pooled number implied.

## Recommendation

1. **Report detection/typology tasks as DT curves, not pooled scalars** —
   especially is_hole/is_crack, whose experimental severities sit at the
   2–6 % stiffness floor. State n_pos at each threshold.
2. **Re-judge future generation variants on the is_bolt DT curve**, where
   there is a real severity range and the metric is well-conditioned.
3. The pooled gate (C1–C5) should be supplemented with a DT-anchored
   criterion (e.g. BA at ≥45 % stiffness reduction) so that improvements on
   *detectable* damage are not masked by near-pristine cases.

Caveat: v1 per-case is a single surviving seed; v2a is 3-seed mean. The DT
*shape* is robust, but the v1↔v2a deltas carry single-seed v1 noise.
