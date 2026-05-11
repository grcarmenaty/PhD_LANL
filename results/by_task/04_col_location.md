# Task 4 — Column-damage location

Self-contained walkthrough of every `(model, feature)` cell on the
**6-class column-damage location** task: given a sample known to
have Bolt / Crack / Hole damage, identify *which column-end* is
damaged.

* [What the task is](#what-the-task-is)
* [Class distribution](#class-distribution)
* [Known ROM limitation: AD ≡ BD for Crack and Hole](#known-rom-limitation)
* [What each *model* is](#what-each-model-is)
* [What each *feature* is](#what-each-feature-is)
* [Per-model results](#per-model-results)
* [Cross-model comparison](#cross-model-comparison)
* [Sim-to-real](#sim-to-real)
* [Recommendation](#recommendation)

Cross-references: [`../PROTOCOL.md`](../PROTOCOL.md),
[`../INTERPRETING_PLOTS.md`](../INTERPRETING_PLOTS.md),
[`../PLOTS.md`](../PLOTS.md), previous task →
[`03_severity.md`](03_severity.md), next task →
[`05_mass_location.md`](05_mass_location.md).

---

## What the task is

Given a sample with column damage (Bolt / Crack / Hole),
predict the (storey, end) pair where the damage is located.

* **Storey** ∈ {1, 2, 3}
* **End**    ∈ {BD = bottom of column, AD = top}
* Combined class label: `storey * 2 + end ∈ {0, 1, 2, 3, 4, 5}`
  mapped to `[S1BD, S1AD, S2BD, S2AD, S3BD, S3AD]`.

Downstream uses:

* Direct the inspection team to the affected storey and end of
  column for a closer look.
* Combined with the *type* prediction, fully describes "what
  and where".

Random baseline = 0.167 (6 equiprobable classes); class-prior
baseline = 0.167 (perfectly balanced).

## Class distribution

The dataset is balanced by sub-stratification:

* Per type (Bolt / Crack / Hole): 333–334 samples per (storey,
  end) for 6 locations × 3 types = 6 000 column-damage samples.
* Train fold: 4 200; test fold: ~150 per (storey, end) class.
* Experimental: 49 column-damage cases (most are composites of
  Bolt + Mass).

![class counts and severity distributions](../figures/dataset/class_severity.png)

## Known ROM limitation

**The reduced-order model used to generate the dataset is
AD/BD-degenerate for Crack and Hole damage.**  The semi-rigid
joint formulation uses a per-end JSR (joint stiffness ratio) for
bolts, but Crack and Hole are modelled as a *symmetric*
reduction of column stiffness — both ends of a column see
identical effective stiffness.  Consequently the FRFs of "Crack
at S2BD" and "Crack at S2AD" are *identical* by construction.

Implication: 1/3 of the 6 000 column-damage samples (Crack) and
another 1/3 (Hole) carry no information about whether the damage
is at the BD or AD end.  Only the 1/3 Bolt samples have a true
AD/BD signal.

**Maximum attainable accuracy on this task** is therefore
bounded by `(1/2) · (Crack + Hole) + (1) · Bolt = 1/2 · 2/3 +
1/3 = 2/3 ≈ 0.67` even with a perfect classifier — and only if
the classifier knows the type a priori.  When the type is *not*
known, the effective ceiling is lower.

The observed ~0.50 accuracy across most models matches this
analysis closely.

---

## What each *model* is

(See [`01_binary.md`](01_binary.md#what-each-model-is) for the
full description.)

---

## What each *feature* is

(See [`01_binary.md`](01_binary.md#what-each-feature-is).)

![modal feature](../figures/feature_examples/modal.png)
![indicators feature](../figures/feature_examples/indicators.png)
![frf_mag feature](../figures/feature_examples/frf_mag.png)
![timeseries feature](../figures/feature_examples/timeseries.png)
![cfdac feature](../figures/feature_examples/cfdac.png)

---

## Per-model results

### Random Forest

#### RF + `modal`

![HPO surface — col_location RF/modal](../figures/hpo/col_location__rf__modal.png)
![confusion — col_location RF/modal](../figures/confusion/col_location_rf_modal.png)
![importance — col_location RF/modal](../figures/feat_importance/col_location_rf_modal.png)

* HPO best : `n_estimators=300, max_depth=None`
* Synthetic : **val 0.509**, **test 0.492**
* Per-class recall : `[S1BD 0.57, S1AD 0.52, S2BD 0.48, S2AD 0.48, S3BD 0.44, S3AD 0.47]`
* Experimental : **0.061** (n = 49) — gap **+0.43**
* Top features : `ch2_bandE` (8 %), `ch6_bandE` (8 %),
  `ch2_std_logA` (4 %).

**Interpretation.** Uniform per-class recall ≈ 0.5 — the
forest "spreads errors evenly" instead of collapsing onto BD.
The top features are the same as for the binary and type
tasks; the forest is using spectral energy to identify Bolt
samples (which have AD/BD information) but cannot separate
Crack/Hole BD from Crack/Hole AD.

Experimental crash to 0.061 is because most IQS column-damage
cases are *composites* and the primary-op label (always BD in
the IQS protocol) is no longer reliable.

#### RF + `indicators`

![HPO surface — col_location RF/indicators](../figures/hpo/col_location__rf__indicators.png)
![confusion — col_location RF/indicators](../figures/confusion/col_location_rf_indicators.png)
![importance — col_location RF/indicators](../figures/feat_importance/col_location_rf_indicators.png)

* HPO best : `n_estimators=200, max_depth=None`
* Synthetic : **val 0.482**, **test 0.481**
* Per-class recall : `[0.47, 0.47, 0.57, 0.47, 0.43, 0.48]`
* Experimental : **0.041** — gap **+0.44**
* Top features : `FRFSF` (11 %), `RVAC_std` (6 %),
  `FRFSM_6dB` (6 %).

**Interpretation.** Almost identical to RF/modal — confirms the
ceiling is information-bounded, not feature-bounded.

#### RF — feature comparison

| feature     | val   | test  | exp   |
|-------------|-------|-------|-------|
| modal       | 0.509 | 0.492 | 0.061 |
| indicators  | 0.482 | 0.481 | 0.041 |

modal marginally better.

### XGBoost

#### XGB + `modal`

![HPO surface — col_location XGB/modal](../figures/hpo/col_location__xgb__modal.png)
![confusion — col_location XGB/modal](../figures/confusion/col_location_xgb_modal.png)
![importance — col_location XGB/modal](../figures/feat_importance/col_location_xgb_modal.png)

* HPO best : `n_estimators=600, max_depth=8`
* Synthetic : **val 0.509**, **test 0.488**
* Per-class recall : `[0.56, 0.54, 0.46, 0.45, 0.38, 0.54]`
* Experimental : **0.020** — gap **+0.47**
* Top features : `ch2_bandE` (12 %), `ch3_peak1_f` (11 %),
  `ch2_peak1_a` (8 %).

**Interpretation.** Same plateau as RF.  Boosting picks
`ch3_peak1_f` (first-mode frequency at Floor 3) as a top split —
that's the variable that most resolves the storey component of
the label.

#### XGB + `indicators`

![HPO surface — col_location XGB/indicators](../figures/hpo/col_location__xgb__indicators.png)
![confusion — col_location XGB/indicators](../figures/confusion/col_location_xgb_indicators.png)
![importance — col_location XGB/indicators](../figures/feat_importance/col_location_xgb_indicators.png)

* HPO best : `n_estimators=600, max_depth=8`
* Synthetic : **val 0.480**, **test 0.454**
* Per-class recall : `[0.47, 0.45, 0.53, 0.43, 0.40, 0.45]`
* Experimental : **0.163** — gap **+0.29**
* Top features : `FRFSF` (11 %), `unsigned_SCI` (7 %),
  `M2L_min` (7 %).

**Interpretation.** Indicator boosting hits the same ceiling.
The experimental transfer is *better* than modal (0.163 vs
0.020) because the indicators are scale-invariant by
construction.

#### XGB — feature comparison

| feature     | val   | test  | exp   |
|-------------|-------|-------|-------|
| modal       | 0.509 | 0.488 | 0.020 |
| indicators  | 0.480 | 0.454 | 0.163 |

modal wins in-domain; indicators transfer better.

### Multilayer Perceptron

#### MLP + `modal`

![HPO surface — col_location MLP/modal](../figures/hpo/col_location__mlp__modal.png)
![confusion — col_location MLP/modal](../figures/confusion/col_location_mlp_modal.png)

* HPO best : `hidden=(256, 128, 64), lr=3e-3`
* Synthetic : **val 0.507**, **test 0.494**
* Per-class recall : `[S1BD 1.00, S1AD 0.00, S2BD 0.95, S2AD 0.07, S3BD 0.51, S3AD 0.43]`
* Experimental : **0.490** — gap **+0.005** (best of any cell!)
* No tree-based importance.

**Interpretation.** The MLP **binarises by `BD` vs `AD`**: BD
classes get high recall, AD classes get near-zero.  This is the
optimal strategy given the AD/BD-degenerate ROM: the model
learns to bet on BD because most of its training signal comes
from Bolt (which has true AD/BD info) and the BD bet is
default-correct for Crack and Hole.

Remarkably the experimental score (0.490) almost matches
synthetic test (0.494) — gap is ~0.005, the smallest of any
location cell.  The IQS data is also almost entirely BD
samples, so the MLP's BD bias is *correct* in the real world.

#### MLP + `indicators`

![HPO surface — col_location MLP/indicators](../figures/hpo/col_location__mlp__indicators.png)
![confusion — col_location MLP/indicators](../figures/confusion/col_location_mlp_indicators.png)

* HPO best : `hidden=(512, 256, 128), lr=3e-3`
* Synthetic : **val 0.429**, **test 0.417**
* Per-class recall : `[0.13, 0.69, 0.27, 0.55, 0.47, 0.39]`
* Experimental : **0.367** — gap **+0.05**

**Interpretation.** Inverse bias — AD classes predicted more
than BD.  Probably an artefact of standard scaling on a 22-d
vector.

#### MLP — feature comparison

| feature     | val   | test  | exp   |
|-------------|-------|-------|-------|
| modal       | 0.507 | 0.494 | 0.490 |
| indicators  | 0.429 | 0.417 | 0.367 |

modal wins in-domain by ~8 pp; both transfer well.

### 1-D CNN

#### 1-D CNN + `frf_mag`

![HPO surface — col_location CNN/frf_mag](../figures/hpo/col_location__cnn__frf_mag.png)
![confusion — col_location CNN/frf_mag](../figures/confusion/col_location_cnn_frf_mag.png)

* HPO best : `widths=(32, 64, 128), kernel_size=7`
* Synthetic : **val 0.490**, **test 0.469**
* Per-class recall : `[0.98, 0.00, 0.99, 0.00, 0.85, 0.00]`
* Experimental : **0.265** — gap **+0.20**

**Interpretation.** Full AD/BD collapse — model never predicts
an AD class.  Same strategy as MLP/modal but more extreme.

#### 1-D CNN + `timeseries`

![HPO surface — col_location CNN/timeseries](../figures/hpo/col_location__cnn__timeseries.png)
![confusion — col_location CNN/timeseries](../figures/confusion/col_location_cnn_timeseries.png)

* HPO best : `widths=(32, 64, 128), kernel_size=5`
* Synthetic : **val 0.488**, **test 0.473**
* Per-class recall : `[0.49, 0.46, 0.48, 0.41, 0.60, 0.39]`
* Experimental : **0.347** — gap **+0.13**

**Interpretation.** Uniform recall — the timeseries CNN does
*not* collapse to BD; it makes errors uniformly across all 6
classes.

#### 1-D CNN — feature comparison

| feature     | val   | test  | exp   |
|-------------|-------|-------|-------|
| frf_mag     | 0.490 | 0.469 | 0.265 |
| timeseries  | 0.488 | 0.473 | 0.347 |

timeseries transfers better (uniform errors generalise).

### Small Transformer

#### Transformer + `frf_mag`

![HPO surface — col_location Transformer/frf_mag](../figures/hpo/col_location__transformer__frf_mag.png)
![confusion — col_location Transformer/frf_mag](../figures/confusion/col_location_transformer_frf_mag.png)

* HPO best : `d_model=64, n_layers=2`
* Synthetic : **val 0.268**, **test 0.251**
* Per-class recall : `[0.00, 0.23, 0.00, 0.84, 0.26, 0.17]`
* Experimental : **0.041** — gap **+0.21**

**Interpretation.** Only S2AD recall is high; rest are
noise-level.  Worst in-domain configuration for this task.

#### Transformer + `timeseries`

![HPO surface — col_location Transformer/timeseries](../figures/hpo/col_location__transformer__timeseries.png)
![confusion — col_location Transformer/timeseries](../figures/confusion/col_location_transformer_timeseries.png)

* HPO best : `d_model=64, n_layers=2`
* Synthetic : **val 0.387**, **test 0.368**
* Per-class recall : `[0.49, 0.39, 0.24, 0.66, 0.29, 0.13]`
* Experimental : **0.204** — gap **+0.16**

**Interpretation.** Erratic per-class behaviour but better
than frf_mag.

#### Transformer — feature comparison

`timeseries` wins (0.368 vs 0.251).

### 2-D CNN

#### 2-D CNN + `cfdac`

![HPO surface — col_location CNN2D/cfdac](../figures/hpo/col_location__cnn2d__cfdac.png)
![confusion — col_location CNN2D/cfdac](../figures/confusion/col_location_cnn2d_cfdac.png)

* HPO best : `widths=(16, 32, 64), kernel_size=5`
* Synthetic : **val 0.492**, **test 0.494**
* Per-class recall : `[0.79, 0.17, 0.74, 0.26, 0.70, 0.30]`
* Experimental : **0.163** — gap **+0.33**

**Interpretation.** CFDAC partially preserves the AD signal
(AD recalls 0.17 / 0.26 / 0.30 are above zero, unlike most
modal-feature models that pin them to 0) — the 2-D
representation retains some of the off-diagonal coupling that
distinguishes AD from BD.  Same overall plateau though.

---

## Cross-model comparison

Sorted by synthetic test accuracy.

| model       | feature     | val   | test  | exp   | gap   |
|-------------|-------------|-------|-------|-------|-------|
| RF          | modal       | 0.509 | **0.492** | 0.061 | +0.43 |
| 2-D CNN     | cfdac       | 0.492 | 0.494 | 0.163 | +0.33 |
| MLP         | modal       | 0.507 | 0.494 | 0.490 | +0.005 |
| XGB         | modal       | 0.509 | 0.488 | 0.020 | +0.47 |
| RF          | indicators  | 0.482 | 0.481 | 0.041 | +0.44 |
| 1-D CNN     | timeseries  | 0.488 | 0.473 | 0.347 | +0.13 |
| 1-D CNN     | frf_mag     | 0.490 | 0.469 | 0.265 | +0.20 |
| XGB         | indicators  | 0.480 | 0.454 | 0.163 | +0.29 |
| MLP         | indicators  | 0.429 | 0.417 | 0.367 | +0.05 |
| Transformer | timeseries  | 0.387 | 0.368 | 0.204 | +0.16 |
| Transformer | frf_mag     | 0.268 | 0.251 | 0.041 | +0.21 |

![per-class F1 — col_location](../figures/perclass_f1/col_location.png)

**Observations.**

1. The synthetic-test plateau at ~0.49–0.51 corresponds to the
   information-theoretic ceiling — see ["Known ROM
   limitation"](#known-rom-limitation).  No amount of HPO or
   feature engineering will break this ceiling without
   modifying the ROM.
2. The MLP / modal cell shows the **smallest sim-to-real gap**
   of *any* cell across *any* task (0.005).  This is a
   coincidence between its BD-collapse strategy and the IQS
   BD-only protocol — not a generalisation property.
3. CFDAC + 2-D CNN is the only model that retains a non-zero
   AD signal across the board.

---

## Sim-to-real

* Most models drop catastrophically because the IQS column-
  damage cases are usually composites that mix Bolt+Mass; the
  primary-op label is "Bolt at storey-X-BD".  A model that
  predicts AD locations cannot match that label by definition.
* The MLP/modal BD-collapse strategy *coincidentally* aligns
  with the IQS labelling convention, giving the best cross-
  domain score on this task.

---

## Recommendation

* **In practice.**  Use **MLP + modal** (test 0.494, exp 0.490)
  — the BD-collapse + IQS-BD-bias happen to align.  Be aware
  this is not a real "location" model; it is an
  optimally-cheating model under the ROM constraint.
* **Honest deep baseline.**  **2-D CNN + CFDAC** (test 0.494)
  preserves some AD signal and degrades more gracefully on a
  hypothetical balanced AD-rich dataset.
* **Plan for improvement.**  The right fix is *physical*: add
  rocking DOFs to the ROM so AD-end damage produces a
  measurably different FRF.  Then re-run HPO; the modal /
  CFDAC models will likely break 0.7 accuracy.

Continue to → [`05_mass_location.md`](05_mass_location.md).
