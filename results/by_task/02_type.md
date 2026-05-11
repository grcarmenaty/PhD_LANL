# Task 2 — Type: Pristine / Bolt / Crack / Hole / Mass

Self-contained walkthrough of every `(model, feature)` cell on the
**5-class damage-type** task.

* [What the task is](#what-the-task-is)
* [Class distribution](#class-distribution)
* [What each *model* is](#what-each-model-is)
* [What each *feature* is](#what-each-feature-is)
* [Per-model results](#per-model-results)
* [Cross-model comparison](#cross-model-comparison)
* [Sim-to-real](#sim-to-real)
* [Recommendation](#recommendation)

Cross-references: [`../PROTOCOL.md`](../PROTOCOL.md),
[`../INTERPRETING_PLOTS.md`](../INTERPRETING_PLOTS.md),
[`../PLOTS.md`](../PLOTS.md), previous task →
[`01_binary.md`](01_binary.md), next task →
[`03_severity.md`](03_severity.md).

---

## What the task is

Given one chirp-driven vibration trial, identify the **damage
mechanism**: Pristine, Bolt loosening, Crack on a column,
drilled Hole on a column, or added Mass on a plate.  Output is
one of 5 class labels.

Downstream uses:

* **Maintenance routing.**  A bolt-loosening alarm and a
  crack alarm trigger different inspection / repair procedures.
* **Decision support.**  Knowing the type narrows the
  subsequent location and severity questions.
* **Synthetic-data validation.**  The classifier scores tell us
  how distinguishable each damage mechanism is in the ROM —
  Crack and Hole both reduce column stiffness, so their
  separability is a non-trivial diagnostic of the dataset.

Random baseline = 0.20 (5 equiprobable classes); class-prior
baseline = 0.20.

## Class distribution

The dataset is perfectly class-balanced:
**2 000 / 2 000 / 2 000 / 2 000 / 2 000** for Pristine / Bolt /
Crack / Hole / Mass.  Train fold 1 400 each; test fold 300 each.

Experimental set, after primary-op label assignment (bolt >
crack > hole > mass > pristine):
**8 / 36 / 6 / 7 / 4** — heavily bolt-skewed because most
composite IQS cases include a `D(xx%) 1BD` bolt loosening.

![class counts and severity distributions](../figures/dataset/class_severity.png)

---

## What each *model* is

(See [`01_binary.md`](01_binary.md#what-each-model-is) for the
full description.  In short: RF / XGB / MLP on flat features;
1-D CNN and Transformer on `frf_mag` / `timeseries`; 2-D CNN on
`cfdac`.)

For the type task all models output **5 logits** and the
training loss is multi-class cross-entropy.

---

## What each *feature* is

(See [`01_binary.md`](01_binary.md#what-each-feature-is) for the
full description.  Synth-vs-real example panels:)

![modal feature](../figures/feature_examples/modal.png)
![indicators feature](../figures/feature_examples/indicators.png)
![frf_mag feature](../figures/feature_examples/frf_mag.png)
![timeseries feature](../figures/feature_examples/timeseries.png)
![cfdac feature](../figures/feature_examples/cfdac.png)

For the type task the most diagnostic features are:

* `ch2_peak1_f`, `ch3_peak1_f`, `ch0_peak1_f` — first-mode
  frequencies at Floors 1, 2, 3 (bolt loosening shifts them
  asymmetrically).
* `ch6_bandE`, `ch2_bandE` — band energies at Floors 2 / 3
  (Mass adds energy at the floor where the mass is placed).
* `unsigned_SCI`, `FRFRMS` — detection-strength scalars (good
  for Pristine vs others but weak at separating Crack from
  Hole).

---

## Per-model results

### Random Forest

#### RF + `modal`

![HPO surface — type RF/modal](../figures/hpo/type__rf__modal.png)
![confusion — type RF/modal](../figures/confusion/type_rf_modal.png)
![importance — type RF/modal](../figures/feat_importance/type_rf_modal.png)

* HPO best : `n_estimators=100, max_depth=None`
* Synthetic : **val 0.815**, **test 0.811**
* Per-class recall : Pristine 0.96 / Bolt 0.85 / Crack 0.64 /
  Hole 0.63 / Mass 0.98
* Experimental : **0.443** (n = 61) — gap **+0.37**
* Top features : `ch6_bandE` (6 %), `ch2_bandE` (5 %),
  `ch2_std_logA` (4 %).

**Interpretation.** Bolt and Mass have characteristic
modal-energy signatures the forest captures.  The **Crack / Hole
pair is the residual confusion**: both reduce column stiffness
similarly, the random-forest splits on `bandE` features cannot
fully separate them.

#### RF + `indicators`

![HPO surface — type RF/indicators](../figures/hpo/type__rf__indicators.png)
![confusion — type RF/indicators](../figures/confusion/type_rf_indicators.png)
![importance — type RF/indicators](../figures/feat_importance/type_rf_indicators.png)

* HPO best : `n_estimators=100, max_depth=None`
* Synthetic : **val 0.757**, **test 0.745**
* Per-class recall : 0.84 / 0.84 / 0.61 / 0.53 / 0.90
* Experimental : **0.164** — gap **+0.58**
* Top features : `unsigned_SCI` (9 %), `FRFRMS` (8 %),
  `RVAC_std` (7 %).

**Interpretation.**  Indicators carry detection signal but
*not* type signal: every indicator is a scalar summary of the
*overall* FRF deviation, so Bolt-vs-Crack-vs-Hole tend to score
similar indicator values.  Pristine recall is still 0.84 because
indicators *do* detect "anything is wrong".

#### RF — feature comparison

| feature     | val   | test  | exp   |
|-------------|-------|-------|-------|
| modal       | 0.815 | 0.811 | 0.443 |
| indicators  | 0.757 | 0.745 | 0.164 |

modal wins by ~7 pp in-domain and by **a factor of 3** on the
experimental data.  The indicator features have larger
sim-to-real drift because the indicators are computed against
the *synthetic* pristine reference — the closer the model
relies on that reference, the worse it transfers.

### XGBoost

#### XGB + `modal`

![HPO surface — type XGB/modal](../figures/hpo/type__xgb__modal.png)
![confusion — type XGB/modal](../figures/confusion/type_xgb_modal.png)
![importance — type XGB/modal](../figures/feat_importance/type_xgb_modal.png)

* HPO best : `n_estimators=300, max_depth=6`
* Synthetic : **val 0.807**, **test 0.822**
* Per-class recall : 0.98 / 0.86 / 0.64 / 0.67 / 0.98
* Experimental : **0.295** — gap **+0.53**
* Top features : `ch2_peak1_f` (13 %), `ch3_peak1_f` (11 %),
  `ch0_peak1_f` (7 %).

**Interpretation.** Boosting picks the **first-mode frequencies
at the three floors** as its top splits — directly mirrors the
physics (loose bolts and cracks shift natural frequencies
differently per floor).

#### XGB + `indicators`

![HPO surface — type XGB/indicators](../figures/hpo/type__xgb__indicators.png)
![confusion — type XGB/indicators](../figures/confusion/type_xgb_indicators.png)
![importance — type XGB/indicators](../figures/feat_importance/type_xgb_indicators.png)

* HPO best : `n_estimators=600, max_depth=6`
* Synthetic : **val 0.774**, **test 0.759**
* Per-class recall : 0.86 / 0.84 / 0.62 / 0.55 / 0.92
* Experimental : **0.148** — gap **+0.61**
* Top features : `unsigned_SCI` (19 %), `FRFRMS` (9 %),
  `M2L_std` (7 %).

**Interpretation.** Boosting concentrates 19 % of importance on
`unsigned_SCI` — but as a single feature it's not enough to
separate the three damage mechanisms.

#### XGB — feature comparison

| feature     | val   | test  | exp   |
|-------------|-------|-------|-------|
| modal       | 0.807 | 0.822 | 0.295 |
| indicators  | 0.774 | 0.759 | 0.148 |

Same direction as RF: modal wins, indicator gap is larger.

### Multilayer Perceptron

#### MLP + `modal`

![HPO surface — type MLP/modal](../figures/hpo/type__mlp__modal.png)
![confusion — type MLP/modal](../figures/confusion/type_mlp_modal.png)

* HPO best : `hidden=(512, 256, 128), lr=3e-3`
* Synthetic : **val 0.869**, **test 0.877**
* Per-class recall : 0.99 / 0.85 / 0.64 / 0.92 / 0.99
* Experimental : **0.443** — gap **+0.43**

**Interpretation.** *The headline result for this task.*  The
MLP is the **only model that breaks 90 % on Hole** — Hole recall
of 0.92 is 28 pp above the next best (0.64).  Crack stays at
0.64.  This means the deep non-linear head *can* read the small
amplitude differences that separate Hole from Crack on the
modal features, while every tree model cannot.

#### MLP + `indicators`

![HPO surface — type MLP/indicators](../figures/hpo/type__mlp__indicators.png)
![confusion — type MLP/indicators](../figures/confusion/type_mlp_indicators.png)

* HPO best : `hidden=(512, 256, 128), lr=3e-3`
* Synthetic : **val 0.703**, **test 0.701**
* Per-class recall : 0.72 / 0.83 / 0.53 / 0.53 / 0.90
* Experimental : **0.066** — gap **+0.63**

**Interpretation.** Same indicator-vector ceiling as the tree
models; the MLP cannot compensate for the missing information.

#### MLP — feature comparison

| feature     | val   | test  | exp   |
|-------------|-------|-------|-------|
| modal       | 0.869 | 0.877 | 0.443 |
| indicators  | 0.703 | 0.701 | 0.066 |

modal wins by 18 pp.

### 1-D CNN

#### 1-D CNN + `frf_mag`

![HPO surface — type CNN/frf_mag](../figures/hpo/type__cnn__frf_mag.png)
![confusion — type CNN/frf_mag](../figures/confusion/type_cnn_frf_mag.png)

* HPO best : `widths=(32, 64, 128), kernel_size=7`
* Synthetic : **val 0.677**, **test 0.689**
* Per-class recall : 0.99 / 0.84 / 0.55 / 0.16 / 0.90
* Experimental : **0.361** — gap **+0.33**

**Interpretation.** Hole recall collapses to **0.16** — the CNN
lumps Hole into Crack (84 % of Hole cases are predicted as
Crack).  This is the worst Crack/Hole confusion across the
model zoo.

#### 1-D CNN + `timeseries`

![HPO surface — type CNN/timeseries](../figures/hpo/type__cnn__timeseries.png)
![confusion — type CNN/timeseries](../figures/confusion/type_cnn_timeseries.png)

* HPO best : `widths=(32, 64, 128), kernel_size=7`
* Synthetic : **val 0.654**, **test 0.657**
* Per-class recall : 0.62 / 0.87 / 0.62 / 0.29 / 0.88
* Experimental : **0.262** — gap **+0.39**

**Interpretation.** Same Hole problem (0.29 recall) plus
Pristine drops to 0.62.  Time-series CNN is the weakest 1-D
model on this task.

#### 1-D CNN — feature comparison

| feature     | val   | test  | exp   |
|-------------|-------|-------|-------|
| frf_mag     | 0.677 | 0.689 | 0.361 |
| timeseries  | 0.654 | 0.657 | 0.262 |

`frf_mag` slightly better on synth and meaningfully better
across-domain.

### Small Transformer

#### Transformer + `frf_mag`

![HPO surface — type Transformer/frf_mag](../figures/hpo/type__transformer__frf_mag.png)
![confusion — type Transformer/frf_mag](../figures/confusion/type_transformer_frf_mag.png)

* HPO best : `d_model=64, n_layers=2`
* Synthetic : **val 0.476**, **test 0.501**
* Per-class recall : 0.56 / 0.69 / 0.27 / 0.41 / 0.58
* Experimental : **0.393** — gap **+0.11**

**Interpretation.** Every class below 70 % recall — the worst
in-domain configuration for this task.  Small sim-to-real gap
because the model never fit the training set strongly to
begin with.

#### Transformer + `timeseries`

![HPO surface — type Transformer/timeseries](../figures/hpo/type__transformer__timeseries.png)
![confusion — type Transformer/timeseries](../figures/confusion/type_transformer_timeseries.png)

* HPO best : `d_model=64, n_layers=2`
* Synthetic : **val 0.557**, **test 0.576**
* Per-class recall : 0.91 / 0.66 / 0.68 / 0.25 / 0.37
* Experimental : **0.295** — gap **+0.28**

**Interpretation.** Pristine is recovered (0.91) but Hole and
Mass drop.

#### Transformer — feature comparison

| feature     | val   | test  | exp   |
|-------------|-------|-------|-------|
| frf_mag     | 0.476 | 0.501 | 0.393 |
| timeseries  | 0.557 | 0.576 | 0.295 |

timeseries wins in-domain; frf_mag transfers slightly better.

### 2-D CNN

#### 2-D CNN + `cfdac`

![HPO surface — type CNN2D/cfdac](../figures/hpo/type__cnn2d__cfdac.png)
![confusion — type CNN2D/cfdac](../figures/confusion/type_cnn2d_cfdac.png)

* HPO best : `widths=(16, 32, 64), kernel_size=5`
* Synthetic : **val 0.796**, **test 0.803**
* Per-class recall : 0.97 / 0.85 / 0.70 / 0.59 / 0.91
* Experimental : **0.426** — gap **+0.38**

**Interpretation.** The 2-D CNN on CFDAC is **the only deep
model that competes with the engineered features** — within
7 pp of the modal MLP.  Hole recall (0.59) is much better than
the 1-D CNN's 0.16, indicating CFDAC preserves the
off-diagonal coupling that distinguishes Hole from Crack.

---

## Cross-model comparison

Sorted by synthetic test accuracy.

| model       | feature     | val   | test  | exp   | gap   |
|-------------|-------------|-------|-------|-------|-------|
| MLP         | modal       | 0.869 | **0.877** | 0.443 | +0.43 |
| XGB         | modal       | 0.807 | 0.822 | 0.295 | +0.53 |
| RF          | modal       | 0.815 | 0.811 | 0.443 | +0.37 |
| 2-D CNN     | cfdac       | 0.796 | 0.803 | 0.426 | +0.38 |
| XGB         | indicators  | 0.774 | 0.759 | 0.148 | +0.61 |
| RF          | indicators  | 0.757 | 0.745 | 0.164 | +0.58 |
| MLP         | indicators  | 0.703 | 0.701 | 0.066 | +0.63 |
| 1-D CNN     | frf_mag     | 0.677 | 0.689 | 0.361 | +0.33 |
| 1-D CNN     | timeseries  | 0.654 | 0.657 | 0.262 | +0.39 |
| Transformer | timeseries  | 0.557 | 0.576 | 0.295 | +0.28 |
| Transformer | frf_mag     | 0.476 | 0.501 | 0.393 | +0.11 |

![per-class F1 — type](../figures/perclass_f1/type.png)

**Observations.**

1. The "Hole column" of the per-class F1 heatmap is
   conspicuously dark for almost every model.  `mlp/modal` is
   the one bright cell.
2. Mass and Pristine are the easiest classes — both have
   characteristic modal-energy signatures.
3. CFDAC + 2-D CNN preserves the structural off-diagonal
   information that distinguishes Crack from Hole, beating
   every 1-D deep model.
4. Indicators are a *detection* feature, not a *type* feature
   — see the t-SNE plot
   [`../figures/embedding/tsne_indicators.png`](../figures/embedding/tsne_indicators.png)
   where Bolt / Crack / Hole are tangled in the same cluster.

---

## Sim-to-real

* Best experimental cell: tie between **MLP+modal** and
  **RF+modal** at **0.443**, roughly half the synthetic test
  accuracy.
* Indicator-feature cells transfer particularly badly because
  the indicators are anchored to the *synthetic* pristine
  reference; the experimental pristine has a different
  calibration.
* Composite IQS cases (e.g. `D(85%) 1BD + Mass First Floor`)
  are mapped to a single primary type for evaluation, so a
  perfectly correct prediction on a composite case still only
  matches one of its true components.  This caps the
  experimental score even for a perfect classifier.

---

## Recommendation

* **Primary classifier for type detection.**  Use
  **MLP + modal** (test 0.877).  It is the only configuration
  that breaks 90 % on Hole.
* **Tree-based alternative** for interpretability:
  **XGB + modal** (test 0.822) with sensor-channel-level
  feature importance.
* **Deep-learning baseline** when the upstream pipeline
  produces CFDAC anyway: **2-D CNN + CFDAC** (test 0.803).
* **Avoid** indicator-only models if cross-domain transfer
  matters — the synthetic-reference anchor leaks badly.

Continue to → [`03_severity.md`](03_severity.md).
