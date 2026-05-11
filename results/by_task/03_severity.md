# Task 3 — Severity regression

Self-contained walkthrough of every `(model, feature)` cell on the
**severity-regression** task: estimate *how much* damage the
structure has, conditional on it being damaged.

* [What the task is](#what-the-task-is)
* [Target distribution](#target-distribution)
* [What each *model* is](#what-each-model-is)
* [What each *feature* is](#what-each-feature-is)
* [Per-model results](#per-model-results)
* [Cross-model comparison](#cross-model-comparison)
* [Sim-to-real](#sim-to-real)
* [Recommendation](#recommendation)

Cross-references: [`../PROTOCOL.md`](../PROTOCOL.md),
[`../INTERPRETING_PLOTS.md`](../INTERPRETING_PLOTS.md),
[`../PLOTS.md`](../PLOTS.md), previous task →
[`02_type.md`](02_type.md), next task →
[`04_col_location.md`](04_col_location.md).

---

## What the task is

Given a damaged trial, estimate a **continuous severity value
in `[0, 1]`**, normalised within each damage type:

* Bolt loosening: `severity = (pct − 5) / (95 − 5)` for pct ∈ [5, 95]
* Crack length: `(mm − 1) / (8 − 1)` for mm ∈ [1, 8]
* Hole diameter: `(mm − 1) / (6 − 1)` for mm ∈ [1, 6]
* Mass: `(kg − 0.1) / (2.5 − 0.1)` for kg ∈ [0.1, 2.5]

Pristine samples are excluded — only 8 000 samples (training
fold 5 600, test fold 1 200).

Downstream uses:

* Prioritise inspection schedule by inferred severity.
* Set thresholds for "immediate action" vs "watch".
* Validate that synthetic and experimental severities live on
  the same monotonic curve.

Random baseline (predict mean) → R² = 0; predict-the-true-mean
on the test fold gives R² = 0 by construction.

## Target distribution

Severity is sampled **uniformly** within each type's bounded
range, so the normalised target is essentially flat on `[0, 1]`
inside the 8 000 damage samples.

![class counts and severity distributions](../figures/dataset/class_severity.png)

---

## What each *model* is

(See [`01_binary.md`](01_binary.md#what-each-model-is) for the
full description.  Regression heads: `RandomForestRegressor`,
`XGBRegressor`, `MLP` with linear output, 1-D CNN / Transformer /
2-D CNN with a single linear output and MSE loss.)

---

## What each *feature* is

(See [`01_binary.md`](01_binary.md#what-each-feature-is).)

![modal feature](../figures/feature_examples/modal.png)
![indicators feature](../figures/feature_examples/indicators.png)
![frf_mag feature](../figures/feature_examples/frf_mag.png)
![timeseries feature](../figures/feature_examples/timeseries.png)
![cfdac feature](../figures/feature_examples/cfdac.png)

For severity specifically, the most predictive variables are:

* `ch2_bandE`, `ch6_bandE` — band energies (energy ≈ amount of
  damage).
* `FRFRMS` — log-FRF root-mean-square deviation (a natural
  severity scalar).
* `ch2_peak1_f` — first-mode frequency at Floor 2 (severity
  monotonically shifts it).
* `GAC_min` — minimum Global Amplitude Criterion (sensitive
  to the deepest amplitude valley).

---

## Per-model results

### Random Forest

#### RF + `modal`

![HPO surface — severity RF/modal](../figures/hpo/severity__rf__modal.png)
![scatter — severity RF/modal](../figures/scatter/severity_rf_modal.png)
![importance — severity RF/modal](../figures/feat_importance/severity_rf_modal.png)

* HPO best : `n_estimators=300, max_depth=None`
* Synthetic : **val R² 0.593**, **test R² 0.573**, MAE 0.130, bias −0.007
* Experimental : R² −0.151 (n = 53), MAE 0.299 — gap **0.72**
* Top features : `ch2_bandE` (8 %), `ch1_mean_logA` (8 %),
  `ch6_bandE` (8 %).

**Interpretation.** *The headline result for severity.*  The
scatter plot tracks the diagonal closely except at the extremes
where leaf-value saturation pulls predictions toward the mean
(visible negative-skew residual histogram).  Top features are
spectral energies at three floors — severity ≈ energy.

#### RF + `indicators`

![HPO surface — severity RF/indicators](../figures/hpo/severity__rf__indicators.png)
![scatter — severity RF/indicators](../figures/scatter/severity_rf_indicators.png)
![importance — severity RF/indicators](../figures/feat_importance/severity_rf_indicators.png)

* HPO best : `n_estimators=300, max_depth=None`
* Synthetic : **val R² 0.498**, **test R² 0.487**, MAE 0.146
* Experimental : R² −0.422, MAE 0.301 — gap **0.91**
* Top features : `FRFRMS` (20 %), `unsigned_SCI` (10 %),
  `M2L_std` (7 %).

**Interpretation.** `FRFRMS` carries 20 % of the importance
alone — it is the natural log-error-magnitude severity scalar.
But the indicator vector simply has less information than the
modal one: the scatter has a much flatter cloud (more
predict-the-mean behaviour).

#### RF — feature comparison

| feature     | val R² | test R² | exp R² |
|-------------|--------|---------|--------|
| modal       | 0.593  | 0.573   | −0.151 |
| indicators  | 0.498  | 0.487   | −0.422 |

modal wins by ~0.09 R² in-domain and by ~0.27 R² cross-domain.

### XGBoost

#### XGB + `modal`

![HPO surface — severity XGB/modal](../figures/hpo/severity__xgb__modal.png)
![scatter — severity XGB/modal](../figures/scatter/severity_xgb_modal.png)
![importance — severity XGB/modal](../figures/feat_importance/severity_xgb_modal.png)

* HPO best : `n_estimators=300, max_depth=8`
* Synthetic : **val R² 0.551**, **test R² 0.532**, MAE 0.137
* Experimental : R² −0.062 — gap **0.60**
* Top features : `ch2_peak1_f` (15 %), `ch1_mean_logA` (11 %),
  `ch2_bandE` (9 %).

**Interpretation.** Wider tails than RF — boosting trees over-
predict at the upper end, where samples are sparse, instead of
saturating.  Experimental R² is the *best* of any model
(−0.062 ≈ predict-the-mean), suggesting boosting trees transfer
slightly better than other models even though they overshoot
in-domain.

#### XGB + `indicators`

![HPO surface — severity XGB/indicators](../figures/hpo/severity__xgb__indicators.png)
![scatter — severity XGB/indicators](../figures/scatter/severity_xgb_indicators.png)
![importance — severity XGB/indicators](../figures/feat_importance/severity_xgb_indicators.png)

* HPO best : `n_estimators=100, max_depth=8`
* Synthetic : **val R² 0.467**, **test R² 0.467**, MAE 0.151
* Experimental : R² −0.242 — gap **0.71**
* Top features : `GAC_min` (12 %), `FRFRMS` (11 %),
  `unsigned_SCI` (10 %).

**Interpretation.** Similar shape to RF on indicators — wider
scatter than modal-feature XGB.

#### XGB — feature comparison

| feature     | val R² | test R² | exp R² |
|-------------|--------|---------|--------|
| modal       | 0.551  | 0.532   | −0.062 |
| indicators  | 0.467  | 0.467   | −0.242 |

modal wins by ~0.07.

### Multilayer Perceptron

#### MLP + `modal`

![HPO surface — severity MLP/modal](../figures/hpo/severity__mlp__modal.png)
![scatter — severity MLP/modal](../figures/scatter/severity_mlp_modal.png)

* HPO best : `hidden=(512, 256, 128), lr=3e-3`
* Synthetic : **val R² 0.551**, **test R² 0.542**, MAE 0.145, bias −0.004
* Experimental : R² −33.19 (catastrophic)
* No tree-based importance.

**Interpretation.** Competitive with XGB in-domain.  The
experimental R² is catastrophically negative because the MLP
predicts severity values well outside `[0, 1]` for the
out-of-distribution experimental inputs — `(pred − true)²`
dominates `Σ(true − mean)²`, dragging R² far below zero.  This
is a clear case for sim-to-real adaptation before deployment.

#### MLP + `indicators`

![HPO surface — severity MLP/indicators](../figures/hpo/severity__mlp__indicators.png)
![scatter — severity MLP/indicators](../figures/scatter/severity_mlp_indicators.png)

* HPO best : `hidden=(512, 256, 128), lr=3e-3`
* Synthetic : **val R² 0.376**, **test R² 0.344**, MAE 0.178
* Experimental : R² −671.8 (catastrophic).

**Interpretation.** Shallow non-linear head on a 22-d vector
cannot do severity well; even worse extrapolation than
modal-feature MLP.

#### MLP — feature comparison

| feature     | val R² | test R² | exp R² |
|-------------|--------|---------|--------|
| modal       | 0.551  | 0.542   | −33.2  |
| indicators  | 0.376  | 0.344   | −671.8 |

modal wins.

### 1-D CNN

#### 1-D CNN + `frf_mag`

![HPO surface — severity CNN/frf_mag](../figures/hpo/severity__cnn__frf_mag.png)
![scatter — severity CNN/frf_mag](../figures/scatter/severity_cnn_frf_mag.png)

* HPO best : `widths=(16, 32, 64), kernel_size=7`
* Synthetic : **val R² 0.253**, **test R² 0.213**, MAE 0.213
* Experimental : R² −4.25 — gap **4.5**.

**Interpretation.** Broad scatter around a flat regression
line — barely beats predict-the-mean.

#### 1-D CNN + `timeseries`

![HPO surface — severity CNN/timeseries](../figures/hpo/severity__cnn__timeseries.png)
![scatter — severity CNN/timeseries](../figures/scatter/severity_cnn_timeseries.png)

* HPO best : `widths=(32, 64, 128), kernel_size=5`
* Synthetic : **val R² 0.258**, **test R² 0.227**, MAE 0.211
* Experimental : R² −22.4 (catastrophic extrapolation).

**Interpretation.** Same shape as `frf_mag` CNN; experimental
predictions diverge.

#### 1-D CNN — feature comparison

Effectively tied in-domain (R² ≈ 0.21–0.23); `frf_mag`
transfers less catastrophically than `timeseries`.

### Small Transformer

#### Transformer + `frf_mag`

![HPO surface — severity Transformer/frf_mag](../figures/hpo/severity__transformer__frf_mag.png)
![scatter — severity Transformer/frf_mag](../figures/scatter/severity_transformer_frf_mag.png)

* HPO best : `d_model=64, n_layers=1`
* Synthetic : **val R² 0.028**, **test R² 0.013**, MAE 0.249
* Experimental : R² −0.039 — gap **0.05**.

**Interpretation.** Flat-line prediction — the model predicts
the dataset mean and the residual histogram is mean ≈ 0 but
wide.  Because it fits no signal at all, the sim-to-real gap is
tiny — the worst kind of "small gap".

#### Transformer + `timeseries`

![HPO surface — severity Transformer/timeseries](../figures/hpo/severity__transformer__timeseries.png)
![scatter — severity Transformer/timeseries](../figures/scatter/severity_transformer_timeseries.png)

* HPO best : `d_model=32, n_layers=2`
* Synthetic : **val R² 0.202**, **test R² 0.168**, MAE 0.222
* Experimental : R² −0.101 — gap **0.27**.

**Interpretation.** Marginal in-domain fit.  Predictions hug
the dataset mean.

#### Transformer — feature comparison

`timeseries` wins (slightly).

### 2-D CNN

#### 2-D CNN + `cfdac`

![HPO surface — severity CNN2D/cfdac](../figures/hpo/severity__cnn2d__cfdac.png)
![scatter — severity CNN2D/cfdac](../figures/scatter/severity_cnn2d_cfdac.png)

* HPO best : `widths=(8, 16, 32), kernel_size=5`
* Synthetic : **val R² 0.399**, **test R² 0.420**, MAE 0.174, bias +0.022
* Experimental : R² −0.211 — gap **0.63**.

**Interpretation.** Slight non-linear bias near 0; spread fans
out for severity > 0.7.  The 2-D CNN learns the gross trend on
CFDAC but loses precision at the extremes — same
saturation-at-extremes behaviour as the tree models on modal.

---

## Cross-model comparison

Sorted by synthetic test R².

| model       | feature     | val R² | test R² | MAE   | exp R² |
|-------------|-------------|--------|---------|-------|--------|
| RF          | modal       | 0.593  | **0.573** | 0.130 | −0.15  |
| MLP         | modal       | 0.551  | 0.542   | 0.145 | −33.2  |
| XGB         | modal       | 0.551  | 0.532   | 0.137 | −0.06  |
| RF          | indicators  | 0.498  | 0.487   | 0.146 | −0.42  |
| XGB         | indicators  | 0.467  | 0.467   | 0.151 | −0.24  |
| 2-D CNN     | cfdac       | 0.399  | 0.420   | 0.174 | −0.21  |
| MLP         | indicators  | 0.376  | 0.344   | 0.178 | −672   |
| 1-D CNN     | timeseries  | 0.258  | 0.227   | 0.211 | −22.4  |
| 1-D CNN     | frf_mag     | 0.253  | 0.213   | 0.213 | −4.25  |
| Transformer | timeseries  | 0.202  | 0.168   | 0.222 | −0.10  |
| Transformer | frf_mag     | 0.028  | 0.013   | 0.249 | −0.04  |

**Observations.**

1. Severity is fundamentally harder than classification —
   even the best model captures only ~57 % of the variance.
2. The R² ceiling is set by the **spectral-energy /
   peak-frequency information** in the modal vector; every
   model that consumes modal sits at R² ≈ 0.53–0.57.
3. Deep models on raw inputs converge to the dataset mean —
   their R² > 0 is mostly cosmetic.
4. The 2-D CNN on CFDAC clearly outperforms 1-D CNN /
   Transformer on raw spectra, again because CFDAC normalises
   the dynamic range by construction.

---

## Sim-to-real

* All cells go *negative* on experimental R² — the experimental
  severity distribution does not match the synthetic uniform
  prior (the IQS protocol uses discrete severity steps:
  11 / 20 / 50 / 85 % bolts; 5 / 8 mm cracks; 4 / 6 mm holes).
* Tree-based models bounded by their leaf values have R² close
  to 0 (mean prediction), so they "fail safely".
* Neural-net models with linear output heads *extrapolate*
  outside `[0, 1]` for experimental inputs — that's the source
  of the catastrophic R² of −33 / −672 for MLP cells.  Clipping
  the predictions to `[0, 1]` post-hoc would put the MLP modal
  cell at R² ≈ −0.5 instead.
* A real deployment would need: (i) post-hoc clipping or a
  bounded output head; (ii) a few labelled experimental
  severity points for sim-to-real finetune.

---

## Recommendation

* **Primary regressor.**  **RF + modal** (test R² 0.573,
  MAE 0.130, exp R² −0.15 — i.e. predict-the-mean on
  experimental data without catastrophic extrapolation).
* **Boosting alternative.**  **XGB + modal** (test R² 0.532)
  for slightly better cross-domain behaviour and feature
  importance interpretability.
* **Deep learning baseline.**  **2-D CNN + CFDAC** (test
  R² 0.420) — the only deep model that does meaningful
  regression.
* **Avoid for deployment.**  MLP regression heads on `modal` /
  `indicators` — they win in-domain but extrapolate
  catastrophically on out-of-distribution experimental data;
  use them only if you add output clipping or a bounded
  activation.

Continue to → [`04_col_location.md`](04_col_location.md).
