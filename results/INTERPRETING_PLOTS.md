# How to read every plot in `results/figures/`

This document walks through *every kind of plot* used in this
project, with worked examples.  After reading it you should be able
to glance at any `figures/<category>/<file>.png` and immediately
know what it tells you and what the colour / axis / annotation
conventions mean.

The companion document [`PLOTS.md`](PLOTS.md) walks through every
*individual* plot one by one.  The per-task documents under
[`by_task/`](by_task/) use the same plots in narrative form.

---

## Plot types in this project

1. [Class-count and severity histograms](#1-class-count-and-severity-histograms)
2. [Example-signal panels (time series, |H(f)|, |CFDAC|)](#2-example-signal-panels)
3. [Synth-vs-real feature panels](#3-synth-vs-real-feature-panels)
4. [Global metric bar charts](#4-global-metric-bar-charts)
5. [Confusion matrices](#5-confusion-matrices)
6. [Per-class F1 heatmaps](#6-per-class-f1-heatmaps)
7. [ROC and Precision-Recall curves](#7-roc-and-precision-recall-curves)
8. [Severity scatter + residual histograms](#8-severity-scatter--residual-histograms)
9. [Feature-importance bar charts](#9-feature-importance-bar-charts)
10. [PCA / t-SNE embeddings](#10-pca--t-sne-embeddings)
11. [HPO response-surface heatmaps](#11-hpo-response-surface-heatmaps)

---

## 1. Class-count and severity histograms

Example: [`figures/dataset/class_severity.png`](figures/dataset/class_severity.png).

![class counts and severity distributions](figures/dataset/class_severity.png)

**Layout.** Two side-by-side panels.

* Left = **bar chart** of `n_samples` per damage type
  (Pristine, Bolt, Crack, Hole, Mass).
* Right = **histogram** of severity values, one colour per
  damage type, stacked on a single axis whose unit changes per
  type (percent loosening for Bolt, mm for Crack/Hole, kg for
  Mass — see the legend).

**How to read.**

* Bar height in the left panel = number of training samples in
  that class.  An even five-bar block means the dataset is
  *class-balanced*.
* Histogram height in the right panel = density of samples with
  that severity within that type's bounded range.  A *flat*
  histogram means severity was sampled uniformly inside the
  physical range.

**Worked example.** Looking at the file linked above:

* All five Left-panel bars are at 2 000 — the dataset has zero
  class imbalance.  Any accuracy below 0.20 (random) is broken.
* The Bolt histogram is flat between 5 and 95 % → bolts span the
  full loosening range, with each percentage equally likely.
  This is the prior the severity regressor sees.

**What conclusions you can draw.**

* If a model gets accuracy ≈ 0.20 → it's predicting the modal
  class, and on this dataset every class is *equally* modal so
  it's literally random guessing.
* The flat severity distributions mean no class is "easier" by
  virtue of having mostly extreme values — every model has to
  cope with the full severity range.

---

## 2. Example-signal panels

Three files: [`timeseries.png`](figures/signals/timeseries.png),
[`frf_mag.png`](figures/signals/frf_mag.png),
[`cfdac.png`](figures/signals/cfdac.png).

### Time series

![example time series](figures/signals/timeseries.png)

**Layout.** Five vertically stacked panels, one per damage class,
each showing the **same chirp-excited acceleration response** over
all 9 sensor channels (overlaid in transparent colour).

**How to read.**

* X-axis = time `t ∈ [0, 4] s`.
* Y-axis = acceleration `m/s²`.  Y-axis units are not log-scaled.
* Each curve is one sensor (S2, S5, S6, S7, S8, S11, S12, S13, S14).

**Worked example.** The Pristine panel shows a smooth sweep
envelope rising as the chirp passes resonance frequencies.  The
Bolt and Crack panels look *almost identical* to Pristine to the
naked eye — the damage signature is a tiny phase / amplitude
shift inside the envelope, not a gross shape difference.

**Conclusion.** Time-domain shape is dominated by the
*excitation*, not by the damage.  A model trained on this
representation must learn to attend to subtle modulations at
specific time stamps.  This is why 1-D CNN and Transformer on
`timeseries` underperform engineered features.

### FRF magnitude `|H(f)|`

![example FRF magnitudes](figures/signals/frf_mag.png)

**Layout.** Same five panels, but the y-axis is `log10|H(f)|` over
5–100 Hz.

**How to read.**

* Sharp peaks = natural frequencies of the structure.
* Valleys between peaks = anti-resonances.
* Per-channel curves overlay so you can see which sensor "feels"
  each mode strongly.

**Worked example.**

* Pristine has the cleanest 3–4 peak envelope.
* Bolt shifts the lowest two peaks down by 1–3 Hz (joint loss
  reduces stiffness, lowering natural frequency).
* Crack flattens the mid-band.
* Mass pushes the floor-mode peak.

**Conclusion.** Damage is *spectral*, not temporal.  Engineered
modal features get this information for free; raw-spectrum CNNs
must rediscover it from log-scale data with high dynamic range.

### CFDAC

![example CFDAC matrices](figures/signals/cfdac.png)

**Layout.** Five 128 × 128 heatmaps, one per damage class.
**Always plotted with `cmap="viridis"` and clipped to `[0, 1]`.**

**How to read.**

* Each cell `(i, j)` is the magnitude of the Complex Frequency-
  Domain Assurance Criterion between this sample's FRF at
  frequency bin `i` and the synthetic pristine mean at bin `j`.
* Diagonal `i == j` should be near 1.0 when the FRF is similar
  to the reference.
* Bright off-diagonal cells indicate that resonances shifted
  *between* frequencies — the brighter the cell at `(i, j)`,
  the more frequency-bin `j` of the damaged FRF "looks like"
  frequency-bin `i` of the reference.

**Worked example.** Pristine is essentially identity (only the
diagonal is bright).  Bolt and Mass show structured off-diagonal
blocks; Crack and Hole show subtler perturbations.

**Conclusion.** CFDAC encodes "which-bin-moved-where" in a
spatially aligned matrix.  This is exactly the inductive bias a
2-D CNN can exploit — and explains why the `cnn2d/cfdac` cell is
the only deep configuration that competes with the engineered
tabular models.

---

## 3. Synth-vs-real feature panels

Side-by-side comparisons in
[`figures/feature_examples/`](figures/feature_examples/).

### Modal feature panel

![modal feature synth vs experimental](figures/feature_examples/modal.png)

**Layout.** 5 rows × 2 columns.  Each row = one damage class
(Pristine, Bolt, Crack, Hole, Mass).  Left column = one synthetic
sample, right column = one experimental sample of the same class.
Each panel is a **bar chart** of the 81-d modal feature vector.

The 81 features are organised as 9 channels × 9 statistics:

```
ch<c>_peak1_f   ch<c>_peak1_a   ch<c>_peak2_f   ch<c>_peak2_a
ch<c>_peak3_f   ch<c>_peak3_a   ch<c>_mean_logA ch<c>_std_logA
ch<c>_bandE
```

Vertical dotted lines mark every 9 features so you can see where
one channel ends and the next begins.

**How to read.**

* Peak frequencies are in Hz; expect the first three to lie in
  ~10–100 Hz.
* Peak amplitudes (`peak*_a`) are `log10|H|` so they are small
  numbers near zero.
* `mean_logA` and `std_logA` capture the gross level / scatter of
  the spectrum at that sensor.
* `bandE` is the integrated squared amplitude in the band.

**Worked example.** Compare the Bolt row: the synth and the
experimental sample should show similar values at the
`peak1_f / peak2_f` positions (both pick up the same resonance
shift).  Channels with strong damage signal will differ more
between Pristine row and Bolt row.

**Conclusion.** When the synth and real bars line up reasonably
in this plot, sim-to-real should be small for any tabular model
on modal features.  Channels where synth ≪ real or vice versa
expose calibration mismatch and will be the first thing a
sim-to-real adaptation step would address.

### Indicators feature panel

![indicators feature synth vs experimental](figures/feature_examples/indicators.png)

**Layout.** Same 5 × 2 grid.  Each panel is a bar chart of the
22-d pymodal damage-indicator vector, x-axis labelled with the
indicator name (`SCI`, `unsigned_SCI`, `DRQ`, `AIGAC`,
`FRFRMS`, `FRFSF`, `FRFSM_6dB`, `ODS_diff`, `r2_imag`, plus
summary statistics of `RVAC`, `GAC`, `M2L`).

**How to read.**

* `SCI / unsigned_SCI` near 0 = no detected change vs reference.
* `FRFRMS` quantifies log-FRF amplitude deviation — higher
  means more damage.
* `RVAC_mean / GAC_mean` near 1 means the damaged FRF correlates
  with the reference.

**Worked example.** Pristine should have near-zero SCI / FRFRMS.
Bolt and Mass rows should light up `FRFRMS` and `unsigned_SCI`.

**Conclusion.** The indicators carry **detection** signal well
but tend to compress *type* information — see the t-SNE plot of
indicators (§10) which puts Bolt / Crack / Hole on the same
manifold.

### FRF magnitude panel

![FRF magnitudes synth vs experimental](figures/feature_examples/frf_mag.png)

**Layout.** Same 5 × 2 grid; semi-log-y `|H(f)|` curves over
5–100 Hz, one curve per sensor.

**How to read.** Resonance peak count and locations should match
between synth and real for the same class — a peak frequency
shift between columns indicates calibration mismatch.

### Time series panel

![time series synth vs experimental](figures/feature_examples/timeseries.png)

**Layout.** Same 5 × 2 grid; raw 4 s acceleration time series,
9 channels overlaid.

**How to read.** Synth signals are noise-free (only structural
variability between rows) — experimental signals include real
sensor noise on top.  This is the visible source of the
sim-to-real gap for any model that consumes `timeseries`.

### CFDAC panel

![CFDAC synth vs experimental](figures/feature_examples/cfdac.png)

**Layout.** 5 rows × 2 columns of 128 × 128 heatmaps.

**How to read.** Synth Pristine ≈ identity by construction.
Experimental Pristine deviates slightly from the synthetic
pristine reference — that small deviation is the sim-to-real bias
that the 2-D CNN must absorb.

---

## 4. Global metric bar charts

Files: [`train_metrics_by_task.png`](figures/train_metrics_by_task.png)
and [`experimental_metrics_by_task.png`](figures/experimental_metrics_by_task.png).

![train metrics by task](figures/train_metrics_by_task.png)
![experimental metrics by task](figures/experimental_metrics_by_task.png)

**Layout.** One sub-panel per task (binary, type, severity,
col_location, mass_location).  Inside each sub-panel:

* X-axis = feature name (`modal`, `indicators`, `frf_mag`,
  `timeseries`, `cfdac`).
* Bars at each x position are coloured one per model
  (RF / XGB / MLP / 1-D CNN / Transformer / 2-D CNN).
* Bar height = test metric (accuracy for classification, R² for
  severity).

**How to read.**

* Taller bar = better.
* If the modal-feature column has tall bars across all models
  while another feature column has short bars, the feature is
  the bottleneck, not the model.
* If for the same feature one model is much taller, the model
  matters more.

**Worked example.** In the synthetic-test version, the `modal`
column of `mass_location` has three nearly-equal bars at ~1.0 (RF
/ XGB / MLP) while the `frf_mag` column has bars at ~0.4 — same
information could in principle be extracted from `frf_mag` but
the deep models don't do it.

The experimental version of the same chart compresses all bars
into ~0.5 – 0.6 — that compression is the sim-to-real gap.

---

## 5. Confusion matrices

Files: [`figures/confusion/<task>_<model>_<feature>.png`](figures/confusion/).

**Layout.** Square `n × n` matrix where rows = **true** class,
columns = **predicted** class.

* Rows = true labels in fixed order (binary: Pristine, Damage;
  type: Pristine, Bolt, Crack, Hole, Mass; col_location:
  S1BD, S1AD, S2BD, S2AD, S3BD, S3AD; mass_location: Base,
  F1, F2, F3).
* Columns = predicted labels in the same order.
* Cell text = absolute count of test samples that fell into
  that `(true, predicted)` cell.
* Cell colour = the same count divided by the row sum, i.e.
  per-row recall ∈ [0, 1].

**Example panel.**

![binary mlp/modal confusion](figures/confusion/binary_mlp_modal.png)

**How to read.**

* A *saturated diagonal* means high recall everywhere
  (every true class is being predicted as itself).
* A *single off-diagonal column lit up* means the model is
  systematically confusing one class for another.
* An *empty column* would mean the model never predicts that
  class.
* An *empty row* would mean there are zero samples of that
  class in the test set (does not happen here — splits are
  stratified).

**Worked example.**

* `binary / mlp / modal` (above): the diagonal cells are
  ≈ 297 / 1 197 with very few off-diagonal counts.  Recall is
  `[0.99, 0.99]`.  Conclusion: near-perfect classifier.
* `binary / transformer / frf_mag` (in
  `figures/confusion/binary_transformer_frf_mag.png`): the
  "predicted Pristine" column is *completely empty*.  The model
  always predicts "Damage", which gives accuracy 0.80 (the
  class prior) but is useless.

**What conclusions you can draw.**

* Overall accuracy is the diagonal-sum divided by the total
  — quoted in the plot title.
* The *Pristine-recall column* in the binary task is what
  distinguishes a real detector from a model that just
  predicts the majority class.
* Asymmetric off-diagonals (e.g. lots of "Crack predicted as
  Hole" but few of the reverse) tell you which damage
  mechanisms the model fails to discriminate.

---

## 6. Per-class F1 heatmaps

Files: [`figures/perclass_f1/<task>.png`](figures/perclass_f1/).

**Layout.** Heatmap with:

* Rows = `(model, feature)` cells, sorted top-to-bottom by mean
  F1 across the columns.
* Columns = task classes (same ordering as the confusion
  matrices).
* Cell colour and text = F1 for that `(model, class)` pair.

**Example.**

![type per-class F1](figures/perclass_f1/type.png)

**How to read.**

* Look at the *top row*: best model overall — its bar of cells
  tells you which classes are hardest even for the best model.
* Look at *columns*: each column of this heatmap is "F1 on this
  class across all models" — a vertical strip of low values
  means that class is hard for everybody and is a target for
  future feature engineering.
* Look at *rows*: each row is "F1 across classes for one
  configuration".  A row that is uniformly low except in one
  column is a model that has collapsed onto a single class.

**Worked example.** In `figures/perclass_f1/type.png` the
**Hole** column is the darkest (lowest F1) for almost every model,
except `mlp/modal` which keeps Hole's F1 ≈ 0.93.  Conclusion:
distinguishing Hole from Crack is the residual hard problem;
modal features have the discriminating information but most
models can't extract it.

---

## 7. ROC and Precision-Recall curves

Files: [`binary_roc.png`](figures/roc/binary_roc.png) and
[`binary_pr.png`](figures/roc/binary_pr.png).

**Layout.**

* ROC: x = false-positive rate, y = true-positive rate.
* PR: x = recall, y = precision.

Every binary classifier produces one curve; AUC is given in the
legend.

**Example.**

![binary ROC overlay](figures/roc/binary_roc.png)

**How to read.**

* ROC: closer to the **top-left corner** is better.  The
  **diagonal line** is random guessing (AUC = 0.5).  AUC = 1.0
  is perfect.
* PR: closer to the **top-right corner** is better.  The
  horizontal asymptote at recall = 1 is the dataset's positive
  prevalence (0.80 here — the fraction of damage cases).

**Worked example.** `mlp/modal` and `xgb/modal` overlap near the
top-left with AUC ≈ 1.00.  `transformer/frf_mag` collapses onto
the diagonal — AUC = 0.5 — which means its "score" is no better
than random.

**What conclusions you can draw.**

* If two models have similar AUC but different curve shapes,
  one of them may have higher precision at high recall —
  important for SHM where false alarms are operationally
  expensive.
* The PR curve compresses harder than the ROC curve when the
  positive class is the majority, so it's the better
  discriminator for the binary task here.

---

## 8. Severity scatter + residual histograms

Files: [`figures/scatter/severity_<model>_<feature>.png`](figures/scatter/).

**Layout.** Two side-by-side panels.

* Left = scatter of `predicted vs true` normalised severity, with
  a black dashed `y = x` reference line.
* Right = histogram of residuals `(pred − true)`.

**Example.**

![severity rf/modal scatter](figures/scatter/severity_rf_modal.png)

**How to read.**

* Points should sit on the `y = x` line if the model predicts
  severity perfectly.
* Vertical scatter at fixed true-severity = irreducible noise
  for that target value.
* A diagonal scatter cloud below the line = systematic under-
  prediction.
* The residual histogram should be Gaussian centred on 0 if the
  errors are unbiased.

**Worked example.** `rf/modal` scatter (above): a clear
positive correlation, R² = 0.57.  Residual histogram is
mean-zero, slightly skewed negative — the model very slightly
under-predicts high severities, which is consistent with trees
hitting the upper limit of leaf values.

**What conclusions you can draw.**

* R² in the title quantifies how much of the severity variance
  the model captures.
* If R² is positive but the cloud is flat (horizontal stripe),
  the model is essentially predicting the dataset mean — useless
  for severity estimation even if R² > 0.
* Bias in the residual histogram (non-zero mean) signals that
  recalibration could improve the model without retraining.

---

## 9. Feature-importance bar charts

Files: [`figures/feat_importance/<task>_<rf|xgb>_<feature>.png`](figures/feat_importance/).

**Layout.** Horizontal bar chart of the top-20 features ranked by
Gini importance (Random Forest) or gain (XGBoost), bars sorted
in descending order top-to-bottom.

**Example.**

![type rf/modal feature importance](figures/feat_importance/type_rf_modal.png)

**How to read.**

* Feature names follow the convention
  `ch<c>_<stat>` for modal (e.g. `ch2_bandE` = band energy at
  sensor S6) or the pymodal indicator name for indicators.
* Bar length = relative importance (sums to 1.0 across all
  features for sklearn ensembles).
* The top-5 typically capture 30 – 50 % of total importance.

**Worked example.** In `type / rf / modal`, the top three are
`ch6_bandE` (6 %), `ch2_bandE` (5 %), `ch2_std_logA` (4 %) —
spectral energy at Floors 2 and 3 dominates type detection.

**What conclusions you can draw.**

* Tells you which sensor channel / indicator carries most of
  the signal for that task.
* Compare across tasks: if `ch2_bandE` is the top feature for
  both binary and type, then Floor-2 acceleration energy is a
  universal damage indicator.
* If two redundant features split importance ~50/50 the tree
  is treating them as substitutes.

---

## 10. PCA / t-SNE embeddings

Files: [`figures/embedding/{pca,tsne}_{modal,indicators}.png`](figures/embedding/).

**Layout.** Scatter plot of 3 000 samples projected from the
feature's full dimensionality to 2 D.

* PCA = linear projection, axes labelled with `% variance
  explained`.  PCA is reversible and the projection axes have a
  precise meaning.
* t-SNE = non-linear projection, axes unit-less.  t-SNE
  preserves local neighbourhood structure but distorts global
  distances.

Points are coloured by damage type (5 colours).

**Example.**

![PCA — modal](figures/embedding/pca_modal.png)

**How to read.**

* Well-separated coloured clouds = the feature is *linearly*
  separable in the original space.
* Tangled clouds in PCA but separated in t-SNE = the feature is
  *non-linearly* separable.
* All classes tangled in both = the feature simply doesn't carry
  the right information.

**Worked example.**

* `pca_modal.png` shows three clearly separable clouds
  (Pristine, Mass, Bolt) plus a tangled Crack-Hole region.  A
  linear classifier on `modal` would already reach ~0.6
  accuracy; the non-linear MLP pushes that up to 0.88.
* `tsne_indicators.png` shows Pristine isolated but all three
  damage mechanisms (Bolt / Crack / Hole) merged.  No model on
  indicators will separate those — that's why indicator-feature
  type-classification tops out around 0.75.

**What conclusions you can draw.**

* Tells you **before training** whether a feature representation
  is going to work for a task.
* Identifies which classes are inherently confusable in a
  representation.

---

## 11. HPO response-surface heatmaps

Files: [`figures/hpo/<task>__<model>__<feature>.png`](figures/hpo/).

**Layout.** 2-D heatmap of the validation metric over the two
hyperparameters that were swept for that `(task, model, feature)`
cell.

* Y-axis = first hyperparameter values.
* X-axis = second hyperparameter values.
* Colour = validation metric (viridis: dark = low, bright =
  high).
* Cell text = exact metric for that hyperparameter combination.

The grid sizes are 9 (3 × 3) for tabular models, 4 (2 × 2) or 6
(3 × 2) for the deep models — see
[`../ml_pipeline/hpo.py`](../ml_pipeline/hpo.py) for the exact
grids.

**Example.**

![binary mlp/modal HPO surface](figures/hpo/binary__mlp__modal.png)

**How to read.**

* A *monotonic gradient* points the way to "more of this is
  better" — useful for telling whether the grid is too small.
* A *flat surface* says HPO did not move the metric — pick the
  smallest / fastest cell.
* An *isolated bright cell* says the optimum is in the interior;
  surrounding cells are noticeably worse.
* If the brightest cell is on a grid *boundary*, the grid was
  too narrow — re-run HPO with extended bounds.

**Worked example.** `binary / mlp / modal` ramps from 0.81 in
the bottom-left (small hidden, small lr) to 0.99 in the top-right
(largest hidden, biggest lr).  Conclusion: a 4 × 4 grid with even
bigger hidden widths might gain another 0.5 pp; the current
3 × 3 is sufficient.

`binary / transformer / frf_mag` is flat at 0.80 everywhere — the
model collapsed to the majority-class baseline regardless of
hyperparameter.  Conclusion: it's not a HPO problem, it's a
*representation* problem.

**What conclusions you can draw.**

* Whether the optimum is inside the grid (good) or on the edge
  (extend the grid).
* Whether HPO is meaningfully moving the metric (gradient is
  bigger than the cell-to-cell noise).
* Whether the architecture *can* learn this task with this
  feature: a uniformly low surface means the binding constraint
  is upstream of the optimiser.

---

## Summary cheat-sheet

| If you want to know …                          | Look at …                            |
|------------------------------------------------|--------------------------------------|
| Whether the model just predicts the majority class | confusion matrix § 5             |
| Which class is hardest for a given model       | confusion matrix § 5 or per-class F1 § 6 |
| Which classes a feature can / cannot separate  | PCA / t-SNE § 10                     |
| Whether HPO would help                          | response surface § 11                |
| Whether a feature is informative for a task     | feature importance § 9 + global bar chart § 4 |
| Whether the model is biased                     | residual histogram § 8               |
| Whether sim-to-real holds                       | the two global bar charts § 4 side by side |
| Which sensor channel carries the damage signal  | top of the feature-importance bar § 9 |

---

Continue with [`PROTOCOL.md`](PROTOCOL.md) for the train / val /
test definitions, [`PLOTS.md`](PLOTS.md) for the per-plot
commentary on all 146 plots, or [`by_task/`](by_task/) for the
use-case-by-use-case narrative.
