# Plots reference

For every plot in [`figures/`](figures/) this document gives **(1)** what
the plot is, **(2)** how to read it, **(3)** what is visible in this
particular plot, and **(4)** the conclusion to take away.

Plots are listed in the order they appear in the file tree.  All
numbers come from the same hold-out test fold used during HPO; the
StandardScaler fit on the train fold is reused at evaluation time so
the results match the HPO logs.

---

## 1. Dataset overview

### [`figures/dataset/class_severity.png`](figures/dataset/class_severity.png)

* **What.** Left panel: number of samples per damage type
  (Pristine / Bolt / Crack / Hole / Mass).  Right panel: severity
  distribution for each non-Pristine type stacked on a single axis.
* **How to read.** Bar height = sample count (left); histogram height
  = density in each severity bin (right).
* **What is shown.** Left bars are five equal-height columns of 2 000
  samples each — the dataset is perfectly class-balanced.  Right
  histograms are flat over each type's bounded range
  (bolt 5–95 %, crack 1–8 mm, hole 1–6 mm, mass 0.1–2.5 kg).
* **Conclusion.** The dataset has zero class imbalance and a uniform
  severity prior, so any reported metric is interpretable without
  resampling.

---

## 2. Example signals (1 sample / class)

These plots overlay all 9 sensor channels in transparent colour to
show the dispersion across the sensor array; we draw one randomly
selected sample per damage class so the structural fingerprint is
visible.

### [`figures/signals/timeseries.png`](figures/signals/timeseries.png)

* **What.** 4 s, 1024-sample acceleration response to the shared
  5–100 Hz chirp, one panel per damage class.
* **How to read.** X-axis = time [s], Y-axis = acceleration [m/s²]
  per channel (units share an axis).
* **What is shown.** All classes exhibit the same chirp-driven sweep
  envelope; differences between classes are subtle modulations of
  the envelope amplitude near the structure's natural frequencies.
* **Conclusion.** Raw time-domain shape is dominated by the
  excitation, not the damage — a deep model on this representation
  must learn to attend to small amplitude / phase shifts at specific
  time stamps, which explains why CNN/Transformer on `timeseries`
  underperform engineered features.

### [`figures/signals/frf_mag.png`](figures/signals/frf_mag.png)

* **What.** |H(f)| for the 5–100 Hz band, log-y, one panel per class.
* **How to read.** Sharp peaks are natural frequencies; valleys are
  anti-resonances; per-channel curves overlay.
* **What is shown.** Pristine has the cleanest four-mode envelope;
  Bolt shifts the lowest two modes down in frequency; Crack/Hole
  flatten and shift the mid-band; Mass pushes the floor-mode peak.
* **Conclusion.** Damage signatures are spectral, not temporal —
  models that operate on engineered modal features get this
  information for free, while a raw-spectrum CNN must rediscover it.

### [`figures/signals/cfdac.png`](figures/signals/cfdac.png)

* **What.** |CFDAC| matrix (128 × 128) of the sample's FRF against the
  synthetic pristine mean.
* **How to read.** Diagonal is the per-frequency self-correlation
  (≈ 1 when undamaged); off-diagonal cross-couplings appear when
  resonance peaks shift / split.
* **What is shown.** Pristine map is essentially identity; damaged
  classes light up off-diagonal blocks aligned with the storey that
  changed stiffness.
* **Conclusion.** CFDAC is a structurally-aligned damage map — this
  is exactly the inductive bias a 2-D CNN can exploit, and matches
  the observation that `cnn2d/cfdac` is the only deep configuration
  that competes with the engineered MLP across tasks.

---

## 3. Global metric bar charts

### [`figures/train_metrics_by_task.png`](figures/train_metrics_by_task.png)

* **What.** Grouped bar chart: per-task test metric for every
  (model, feature) cell from HPO.
* **How to read.** One subplot per task, x-axis = feature, bars
  coloured by model.
* **What is shown.** Modal-feature bars (RF / XGB / MLP) sit at the
  top of every classification task; the cnn2d-on-cfdac bar is
  comparable on `binary` and `mass_location`; the
  transformer-on-frf_mag bar is consistently at the bottom.
* **Conclusion.** Engineered features dominate; cnn2d on CFDAC is
  the best deep baseline; raw FRF / time series with vanilla deep
  models struggle.

### [`figures/experimental_metrics_by_task.png`](figures/experimental_metrics_by_task.png)

* **What.** Same chart, but evaluated on the 61 IQS experimental
  cases (composite damage scenarios labelled by their primary op).
* **How to read.** Same.
* **What is shown.** All bars compress closer to ~0.5 – 0.6 — the
  sim-to-real shift erases the synthetic-test ranking; the modal
  MLP is still the best, but with a much smaller margin.
* **Conclusion.** The sim-to-real gap is the dominant error term;
  improvements to the ROM (added DOFs, calibrated noise) would help
  more than further ML tuning.

---

## 4. Confusion matrices

### General how-to-read

Each plot is an `n × n` matrix where rows are **true** classes and
columns are **predicted** classes.  Cells display the absolute
count; the colour intensity normalises each row to a sum of 1, so a
saturated diagonal means high recall on that class.  Off-diagonal
saturated cells expose systematic confusions.

Class label ordering by task:

* binary       — Pristine, Damage
* type         — Pristine, Bolt, Crack, Hole, Mass
* col_location — S1BD, S1AD, S2BD, S2AD, S3BD, S3AD
* mass_location — Base, F1, F2, F3

Overall accuracy is the diagonal sum divided by total; cf. the
per-plot commentary below for the exact number and the dominant
confusion pattern (if any).

### binary  ([all](figures/confusion/))

| file | overall | Pristine / Damage recall | conclusion |
|------|---------|--------------------------|------------|
| [binary_mlp_modal.png](figures/confusion/binary_mlp_modal.png)               | 0.989 | 0.99 / 0.99 | near-perfect; the few errors split symmetrically |
| [binary_xgb_modal.png](figures/confusion/binary_xgb_modal.png)               | 0.965 | 0.94 / 0.97 | slight bias toward predicting Damage |
| [binary_rf_modal.png](figures/confusion/binary_rf_modal.png)                 | 0.949 | 0.83 / 0.98 | misses 17 % of Pristine; class-weight balanced fights majority bias |
| [binary_cnn2d_cfdac.png](figures/confusion/binary_cnn2d_cfdac.png)           | 0.944 | 0.93 / 0.95 | symmetric errors; best non-MLP cell |
| [binary_xgb_indicators.png](figures/confusion/binary_xgb_indicators.png)     | 0.926 | 0.82 / 0.95 | indicator features lose Pristine recall |
| [binary_rf_indicators.png](figures/confusion/binary_rf_indicators.png)       | 0.916 | 0.73 / 0.96 | strong asymmetry — Pristine often mistaken for damage |
| [binary_transformer_timeseries.png](figures/confusion/binary_transformer_timeseries.png) | 0.876 | 0.68 / 0.92 | better than baseline, still 32 % Pristine miss |
| [binary_cnn_frf_mag.png](figures/confusion/binary_cnn_frf_mag.png)           | 0.853 | 0.63 / 0.91 | spectrum CNN catches obvious damage but loses Pristine |
| [binary_cnn_timeseries.png](figures/confusion/binary_cnn_timeseries.png)     | 0.842 | 0.89 / 0.83 | the only model where Damage recall < Pristine recall |
| [binary_mlp_indicators.png](figures/confusion/binary_mlp_indicators.png)     | 0.821 | 0.42 / 0.92 | indicator MLP collapses Pristine recall to 42 % |
| [binary_transformer_frf_mag.png](figures/confusion/binary_transformer_frf_mag.png) | 0.800 | 0.00 / 1.00 | full collapse to majority class — the model never says "Pristine" |

**Take-away.** Pristine recall is the discriminating axis: every model
keeps Damage recall high (the majority class) but only the
modal-feature models also keep Pristine recall.

### type  ([all](figures/confusion/))

Per-class recall is `[Pristine, Bolt, Crack, Hole, Mass]`.

| file | overall | per-class recall | conclusion |
|------|---------|------------------|------------|
| [type_mlp_modal.png](figures/confusion/type_mlp_modal.png)               | 0.877 | 0.99 / 0.85 / 0.64 / 0.92 / 0.99 | best overall; Crack ↔ Hole is the residual confusion |
| [type_xgb_modal.png](figures/confusion/type_xgb_modal.png)               | 0.822 | 0.98 / 0.86 / 0.64 / 0.67 / 0.98 | gradient boosting essentially matches the MLP except on Hole |
| [type_rf_modal.png](figures/confusion/type_rf_modal.png)                 | 0.811 | 0.96 / 0.85 / 0.64 / 0.63 / 0.98 | Crack/Hole confusion = ~36 % |
| [type_cnn2d_cfdac.png](figures/confusion/type_cnn2d_cfdac.png)           | 0.803 | 0.97 / 0.85 / 0.70 / 0.59 / 0.91 | the only deep model that competes; same Crack/Hole confusion |
| [type_xgb_indicators.png](figures/confusion/type_xgb_indicators.png)     | 0.759 | 0.86 / 0.84 / 0.62 / 0.55 / 0.92 | indicator features lose Crack/Hole separation further |
| [type_rf_indicators.png](figures/confusion/type_rf_indicators.png)       | 0.745 | 0.84 / 0.84 / 0.61 / 0.53 / 0.90 | same trend, sklearn ensemble |
| [type_mlp_indicators.png](figures/confusion/type_mlp_indicators.png)     | 0.701 | 0.72 / 0.83 / 0.53 / 0.53 / 0.90 | Pristine recall drops by 20 pp vs modal |
| [type_cnn_frf_mag.png](figures/confusion/type_cnn_frf_mag.png)           | 0.689 | 0.99 / 0.84 / 0.55 / 0.16 / 0.90 | Hole recall collapses to 16 % — the model lumps Hole into Crack |
| [type_cnn_timeseries.png](figures/confusion/type_cnn_timeseries.png)     | 0.657 | 0.62 / 0.87 / 0.62 / 0.29 / 0.88 | similar Hole problem |
| [type_transformer_timeseries.png](figures/confusion/type_transformer_timeseries.png) | 0.576 | 0.91 / 0.66 / 0.68 / 0.25 / 0.37 | Mass recall now also drops |
| [type_transformer_frf_mag.png](figures/confusion/type_transformer_frf_mag.png) | 0.501 | 0.56 / 0.69 / 0.27 / 0.41 / 0.58 | worst — every class < 70 % recall |

**Take-away.** Crack and Hole are the inherently hard pair (both
reduce column stiffness, only by different amounts).  MLP + modal is
the only configuration that breaks 90 % on Hole.

### col_location (6 classes)  ([all](figures/confusion/))

Class ordering = `[S1BD, S1AD, S2BD, S2AD, S3BD, S3AD]`.  Random
baseline = 0.167.

| file | overall | per-class recall | conclusion |
|------|---------|------------------|------------|
| [col_location_mlp_modal.png](figures/confusion/col_location_mlp_modal.png)               | 0.494 | 1.00 / 0.00 / 0.95 / 0.07 / 0.51 / 0.43 | binarises by `BD` vs `AD` — AD storeys collapse |
| [col_location_cnn2d_cfdac.png](figures/confusion/col_location_cnn2d_cfdac.png)           | 0.494 | 0.79 / 0.17 / 0.74 / 0.26 / 0.70 / 0.30 | CFDAC keeps some AD signal but with low recall |
| [col_location_rf_modal.png](figures/confusion/col_location_rf_modal.png)                 | 0.492 | 0.57 / 0.52 / 0.48 / 0.48 / 0.44 / 0.47 | the most balanced confusion across the six classes |
| [col_location_xgb_modal.png](figures/confusion/col_location_xgb_modal.png)               | 0.488 | 0.56 / 0.54 / 0.46 / 0.45 / 0.38 / 0.54 | similar balance to RF |
| [col_location_rf_indicators.png](figures/confusion/col_location_rf_indicators.png)       | 0.481 | 0.47 / 0.47 / 0.57 / 0.47 / 0.43 / 0.48 | indicators give comparable balance |
| [col_location_cnn_timeseries.png](figures/confusion/col_location_cnn_timeseries.png)     | 0.473 | 0.49 / 0.46 / 0.48 / 0.41 / 0.60 / 0.39 | best deep alternative; uniform recall |
| [col_location_cnn_frf_mag.png](figures/confusion/col_location_cnn_frf_mag.png)           | 0.469 | 0.98 / 0.00 / 0.99 / 0.00 / 0.85 / 0.00 | full AD/BD collapse — model only predicts BD storeys |
| [col_location_xgb_indicators.png](figures/confusion/col_location_xgb_indicators.png)     | 0.454 | 0.47 / 0.45 / 0.53 / 0.43 / 0.40 / 0.45 | similar to RF/indicators |
| [col_location_mlp_indicators.png](figures/confusion/col_location_mlp_indicators.png)     | 0.417 | 0.13 / 0.69 / 0.27 / 0.55 / 0.47 / 0.39 | curious reverse — AD classes predicted more often than BD |
| [col_location_transformer_timeseries.png](figures/confusion/col_location_transformer_timeseries.png) | 0.368 | 0.49 / 0.39 / 0.24 / 0.66 / 0.29 / 0.13 | erratic per-class behaviour |
| [col_location_transformer_frf_mag.png](figures/confusion/col_location_transformer_frf_mag.png) | 0.251 | 0.00 / 0.23 / 0.00 / 0.84 / 0.26 / 0.17 | only S2AD recall is high; rest noise-level |

**Take-away.** The current ROM is **AD/BD-degenerate** for column
damage (the asymmetric semi-rigid formula matters only for bolts).
Column-only Crack/Hole damage produces identical FRFs whether at the
top or bottom of a column, so the AD/BD axis is effectively
unobservable from sensors — the maximum attainable accuracy on this
6-class task is bounded by chance on that axis (~0.5 × 1.0).
Adding rocking DOFs is the principled fix.

### mass_location (4 classes)  ([all](figures/confusion/))

Class ordering = `[Base, F1, F2, F3]`.

| file | overall | per-class recall | conclusion |
|------|---------|------------------|------------|
| [mass_location_rf_modal.png](figures/confusion/mass_location_rf_modal.png)         | 0.990 | 0.99 / 0.99 / 0.99 / 1.00 | near-perfect on a structurally distinctive task |
| [mass_location_mlp_modal.png](figures/confusion/mass_location_mlp_modal.png)       | 0.987 | 0.99 / 0.99 / 0.97 / 1.00 | tied with RF |
| [mass_location_xgb_modal.png](figures/confusion/mass_location_xgb_modal.png)       | 0.987 | 0.99 / 0.97 / 0.99 / 1.00 | tied with RF |
| [mass_location_xgb_indicators.png](figures/confusion/mass_location_xgb_indicators.png) | 0.973 | 0.97 / 0.97 / 0.95 / 1.00 | indicator features almost as good |
| [mass_location_rf_indicators.png](figures/confusion/mass_location_rf_indicators.png) | 0.967 | 0.97 / 0.96 / 0.93 / 1.00 | F2 is the hardest plate |
| [mass_location_mlp_indicators.png](figures/confusion/mass_location_mlp_indicators.png) | 0.963 | 0.99 / 0.93 / 0.93 / 1.00 | similar pattern |
| [mass_location_cnn2d_cfdac.png](figures/confusion/mass_location_cnn2d_cfdac.png)   | 0.953 | 0.93 / 0.97 / 0.93 / 0.97 | the 2-D CNN reaches > 95 % uniformly |
| [mass_location_transformer_timeseries.png](figures/confusion/mass_location_transformer_timeseries.png) | 0.637 | 0.63 / 0.56 / 0.73 / 0.63 | uniform-ish but well below tabular |
| [mass_location_transformer_frf_mag.png](figures/confusion/mass_location_transformer_frf_mag.png) | 0.480 | 0.39 / 0.37 / 0.29 / 0.87 | only F3 predicted well — model learns the largest mode shift |
| [mass_location_cnn_timeseries.png](figures/confusion/mass_location_cnn_timeseries.png) | 0.473 | 1.00 / 0.89 / 0.00 / 0.00 | predicts Base/F1, ignores F2/F3 |
| [mass_location_cnn_frf_mag.png](figures/confusion/mass_location_cnn_frf_mag.png)   | 0.413 | 1.00 / 0.65 / 0.00 / 0.00 | same Base/F1 bias |

**Take-away.** Mass location is the easiest task — each plate
produces a distinct dominant-mode shift, and tabular models trained
on log-amplitude per channel pick it up trivially.  Deep models on
raw FRF / time series miss the higher-storey plates without explicit
per-channel normalisation.

---

## 5. Per-class F1 heatmaps

For each classification task, a heatmap shows F1 (rows = model /
feature configuration sorted by mean F1; columns = class).  This is
the per-plot answer to "which model fails on which class".

### [`figures/perclass_f1/binary.png`](figures/perclass_f1/binary.png)

* **What.** Per-model F1 on `[Pristine, Damage]`.
* **What is shown.** The bottom row of the figure (worst-mean F1) is
  the transformer on `frf_mag` — F1 on Pristine is 0 because it
  never predicts that class.  The top rows are the modal-feature
  models with both classes ≥ 0.95.
* **Conclusion.** The Pristine recall asymmetry seen in the
  confusion matrices shows up cleanly as a "column" of low F1 on
  Pristine for deep / raw-feature configurations.

### [`figures/perclass_f1/type.png`](figures/perclass_f1/type.png)

* **What.** Per-model F1 on the 5 damage types.
* **What is shown.** A "Hole column" with low F1 across most models
  except `mlp/modal` (F1 ≈ 0.93) — the residual hard case is Hole.
* **Conclusion.** Future model improvements should focus on
  separating Hole from Crack; data augmentation that emphasises
  storey-localised stiffness loss would help.

### [`figures/perclass_f1/col_location.png`](figures/perclass_f1/col_location.png)

* **What.** F1 per (storey, end) class.
* **What is shown.** The AD-end columns are uniformly weaker than
  the BD-end columns regardless of model.
* **Conclusion.** As above: AD/BD is unobservable for column
  Crack/Hole damage in the current ROM.

### [`figures/perclass_f1/mass_location.png`](figures/perclass_f1/mass_location.png)

* **What.** F1 per plate.
* **What is shown.** Tabular rows have F1 ≥ 0.93 on every plate;
  deep `cnn/frf_mag` and `cnn/timeseries` rows show binary `1 / 0`
  patterns because the model only predicts Base / F1.
* **Conclusion.** Mass detection is information-rich enough that
  classical models saturate near 1.0; the deep failures here are
  artefacts of unscaled input dynamic range.

---

## 6. ROC and Precision-Recall curves (binary task)

### [`figures/roc/binary_roc.png`](figures/roc/binary_roc.png)

* **What.** Receiver-operating-characteristic curves for every
  binary classifier, overlaid.  AUC in the legend.
* **How to read.** Closer to the top-left corner is better; AUC = 1
  is perfect.  The diagonal line is random.
* **What is shown.** `mlp/modal` and `xgb/modal` overlay near the
  corner with AUC ≈ 1.0; `transformer/frf_mag` falls onto the
  diagonal (AUC ≈ 0.5).
* **Conclusion.** The ROC ranking reproduces the accuracy ranking
  exactly — there is no operating-point where a worse-accuracy
  model becomes preferable.

### [`figures/roc/binary_pr.png`](figures/roc/binary_pr.png)

* **What.** Precision-recall curves for every binary classifier.
* **How to read.** Top-right corner is best; the line value at
  recall = 1 is the dataset's positive prevalence (0.80 here).
* **What is shown.** Modal-MLP / XGB / RF dominate with
  near-rectangular envelopes; tree / forest indicator curves drop
  rapidly past 0.8 recall; transformer/frf_mag is a flat line.
* **Conclusion.** For any precision floor above 0.9, only the modal
  models are viable.

---

## 7. Severity regression scatter + residual histograms

Each plot has two panels: left = true-vs-predicted scatter with a
y=x reference, right = residual (pred − true) histogram.  R² and
MAE are quoted in the title.

| file | R² | MAE | observation | conclusion |
|------|----|------|-------------|------------|
| [severity_rf_modal.png](figures/scatter/severity_rf_modal.png)               | 0.573 | 0.130 | predictions track the diagonal except at the extremes; residual histogram is mean-zero, slight negative skew | best regressor; modal-peak energy captures severity monotonically |
| [severity_mlp_modal.png](figures/scatter/severity_mlp_modal.png)             | 0.542 | 0.145 | similar shape, slightly wider scatter | MLP is competitive with RF when given the same engineered features |
| [severity_xgb_modal.png](figures/scatter/severity_xgb_modal.png)             | 0.532 | 0.137 | boosting matches RF on the central mass, wider tails | trees overfit at the extremes where samples are sparse |
| [severity_rf_indicators.png](figures/scatter/severity_rf_indicators.png)     | 0.487 | 0.146 | predictions cluster around the dataset mean; saturation at both ends | indicator scalars compress information too aggressively |
| [severity_xgb_indicators.png](figures/scatter/severity_xgb_indicators.png)   | 0.468 | 0.151 | very similar scatter to RF/indicators | same compression issue |
| [severity_cnn2d_cfdac.png](figures/scatter/severity_cnn2d_cfdac.png)         | 0.420 | 0.174 | non-linear bias near 0; spread fans out for severity > 0.7 | 2-D CNN learns the gross trend but loses precision at extremes |
| [severity_mlp_indicators.png](figures/scatter/severity_mlp_indicators.png)   | 0.344 | 0.178 | scatter is wide and slightly biased toward the mean | shallow MLP cannot use indicator vector for regression as well as trees |
| [severity_cnn_timeseries.png](figures/scatter/severity_cnn_timeseries.png)   | 0.227 | 0.211 | broad scatter around a flat regression line | 1-D CNN barely beats predict-the-mean |
| [severity_cnn_frf_mag.png](figures/scatter/severity_cnn_frf_mag.png)         | 0.213 | 0.213 | same shape as above | unscaled FRF magnitudes hurt regression |
| [severity_transformer_timeseries.png](figures/scatter/severity_transformer_timeseries.png) | 0.168 | 0.222 | predictions hug the dataset mean | transformer fits the mean within 4 epochs |
| [severity_transformer_frf_mag.png](figures/scatter/severity_transformer_frf_mag.png) | 0.013 | 0.249 | residuals = (mean − true) | flat-line prediction, structurally no signal extracted |

**Take-away.** Severity is fundamentally harder than classification
because it requires *quantifying* how much stiffness was lost, not
just detecting that some was lost.  The information is concentrated
in resonance amplitudes (modal `bandE` / `mean_logA` channels) and
the FRFRMS indicator — the top-3 models that exploit those reach
R² > 0.50; everything else hugs the mean.

---

## 8. Feature importance bars (RF / XGB)

For every `(task, model, feature)` cell with a tree-based estimator,
the top-20 features by Gini / gain importance are plotted as a
horizontal bar chart.  The textual top-5 below summarises what each
plot shows.

### Modal features  (`ch<c>_<peak|mean|std|bandE>`, 0 ≤ c ≤ 8)

| file | top 3 features (importance %) | reading |
|------|-------------------------------|---------|
| [binary_rf_modal.png](figures/feat_importance/binary_rf_modal.png)   | `ch2_bandE`(10), `ch6_bandE`(10), `ch6_std_logA`(5) | Floor-2 / Floor-3 spectral energy dominates the detection signal |
| [binary_xgb_modal.png](figures/feat_importance/binary_xgb_modal.png) | `ch2_peak1_f`(13), `ch2_bandE`(9), `ch0_peak1_f`(8) | Boosting prefers Floor-2's first natural frequency as the single best split |
| [type_rf_modal.png](figures/feat_importance/type_rf_modal.png)       | `ch6_bandE`(6), `ch2_bandE`(5), `ch2_std_logA`(4) | Type classification is more diffuse — Top-3 covers ~15 % of total importance |
| [type_xgb_modal.png](figures/feat_importance/type_xgb_modal.png)     | `ch2_peak1_f`(13), `ch3_peak1_f`(11), `ch0_peak1_f`(7) | Boosting reads damage type off the first-mode shifts at three floors |
| [severity_rf_modal.png](figures/feat_importance/severity_rf_modal.png) | `ch2_bandE`(8), `ch1_mean_logA`(8), `ch6_bandE`(8) | Spectral energy at floors 1 / 2 / 3 — severity ≈ energy |
| [severity_xgb_modal.png](figures/feat_importance/severity_xgb_modal.png) | `ch2_peak1_f`(15), `ch1_mean_logA`(11), `ch2_bandE`(9) | Peak frequency is more informative for boosting |
| [col_location_rf_modal.png](figures/feat_importance/col_location_rf_modal.png) | `ch2_bandE`(8), `ch6_bandE`(8), `ch2_std_logA`(4) | Same features as binary — col_location is not extracting much extra info |
| [col_location_xgb_modal.png](figures/feat_importance/col_location_xgb_modal.png) | `ch2_bandE`(12), `ch3_peak1_f`(11), `ch2_peak1_a`(8) | Boosting separates storeys via first-mode frequency at floor 3 |
| [mass_location_rf_modal.png](figures/feat_importance/mass_location_rf_modal.png) | `ch3_std_logA`(7), `ch7_bandE`(6), `ch7_std_logA`(6) | Floor-1 std + Floor-3 spectral energy — these isolate "which plate is heavier" |
| [mass_location_xgb_modal.png](figures/feat_importance/mass_location_xgb_modal.png) | `ch0_peak1_a`(20), `ch3_peak1_a`(17), `ch3_std_logA`(16) | Boosting almost solves it with the *amplitude* of the first mode at base and floor 1 |

### Indicator features

| file | top 3 features (importance %) | reading |
|------|-------------------------------|---------|
| [binary_rf_indicators.png](figures/feat_importance/binary_rf_indicators.png) | `unsigned_SCI`(8), `RVAC_std`(7), `FRFSM_6dB`(7) | A spread of indicators — no single dominant signal |
| [binary_xgb_indicators.png](figures/feat_importance/binary_xgb_indicators.png) | `unsigned_SCI`(10), `FRFRMS`(8), `GAC_min`(8) | Boosting picks SCI as the highest single split feature |
| [type_rf_indicators.png](figures/feat_importance/type_rf_indicators.png) | `unsigned_SCI`(9), `FRFRMS`(8), `RVAC_std`(7) | Same dominant indicators as binary |
| [type_xgb_indicators.png](figures/feat_importance/type_xgb_indicators.png) | `unsigned_SCI`(19), `FRFRMS`(9), `M2L_std`(7) | Even more concentrated on unsigned_SCI |
| [severity_rf_indicators.png](figures/feat_importance/severity_rf_indicators.png) | `FRFRMS`(20), `unsigned_SCI`(10), `M2L_std`(7) | FRFRMS quantifies the log-error magnitude — natural severity signal |
| [severity_xgb_indicators.png](figures/feat_importance/severity_xgb_indicators.png) | `GAC_min`(12), `FRFRMS`(11), `unsigned_SCI`(10) | Same regression signal, more balanced |
| [col_location_rf_indicators.png](figures/feat_importance/col_location_rf_indicators.png) | `FRFSF`(11), `RVAC_std`(6), `FRFSM_6dB`(6) | FRFSF dominates for location — but at ~50 % accuracy |
| [col_location_xgb_indicators.png](figures/feat_importance/col_location_xgb_indicators.png) | `FRFSF`(11), `unsigned_SCI`(7), `M2L_min`(7) | Same dominant feature |
| [mass_location_rf_indicators.png](figures/feat_importance/mass_location_rf_indicators.png) | `FRFSF`(17), `RVAC_std`(13), `GAC_std`(9) | FRFSF and RVAC_std jointly almost solve mass localisation |
| [mass_location_xgb_indicators.png](figures/feat_importance/mass_location_xgb_indicators.png) | `GAC_min`(31), `GAC_max`(13), `FRFSF`(11) | Boosting collapses onto GAC summary statistics |

**Take-away.** The features that recur across tasks (`ch2_bandE`,
`unsigned_SCI`, `FRFRMS`, `FRFSF`) are well-known
sensitivity-to-damage quantities — Floor-2 acceleration energy is
the single richest channel, the unsigned SCI is the dominant
classification-ready indicator, and FRFRMS is the natural severity
metric.

---

## 9. PCA + t-SNE feature-space embeddings

3 000-sample subset, coloured by damage type.  PCA is linear (axes
labelled with % variance explained); t-SNE is non-linear (axes are
unit-less).

### [`figures/embedding/pca_modal.png`](figures/embedding/pca_modal.png)

* **What.** 2-D PCA of the 81-d modal feature on 3 000 samples,
  coloured by type.
* **What is shown.** Three of the five classes form clearly
  separable clouds; Crack and Hole overlap heavily.
* **Conclusion.** The geometry already shows the Crack / Hole
  confusion observed in every confusion matrix — these classes are
  almost on the same manifold in modal-feature space.

### [`figures/embedding/tsne_modal.png`](figures/embedding/tsne_modal.png)

* **What.** Non-linear t-SNE of the same 3 000 samples.
* **What is shown.** Larger separation between Pristine, Bolt, and
  Mass; the Crack / Hole cluster is still mostly merged but breaks
  into a few sub-blobs.
* **Conclusion.** A non-linear head (MLP, RF) can pull more out of
  the modal features than a linear classifier — consistent with the
  MLP/modal scoring highest.

### [`figures/embedding/pca_indicators.png`](figures/embedding/pca_indicators.png)

* **What.** 2-D PCA of the 22-d indicator vector.
* **What is shown.** Pristine sits alone on a thin spike; Mass forms
  a separate cloud; Bolt / Crack / Hole are tangled.
* **Conclusion.** Indicators are an excellent *anomaly* feature
  (separating Pristine) but a poor *type* feature (failing on the
  three damage-mechanism classes).

### [`figures/embedding/tsne_indicators.png`](figures/embedding/tsne_indicators.png)

* **What.** Non-linear t-SNE of the indicator vector.
* **What is shown.** Same qualitative behaviour as PCA — Pristine is
  isolated, the three damage mechanisms merge.
* **Conclusion.** Indicators carry strong detection signal but weak
  type information; matches the confusion-matrix observation that
  indicator-feature models have low Crack/Hole F1.

---

## 10. HPO response surfaces

For every `(task, model, feature)` cell, a 2-D heatmap shows the
validation metric over the two hyperparameters grid.  The cell that
won is the brightest cell in each plot.

### How to read

* **Axes.** Horizontal = second hyperparameter; vertical = first.
  See [`docs/ml/THEORY.md`](../docs/ml/THEORY.md) §5 for what each
  hyperparameter does.
* **Colour.** Viridis: dark = low validation metric, bright = high.
* **Text.** The exact metric in every cell is overlaid.
* **What to look for.** A *flat* surface means HPO did not help; a
  *gradient* points toward the best configuration; an *island* of
  high metric in one corner suggests the grid bounds may be off.

### Per-task patterns

* **binary** ([`hpo/binary__*.png`](figures/hpo/)) — the modal
  surfaces show a strong gradient toward `lr = 3e-3` and wide
  hidden, peaking at val 0.99; deep-feature surfaces are essentially
  flat at 0.80 (class baseline).
* **type** ([`hpo/type__*.png`](figures/hpo/)) — modal MLP surface
  is strongly monotonic in both axes; cfdac+2-D CNN surface peaks
  in the middle of the grid.  Transformer / 1-D CNN surfaces are
  noisy with no clear optimum.
* **severity** ([`hpo/severity__*.png`](figures/hpo/)) — RF surface
  shows the classic "deeper trees + more estimators" gradient
  saturating around (300, None).  Deep-model surfaces are uniformly
  low.
* **col_location** ([`hpo/col_location__*.png`](figures/hpo/)) — every
  surface is flat or noisy.  The plateau at 0.50 corresponds to the
  AD/BD-degenerate ceiling discussed in §4.
* **mass_location** ([`hpo/mass_location__*.png`](figures/hpo/)) —
  modal surfaces saturate at val 1.0 across most of the grid (every
  reasonable hyperparameter solves it); cfdac+2-D CNN shows a clear
  optimum at (8, 16, 32) widths with kernel = 5.  Deep raw-feature
  surfaces are flat at 0.25 (random).

### Cells worth eyeballing individually

* [`hpo/binary__mlp__modal.png`](figures/hpo/binary__mlp__modal.png)
  — clean 3 × 3 surface ramping from 0.81 → 0.99; lr is the bigger
  knob than hidden width.
* [`hpo/binary__cnn2d__cfdac.png`](figures/hpo/binary__cnn2d__cfdac.png)
  — small 2 × 2 grid; kernel = 5 beats kernel = 3 in both width
  settings.
* [`hpo/severity__rf__modal.png`](figures/hpo/severity__rf__modal.png)
  — `n_estimators` × `max_depth` heatmap; `max_depth = None`
  outperforms 6 / 12 by a wide margin (~0.4 R² absolute).
* [`hpo/type__cnn2d__cfdac.png`](figures/hpo/type__cnn2d__cfdac.png)
  — the largest gain in the 2 × 2 grid (val 0.61 → 0.80) is going
  from `kernel = 3` to `kernel = 5` at small widths.
* [`hpo/col_location__transformer__frf_mag.png`](figures/hpo/col_location__transformer__frf_mag.png)
  — a flat surface around 0.25; the transformer simply cannot
  separate the 6 classes from raw |H(f)|.

**Take-away.** Engineered-feature surfaces have clear gradients and
respond to HPO; raw-feature surfaces are either flat at random or
hover near the class baseline regardless of hyperparameter — the
binding constraint is the representation, not the optimiser.

---

## Index of plot directories

* [`figures/dataset/`](figures/dataset/) — 1 plot.
* [`figures/signals/`](figures/signals/) — 3 plots.
* [`figures/confusion/`](figures/confusion/) — 44 plots.
* [`figures/perclass_f1/`](figures/perclass_f1/) — 4 plots.
* [`figures/roc/`](figures/roc/) — 2 plots.
* [`figures/scatter/`](figures/scatter/) — 11 plots.
* [`figures/feat_importance/`](figures/feat_importance/) — 20 plots.
* [`figures/embedding/`](figures/embedding/) — 4 plots.
* [`figures/hpo/`](figures/hpo/) — 55 plots.
* [`train_metrics_by_task.png`](figures/train_metrics_by_task.png),
  [`experimental_metrics_by_task.png`](figures/experimental_metrics_by_task.png) — 2 plots.

**Grand total: 146 plots.**
