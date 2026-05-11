# Shared introduction (referenced from every task document)

> Each task document is self-contained.  The "What each model is"
> and "What each feature is" sections below are duplicated verbatim
> in every task doc so the reader does not have to jump between
> files.  This template is the canonical version — if a section
> needs an edit, change it here first.

## What each *model* is

Six model families are evaluated.  Same architecture is used across
tasks; only the head changes between classification and regression.

### Random Forest (RF)

* **What it is.**  An ensemble of decision trees built by
  bootstrapping samples and randomly subsampling features at each
  split.  Final prediction = majority vote (classification) or
  mean (regression) across the trees.
* **How it works.**  Each tree greedily splits the input space to
  minimise Gini impurity (classification) or variance (regression).
  Trees overfit individually; aggregating them reduces variance.
* **Why for this benchmark.**  Strong tabular baseline with no
  scaling sensitivity, fast to train, and gives feature importance
  for free.  Implemented with `sklearn.ensemble.RandomForestClassifier
  / RandomForestRegressor`, `class_weight="balanced"` for
  classification.

### XGBoost (XGB)

* **What it is.**  Gradient-boosted regression trees — trees are
  trained sequentially on the residuals of the previous trees.
* **How it works.**  Each new tree is fit to the gradient of the
  loss with respect to the current prediction; learning rate scales
  how much each tree contributes.
* **Why for this benchmark.**  Usually the best tabular method on
  small-to-medium structured datasets, handles non-linear
  interactions, and provides gain-based importance.  Implemented
  with `xgboost.XGBClassifier / XGBRegressor`, learning rate 0.1.

### Multilayer Perceptron (MLP)

* **What it is.**  Stack of fully-connected layers with non-linear
  activations (GELU here).
* **How it works.**  Three hidden layers
  `(d_in → 256 → 128 → 64 → d_out)` (default) or
  `(d_in → 512 → 256 → 128 → d_out)` after HPO.  Dropout 0.2
  between layers, AdamW optimiser, cosine LR schedule.
* **Why for this benchmark.**  Universal function approximator on
  fixed-length feature vectors; what HPO consistently picks as the
  best non-tree model on `modal` and `indicators`.

### 1-D Convolutional Neural Network (1-D CNN)

* **What it is.**  Convolutional stack along the time / frequency
  axis; treats each sensor as a channel.
* **How it works.**  Three `Conv1d + BN + GELU + MaxPool1d` blocks
  with widths `(32, 64, 128)` (default) or `(16, 32, 64)` per HPO,
  then global average pool + MLP head.  Kernel size 5 or 7.
* **Why for this benchmark.**  Standard backbone for vibration
  time series; tests whether a small CNN can learn the
  damage signature without engineered features.

### Small Transformer

* **What it is.**  Stack of self-attention encoder layers with a
  CLS-token head.
* **How it works.**  Strided 1-D convolution downsamples the long
  sequence (1024 → 64 tokens) before tokens enter
  `nn.TransformerEncoder`.  Two encoder layers, four heads,
  `d_model = 48` (default) or 64 after HPO.
* **Why for this benchmark.**  Modern alternative to the 1-D CNN
  on the same inputs; mainly here to test whether attention beats
  convolution on this small, low-noise dataset.

### 2-D Convolutional Neural Network (2-D CNN)

* **What it is.**  Convolutional stack over the 128 × 128 CFDAC
  matrix; the *only* model that consumes a 2-D matricial feature.
* **How it works.**  Strided 7 × 7 stem (stride 4) compresses the
  128 × 128 input to 32 × 32, then three
  `Conv2d + BN + GELU + MaxPool2d` blocks with widths
  `(16, 32, 64)` (default) or `(8, 16, 32)` after HPO.  Kernel
  size 3 or 5.
* **Why for this benchmark.**  CFDAC is a structurally-aligned
  damage map; a 2-D CNN can pool across local frequency bands
  the same way a 2-D CNN on an image pools across local pixels.

## What each *feature* is

Every model is paired with the feature representation(s) it is
shape-compatible with.

### `modal` — 81-d engineered modal-peak vector

For each of the 9 sensor channels, in order:

```
peak1_freq  peak1_amp  peak2_freq  peak2_amp  peak3_freq  peak3_amp
mean_log_amp  std_log_amp  band_energy
```

So `ch<c>_<stat>` is feature index `9·c + s` for stat index `s`.
The three peaks are found by `argpartition` on `|H(f)|` of the
sample.  This is a deliberate imitation of the natural-frequency
+ amplitude features a human modal analyst would write down.

Synth + real examples side by side:

![modal feature synth vs experimental](../figures/feature_examples/modal.png)

### `indicators` — 22-d pymodal damage-indicator vector

Every scalar is computed by `pymodal.utils.<name>` against the
synthetic pristine mean FRF.  Entries:

| idx | name                | meaning                                      |
|-----|---------------------|----------------------------------------------|
| 0   | `SCI`               | signed Structural Change Indicator           |
| 1   | `unsigned_SCI`      | `1 − |Pearson(CFDAC_ref, CFDAC_dmg)|`        |
| 2   | `DRQ`               | mean of the RVAC vector                      |
| 3   | `AIGAC`             | mean of the GAC vector                       |
| 4   | `FRFRMS`            | log-FRF RMS deviation                        |
| 5   | `FRFSF`             | FRF Shape Factor                             |
| 6   | `FRFSM_6dB`         | Standard Mean with 6 dB band                 |
| 7   | `ODS_diff`          | `Σ|FRF − Ref|` ODS-difference                |
| 8   | `r2_imag`           | R² of `Im(FRF)` against `Im(Ref)`            |
| 9–12 | `RVAC_{mean,std,min,max}` | summaries of per-frequency RVAC      |
| 13–16 | `GAC_{mean,std,min,max}`  | summaries of GAC                      |
| 17–21 | `M2L_{mean,std,min,max,abs_sum}` | summaries of M2L              |

![indicators feature synth vs experimental](../figures/feature_examples/indicators.png)

### `frf_mag` — `(N_f, 9)` `|H(f)|` spectrum

The accelerance magnitude on the 5–100 Hz band (≈ 381 bins at
0.25 Hz resolution).  Consumed only by 1-D CNN and Transformer.

![frf_mag synth vs experimental](../figures/feature_examples/frf_mag.png)

### `timeseries` — `(1024, 9)` raw acceleration

The 4 s chirp response sampled at 256 Hz.

![timeseries synth vs experimental](../figures/feature_examples/timeseries.png)

### `cfdac` — `(2, 128, 128)` real / imag CFDAC matrix

The Complex Frequency-Domain Assurance Criterion of the sample's
FRF against the synthetic pristine mean, decimated to 128 frequency
bins inside the 5–100 Hz band.  Consumed only by the 2-D CNN.

![cfdac synth vs experimental](../figures/feature_examples/cfdac.png)
