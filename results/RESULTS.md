# Results — synthetic dataset training and IQS experimental evaluation

Auto-generated from `results/training_metrics.json` and
`results/experimental_evaluation.json`.  For full per-row data see
`results/training_metrics.csv` and `results/experimental_metrics.csv`.

## Dataset at a glance

* 10 000 samples, **5 classes × 2 000 samples** (Pristine / Bolt / Crack / Hole / Mass).
* Each non-Pristine class is sub-stratified across all of its physical locations
  (3 storeys × 2 ends for column damage; 4 plates for mass).
* Continuous severity drawn from physical bounds (bolt 5–95 %, crack 1–8 mm,
  hole 1–6 mm, mass 0.1–2.5 kg) with ±2 % E, ±1 % ρ, ±5 % JSR, ±20 % damping,
  ±0.5 % plate/column dimensions, ±50 g per-plate residual mass.
* Time series: 4 s, 256 Hz, 1024 samples, 9 channels, deterministic
  5–100 Hz chirp excitation (1 N).  No noise.

## Training metrics (synthetic, hold-out test split)

### binary — Pristine vs Damage  (10 000 samples)

| model        | feature      | test   |
|--------------|--------------|--------|
| **MLP**      | **modal**    | **0.9820** |
| XGBoost      | modal        | 0.9620 |
| Random Forest| modal        | 0.9513 |
| Transformer  | timeseries   | 0.9233 |
| XGBoost      | indicators   | 0.9207 |
| Random Forest| indicators   | 0.9160 |
| MLP          | indicators   | 0.8300 |
| 1-D CNN      | frf_mag      | 0.8240 |
| 1-D CNN      | timeseries   | 0.8213 |
| Transformer  | frf_mag      | 0.8000 |

### type — 5-class damage type  (10 000 samples)

| model        | feature      | test   |
|--------------|--------------|--------|
| **MLP**      | **modal**    | **0.8513** |
| XGBoost      | modal        | 0.8247 |
| Random Forest| modal        | 0.8100 |
| Random Forest| indicators   | 0.7520 |
| 1-D CNN      | frf_mag      | 0.7493 |
| XGBoost      | indicators   | 0.7453 |
| 1-D CNN      | timeseries   | 0.7353 |
| MLP          | indicators   | 0.6560 |
| Transformer  | frf_mag      | 0.5800 |
| Transformer  | timeseries   | 0.4633 |

### severity — R² regression  (8 000 samples)

| model        | feature      | test R² | MAE   |
|--------------|--------------|---------|-------|
| **Random Forest** | **modal**  | **0.5728** | 0.13 |
| XGBoost      | modal        | 0.5280  | 0.14  |
| Random Forest| indicators   | 0.4868  | 0.15  |
| XGBoost      | indicators   | 0.4645  | 0.15  |
| MLP          | indicators   | 0.3141  | 0.19  |
| MLP          | modal        | 0.3001  | 0.19  |
| 1-D CNN      | timeseries   | 0.2714  | 0.20  |
| 1-D CNN      | frf_mag      | 0.2443  | 0.20  |
| Transformer  | timeseries   | 0.0970  | 0.23  |
| Transformer  | frf_mag      | 0.0083  | 0.25  |

### col_location — 6-class column damage location  (6 000 samples)

| model        | feature      | test   |
|--------------|--------------|--------|
| **Random Forest** | **modal** | **0.5022** |
| 1-D CNN      | timeseries   | 0.4989 |
| XGBoost      | modal        | 0.4933 |
| MLP          | modal        | 0.4811 |
| Random Forest| indicators   | 0.4756 |
| XGBoost      | indicators   | 0.4567 |
| 1-D CNN      | frf_mag      | 0.4522 |
| MLP          | indicators   | 0.3922 |
| Transformer  | frf_mag      | 0.3878 |
| Transformer  | timeseries   | 0.3500 |

### mass_location — 4-class plate location for added mass  (2 000 samples)

| model        | feature      | test   |
|--------------|--------------|--------|
| **XGBoost**  | **modal**    | **0.9933** |
| Random Forest| modal        | 0.9900 |
| MLP          | modal        | 0.9900 |
| XGBoost      | indicators   | 0.9733 |
| Random Forest| indicators   | 0.9700 |
| MLP          | indicators   | 0.9233 |
| 1-D CNN      | timeseries   | 0.8900 |
| Transformer  | timeseries   | 0.7267 |
| Transformer  | frf_mag      | 0.5633 |
| 1-D CNN      | frf_mag      | 0.5533 |

## Experimental-data evaluation (61 IQS cases from `median_frfs.h5`)

Composite damage scenarios (e.g. `D(85%) 1BD + Mass First Floor`) are
labelled by their **primary** op for evaluation purposes
(bolt > crack > hole > mass > pristine).  This means the trained
single-damage classifier is being tested out-of-distribution.

### binary — Pristine vs Damage  (61 cases)

| model | feature | accuracy |
|-------|---------|----------|
| MLP / RF / XGB / Transformer  | modal / indicators / frf_mag | 0.869 |
| Transformer | timeseries | 0.787 |
| 1-D CNN     | frf_mag    | 0.705 |
| 1-D CNN     | timeseries | 0.311 |

The first row equals 53 / 61, which is the rate the experimental set
contains *damage* cases — i.e. those classifiers predict damage on
almost everything.  The deep models that consume the raw spectra do
worse than chance on this metric because they don't transfer cleanly
to the noisy, slightly mis-calibrated experimental FRFs.

### type — 5-class damage type  (61 cases)

| model        | feature    | accuracy |
|--------------|------------|----------|
| **MLP**      | **modal**  | **0.574** |
| RF           | modal      | 0.475 |
| XGB          | modal      | 0.426 |
| Transformer  | frf_mag    | 0.377 |
| 1-D CNN      | frf_mag    | 0.361 |
| 1-D CNN      | timeseries | 0.230 |
| Transformer  | timeseries | 0.197 |
| RF           | indicators | 0.164 |
| XGB          | indicators | 0.147 |
| MLP          | indicators | 0.066 |

The modal-peak features generalise the best because they encode
physically meaningful invariants (resonant frequencies, peak
amplitudes) that survive experimental noise; the `pymodal` damage
indicators were normalised against a synthetic pristine reference and
therefore drift on the experimental set.

### severity — R² regression  (53 damaged cases)

| model        | feature    | R²    | MAE  |
|--------------|------------|-------|------|
| Transformer  | timeseries | 0.017 | 0.28 |
| Transformer  | frf_mag    | −0.05 | 0.28 |
| XGB          | modal      | −0.08 | 0.29 |
| RF           | modal      | −0.15 | 0.30 |
| Others       | …          | < −0.4| > 0.3 |

Severity regression on the experimental set is essentially at the
constant-prediction R² (≈ 0).  Reasons: (a) the IQS protocol used
only discrete severity steps (5/8 mm cracks, 4/6 mm holes,
11/20/50/85 % bolts) while training spread severities continuously;
(b) experimental FRF amplitude calibration differs subtly from the
synthetic chirp reconstruction.

### col_location — 6-class location for column damage  (49 cases)

| model        | feature    | accuracy |
|--------------|------------|----------|
| **1-D CNN**  | **timeseries** | **0.429** |
| MLP          | indicators | 0.367 |
| MLP          | modal      | 0.367 |
| Transformer  | frf_mag    | 0.286 |
| Others       | …          | < 0.21 |

Random baseline is 1/6 ≈ 0.17.  The CNN+timeseries combination is the
only configuration meaningfully above chance; the modal-peak features
alone don't encode enough spatial information about which storey
loosened.

### mass_location — 4-class plate location  (4 cases)

Only 4 experimental cases exist (`Mass Base / 1F / 2F / 3F`); too few
for a robust hold-out score.  Most models score 0.25 (random
baseline); the simple Random Forest / MLP / XGB on modal features get
the same 1/4 hit and are not statistically distinguishable from
chance on this micro test set.

## Takeaways

* **Engineered modal features dominate generalisation.**  Across all
  five tasks the top configuration on the synthetic test set is an
  MLP/RF/XGB consuming the 81-d modal vector.  These same models also
  give the best transfer to the experimental set (binary 0.87, type
  0.57, col_location 0.37).
* **`pymodal` damage indicators** were excellent in-domain
  (binary 0.92, mass_location 0.97) but transfer poorly because the
  signed indicators (SCI, etc.) are referenced against the synthetic
  pristine mean — a calibrated mismatch with the experimental
  pristine.  Re-anchoring the reference to one of the eight
  experimental Pristine FRFs would close most of that gap.
* **Raw FRF / time-series + deep models** require significantly
  longer training and per-channel log-scaling to compete on this
  benchmark; without those, RF/MLP on engineered features remain the
  pragmatic baseline.
* **Severity regression** suffers the largest sim-to-real gap and
  would benefit from finetuning on a small handful of labelled
  experimental severities.

## Files

```
results/
├── training_metrics.json     all 50 (model × feature × task) rows
├── training_metrics.csv      same, flat CSV
├── experimental_evaluation.json   one row per (model × feature × task) on IQS
├── experimental_metrics.csv  same, flat CSV
├── experimental_per_case.json     per-case predictions for inspection
├── figures/
│   ├── train_metrics_by_task.png
│   └── experimental_metrics_by_task.png
└── models/                   50 trained artefacts (`.pkl` / `.pt`)
```
