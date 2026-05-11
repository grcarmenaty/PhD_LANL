# LANL 3SBB synthetic damage dataset & ML pipeline

A 10 000-sample synthetic dataset of vibration time series derived from
the calibrated 3SBB reduced-order model, paired with classification /
regression models for damage detection, type, location and severity.

## Layout

```
ml_pipeline/
├── case_design.py      enumerate the 5 damage types and 22 physical locations
├── variation.py        per-sample sampling of material / geometry / damage params
├── generate_dataset.py time-series generator → dataset/chunk_XXXX.h5
├── features.py         vibrational-feature extraction → dataset/features.h5
├── tasks.py            5 ML targets (binary, type, severity, col_location, mass_location)
├── models.py           MLP, 1-D CNN, small Transformer
├── train.py            per-(task, model, feature) training driver
└── evaluate.py         test trained models on median_frfs.h5
```

## Dataset

* **10 000 samples**, **5 classes × 2 000 samples each** (Pristine /
  Bolt / Crack / Hole / Mass).  Within each non-Pristine type the
  2 000 samples are sub-stratified across all physical locations of
  that type (3 storeys × 2 ends for column damage; 4 plates for mass).
* Severity is **continuously sampled** within bounded ranges:
  * Bolt loosening: 5 – 95 %
  * Crack size:     1 – 8 mm
  * Hole diameter:  1 – 6 mm
  * Mass:           0.1 – 2.5 kg
* Each sample additionally varies in Young's modulus (±2 %), density
  (±1 %), joint stiffness (±5 %), modal damping (±20 %), plate and
  column dimensions (±0.5 %), and per-plate residual masses (±50 g).
  **No measurement noise** is injected — all spread is physical.
* Inputs are a deterministic 5 – 100 Hz linear chirp (1 N amplitude,
  4 s, 256 Hz sampling, 1024 samples per channel).
* 9-channel acceleration is computed by `H(f) · F(f) → IFFT`, where
  `H(f)` comes from the calibrated semi-rigid shear-building model
  (`reduced_model_semirigid.compute_frf_matrix`).
* On-disk: **20 HDF5 chunks ≈ 8.4 MB each** (`dataset/chunk_XXXX.h5`)
  – well under the 20 MB cap.  Manifest: `dataset/manifest.json`.

### Per-chunk schema

```
signals      (n, 1024, 9)   float32   m/s²
time         (1024,)         float32   s
excitation   (1024,)         float32   N
freqs        (513,)          float32   Hz   (rfft bins)
labels/sample_id   int32
labels/type_code   int8      0..4
labels/storey      int8      -1 = N/A
labels/end         int8      -1 = N/A  | 0=BD,1=AD (column damage) | 0..3 plate (mass)
labels/severity    float32   percent / mm / kg
params/*           float32   per-sample physical jitter factors
```

## Features (`dataset/features.h5`, ≈ 340 MB)

Five representations stored per sample:

* `timeseries`  (10000, 1024, 9) float32 — raw acceleration
* `frf_mag`     (10000, N_F, 9)  float32 — |H(f)| (5–100 Hz band)
* `frf_real`, `frf_imag` (10000, N_F, 9)  float32 — complex FRF
* `modal`       (10000, 81)      float32 — peak freqs/amps + summary stats per channel
* `indicators`  (10000, 22)      float32 — `pymodal` damage indicators
  (SCI, unsigned_SCI, DRQ, AIGAC, FRFRMS, FRFSF, FRFSM_6dB, ODS_diff,
   r2_imag, plus mean/std/min/max summaries of RVAC, GAC, M2L)

Reference FRF for the indicators is the mean of all 2 000 Pristine
samples, stored under `reference/frf_complex` (and `frf_mag`).

## Tasks

| task            | n samples | kind | target                                       |
|-----------------|-----------|------|----------------------------------------------|
| binary          | 10 000    | cls  | 0 = Pristine, 1 = any damage                 |
| type            | 10 000    | cls  | Pristine / Bolt / Crack / Hole / Mass (5)    |
| severity        | 8 000     | reg  | severity normalised to [0, 1] per type       |
| col_location    | 6 000     | cls  | 3 storeys × 2 ends  →  6 classes             |
| mass_location   | 2 000     | cls  | 4 plate indices                              |

## Models

| arch          | features it consumes              |
|---------------|-----------------------------------|
| Random Forest | modal, indicators                 |
| XGBoost       | modal, indicators                 |
| MLP           | modal, indicators                 |
| 1-D CNN       | frf_mag, timeseries               |
| Transformer   | frf_mag, timeseries               |

Splits: 70 % train / 15 % val / 15 % test, stratified for
classification.

## Running

```bash
# 1. Generate the dataset (~30 s)
python ml_pipeline/generate_dataset.py

# 2. Extract all feature representations (~45 s)
python ml_pipeline/features.py

# 3. Train every (task, model, feature) combination (~15 min)
python ml_pipeline/train.py --epochs 8

# 4. Test trained models against the IQS experimental dataset
python ml_pipeline/evaluate.py
```

Results land in `results/training_metrics.json` and
`results/experimental_evaluation.json`; trained artefacts under
`results/models/`.
