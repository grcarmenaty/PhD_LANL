# LANL 3SBB synthetic-dataset ML pipeline — theory & methodology

This document describes the full design space of the
`ml_pipeline/` module: what each feature *means*, why every model
architecture was selected, the exact hyperparameter grids used during
optimisation, the software stack, and the references behind each
choice.  It complements `ml_pipeline/README.md` (operational quick-
start) and [`../../results/RESULTS.md`](../../results/RESULTS.md) (numerical findings).

---

## 1 Problem statement

The IQS lab at Los Alamos National Laboratory measured the
frequency-response functions (FRFs) of the 3-Storey Bookcase
Benchmark (3SBB) for 61 structural-damage scenarios.  The calibrated
reduced-order model in `reduced_model_semirigid.py` reproduces those
FRFs in milliseconds, exposing only a small number of physical knobs
(Young's modulus, joint stiffness ratio, plate masses, column
factors, damping).

We use that ROM as a **digital twin**: by perturbing its parameters
within physically plausible bounds and applying the standard damage
operators (bolt loosening, column crack, column hole, plate mass),
we obtain an arbitrarily large library of *labelled* synthetic FRFs.
Time-domain responses to a deterministic shaker chirp are then
reconstructed from those FRFs and used as the raw input for the ML
pipeline.

Five inference tasks are studied:

| task            | type         | classes        | n samples | random baseline |
|-----------------|--------------|----------------|-----------|-----------------|
| binary          | classification | 2            | 10 000    | 0.50 / 0.80*    |
| type            | classification | 5            | 10 000    | 0.20            |
| severity        | regression     | continuous   |  8 000    | n/a             |
| col_location    | classification | 6            |  6 000    | 0.17            |
| mass_location   | classification | 4            |  2 000    | 0.25            |

*`binary` baseline 0.80 = predicting the majority class (damage); 0.50 = uniform.*

---

## 2 Generative parametrisation

Each sample draws independent perturbations of the calibrated
pristine geometry (see `ml_pipeline/variation.py`):

| parameter            | distribution         | rationale |
|----------------------|----------------------|-----------|
| Young's modulus E    | U(0.98, 1.02) · E₀   | spec tolerance for 6061-T6 aluminium [Hibbeler 2018] |
| density ρ            | U(0.99, 1.01) · ρ₀   | rolling / casting density spread |
| joint stiffness JSR  | U(0.95, 1.05) · JSR₀ | bolted-joint preload variation [Coletta & Adams 2017] |
| modal damping        | U(0.80, 1.20) · ζ₀   | wide because damping is least repeatable |
| plate dimensions Lx,Ly,Lz | U(0.995, 1.005)·L₀ | mill-finish tolerance |
| column dimensions    | U(0.995, 1.005)·L₀  | extrusion tolerance |
| base-extra mass      | U(−100, +100) g     | bolt heads, accelerometer cables |
| plate residual mass  | U(−50, +50) g       | per-plate residuals |
| damage parameter     | U over bounded range | bolt 5–95 %, crack 1–8 mm, hole 1–6 mm, mass 0.1–2.5 kg |

The damage operator selected for the sample sets the relevant
storey-stiffness ratio (or per-end JSR for bolts) according to
piece-wise-linear interpolations of the calibration anchors in
`damage_scenarios.py`.  Hole and crack ratios use the linearisations
`ratio_hole(d_mm) = 1 − 0.005·d_mm` and
`ratio_crack(s_mm) = 1 − 0.008·s_mm` respectively; bolt JSR uses the
six-anchor table `(5, 11, 20, 50, 85, 95) → (0.94, 0.85, 0.70, 0.55,
0.39, 0.30)` linearly interpolated.

There is **no measurement noise added** — every sample-to-sample
variation is a physical parameter of a different structural
realisation.  This was a deliberate scoping decision: it lets the ML
pipeline learn the structural fingerprint without confounding with
sensor-noise modelling, and the sim-to-real noise gap is left as the
test-set domain shift in `evaluate.py`.

---

## 3 Excitation and time-series synthesis

* `F(t) = sin(2π·[f₀·t + ½·k·t²])` with `f₀ = 2 Hz`, `k = (100−2)/4 s⁻¹`,
  giving a 2–100 Hz linear chirp.
* `fs = 256 Hz`, `N_t = 1024` samples → 4 s acquisition.  Nyquist
  (128 Hz) sits comfortably above the analysis band (5–100 Hz used
  downstream).
* For sample *i* the acceleration response at sensor *j* is
  `y_{ij}(t) = ℱ⁻¹{H_{ij}(f) · ℱ{F(t)}}` where `H_{ij}` is the
  calibrated `(n_freq, 9, 1)` accelerance matrix from
  `compute_frf_matrix`.  Because `F(t)` is deterministic and shared,
  every sample's variability is purely structural.

This reconstruction is mathematically equivalent to the chirp-input
test the LANL group performs on the real structure; the only
difference is the absence of noise.

---

## 4 Feature representations

The five feature views stored in `dataset/features.h5` are the
canonical Brüel & Kjær / `pymodal` representations a vibration
engineer would compute.  Each has a different inductive bias.

### 4.1 `timeseries` — `(N_t, 9)` raw acceleration

Direct output of the time-domain reconstruction above.  Suited to
1-D CNN and Transformer architectures that learn temporal filters.

### 4.2 `frf_mag` — `(N_f, 9)` |H(f)| on the 5–100 Hz band

`|H(f)|` is the standard frequency-domain damage diagnostic; peaks
mark natural frequencies and their shifts mark damage.  Stored on
the `rfft` bin grid restricted to 5–100 Hz (≈381 bins at 0.25 Hz
resolution).

### 4.3 `frf_real`, `frf_imag` — `(N_f, 9)` real/imag of H(f)

Same band as `frf_mag` but with full complex information so that
deep models can in principle exploit phase, residue signs and
imaginary-part curvature.

### 4.4 `modal` — 81-d vector of peak-based modal features

For each of the 9 channels, in order:

```
peak1_freq, peak1_amp, peak2_freq, peak2_amp, peak3_freq, peak3_amp,
mean_log_amp, std_log_amp, band_energy
```

The three peaks are found by `argpartition` on `|H(f)|`.  This is an
explicit imitation of the manual "natural-frequency + amplitude"
features a modal analyst writes down by hand.  Total length = 9·9 = 81.

### 4.5 `indicators` — 22-d vector of pymodal damage indicators

Every entry below is computed by `pymodal.utils.<name>` (or, in the
1-D case, summarised by `np.{mean, std, min, max}`) using the
*synthetic pristine mean* `H_ref(f)` as the reference.  Symbols
follow the `pymodal` README and the originating papers.

| idx | name        | type   | definition (short) |
|-----|-------------|--------|--------------------|
| 0   | `SCI`       | scalar | Structural Change Indicator: `sign·(1 − |PCC(CFDAC_ref, CFDAC_dmg)|)` [García-Macías 2020] |
| 1   | `unsigned_SCI` | scalar | `1 − |PCC|` of the same CFDAC pair |
| 2   | `DRQ`       | scalar | mean of the RVAC vector — Detection / Repeatability Quotient [Heyns 1998] |
| 3   | `AIGAC`     | scalar | mean of the GAC vector — Average Integrated GAC |
| 4   | `FRFRMS`    | scalar | log-FRF RMS difference, `√Σ((logFRF − logRef)/logRef)²` |
| 5   | `FRFSF`    | scalar  | FRF Shape Factor: `Σ|ref|/Σ|dmg|` |
| 6   | `FRFSM_6dB`| scalar  | Standard Mean with 6 dB band: Gaussian-weighted dB-error in 6 dB |
| 7   | `ODS_diff` | scalar  | `Σ|FRF − Ref|` ODS-difference indicator |
| 8   | `r2_imag`  | scalar  | R² of `Im(FRF)` against `Im(Ref)` |
| 9–12  | `RVAC_{mean,std,min,max}` | scalars | summaries of the per-freq RVAC vector [Heyns 1998] |
| 13–16 | `GAC_{mean,std,min,max}`  | scalars | summaries of the Global Amplitude Criterion |
| 17–21 | `M2L_{mean,std,min,max,abs_sum}` | scalars | summaries of the M2L damage-localisation vector [Fernández Esmerats 2022] |

`PCC` is Pearson product-moment correlation between the flattened
CFDAC matrices.  The 2-D CFDAC matrix itself
`CFDAC(ω_i, ω_j) = (FRF · Ref*) ∘ … / normalisation` is the kernel
behind the SCI / M2L indicators; we feed the *raw* CFDAC into the
2-D CNN below (§ 5.6) so the network can rediscover its own
indicators from the matrix instead of relying on the analytically
defined scalars.

### 4.6 `cfdac` — `(128, 128, 2)` CFDAC real/imag per sample

The Complex Frequency-Domain Assurance Criterion is

```
CFDAC(i, j) = ( Σ_k FRF_k(ω_i) · Ref*_k(ω_j) )² /
              ( Σ_k |FRF_k(ω_i)|² · Σ_k |Ref_k(ω_j)|² )
```

a 2-D structural-change signature widely used in SHM
[Pastor & Binda 2012].  We decimate the 5–100 Hz band to 128
frequency bins (≈ 0.75 Hz resolution) so the matrix is small enough
to feed a 2-D CNN as a `(2, 128, 128)` tensor (channel = `[real, imag]`).

---

## 5 Model architectures

Every model is committed to `ml_pipeline/models.py` and trained with
the same hold-out split (70 / 15 / 15, stratified for classification,
random for regression, seed 20260511).

### 5.1 Random Forest  (scikit-learn 1.x)

`RandomForestClassifier` / `RandomForestRegressor`, default 200/300
trees, Gini / MSE split criterion, `class_weight="balanced"` for
classification.  Why: strong tabular baseline, no scaling needed,
out-of-the-box class balance, fast on 81-d / 22-d inputs.

HPO grid (each task × feature):

| hyperparameter | values                     |
|----------------|----------------------------|
| `n_estimators` | 100, 200, 300              |
| `max_depth`    | 6, 12, None                |

### 5.2 XGBoost  (xgboost 2.x)

`XGBClassifier` / `XGBRegressor`, default 300 estimators, depth 6,
learning rate 0.1.  Gradient-boosted trees with sigmoidal / softmax
heads.  HPO grid:

| hyperparameter | values                     |
|----------------|----------------------------|
| `n_estimators` | 100, 300, 600              |
| `max_depth`    | 4, 6, 8                    |

### 5.3 Multilayer Perceptron  (PyTorch 2.x)

Three hidden layers `(256, 128, 64)` with `GELU` activations and
`Dropout(0.2)` between layers, AdamW (`lr = 1e-3, weight_decay =
1e-4`), `CosineAnnealingLR`.  CrossEntropyLoss for classification,
MSELoss for regression.  Input is the flattened feature vector
(modal 81-d or indicators 22-d), standardised with `StandardScaler`
fit on the training fold.

HPO grid:

| hyperparameter   | values                  |
|------------------|-------------------------|
| `hidden`         | (128, 64), (256, 128, 64), (512, 256, 128) |
| `lr`             | 5e-4, 1e-3, 3e-3        |

### 5.4 1-D CNN  (PyTorch 2.x)

`Conv1d → BatchNorm1d → GELU → MaxPool1d` blocks (3 blocks with
widths `(32, 64, 128)`, kernel = 7, padding kept), then global
average pool, two-layer MLP head `(128 → n_out)`.  AdamW, cosine LR.
Sequence inputs `(B, C=9, L)` where `L = 1024` for `timeseries` or
~381 for `frf_mag`.  No log-scaling on `frf_mag`; standardisation
is performed implicitly by BatchNorm.

HPO grid:

| hyperparameter  | values                                     |
|-----------------|--------------------------------------------|
| `widths`        | (16, 32, 64), (32, 64, 128), (64, 128, 256) |
| `kernel_size`   | 5, 7, 9                                    |
| `lr`            | 1e-3, 3e-3                                 |

### 5.5 Small Transformer  (PyTorch 2.x)

Strided `Conv1d` projection (kernel = stride = `downsample`)
reduces the 1024-sample (or ~381-bin) sequence to a small token
sequence of length `L / downsample`.  A learnable CLS token is
prepended.  `nn.TransformerEncoder` with `batch_first=True`,
`activation="gelu"`, two encoder layers, four heads, `d_model = 48`,
`dim_feedforward = 4 · d_model`, dropout 0.1.  Head is `LayerNorm`
+ `Linear(d_model, n_out)`.

HPO grid:

| hyperparameter | values             |
|----------------|--------------------|
| `d_model`      | 32, 48, 64         |
| `n_layers`     | 1, 2, 3            |
| `downsample`   | 8, 16              |

### 5.6 2-D CNN  (new — operates on CFDAC)

Three `Conv2d → BatchNorm2d → GELU → MaxPool2d` blocks with widths
`(16, 32, 64)`, kernel = 5, padding kept.  Reduces a `(2, 128, 128)`
CFDAC tensor to `(64, 16, 16)`, global average pool, then a
two-layer MLP head `(64 → 64 → n_out)`.  AdamW, cosine LR, 6 epochs,
batch = 64.  This is the matricial analogue of the 1-D CNN and is
the architecture meant for the 2-D damage indicators (CFDAC, FDAC).

HPO grid:

| hyperparameter | values                                    |
|----------------|-------------------------------------------|
| `widths`       | (8, 16, 32), (16, 32, 64), (32, 64, 128) |
| `kernel_size`  | 3, 5, 7                                  |
| `lr`           | 1e-3, 3e-3                               |

---

## 6 Hyperparameter optimisation

`ml_pipeline/hpo.py` performs an exhaustive grid search per
`(task, model, feature)` cell.  Each trial trains on the train
fold and is scored on the validation fold; the best trial is
re-evaluated on the test fold and saved as the final artefact in
[`../../results/models/`](../../results/models/).  Every trial's metrics, hyperparameters and
runtime are logged to [`../../results/hpo/`](../../results/hpo/)
so that response surfaces can be plotted post-hoc.

Response-surface plots in [`../../results/figures/hpo/`](../../results/figures/hpo/) are generated by
`ml_pipeline/plot_results.py`: for every `(task, model, feature)`
cell the validation metric is reshaped into a 2-D grid over the two
varying hyperparameters and rendered as an `imshow` heatmap.

---

## 7 Evaluation protocol

The synthetic test split is stratified (15 % of 10 000 = 1 500 rows
for tasks that cover the whole dataset).  Classification metric is
accuracy; regression metric is R² with MAE as a secondary readout.
For multi-class problems we also compute one-vs-rest ROC AUC and
macro F1 (see [`../../results/figures/`](../../results/figures/)).

The cross-domain (IQS experimental) evaluation in
`ml_pipeline/evaluate.py` builds the same feature stack from
`median_frfs.h5`, parses each case name into our 5-class taxonomy,
and applies every trained classifier.  Composite cases such as
`D(85%) 1BD + Mass First Floor` are mapped to a single *primary*
op via the priority `bolt > crack > hole > mass > pristine`; this
deliberately tests how the single-damage classifier behaves on
out-of-distribution multi-damage signatures.

---

## 8 Software stack

| package    | version  | purpose                              |
|------------|----------|--------------------------------------|
| python     | 3.11     | runtime                              |
| numpy      | 2.x      | linear algebra                       |
| scipy      | 1.13     | eigenvalue, FFT helpers              |
| h5py       | 3.10     | HDF5 chunk storage                   |
| scikit-learn | 1.5    | RF, scaling, splits, metrics         |
| xgboost    | 2.x      | gradient boosting                    |
| torch      | 2.11+cpu | MLP, CNN, Transformer, 2-D CNN       |
| matplotlib | 3.9      | plots                                |
| pymodal    | 0.2.0    | FRF / CFDAC / SCI utilities          |
| pyFRF      | 0.40     | H1 estimator (via `pymodal`)         |

---

## 9 References

* Brincker & Ventura, *Introduction to Operational Modal Analysis*, Wiley 2015.
* Coletta, J. & Adams, D. E., "Sensitivity of bolted-joint stiffness …", *J. Sound Vib.*, 2017.
* Fernández Esmerats, J., *M2L: A damage-localisation indicator
  derived from the CFDAC matrix*, MSc thesis, EEBE 2022.
* García-Macías, E. et al., *Frequency-Domain SCI*, *Mech. Sys. Sig. Process*, 2020.
* Heyns, P.S., *RVAC and DRQ for damage detection*, *J. Acoustic Emission*, 1998.
* Hibbeler, R. C., *Mechanics of Materials*, 10th ed., Pearson 2018.
* Pastor, M. & Binda, M., *Modal Assurance Criterion*, *Procedia Engineering* 48, 2012.
