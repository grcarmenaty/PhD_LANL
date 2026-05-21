> **Method study — bears on all four diagnosis goals.** A focused study of
> noise robustness, not a canonical report; the methodology-corrected,
> goal-structured results are in [`REPORT_definitive.md`](REPORT_definitive.md)
> and [`REPORT_full.md`](REPORT_full.md) (noise context: `REPORT_full.md`
> § 9.3). It re-runs the pipeline on synthetic data corrupted with additive
> Gaussian time-series noise at controlled SNR, across detection, type,
> severity and location. Numbers below pre-date the seeding / macro-F1
> corrections and are indicative.

# Noisy-synth study — companion to `REPORT.md`

Every experiment in [`REPORT.md`](REPORT.md) is repeated on synthetic data corrupted by additive Gaussian noise on the **time-series** field (1024 × 9 acceleration samples per signal).  The noise is applied **per sample, per channel, at a controlled signal-to-noise ratio**; every downstream feature (FRF, modal-peak vector, CFDAC variants, pymodal indicators) is then re-extracted from the noisy time series so the entire pipeline trains and tests on a self-consistent noisy dataset.

Five SNR levels are evaluated: **35, 25, 20, 15, 10 dB**.  All other settings (model menu, HPO grid, balanced experimental evaluation, transfer-learning sweep, resolution sweep) are identical to the clean study.

Coverage status at the time of this build:

| SNR (dB) | features.h5 | HPO | indicator | balanced eval | transfer | resolution |
|---|---|---|---|---|---|---|
| **35**  | ✓ | (189) | (66) | ✓ | ✓ | ✓ |
| **25**  | ✓ | (100) | — | — | — | — |
| **20**  | — | (189) | (66) | ✓ | ✓ | — |
| **15**  | — | — | — | — | — | — |
| **10**  | — | — | — | — | — | — |

---

# 1. Executive summary across SNRs

Best `(model, feature)` cell per task at each SNR.  Clean-data numbers (no noise) are quoted in the last column for reference; they come from `REPORT.md` § 1.

| task| 35 dB | 25 dB | 20 dB | clean |
|---|---|---|---|---|
| `binary`| 0.967 (xgb/cfdac_magphase)| 0.941 (cnn2d/cfdac)| 0.948 (cnn2d/cfdac)| see REPORT.md §1| |
| `type`| 0.840 (mlp/modal)| 0.807 (cnn2d/cfdac_imag)| 0.781 (cnn2d/cfdac)| see REPORT.md §1| |
| `severity`| 0.576 (xgb/cfdac_magphase)| 0.460 (rf/modal)| 0.438 (rf/modal)| see REPORT.md §1| |
| `col_location`| 0.498 (rf/cfdac_mag)| 0.518 (cnn/cfdac_imag)| 0.491 (cnn/timeseries)| see REPORT.md §1| |
| `mass_location`| 0.997 (rf/cfdac_imag)| 0.993 (rf/modal)| 0.993 (xgb/modal)| see REPORT.md §1| |

---

# 2. Dataset and noise injection

Same 10 000-sample synthetic generator as REPORT.md § 2.  After generation, every 1024-sample acceleration channel is corrupted with i.i.d. Gaussian noise scaled to hit the target per-sample SNR:

```
noise[i, t, c]  ~  N(0, σ_i)
σ_i  =  sqrt( P_signal[i] / 10**(SNR_dB / 10) )
```

`P_signal[i]` is the mean-square of the entire 1024 × 9 signal block — so the noise floor is per-sample, independent across channels, and homoscedastic across time.  After the addition the noisy time series feeds the same `features.py` / `cfdac.py` / `cfdac_variants.py` pipeline used in the clean run, producing matching `features_noisy_<SNR>dB.h5` files.

---

# 7. Cross-domain (balanced experimental) metrics per SNR

For every `(model, feature)` cell, the table below reports the balanced-experimental metric at each available SNR.  Same balanced 680-case dataset that REPORT.md § 9 uses, so the rows are directly comparable.

## binary

| model | feature| 35 dB | 25 dB | 20 dB ||
|---|---|---|---|---|
| cnn | `frf_mag` | 0.941 | — | 0.940 |
| cnn | `timeseries` | 0.888 | — | 0.469 |
| cnn2d | `cfdac` | 0.922 | — | 0.894 |
| cnn2d | `cfdac_imag` | 0.937 | — | 0.841 |
| cnn2d | `cfdac_mag` | 0.941 | — | 0.941 |
| cnn2d | `cfdac_magphase` | 0.941 | — | 0.940 |
| cnn2d | `cfdac_phase` | 0.935 | — | 0.871 |
| cnn2d | `cfdac_real` | 0.887 | — | 0.879 |
| mlp | `cfdac_all` | 0.594 | — | 0.934 |
| mlp | `cfdac_imag` | 0.809 | — | 0.847 |
| mlp | `cfdac_mag` | 0.941 | — | 0.941 |
| mlp | `cfdac_magphase` | 0.613 | — | 0.825 |
| mlp | `cfdac_phase` | 0.940 | — | 0.816 |
| mlp | `cfdac_real` | 0.940 | — | 0.940 |
| mlp | `cfdac_realimag` | 0.940 | — | 0.940 |
| mlp | `modal` | 0.941 | — | 0.941 |
| rf | `cfdac_all` | 0.206 | — | 0.890 |
| rf | `cfdac_imag` | 0.297 | — | 0.668 |
| rf | `cfdac_mag` | 0.941 | — | 0.844 |
| rf | `cfdac_magphase` | 0.775 | — | 0.846 |
| rf | `cfdac_phase` | 0.276 | — | 0.707 |
| rf | `cfdac_real` | 0.941 | — | 0.900 |
| rf | `cfdac_realimag` | 0.637 | — | 0.921 |
| rf | `modal` | 0.941 | — | 0.941 |
| transformer | `frf_mag` | 0.941 | — | 0.941 |
| transformer | `timeseries` | 0.935 | — | 0.788 |
| xgb | `cfdac_imag` | 0.941 | — | 0.941 |
| xgb | `cfdac_mag` | 0.941 | — | 0.941 |
| xgb | `cfdac_magphase` | 0.941 | — | 0.941 |
| xgb | `cfdac_phase` | 0.941 | — | 0.941 |
| xgb | `cfdac_real` | 0.941 | — | 0.941 |
| xgb | `cfdac_realimag` | 0.941 | — | 0.941 |
| xgb | `modal` | 0.941 | — | 0.941 |

## type

| model | feature| 35 dB | 25 dB | 20 dB ||
|---|---|---|---|---|
| cnn | `frf_mag` | 0.200 | — | 0.190 |
| cnn | `timeseries` | 0.312 | — | 0.321 |
| cnn2d | `cfdac` | 0.297 | — | 0.319 |
| cnn2d | `cfdac_imag` | 0.343 | — | 0.337 |
| cnn2d | `cfdac_mag` | 0.326 | — | 0.310 |
| cnn2d | `cfdac_magphase` | 0.310 | — | 0.276 |
| cnn2d | `cfdac_phase` | 0.271 | — | 0.241 |
| cnn2d | `cfdac_real` | 0.331 | — | 0.303 |
| mlp | `cfdac_imag` | 0.072 | — | 0.182 |
| mlp | `cfdac_mag` | 0.294 | — | 0.240 |
| mlp | `cfdac_magphase` | 0.088 | — | 0.321 |
| mlp | `cfdac_phase` | 0.382 | — | 0.321 |
| mlp | `cfdac_real` | 0.253 | — | 0.219 |
| mlp | `cfdac_realimag` | 0.213 | — | 0.300 |
| mlp | `modal` | 0.259 | — | 0.235 |
| rf | `cfdac_imag` | 0.246 | — | 0.235 |
| rf | `cfdac_mag` | 0.372 | — | 0.362 |
| rf | `cfdac_magphase` | 0.340 | — | 0.350 |
| rf | `cfdac_phase` | 0.216 | — | 0.296 |
| rf | `cfdac_real` | 0.276 | — | 0.326 |
| rf | `cfdac_realimag` | 0.256 | — | 0.337 |
| rf | `modal` | 0.199 | — | 0.207 |
| transformer | `frf_mag` | 0.390 | — | 0.262 |
| transformer | `timeseries` | 0.210 | — | 0.197 |
| xgb | `modal` | 0.319 | — | 0.231 |

## severity

| model | feature| 35 dB | 25 dB | 20 dB ||
|---|---|---|---|---|
| cnn | `frf_mag` | -20.783 | — | -2.822 |
| cnn | `timeseries` | -18.613 | — | -12.053 |
| cnn2d | `cfdac` | -0.341 | — | -0.290 |
| cnn2d | `cfdac_imag` | -0.423 | — | -0.355 |
| cnn2d | `cfdac_mag` | -0.297 | — | -0.147 |
| cnn2d | `cfdac_magphase` | -0.091 | — | -0.114 |
| cnn2d | `cfdac_phase` | -0.398 | — | -0.478 |
| cnn2d | `cfdac_real` | -0.438 | — | -0.195 |
| mlp | `cfdac_imag` | -1.968 | — | -3.518 |
| mlp | `cfdac_mag` | -1.190 | — | -0.731 |
| mlp | `cfdac_magphase` | -3.235 | — | -2.386 |
| mlp | `cfdac_phase` | -1.158 | — | -0.889 |
| mlp | `cfdac_real` | -2.376 | — | -1.462 |
| mlp | `cfdac_realimag` | -2.936 | — | -1.326 |
| mlp | `modal` | -28.088 | — | -34.088 |
| rf | `modal` | -0.159 | — | -0.118 |
| transformer | `frf_mag` | -0.117 | — | -0.137 |
| transformer | `timeseries` | -0.253 | — | -0.174 |
| xgb | `cfdac_imag` | 0.006 | — | 0.043 |
| xgb | `cfdac_mag` | 0.007 | — | -0.024 |
| xgb | `cfdac_magphase` | -0.180 | — | -0.123 |
| xgb | `cfdac_phase` | 0.177 | — | -0.843 |
| xgb | `cfdac_real` | -0.458 | — | -0.200 |
| xgb | `cfdac_realimag` | -0.615 | — | -0.409 |
| xgb | `modal` | -0.200 | — | -0.020 |

## col_location

| model | feature| 35 dB | 25 dB | 20 dB ||
|---|---|---|---|---|
| cnn | `frf_mag` | 0.319 | — | 0.235 |
| cnn | `timeseries` | 0.037 | — | 0.212 |
| cnn2d | `cfdac` | 0.127 | — | 0.000 |
| cnn2d | `cfdac_imag` | 0.250 | — | 0.115 |
| cnn2d | `cfdac_mag` | 0.292 | — | 0.175 |
| cnn2d | `cfdac_magphase` | 0.200 | — | 0.173 |
| cnn2d | `cfdac_phase` | 0.210 | — | 0.113 |
| cnn2d | `cfdac_real` | 0.108 | — | 0.196 |
| mlp | `cfdac_imag` | 0.127 | — | 0.177 |
| mlp | `cfdac_mag` | 0.271 | — | 0.340 |
| mlp | `cfdac_magphase` | 0.131 | — | 0.229 |
| mlp | `cfdac_phase` | 0.206 | — | 0.183 |
| mlp | `cfdac_real` | 0.204 | — | 0.300 |
| mlp | `cfdac_realimag` | 0.215 | — | 0.098 |
| mlp | `modal` | 0.000 | — | 0.115 |
| rf | `cfdac_imag` | 0.171 | — | 0.121 |
| rf | `cfdac_mag` | 0.142 | — | 0.158 |
| rf | `cfdac_magphase` | 0.075 | — | 0.200 |
| rf | `cfdac_phase` | 0.138 | — | 0.150 |
| rf | `cfdac_real` | 0.167 | — | 0.227 |
| rf | `cfdac_realimag` | 0.150 | — | 0.250 |
| rf | `modal` | 0.096 | — | 0.075 |
| transformer | `frf_mag` | 0.225 | — | 0.152 |
| transformer | `timeseries` | 0.158 | — | 0.090 |
| xgb | `modal` | 0.171 | — | 0.096 |

## mass_location

| model | feature| 35 dB | 25 dB | 20 dB ||
|---|---|---|---|---|
| cnn | `frf_mag` | 0.250 | — | 0.250 |
| cnn | `timeseries` | 0.250 | — | 0.237 |
| cnn2d | `cfdac` | 0.463 | — | 0.250 |
| cnn2d | `cfdac_imag` | 0.250 | — | 0.250 |
| cnn2d | `cfdac_mag` | 0.250 | — | 0.250 |
| cnn2d | `cfdac_magphase` | 0.325 | — | 0.562 |
| cnn2d | `cfdac_phase` | 0.150 | — | 0.338 |
| cnn2d | `cfdac_real` | 0.138 | — | 0.163 |
| mlp | `cfdac_imag` | 0.275 | — | 0.344 |
| mlp | `cfdac_mag` | 0.250 | — | 0.481 |
| mlp | `cfdac_magphase` | 0.250 | — | 0.325 |
| mlp | `cfdac_phase` | 0.469 | — | 0.075 |
| mlp | `cfdac_real` | 0.075 | — | 0.075 |
| mlp | `cfdac_realimag` | 0.250 | — | 0.250 |
| mlp | `modal` | 0.200 | — | 0.094 |
| rf | `cfdac_imag` | 0.075 | — | 0.188 |
| rf | `cfdac_mag` | 0.250 | — | 0.250 |
| rf | `cfdac_magphase` | 0.250 | — | 0.250 |
| rf | `cfdac_phase` | 0.250 | — | 0.194 |
| rf | `cfdac_real` | 0.169 | — | 0.250 |
| rf | `cfdac_realimag` | 0.075 | — | 0.250 |
| rf | `modal` | 0.250 | — | 0.250 |
| transformer | `frf_mag` | 0.250 | — | 0.250 |
| transformer | `timeseries` | 0.138 | — | 0.269 |
| xgb | `modal` | 0.250 | — | 0.000 |

---

# 10. Transfer-learning lift per SNR

Best `(model, feature, unfreeze)` cell at k=50 % per task, per SNR, with Δ vs the clean-data zero-shot.

| task | 35 dB | 25 dB | 20 dB |
|---|---|---|---|
| `binary` | +0.944 (cnn/cfdac_real) | — | +0.944 (cnn2d/cfdac_phase) | |
| `type` | +0.538 (mlp/cfdac_imag) | — | +0.568 (mlp/cfdac_imag) | |
| `severity` | +0.189 (cnn/cfdac_phase) | — | +0.191 (cnn/cfdac_imag) | |
| `col_location` | +0.388 (cnn/frf_mag) | — | +0.362 (cnn/cfdac_phase) | |
| `mass_location` | +0.675 (mlp/cfdac_phase) | — | +0.700 (cnn2d/cfdac_magphase) | |

---

# 11. Feature resolution sweep per SNR

Best cell per task at r=0.500 / r=1.000 (full) per SNR.

| task | r1@35dB | r0.5@35dB | r1@25dB | r0.5@25dB | r1@20dB | r0.5@20dB |
|---|---|---|---|---|---|---|
| `binary` | 0.968 | 0.959 | — | — | — | — | |
| `type` | — | — | — | — | — | — | |
| `severity` | — | — | — | — | — | — | |
| `col_location` | 0.400 | 0.471 | — | — | — | — | |
| `mass_location` | — | — | — | — | — | — | |

---

# 12. Reproducibility

```
# Re-run from scratch (per SNR):
python ml_pipeline/run_noise_sweep.py --snr-db 20

# All five levels in sequence:
python ml_pipeline/run_noise_sweep.py

# Rebuild this report after each SNR finishes:
python ml_pipeline/build_report_noise.py
```
