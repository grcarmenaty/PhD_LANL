# Noisy-synth study — companion to `REPORT.md`

Every experiment in [`REPORT.md`](REPORT.md) is repeated on synthetic data corrupted by additive Gaussian noise on the **time-series** field (1024 × 9 acceleration samples per signal).  The noise is applied **per sample, per channel, at a controlled signal-to-noise ratio**; every downstream feature (FRF, modal-peak vector, CFDAC variants, pymodal indicators) is then re-extracted from the noisy time series so the entire pipeline trains and tests on a self-consistent noisy dataset.

Five SNR levels are evaluated: **35, 25, 20, 15, 10 dB**.  All other settings (model menu, HPO grid, balanced experimental evaluation, transfer-learning sweep, resolution sweep) are identical to the clean study.

Coverage status at the time of this build:

| SNR (dB) | features.h5 | HPO | indicator | balanced eval | transfer | resolution |
|---|---|---|---|---|---|---|
| **35**  | — | — | — | — | — | — |
| **25**  | — | — | — | — | — | — |
| **20**  | ✓ | (189) | (66) | ✓ | ✓ | — |
| **15**  | — | — | — | — | — | — |
| **10**  | — | — | — | — | — | — |

---

# 1. Executive summary across SNRs

Best `(model, feature)` cell per task at each SNR.  Clean-data numbers (no noise) are quoted in the last column for reference; they come from `REPORT.md` § 1.

| task| 20 dB | clean |
|---|---|---|
| `binary`| 0.948 (cnn2d/cfdac)| see REPORT.md §1| |
| `type`| 0.781 (cnn2d/cfdac)| see REPORT.md §1| |
| `severity`| 0.438 (rf/modal)| see REPORT.md §1| |
| `col_location`| 0.491 (cnn/timeseries)| see REPORT.md §1| |
| `mass_location`| 0.993 (xgb/modal)| see REPORT.md §1| |

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

| model | feature| 20 dB ||
|---|---|---|
| cnn | `frf_mag` | 0.940 |
| cnn | `timeseries` | 0.469 |
| cnn2d | `cfdac` | 0.894 |
| cnn2d | `cfdac_imag` | 0.841 |
| cnn2d | `cfdac_mag` | 0.941 |
| cnn2d | `cfdac_magphase` | 0.940 |
| cnn2d | `cfdac_phase` | 0.871 |
| cnn2d | `cfdac_real` | 0.879 |
| mlp | `cfdac_all` | 0.934 |
| mlp | `cfdac_imag` | 0.847 |
| mlp | `cfdac_mag` | 0.941 |
| mlp | `cfdac_magphase` | 0.825 |
| mlp | `cfdac_phase` | 0.816 |
| mlp | `cfdac_real` | 0.940 |
| mlp | `cfdac_realimag` | 0.940 |
| mlp | `modal` | 0.941 |
| rf | `cfdac_all` | 0.890 |
| rf | `cfdac_imag` | 0.668 |
| rf | `cfdac_mag` | 0.844 |
| rf | `cfdac_magphase` | 0.846 |
| rf | `cfdac_phase` | 0.707 |
| rf | `cfdac_real` | 0.900 |
| rf | `cfdac_realimag` | 0.921 |
| rf | `modal` | 0.941 |
| transformer | `frf_mag` | 0.941 |
| transformer | `timeseries` | 0.788 |
| xgb | `cfdac_imag` | 0.941 |
| xgb | `cfdac_mag` | 0.941 |
| xgb | `cfdac_magphase` | 0.941 |
| xgb | `cfdac_phase` | 0.941 |
| xgb | `cfdac_real` | 0.941 |
| xgb | `cfdac_realimag` | 0.941 |
| xgb | `modal` | 0.941 |

## type

| model | feature| 20 dB ||
|---|---|---|
| cnn | `frf_mag` | 0.190 |
| cnn | `timeseries` | 0.321 |
| cnn2d | `cfdac` | 0.319 |
| cnn2d | `cfdac_imag` | 0.337 |
| cnn2d | `cfdac_mag` | 0.310 |
| cnn2d | `cfdac_magphase` | 0.276 |
| cnn2d | `cfdac_phase` | 0.241 |
| cnn2d | `cfdac_real` | 0.303 |
| mlp | `cfdac_imag` | 0.182 |
| mlp | `cfdac_mag` | 0.240 |
| mlp | `cfdac_magphase` | 0.321 |
| mlp | `cfdac_phase` | 0.321 |
| mlp | `cfdac_real` | 0.219 |
| mlp | `cfdac_realimag` | 0.300 |
| mlp | `modal` | 0.235 |
| rf | `cfdac_imag` | 0.235 |
| rf | `cfdac_mag` | 0.362 |
| rf | `cfdac_magphase` | 0.350 |
| rf | `cfdac_phase` | 0.296 |
| rf | `cfdac_real` | 0.326 |
| rf | `cfdac_realimag` | 0.337 |
| rf | `modal` | 0.207 |
| transformer | `frf_mag` | 0.262 |
| transformer | `timeseries` | 0.197 |
| xgb | `modal` | 0.231 |

## severity

| model | feature| 20 dB ||
|---|---|---|
| cnn | `frf_mag` | -2.822 |
| cnn | `timeseries` | -12.053 |
| cnn2d | `cfdac` | -0.290 |
| cnn2d | `cfdac_imag` | -0.355 |
| cnn2d | `cfdac_mag` | -0.147 |
| cnn2d | `cfdac_magphase` | -0.114 |
| cnn2d | `cfdac_phase` | -0.478 |
| cnn2d | `cfdac_real` | -0.195 |
| mlp | `cfdac_imag` | -3.518 |
| mlp | `cfdac_mag` | -0.731 |
| mlp | `cfdac_magphase` | -2.386 |
| mlp | `cfdac_phase` | -0.889 |
| mlp | `cfdac_real` | -1.462 |
| mlp | `cfdac_realimag` | -1.326 |
| mlp | `modal` | -34.088 |
| rf | `modal` | -0.118 |
| transformer | `frf_mag` | -0.137 |
| transformer | `timeseries` | -0.174 |
| xgb | `cfdac_imag` | 0.043 |
| xgb | `cfdac_mag` | -0.024 |
| xgb | `cfdac_magphase` | -0.123 |
| xgb | `cfdac_phase` | -0.843 |
| xgb | `cfdac_real` | -0.200 |
| xgb | `cfdac_realimag` | -0.409 |
| xgb | `modal` | -0.020 |

## col_location

| model | feature| 20 dB ||
|---|---|---|
| cnn | `frf_mag` | 0.235 |
| cnn | `timeseries` | 0.212 |
| cnn2d | `cfdac` | 0.000 |
| cnn2d | `cfdac_imag` | 0.115 |
| cnn2d | `cfdac_mag` | 0.175 |
| cnn2d | `cfdac_magphase` | 0.173 |
| cnn2d | `cfdac_phase` | 0.113 |
| cnn2d | `cfdac_real` | 0.196 |
| mlp | `cfdac_imag` | 0.177 |
| mlp | `cfdac_mag` | 0.340 |
| mlp | `cfdac_magphase` | 0.229 |
| mlp | `cfdac_phase` | 0.183 |
| mlp | `cfdac_real` | 0.300 |
| mlp | `cfdac_realimag` | 0.098 |
| mlp | `modal` | 0.115 |
| rf | `cfdac_imag` | 0.121 |
| rf | `cfdac_mag` | 0.158 |
| rf | `cfdac_magphase` | 0.200 |
| rf | `cfdac_phase` | 0.150 |
| rf | `cfdac_real` | 0.227 |
| rf | `cfdac_realimag` | 0.250 |
| rf | `modal` | 0.075 |
| transformer | `frf_mag` | 0.152 |
| transformer | `timeseries` | 0.090 |
| xgb | `modal` | 0.096 |

## mass_location

| model | feature| 20 dB ||
|---|---|---|
| cnn | `frf_mag` | 0.250 |
| cnn | `timeseries` | 0.237 |
| cnn2d | `cfdac` | 0.250 |
| cnn2d | `cfdac_imag` | 0.250 |
| cnn2d | `cfdac_mag` | 0.250 |
| cnn2d | `cfdac_magphase` | 0.562 |
| cnn2d | `cfdac_phase` | 0.338 |
| cnn2d | `cfdac_real` | 0.163 |
| mlp | `cfdac_imag` | 0.344 |
| mlp | `cfdac_mag` | 0.481 |
| mlp | `cfdac_magphase` | 0.325 |
| mlp | `cfdac_phase` | 0.075 |
| mlp | `cfdac_real` | 0.075 |
| mlp | `cfdac_realimag` | 0.250 |
| mlp | `modal` | 0.094 |
| rf | `cfdac_imag` | 0.188 |
| rf | `cfdac_mag` | 0.250 |
| rf | `cfdac_magphase` | 0.250 |
| rf | `cfdac_phase` | 0.194 |
| rf | `cfdac_real` | 0.250 |
| rf | `cfdac_realimag` | 0.250 |
| rf | `modal` | 0.250 |
| transformer | `frf_mag` | 0.250 |
| transformer | `timeseries` | 0.269 |
| xgb | `modal` | 0.000 |

---

# 10. Transfer-learning lift per SNR

Best `(model, feature, unfreeze)` cell at k=50 % per task, per SNR, with Δ vs the clean-data zero-shot.

| task | 20 dB |
|---|---|
| `binary` | +0.944 (cnn2d/cfdac_phase) | |
| `type` | +0.568 (mlp/cfdac_imag) | |
| `severity` | +0.191 (cnn/cfdac_imag) | |
| `col_location` | +0.362 (cnn/cfdac_phase) | |
| `mass_location` | +0.700 (cnn2d/cfdac_magphase) | |

---

# 11. Feature resolution sweep per SNR

Best cell per task at r=0.500 / r=1.000 (full) per SNR.

| task | r1@20dB | r0.5@20dB |
|---|---|---|
| `binary` | — | — | |
| `type` | — | — | |
| `severity` | — | — | |
| `col_location` | — | — | |
| `mass_location` | — | — | |

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
