| model | feature | best indicator (exp R²) | severity exp R² | type exp acc | col_loc exp acc | mass_loc exp acc | binary exp acc |
|---|---|---|---|---|---|---|---|
| cnn | `frf_mag` | — | -5.11 | 0.39 | 0.29 | 0.25 | 0.94 |
| cnn | `timeseries` | — | -34.77 | 0.20 | 0.19 | 0.25 | 0.29 |
| cnn2d | `cfdac` | — | -0.43 | 0.30 | 0.25 | 0.25 | 0.93 |
| cnn2d | `cfdac_all` | — | -0.39 | — | 0.23 | 0.34 | 0.94 |
| cnn2d | `cfdac_imag` | — | -0.62 | 0.32 | 0.17 | 0.25 | 0.93 |
| cnn2d | `cfdac_mag` | — | -0.88 | 0.40 | 0.26 | 0.25 | 0.93 |
| cnn2d | `cfdac_magphase` | — | -0.05 | 0.22 | 0.11 | 0.25 | 0.94 |
| cnn2d | `cfdac_phase` | — | -0.02 | 0.24 | 0.21 | 0.25 | 0.94 |
| cnn2d | `cfdac_real` | — | -0.19 | 0.36 | 0.18 | 0.16 | 0.90 |
| cnn3d | `cfdac3d_all` | — | -0.26 | 0.26 | 0.21 | 0.25 | 0.94 |
| cnn3d | `cfdac3d_magphase` | — | -0.17 | 0.24 | 0.25 | 0.25 | 0.94 |
| cnn3d | `cfdac3d_realimag` | — | -0.55 | 0.34 | 0.23 | 0.25 | 0.88 |
| mlp | `modal` | `M2L_min` (+0.21) | -52.59 | 0.25 | 0.29 | 0.25 | 0.94 |
| rf | `modal` | `M2L_abs_sum` (+0.78) | -0.04 | 0.27 | 0.08 | 0.25 | 0.94 |
| transformer | `frf_mag` | — | -0.13 | 0.35 | 0.11 | 0.25 | 0.94 |
| transformer | `timeseries` | — | -0.25 | 0.25 | 0.21 | 0.07 | 0.84 |
| xgb | `modal` | `M2L_abs_sum` (+0.78) | +0.02 | 0.22 | 0.10 | 0.25 | 0.94 |
