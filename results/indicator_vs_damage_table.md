| model | feature | best indicator (exp R²) | severity exp R² | type exp acc | col_loc exp acc | mass_loc exp acc | binary exp acc |
|---|---|---|---|---|---|---|---|
| cnn | `frf_mag` | — | -11885453312.00 | 0.33 | 0.24 | 0.26 | 0.82 |
| cnn | `timeseries` | — | -5450889216.00 | 0.29 | 0.30 | 0.26 | 0.42 |
| cnn2d | `cfdac` | — | -0.21 | 0.41 | 0.17 | 0.17 | 0.82 |
| cnn2d | `cfdac_all` | — | -0.38 | — | 0.16 | 0.28 | 0.82 |
| cnn2d | `cfdac_imag` | — | -0.58 | 0.33 | 0.10 | 0.17 | 0.82 |
| cnn2d | `cfdac_mag` | — | -0.91 | 0.47 | 0.42 | 0.26 | 0.82 |
| cnn2d | `cfdac_magphase` | — | -0.02 | 0.27 | 0.12 | 0.17 | 0.82 |
| cnn2d | `cfdac_phase` | — | -0.03 | 0.18 | 0.23 | 0.17 | 0.82 |
| cnn2d | `cfdac_real` | — | -0.10 | 0.42 | 0.21 | 0.22 | 0.80 |
| cnn3d | `cfdac3d_all` | — | -0.26 | 0.23 | 0.13 | 0.26 | 0.82 |
| cnn3d | `cfdac3d_magphase` | — | -0.13 | 0.13 | 0.17 | 0.17 | 0.82 |
| cnn3d | `cfdac3d_realimag` | — | -0.65 | 0.34 | 0.15 | 0.17 | 0.78 |
| mlp | `indicators` | — | -48110543046268944384.00 | 0.09 | 0.37 | 0.26 | 0.82 |
| mlp | `modal` | `r2_imag` (+0.61) | -24188615174025006546944.00 | 0.38 | 0.45 | 0.26 | 0.82 |
| rf | `indicators` | — | -0.51 | 0.18 | 0.05 | 0.27 | 0.82 |
| rf | `modal` | `M2L_abs_sum` (+0.76) | -0.17 | 0.42 | 0.10 | 0.17 | 0.82 |
| transformer | `frf_mag` | — | -0.02 | 0.35 | 0.04 | 0.17 | 0.82 |
| transformer | `timeseries` | — | -0.08 | 0.29 | 0.20 | 0.06 | 0.75 |
| xgb | `indicators` | — | -0.28 | 0.16 | 0.16 | 0.09 | 0.82 |
| xgb | `modal` | `M2L_abs_sum` (+0.74) | -0.04 | 0.33 | 0.06 | 0.26 | 0.82 |
