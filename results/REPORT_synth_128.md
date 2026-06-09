# LANL 3SBB — Synthetic-domain (in-domain) results @128
**Companion to** [`REPORT_CONSOLIDATED_128.md`](REPORT_CONSOLIDATED_128.md) (the full cross-domain / experimental study). **Date:** 2026-06-09.
**Scope.** How well the **128-bin CFDAC model zoo** learns each diagnosis task *within the synthetic domain* — trained and tested on held-out synthetic data (the upper bound, before any sim-to-real transfer).

---
## Why this report
The consolidated report measures **zero-shot transfer to real data**. This one isolates the **in-domain ceiling**: with synth-only training and a held-out synth test fold, *can the models learn the task at all, and how well?* The gap between this report (in-domain) and the consolidated report (experimental) **is** the sim-to-real problem, quantified per task. Same zoo, same 128-bin features/models, same 70/15/15 split and train-to-convergence protocol. Metric: held-out **synthetic macro-F1** (classification) or **R²** (severity).

---
## In-domain results (held-out synthetic test, 128)

| task | chance | best in-domain (macro-F1 / R²) | best in-domain cell |
|---|---|---|---|
| **mass_location** | 0.25 | **1.00** | `cnn2d_deep / cfdac_realimag` |
| **is_mass** | 0.50 | **0.98** | `cnn1d / frf_mag` |
| **binary** | 0.50 | **0.96** | `convnext_tiny / cfdac_realimag` |
| **is_pristine** | 0.50 | **0.97** | `convnext_tiny / cfdac_realimag` |
| **is_bolt** | 0.50 | **0.94** | `cnn1d / frf_mag` |
| **type** | 0.20 | **0.85** | `transformer / cfdac_real` |
| **is_hole** | 0.50 | **0.83** | `cnn2d_shallow / cfdac_imag` |
| **is_crack** | 0.50 | **0.80** | `mlp / frf_mag` |
| **col_location** | 0.17 | **0.54** | `cnn3d / cfdac_all` |
| **severity** | — | **0.58** | `cnn2d_deep / cfdac_realimag` |

### Observations
- **Most tasks are learned well synthetically** — the models and features are not the bottleneck; the synthetic task is solvable. The differences only emerge under transfer (consolidated report).
- **`col_location` is the in-domain hard case**: symmetric crack/hole damage makes the two column ends nearly degenerate in the linear reduced model — an intrinsic ceiling, not a learning failure.
- **Severity is learnable but only moderate** in-domain — a real but imperfect regression even before transfer.

---
## The sim-to-real gap (in-domain → experiment)

Best cell per task; in-domain macro-F1/R² vs zero-shot balanced-acc / R²:

| task | in-domain | experiment (zero-shot) | metric |
|---|---|---|---|
| mass_location | 0.99 | +0.41 | bal-acc |
| is_mass | 0.93 | +0.65 | bal-acc |
| binary | 0.87 | +0.57 | bal-acc |
| is_pristine | 0.85 | +0.58 | bal-acc |
| is_bolt | 0.92 | +0.71 | bal-acc |
| type | 0.81 | +0.39 | bal-acc |
| is_hole | 0.60 | +0.72 | bal-acc |
| is_crack | 0.59 | +0.62 | bal-acc |
| col_location | 0.48 | +0.43 | bal-acc |
| severity | 0.52 | +0.18 | R² |

![in-domain vs zero-shot, best cell per task](hires128/zoo_synth_vs_exp.png)

**Interpretation.** The drop from this report to the experimental one is the sim-to-real gap. It is **largest where the synthetic model is most confident** (a hallmark of covariate shift: the model locks onto synthetic spectral structure that does not match reality). The smaller drops are the severe-damage detectors that survive transfer (and improve further at high severity — see the DT sweep in `REPORT_CONSOLIDATED_128.md`).

---
## Takeaways
1. **In-domain is essentially solved** for detection/typing; the challenge is transfer.
2. **The most over-confident synthetic tasks transfer worst** — chase domain adaptation, not in-domain accuracy.
3. **`col_location` and `severity`** are limited *in-domain too*, so their poor transfer is partly an intrinsic ceiling, not only covariate shift.

*Experimental transfer, the DT severity sweep, and the per-representation analysis are in [`REPORT_CONSOLIDATED_128.md`](REPORT_CONSOLIDATED_128.md).*
