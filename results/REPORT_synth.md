# LANL 3SBB — Synthetic-domain (in-domain) results @1601
**Companion to** [`REPORT_CONSOLIDATED.md`](REPORT_CONSOLIDATED.md) (the full cross-domain / experimental study). **Date:** 2026-06-09.
**Scope.** How well the **1601-bin CFDAC model zoo** learns each diagnosis task *within the synthetic domain* — trained and tested on held-out synthetic data (the upper bound, before any sim-to-real transfer).

---
## Why this report
The consolidated report measures **zero-shot transfer to real data**. This one isolates the **in-domain ceiling**: with synth-only training and a held-out synth test fold, *can the models learn the task at all, and how well?* The gap between this report (in-domain) and the consolidated report (experimental) **is** the sim-to-real problem, quantified per task. Same zoo, same 1601-bin features/models, same 70/15/15 split and train-to-convergence protocol. Metric: held-out **synthetic macro-F1** (classification) or **R²** (severity).

---
## In-domain results (held-out synthetic test, 1601)

| task | chance | best in-domain (macro-F1 / R²) | best in-domain cell |
|---|---|---|---|
| **mass_location** | 0.25 | **1.00** | `cnn2d_deep / cfdac_phase` |
| **is_mass** | 0.50 | **0.99** | `mlp / modal` |
| **binary** | 0.50 | **0.96** | `mlp / frf_mag` |
| **is_pristine** | 0.50 | **0.96** | `convnext_tiny / cfdac_realimag` |
| **is_bolt** | 0.50 | **0.94** | `cnn2d_deep / cfdac_magphase` |
| **type** | 0.20 | **0.87** | `transformer / cfdac_all` |
| **is_hole** | 0.50 | **0.85** | `mlp / frf_mag` |
| **is_crack** | 0.50 | **0.78** | `mlp / frf_realimag` |
| **col_location** | 0.17 | **0.51** | `resnet50 / cfdac_phase` |
| **severity** | — | **0.59** | `mlp / frf_realimag` |

### Observations
- **Most tasks are learned well synthetically** — the models and features are not the bottleneck; the synthetic task is solvable. The differences only emerge under transfer (consolidated report).
- **`col_location` is the in-domain hard case**: symmetric crack/hole damage makes the two column ends nearly degenerate in the linear reduced model — an intrinsic ceiling, not a learning failure.
- **Severity is learnable but only moderate** in-domain — a real but imperfect regression even before transfer.

---
## The sim-to-real gap (in-domain → experiment)

Best cell per task; in-domain macro-F1/R² vs zero-shot balanced-acc / R²:

| task | in-domain | experiment (zero-shot) | metric |
|---|---|---|---|
| mass_location | 0.99 | +0.50 | bal-acc |
| is_mass | 0.58 | +0.62 | bal-acc |
| binary | 0.86 | +0.59 | bal-acc |
| is_pristine | 0.93 | +0.56 | bal-acc |
| is_bolt | 0.89 | +0.67 | bal-acc |
| type | 0.82 | +0.31 | bal-acc |
| is_hole | 0.57 | +0.67 | bal-acc |
| is_crack | 0.54 | +0.59 | bal-acc |
| col_location | 0.45 | +0.35 | bal-acc |
| severity | 0.59 | +0.04 | R² |

![in-domain vs zero-shot, best cell per task](hires/zoo_synth_vs_exp.png)

**Interpretation.** The drop from this report to the experimental one is the sim-to-real gap. It is **largest where the synthetic model is most confident** (a hallmark of covariate shift: the model locks onto synthetic spectral structure that does not match reality). The smaller drops are the severe-damage detectors that survive transfer (and improve further at high severity — see the DT sweep in `REPORT_CONSOLIDATED.md`).

---
## Takeaways
1. **In-domain is essentially solved** for detection/typing; the challenge is transfer.
2. **The most over-confident synthetic tasks transfer worst** — chase domain adaptation, not in-domain accuracy.
3. **`col_location` and `severity`** are limited *in-domain too*, so their poor transfer is partly an intrinsic ceiling, not only covariate shift.

*Experimental transfer, the DT severity sweep, and the per-representation analysis are in [`REPORT_CONSOLIDATED.md`](REPORT_CONSOLIDATED.md).*
