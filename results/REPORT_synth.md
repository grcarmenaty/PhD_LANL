# LANL 3SBB — Synthetic-domain training results (pre-transfer)
**Companion to** [`REPORT_CONSOLIDATED.md`](REPORT_CONSOLIDATED.md). Date: 2026-06-02.

This report shows **in-distribution** performance: every model is trained on synthetic data and scored on a **held-out synthetic test fold** — *before* any contact with the real experimental set. These are the numbers to compare against the cross-domain (experimental) numbers in the consolidated report; the difference between them is the sim-to-real gap.

- Classification metric: accuracy on the synthetic test fold (70/15/15 stratified split).
- Regression (`severity`): R² on the synthetic test fold.
- Bespoke numbers are HPO-tuned on the **v1** baseline physics (the 5 original tasks). Vision numbers span **v1/v2/v2a × seeds 42/101/202** and all tasks attempted, and grow as the sweep completes.

---
## 1. Bespoke models — synthetic val/test (v1, HPO-tuned)
Source: `results/training_metrics.json` (55 cells).

### `binary` — chance ≈ 0.5 (balanced) / 0.825 acc by majority-class

| model | feature | synth val (accuracy) | synth test (accuracy) | runtime (s) |
|---|---|---|---|---|
| mlp | modal | 0.995 | **0.989** | 2 |
| xgb | modal | 0.975 | **0.965** | 1 |
| rf | modal | 0.958 | **0.949** | 4 |
| cnn2d | cfdac | 0.961 | **0.944** | 15 |
| xgb | indicators | 0.919 | **0.926** | 1 |
| rf | indicators | 0.924 | **0.916** | 1 |
| transformer | timeseries | 0.890 | **0.876** | 33 |
| cnn | frf_mag | 0.839 | **0.853** | 5 |
| cnn | timeseries | 0.845 | **0.842** | 30 |
| mlp | indicators | 0.826 | **0.821** | 2 |
| transformer | frf_mag | 0.800 | **0.800** | 3 |

### `col_location` — chance ≈ 0.11 (9-class)

| model | feature | synth val (accuracy) | synth test (accuracy) | runtime (s) |
|---|---|---|---|---|
| cnn2d | cfdac | 0.492 | **0.494** | 9 |
| mlp | modal | 0.507 | **0.494** | 1 |
| rf | modal | 0.509 | **0.492** | 2 |
| xgb | modal | 0.509 | **0.488** | 2 |
| rf | indicators | 0.482 | **0.481** | 1 |
| cnn | timeseries | 0.488 | **0.473** | 7 |
| cnn | frf_mag | 0.489 | **0.469** | 6 |
| xgb | indicators | 0.479 | **0.454** | 3 |
| mlp | indicators | 0.430 | **0.417** | 1 |
| transformer | timeseries | 0.387 | **0.368** | 18 |
| transformer | frf_mag | 0.267 | **0.251** | 4 |

### `mass_location` — chance ≈ 0.33 (3-class)

| model | feature | synth val (accuracy) | synth test (accuracy) | runtime (s) |
|---|---|---|---|---|
| rf | modal | 1.000 | **0.990** | 0 |
| mlp | modal | 1.000 | **0.987** | 0 |
| xgb | modal | 1.000 | **0.987** | 0 |
| xgb | indicators | 0.990 | **0.973** | 1 |
| rf | indicators | 0.980 | **0.967** | 0 |
| mlp | indicators | 0.977 | **0.963** | 1 |
| cnn2d | cfdac | 0.977 | **0.953** | 2 |
| transformer | timeseries | 0.683 | **0.637** | 6 |
| transformer | frf_mag | 0.477 | **0.480** | 1 |
| cnn | timeseries | 0.477 | **0.473** | 6 |
| cnn | frf_mag | 0.427 | **0.413** | 1 |

### `severity` — chance ≈ R²=0 = predict-mean

| model | feature | synth val (R2) | synth test (R2) | runtime (s) |
|---|---|---|---|---|
| rf | modal | 0.593 | **0.573** | 18 |
| mlp | modal | 0.551 | **0.542** | 1 |
| xgb | modal | 0.551 | **0.532** | 3 |
| rf | indicators | 0.498 | **0.487** | 7 |
| xgb | indicators | 0.467 | **0.468** | 0 |
| cnn2d | cfdac | 0.398 | **0.420** | 7 |
| mlp | indicators | 0.376 | **0.344** | 1 |
| cnn | timeseries | 0.258 | **0.227** | 25 |
| cnn | frf_mag | 0.253 | **0.213** | 4 |
| transformer | timeseries | 0.202 | **0.168** | 16 |
| transformer | frf_mag | 0.028 | **0.013** | 3 |

### `type` — chance ≈ 0.20 (5-class)

| model | feature | synth val (accuracy) | synth test (accuracy) | runtime (s) |
|---|---|---|---|---|
| mlp | modal | 0.869 | **0.877** | 2 |
| xgb | modal | 0.807 | **0.822** | 9 |
| rf | modal | 0.815 | **0.811** | 1 |
| cnn2d | cfdac | 0.796 | **0.803** | 14 |
| xgb | indicators | 0.774 | **0.759** | 4 |
| rf | indicators | 0.757 | **0.745** | 2 |
| mlp | indicators | 0.703 | **0.701** | 2 |
| cnn | frf_mag | 0.677 | **0.689** | 11 |
| cnn | timeseries | 0.654 | **0.657** | 35 |
| transformer | timeseries | 0.557 | **0.576** | 30 |
| transformer | frf_mag | 0.476 | **0.501** | 10 |


---
## 2. Vision backbones — synthetic test (running sweep)
Source: `meta.synth_test` of every per-case JSON produced so far — **3 / 810** vision cells complete.

**Coverage (cells done per variant×seed, of 90):**

| variant | seed42 | seed101 | seed202 |
|---|---|---|---|
| v1 | 3 | 0 | 0 |
| v2 | 0 | 0 | 0 |
| v2a | 0 | 0 | 0 |

### `is_bolt` — chance ≈ 0.5

| backbone | feature | v1 | v2 | v2a |
|---|---|---|---|---|
| convnext_tiny | cfdac_all | 0.827 (n=1) | — | — |

### `is_hole` — chance ≈ 0.5

| backbone | feature | v1 | v2 | v2a |
|---|---|---|---|---|
| convnext_tiny | cfdac_all | 0.427 (n=1) | — | — |

### `type` — chance ≈ 0.20 (5-class)

| backbone | feature | v1 | v2 | v2a |
|---|---|---|---|---|
| convnext_tiny | cfdac_all | 0.480 (n=1) | — | — |


---
## 3. How to read this against the consolidated (experimental) report
- **High synth test + low experimental score = sim-to-real gap**, not a failure to learn. Most cells here fit the synthetic task well; the consolidated report shows how little of that survives zero-shot transfer to the real structure.
- The synthetic test fold and the synthetic train fold come from the *same* generator, so these numbers are optimistic by construction — they are the ceiling, not the deployment estimate.
- For the cross-domain story, per-cell and per-variant, see [`REPORT_CONSOLIDATED.md`](REPORT_CONSOLIDATED.md).
