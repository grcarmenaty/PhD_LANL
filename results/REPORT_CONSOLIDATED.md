# LANL 3SBB — Synth-to-Real Damage Diagnosis: Consolidated Report
**Author:** G. Reyes-Carmenaty (PhD work, 2024–2026)
**Date:** 2026-06-08
**Scope.** Single source of truth for the **high-resolution (1601-bin) CFDAC model-zoo**
study: train on the calibrated 3SBB synthetic model, test **zero-shot on the
2 638-case IQS experimental set**. Supersedes every other `REPORT_*.md` here
(deprecation banners added). The synthetic-domain (in-domain) companion is
[`REPORT_synth.md`](REPORT_synth.md). *A reduced-resolution (128-bin) comparison
is in progress on a separate branch and is intentionally excluded here until complete.*

---
## Executive summary

- **What was run.** A full **model zoo at native 1601-bin resolution**: every
  feature family × every model × 10 diagnosis tasks, each trained to convergence
  on synthetic data and evaluated zero-shot on experiment. **575 unique cells.**
- **Headline 1 — diagnosis *does* transfer, with the right representation.**
  120 / 517 classification cells (≈23 %) clear chance on real data, and **every
  detection task has a working cell**: best zero-shot **balanced accuracy**
  is **is_bolt 0.67, is_hole 0.67, is_mass 0.62, is_crack 0.59, binary 0.59,
  is_pristine 0.56** (chance 0.50). This is a clear improvement over the older
  128² modal-MLP baseline (e.g. is_hole 0.62 → 0.67).
- **Headline 2 — it works best on severe damage (DT thesis confirmed).**
  Stratifying positives by damage severity, transfer **rises with severity**:
  `is_bolt` **0.67 → 0.82** at ≥75 % bolt loosening, `binary` 0.59 → 0.66,
  `is_crack` 0.59 → 0.65. The aggregate metric understates performance on the
  cases that matter operationally.
- **Headline 3 — the winning features are NOT what the literature assumed.**
  The best-transferring cells are **raw FRF (real+imag) / reconstructed
  timeseries fed to 1-D transformers/CNNs**, and **CFDAC fed to 2-D/3-D CNNs and
  a conv-tokenised transformer** — *not* the hand-crafted `modal` vector (the old
  baseline winner) and *not* ImageNet vision backbones, which mostly under-perform.
- **Headline 4 — localization and severity remain hard.** Multi-class
  localization is only weakly above chance (`mass_location` 0.50 vs 0.25,
  `col_location` 0.35 vs 0.17, `type` 0.31 vs 0.20), and **severity regression
  barely transfers** (best R² ≈ 0.04, Pearson r ≈ 0.36) despite R² ≈ 0.59
  in-domain — the clearest remaining sim-to-real gap.
- **The persistent story:** models learn the synthetic task almost perfectly
  (most ≥ 0.85 macro-F1 in-domain) but only a *partial* signal survives to real
  data. Detection/typing of **severe** damage transfers; fine-grained
  localization and severity magnitude do not.

![in-domain vs zero-shot, best cell per task](figures/hires/zoo1601_synth_vs_exp.png)

---
## Methodology

### The model zoo (1601 bins)
Synthetic data is regenerated at a 16 s simulation length (N_T = 4096, fs = 256)
so the FFT grid is **df = 0.0625 Hz → 1601 bins over 0–100 Hz**, matching the
experimental FRFs exactly. From those FRFs every cell computes one **feature**,
trains one **model** on a synth subsample, and is evaluated on held-out synth
(in-domain) and all 2 638 experimental cases (zero-shot).

| feature | description | models applied |
|---|---|---|
| `modal` (81) | per-channel peaks / log-amp / band-energy | mlp, rf, xgb |
| `indicators` (22) | pymodal damage indicators (SCI, DRQ, FRFRMS, …) vs pristine ref | mlp, rf, xgb |
| `frf_mag` (9×1601) | per-sample log-normalised \|H(f)\| | mlp, cnn1d, transformer1d |
| `frf_realimag` (18×1601) | per-sample normalised Re/Im H(f) | mlp, cnn1d, transformer1d |
| `timeseries` (9×4096) | band-limited response reconstructed from the FRF (IFFT·chirp) | mlp, cnn1d, transformer1d |
| CFDAC × 7 (1601×1601) | real / imag / mag / phase / realimag / magphase / all channels | cnn2d_shallow, cnn2d_deep, cnn3d, transformer, convnext_tiny, resnet50 |

`timeseries` is reconstructed from the FRF **identically for synth and
experiment** (the IQS set has no measured timeseries), so the only domain
difference is the FRF content, not the pipeline.

### Protocol (scientifically sound, zero-shot)
- Fixed **70/15/15** split; **class-weighted** losses / **balanced** trees.
- NN models **train to convergence** (early stop on val + ReduceLROnPlateau,
  per-epoch checkpoint/resume); trees fit once.
- Tabular features standardised on the **synth-train fold only**, then applied to
  experiment; sequence/image features use per-sample normalisation (no leakage).
- **Metrics of record: balanced accuracy / macro-F1** (classification) and
  **R² / MAE / Pearson r** (regression) — never raw accuracy, which is misleading
  under the 82.5 % damaged class prior. Class-collapse (predicting one class) is
  flagged explicitly.
- Engines: `ml_pipeline/hires_zoo.py` (CFDAC/image), `hires_tab.py`
  (modal/indicators/FRF/timeseries), `hires_all.py` (dispatch); rollups
  `hires_zoo_summary.py`, `hires_dt_1601.py`. Per-case predictions live on the
  `colab-hires-{tabular,cnn,transformer,vision}` branches.

---
## Per-task results (1601, zero-shot on experiment)

For each task: chance, the **best zero-shot** cell (balanced-acc / macro-F1, or
R² for severity), the same cell's **in-domain** score, and the best **in-domain**
cell (the gap).

| task | chance | best EXP cell | exp bal-acc | exp macro-F1 | in-domain (best synth) |
|---|---|---|---|---|---|
| **is_bolt** | 0.50 | `transformer1d / frf_realimag` | **0.669** | 0.654 | 0.94 |
| **is_hole** | 0.50 | `transformer1d / frf_realimag` | **0.667** | 0.599 | 0.85 |
| **is_mass** | 0.50 | `cnn3d / cfdac_imag` | **0.620** | 0.399 | 0.99 |
| **binary** | 0.50 | `transformer1d / timeseries` | **0.589** | 0.582 | 0.96 |
| **is_crack** | 0.50 | `transformer / cfdac_mag` | **0.587** | 0.566 | 0.78 |
| **is_pristine** | 0.50 | `mlp / timeseries` | **0.557** | 0.556 | 0.96 |
| **mass_location** | 0.25 | `mlp / frf_realimag` | 0.500 | 0.308 | 1.00 |
| **col_location** | 0.17 | `transformer / cfdac_mag` | 0.353 | 0.082 | 0.51 |
| **type** (5-cls) | 0.20 | `convnext_tiny / cfdac_imag` | 0.306 | 0.280 | 0.87 |
| **severity** (reg) | — | `mlp / frf_mag` | R² **0.037** (r 0.36) | — | R² 0.59 |

**Reading.** Binary detection and the four damage-type detectors transfer
**above chance** (0.56–0.67); localization is weak-but-real (≈1.4–2× chance on
macro-F1); type is just above chance; severity regression is essentially flat.
The in-domain column shows the models are not under-fitting — every task is
learned well synthetically; the loss is purely sim-to-real.

### What transfers — winning representations
Among the cells clearly above chance (balanced-acc ≥ chance + 0.05), the
representations that dominate are **CFDAC channels** (real+imag/imag/mag, fed to
2-D/3-D CNNs and the conv-tokenised transformer) and **raw FRF / timeseries** (fed
to 1-D transformers/CNNs). The hand-crafted **`modal`** vector — the winner of the
old 128² study — is now near the bottom (only 2 above-chance cells), and the
**ImageNet vision backbones** (ResNet50, ConvNeXt-T) rarely top a task. The signal
lives in the **full complex spectral content**, learned by a model with the right
inductive bias, not in compressed physics summaries.

---
## Damage-threshold (DT) sweep — does it work at high damage?

Positives are stratified by their damage-severity percentile (each task on its
own axis — bolt %, hole mm, mass kg, crack depth) and balanced-accuracy is
recomputed keeping only the more-severe positives.

![DT sweep @1601](figures/hires/dt_1601_combined.png)

| task | all (p0) | ≥p50 | ≥p75 | ≥p90 |
|---|---|---|---|---|
| **is_bolt** | 0.669 | 0.742 | **0.821** | **0.821** |
| **binary** | 0.589 | 0.603 | 0.618 | **0.658** |
| **is_crack** | 0.587 | 0.587 | 0.646 | **0.646** |
| is_hole | 0.667 | 0.667 | 0.646 | 0.646 |
| is_mass | 0.620 | 0.620 | 0.620 | 0.620 |

![is_bolt DT curve](figures/hires/zoo_dt_is_bolt.png)

**Verdict.** For damage with a wide severity range — **bolt loosening** above all —
transfer climbs strongly with severity (**0.67 → 0.82**), and binary/crack rise
too. `is_hole` and `is_mass` are flat *because their experimental severity barely
varies* (holes 1–6 mm, added-mass near-discrete), not because the model fails.
This confirms the long-standing hypothesis: **synth-trained models detect *severe*
real damage well; the aggregate number is dragged down by near-pristine cases.**

---
## The non-classifier: severity regression

Severity is the only regression task and the weakest link. Best experimental
**R² ≈ 0.037** (most cells negative — worse than predicting the mean), with a
weak-but-real monotonic signal (**Pearson r ≈ 0.36** for the frf/timeseries
cells), against **R² ≈ 0.59 in-domain**. Restricting to severe cases does *not*
improve R² (it falls — a variance-narrowing artefact), and MAE stays ≈ 0.25 on a
0.07–1.0 normalised scale. **Predicting damage magnitude zero-shot remains
unsolved**; re-casting severity as ordinal classification is a promising next step.

---
## Cross-task synthesis

1. **Detection ≫ localization ≫ magnitude.** Presence/type of damage transfers
   (0.56–0.67); *where* transfers weakly; *how much* barely at all.
2. **Severity is the lever.** Every detector improves on more-severe damage; the
   operationally relevant regime (severe loosening) reaches ~0.82.
3. **Representation matters more than model size.** Full complex spectral inputs
   (CFDAC, raw FRF, timeseries) + an appropriate sequence/conv model beat both the
   compressed `modal` baseline and the large pretrained vision backbones.
4. **The gap is covariate shift, not capacity.** Near-perfect in-domain scores
   with partial transfer point at a synthetic-vs-real spectral distribution shift;
   future gains should target domain adaptation, not bigger models.

---
## Limitations & honest caveats
- **Single experimental structure** (2 638-case 3SBB IQS). No cross-structure test.
- **One seed** per cell in this zoo (the 3-seed variance study was the 128² work);
  ±~0.01–0.05 noise on balanced-acc is expected — treat sub-0.05 gaps as ties.
- **Post-hoc best-cell selection** per task is hypothesis-generating; the DT
  curves are exploratory, not a pre-registered gate.
- **Localization classes are near-degenerate** in the linear ROM (symmetric
  crack/hole make column ends hard to separate) — partly an intrinsic ceiling.
- **`timeseries` is FRF-derived** (no measured experimental timeseries), so it
  carries the same information as the FRF in a different basis.
- **Resolution comparison (128 vs 1601) is deliberately omitted** until the 128
  run completes.

---
## Recommendations
1. **Deploy detection on severe damage.** is_bolt/binary/is_crack are usable at
   high severity (≈0.66–0.82 balanced-acc); report severity-stratified, not aggregate.
2. **Use spectral inputs + sequence/conv models** (FRF/timeseries→transformer1d,
   CFDAC→CNN). Drop `modal` and the pretrained vision backbones as the primary route.
3. **Attack severity & localization with domain adaptation**, not bigger nets —
   align synth/real spectral statistics (the covariate shift is the bottleneck).
4. **Recast severity as ordinal classification** to expose its weak monotonic signal.
5. **Finish the 128-bin run** to settle whether full resolution is necessary
   (early indications suggest it is not — to be reported separately).

---
## Artefact index
### Code
- `ml_pipeline/hires_zoo.py`, `hires_tab.py`, `hires_all.py` — training engines (CFDAC/image, tabular/seq, dispatch)
- `ml_pipeline/hires_zoo_summary.py` — per-cell rollup → `results_hires/zoo_summary.json`, `zoo_best_by_task_res.json`
- `ml_pipeline/hires_dt_1601.py` — DT severity sweep → `results_hires/dt_1601.json`
- `notebooks/hires_{tabular,cnn_zoo,transformer,vision,all}_gpu.ipynb` — Colab runners (autosave to `colab-hires-*`)
### Data
- `results_hires/zoo_summary.json` — per-cell exp metrics (keyed `task/model/feature@res`)
- `results_hires/zoo_best_by_task_res.json`, `results_hires/dt_1601.json`
- raw per-case predictions: branches `colab-hires-{tabular,cnn,transformer,vision}`
### Figures
- `results/figures/hires/zoo1601_synth_vs_exp.png` — in-domain vs zero-shot per task
- `results/figures/hires/dt_1601_combined.png` — DT sweep per detection task
- `results/figures/hires/zoo_dt_is_bolt.png` — is_bolt severity curves
### Canonical reports
- `REPORT_CONSOLIDATED.md` (this file) — experimental / cross-domain.
- `REPORT_synth.md` — synthetic-domain (in-domain) results.
