# Sim-to-real damage diagnosis on the LANL 3SBB — definitive report

How well damage diagnosis trained **only on synthetic data** transfers,
**zero-shot**, to **real** measurements of the LANL 3-Storey Bookcase
Benchmark (3SBB) — reported **goal by goal**, with the exact training
conditions for each.

The pipeline trains ML models on **10 000 synthetic** finite-element
samples and evaluates them on **2 638 real** IQS experimental cases. This
is the methodology-corrected edition: while implementing the previous
draft's recommendations, two correctness problems were found in the
evaluation (an unseeded training loop, an accuracy-only metric on an
imbalanced test set) — both are fixed and every number here is recomputed.
The exhaustive catalogue is in [`REPORT_full.md`](REPORT_full.md); the
chronological ablation table is in [`ablation_log.json`](ablation_log.json).

## The four goals — at a glance

| # | goal | task(s) | best synth-only result (seeded, zero-shot) | transfers? |
|---|---|---|---|---|
| 1 | **Damage detection** | `binary` (damage vs pristine) | balanced acc 0.585 (`cnn2d/cfdac_real`) | **weak** — no usable operating point |
| 2 | **Damage type** | `type` (5-class) | macro-F1 0.25, balanced acc 0.33 | **weakly** |
| 2 | | `type` one-vs-rest (modal-MLP) | 3-seed mean balanced acc 0.62–0.66 ± ≤ 0.03 (`is_hole / mlp / modal` 0.661 ± 0.004, § 5.6) | **yes — robust, strongest in study** |
| 3 | **Damage severity** | `severity` (regression) | R² 0.132 ± 0.013 (`mlp / cfdac_realimag`, 3-seed) | **weakly** |
| 4 | **Damage location** | `col_location` (column-end) | macro-F1 0.17, balanced acc 0.27 | **no** |
| 4 | | `mass_location` (mass-plate) | macro-F1 0.45, balanced acc 0.51 | **partly** |

> **One-paragraph summary.** A **3-seed multi-seed validation**
> (`multiseed_summary.json`) reshaped the iteration-3 headline twice. The
> single-seed CFDAC-CNN "best" cells *were* flukes (`is_bolt cnn2d/cfdac_real`
> BA 0.71 collapsed to 0.49 / 0.51 on the other seeds), **but the
> **modal-MLP one-vs-rest bank is robust**: `is_hole / mlp / modal`
> BA 0.661 ± 0.004, `is_bolt / mlp / cfdac_real` 0.631 ± 0.015,
> `is_mass / mlp / modal` 0.628 ± 0.011, `is_crack / mlp / modal`
> 0.615 ± 0.026. **One-vs-rest type detection (via modal-MLP) is the
> strongest robust transfer in the study.** **Mass-plate location**
> (`mlp / cfdac_imag`, macro-F1 0.42 ± 0.07, balanced acc 0.48 ± 0.08,
> ≈ 2× chance) is comparable in normalised lift. 5-class type transfers
> weakly (0.27 ± 0.02); severity weakly (R² 0.132 ± 0.013, `mlp / cfdac_realimag`);
> detection marginal (BA 0.52 ± 0.05); column-end location does not
> transfer. Measured macro-F1 noise band: median sd 0.011, p90 sd 0.07
> (244 cells × 3 seeds) — torch-cell only: median 0.016, p90 0.086.

---

## 1. The problem — why accuracy is the wrong metric

The 2 638-case IQS experimental set is sharply imbalanced:

| class | Bolt | Pristine | Crack | Hole | Mass |
|---|---|---|---|---|---|
| count | 1 338 | 462 | 320 | 280 | 238 |
| share | 50.7 % | 17.5 % | 12.1 % | 10.6 % | 9.0 % |

A classifier that always predicts Bolt therefore scores **0.507 accuracy on
`type`** and **0.825 on `binary`** with *zero* discriminative skill. Raw
accuracy is gamed by this imbalance. The honest metrics, used throughout
this report, are:

* **macro-F1** — unweighted mean of per-class F1; not inflated by a large
  majority class.
* **balanced accuracy** — mean per-class recall; **chance = 1 / n_classes**
  (binary 0.500, type 0.200, col_location 0.167, mass_location 0.250).

A model whose balanced accuracy equals chance has learned nothing
transferable, whatever its raw accuracy.

---

## 2. Methodology corrections

Both fixes were made *before* re-deriving any result (commit `a40ed6d`).

* **2.1 Deterministic training.** `hpo.py` seeded the data split and the
  sklearn models but **never seeded PyTorch** — every cnn/transformer/cnn2d
  cell was a single unreproducible draw. `_train_torch` now seeds torch,
  numpy and the `DataLoader`; `hpo_cfdac_*` likewise (commit `f3ceeaf`).
  Re-running against the report-era artefacts shows nominally-identical
  cells differ by up to ≈ 0.07 macro-F1 — a mix of seed noise and the fix.
* **2.2 Honest metrics.** `evaluate_full_experimental` recorded only
  accuracy; it now also records **macro-F1** and **balanced accuracy** per
  cell. Re-scoring the report-era models with these
  (`experimental_full_evaluation_basescore.json`) is what exposes the
  class-prior collapse behind the old accuracy headlines.

---

## 3. Shared training setup

Every goal in §§ 4–7 uses the recipe below; the per-goal sections state
only what differs (feature, model, hyperparameters).

| element | setting |
|---|---|
| **synthetic data** | 10 000 FE samples — 2 000 each of Pristine / Bolt / Crack / Hole / Mass; 9 accelerometer channels; FRF band 5–100 Hz |
| **features** | `modal` (81-dim descriptor), `frf_mag` (381×9), `cfdac_*` (128×128 Complex Frequency-Domain Assurance Criterion matrices and projections) |
| **split** | stratified 70 / 15 / 15 → 7 000 train / 1 500 val / 1 500 test (`make_split`, sklearn `train_test_split`, `random_state` 20260511). Severity uses the 8 000 damage-only samples → 5 600 / 1 200 / 1 200 |
| **HPO** | exhaustive grid search per (task, model, feature) cell; best configuration chosen by the **validation** metric |
| **torch training** | AdamW (weight decay 10⁻⁴), CosineAnnealingLR, **4 epochs**, batch 64, loss = cross-entropy (classification) / MSE (regression), MLP dropout 0.2 |
| **seed** | 20260511 — torch + numpy + `DataLoader` + split + sklearn `random_state` |
| **preprocessing** | P0.1 — CFDAC computed against the experimental-Pristine reference (mean of the 462 IQS Pristine cases); P0.3 — `StandardScaler` for modal MLP/sklearn cells fit on the experimental-Pristine subset; P1.1 — per-sample input normalisation applied identically to synth training and experimental inference |
| **evaluation** | the synth-trained model is run **zero-shot** on all 2 638 experimental cases; metrics: macro-F1, balanced accuracy, accuracy (classification) / R², MAE (regression) |

Evidence artefacts: `experimental_full_evaluation_plain.json` (seeded,
60 cells), `_aug.json` (seeded augmented arm, 50 cells),
`_basescore.json` (report-era models re-scored, 78 cells).

### 3.1 The cell grid — which model pairs with which feature

A **cell** is one (task × model × feature) combination — e.g.
`type / mlp / modal`. The model family is fixed by the feature's tensor
rank: a flat vector goes to a tree ensemble or an MLP; a frequency×channel
*sequence* to a 1-D CNN or Transformer; a 128×128 CFDAC *matrix* to a 2-D
CNN; a stacked 3-D CFDAC tensor to a 3-D CNN.

| feature | tensor shape | model(s) paired with it |
|---|---|---|
| `modal` | 81-vector (flat) | `rf`, `xgb`, `mlp` |
| `frf_mag` | 381×9 sequence | `cnn` (1-D), `transformer` |
| `timeseries` (legacy) | 1024×9 sequence | `cnn` (1-D), `transformer` |
| `cfdac` (legacy, real+imag) | 128×128×2 matrix | `cnn2d` |
| `cfdac_real` | 128×128 matrix | `cnn2d` (also `mlp`, flattened) |
| `cfdac_imag` / `cfdac_mag` / `cfdac_phase` | 128×128 matrix | `cnn2d` |
| `cfdac_realimag` / `cfdac_magphase` | 128×128×2 matrix | `cnn2d` |
| `cfdac_all` | 128×128×4 matrix | `cnn2d` |
| `cfdac3d_realimag` / `cfdac3d_magphase` / `cfdac3d_all` | 3-D stack | `cnn3d` |

Models: `rf`, `xgb` — sklearn tree ensembles; `mlp` — multilayer
perceptron; `cnn` — 1-D conv stack; `transformer` — small encoder; `cnn2d`
/ `cnn3d` — 2-/3-D conv stacks. Across the five tasks this yields the
78-cell report-era sweep (`_basescore.json`) and the 60-cell seeded sweep
(`_plain.json`, the `modal` / `frf_mag` / `cfdac`-legacy subset that
`hpo.py` covers; the CFDAC variants come from `hpo_cfdac_*.py`).

### 3.2 Deployment data assumption — what experimental data is available

"Synth-only" here means a specific, realistic constraint: **before
deployment the only experimental data available is a reference measurement
of the *healthy* structure.** The pristine 3SBB can be measured freely, but
there are *no* measurements of it in any damaged state — diagnosing unseen
damage is the whole task. Methods are classified by what they need:

| method | experimental data needed | within the assumption? |
|---|---|---|
| P0.1 CFDAC reference, P0.3 scaler, P1.1, the whole synth-only pipeline (§§ 4–8) | pristine reference only (462 healthy measurements, averaged to one reference) | **yes** |
| Recommendation 4 (§ 10) — SSL pretrain on "unlabelled experimental data" | unlabelled measurements of the **damaged** structure | **no** |
| Joint synth+exp fine-tune (`REPORT_full.md` § 9.2) | **labelled** measurements of the damaged structure | no — post-deployment only |

The SSL proposal pretrains on all 2 638 experimental cases — but 2 176 of
those are damaged-structure measurements a genuine pre-deployment scenario
does not have. SSL pretrain is therefore **not a synth-only method** under
this assumption; § 10 is corrected accordingly. P0.1 / P0.3 use only the
462 pristine measurements and stay within it.

---

## 4. Goal 1 — Damage detection

**Question:** is the structure damaged at all? Task `binary` — two classes,
Pristine vs any damage. The experimental set is 82.5 % damage.

### 4.1 Recommended training configuration

| element | value |
|---|---|
| training data | 10 000 synthetic samples (7 000 / 1 500 / 1 500) |
| feature | `cfdac_magphase` — 128×128×2 CFDAC magnitude+phase stack |
| model | `MLP`, hidden (256, 128, 64), dropout 0.2 (flattened CFDAC input) |
| optimiser | AdamW, lr 1×10⁻³, weight decay 10⁻⁴ |
| schedule / epochs / batch | CosineAnnealing / 4 / 64 |
| loss | cross-entropy |
| preprocessing | P0.1 experimental-Pristine CFDAC reference; P1.1 per-sample normalisation |
| HPO selection | grid hidden × lr (9 configs, `hpo_cfdac_allmodels.py`); best by validation accuracy |
| seed | 20260511 |

### 4.2 Result (seeded, zero-shot on 2 638 real cases)

| synth val | synth test | exp accuracy | exp macro-F1 | exp balanced acc |
|---|---|---|---|---|
| 0.80 | 0.80 | 0.764 | 0.542 | **0.540** |

Per-class recall: Damage 0.88, Pristine 0.19 — the cell flags 19 % of
Pristine cases, where a "predict-damage" collapse would flag 0 %.

### 4.3 Comparison (seeded sweep, `_seeded.json`)

| model / feature | accuracy | macro-F1 | balanced acc | per-class recall |
|---|---|---|---|---|
| cnn2d / cfdac_real | 0.49 | 0.46 | **0.585** | Pristine 0.73 / Damage 0.44 |
| mlp / cfdac_magphase | 0.76 | **0.54** | 0.540 | Pristine 0.19 / Damage 0.88 |
| mlp / modal | 0.82 | 0.49 | 0.515 | — |
| transformer / frf_mag | 0.82 | 0.47 | 0.505 | — |

### 4.4 Verdict

**Weakly transfers — but no operationally usable cell.** The best
balanced-accuracy cell, `cnn2d/cfdac_real` (0.585), clears the 0.500 chance
level by +0.085 — outside the noise band, so detection carries *some*
genuine discriminative signal. But the metric and the operating point
disagree: `cnn2d/cfdac_real` reaches 0.585 by leaning Pristine — it catches
73 % of Pristine cases but **misses 56 % of real damage**, useless as a
safety detector; `mlp/cfdac_magphase` (best macro-F1) instead catches 88 %
of damage but flags 81 % of Pristine as damaged. Detection has weak
cross-domain signal yet no cell trades recall against false alarms well
enough to deploy.

### 4.5 Evaluation on Pristine + severe-damage only

A deployment-relevant restriction: evaluate only on cases that are either
Pristine or *clearly* damaged — Pristine ∪ {damage with per-type-normalised
severity ≥ τ} (`mlp/modal`; artefact `results/severity_inclusive_eval.json`,
regenerate via `python -m ml_pipeline.severity_inclusive_eval`):

| test set | n | accuracy | macro-F1 | balanced acc |
|---|---|---|---|---|
| all cases | 2 638 | 0.825 | 0.482 | 0.513 |
| Pristine + damage τ ≥ 0.5 | 1 520 | 0.705 | 0.444 | 0.516 |
| Pristine + damage τ ≥ 0.7 | 1 240 | 0.640 | 0.420 | 0.516 |

Balanced accuracy is flat at ≈ 0.51 at every threshold — restricting to
severe damage does **not** help detection. Raw accuracy *falls* only
because Pristine becomes a larger share of a smaller set.

---

## 5. Goal 2 — Damage type assessment

**Question:** what *kind* of damage? Task `type` — 5 classes (Pristine,
Bolt, Crack, Hole, Mass).

### 5.1 Recommended training configuration

| element | value |
|---|---|
| training data | 10 000 synthetic samples (7 000 / 1 500 / 1 500) |
| feature | `modal` — 81-dim modal descriptor |
| model | `MLP`, hidden (512, 256, 128), dropout 0.2 |
| optimiser | AdamW, lr 3×10⁻³, weight decay 10⁻⁴ |
| schedule / epochs / batch | CosineAnnealing / 4 / 64 |
| loss | cross-entropy |
| preprocessing | P0.3 — `StandardScaler` on the experimental-Pristine subset |
| HPO selection | grid hidden × lr (9 configs); best by validation accuracy |
| seed | 20260511 |

### 5.2 Result (seeded, zero-shot on 2 638 real cases)

| synth val | synth test | exp accuracy | exp macro-F1 | exp balanced acc |
|---|---|---|---|---|
| 0.867 | 0.880 | 0.349 | **0.250** | 0.331 |

(The report-era `mlp/modal` model re-scored gives macro-F1 0.296 / balanced
acc 0.371 — the ≈ 0.05 gap is the § 2.1 seed-noise band.)

### 5.3 Comparison (seeded sweep, `_seeded.json`)

| model / feature | accuracy | macro-F1 | balanced acc | note |
|---|---|---|---|---|
| **mlp / modal** | 0.35 | **0.25** | 0.33 | best seeded cell |
| mlp / cfdac_mag | 0.34 | 0.22 | 0.26 | |
| cnn2d / cfdac_real | 0.31 | 0.21 | 0.26 | deep CFDAC — weak |
| cnn / frf_mag | **0.51** | 0.14 | 0.20 | **class-prior collapse** — predicts Bolt for all 2 638 cases |

The previous draft's headline "type 0.507" is the bottom row: a degenerate
classifier whose balanced accuracy (0.200) is *exactly* 5-class chance.

### 5.4 Verdict

**Transfers weakly.** Only the modal feature with a plain MLP carries real
signal — macro-F1 ≈ 0.25–0.30, balanced accuracy ≈ 0.33–0.37 against a
0.20 chance level. The deep CFDAC models (cnn2d/cnn3d) achieve higher
*accuracy* purely by collapsing onto the majority class; their balanced
accuracy is at or near chance. Type assessment is above chance but far from
usable.

### 5.5 Evaluation on Pristine + severe-damage only

The same restriction as § 4.5 — Pristine ∪ {damage with per-type-normalised
severity ≥ τ} — a scenario where damage, if present, is significant
(artefact `results/severity_inclusive_eval.json`):

| cell | test set | n | accuracy | macro-F1 | balanced acc |
|---|---|---|---|---|---|
| mlp / modal | all cases | 2 638 | 0.373 | 0.296 | 0.371 |
| mlp / modal | Pristine + τ ≥ 0.7 | 1 240 | 0.390 | 0.219 | 0.306 |
| cnn2d / cfdac_real | all cases | 2 638 | 0.324 | 0.172 | 0.189 |
| cnn2d / cfdac_real | Pristine + τ ≥ 0.7 | 1 240 | 0.411 | 0.206 | 0.307 |

Restricting to severe damage helps the 2-D CNN / CFDAC cell modestly
(balanced accuracy 0.19 → 0.31) but *lowers* the modal-MLP cell's macro-F1
(0.30 → 0.22). A previously-reported "≈ 0.66 accuracy at high severity" was
raw accuracy on a *damage-only* subset — a class-distribution shift, not a
real skill gain. Under macro-F1 / balanced accuracy on the Pristine-
inclusive set there is at most a small genuine effect, and only for one
CFDAC cell — not the breakthrough the accuracy figure suggested.

### 5.6 One-vs-rest type detection — robust transfer via modal-MLP

The 5-class `type` task transfers weakly (§ 5.2, macro-F1 0.25). The
**one-vs-rest decomposition** — five binary "is the damage type X?"
classifiers — transfers *substantially better*, with a critical multi-seed
caveat: the **CFDAC-CNN best cells were single-seed flukes** (an earlier
draft made these the headline), while **modal-MLP and frf_mag/transformer
cells transfer robustly** across all three seeds. 3-seed best **robust**
cell per sub-task (filter: sd < 0.05), from `multiseed_summary.json`:

| one-vs-rest task | best robust cell | BA mean | BA sd | seeds (default / 101 / 202) |
|---|---|---|---|---|
| `is_hole` | **mlp / modal** | **0.661** | 0.004 | 0.665 / 0.659 / 0.658 |
| `is_bolt` | **mlp / cfdac_real** | **0.631** | 0.015 | 0.635 / 0.645 / 0.615 |
| `is_mass` | **mlp / modal** | **0.628** | 0.011 | 0.616 / 0.638 / 0.629 |
| `is_crack` | **mlp / modal** | **0.615** | 0.026 | 0.603 / 0.645 / 0.597 |
| `is_pristine` | mlp / modal | 0.517 | 0.003 | 0.513 / 0.519 / 0.518 |

**Four of five sub-tasks transfer robustly at balanced accuracy 0.62–0.66
(≈ 2× chance) with sd ≤ 0.03** — `is_pristine` is the only one stuck near
chance. Normalised lift (`(BA − 0.5) / (1 − 0.5)`) of 0.24–0.32 — comparable
to mass-plate location's lift (0.30, § 7.2). **This makes one-vs-rest type
detection the strongest *robust* transfer in the study**, tied with
mass-plate location in normalised lift but with substantially smaller
per-cell sd (≤ 0.03 vs 0.07).

The earlier "best" CFDAC-CNN cells were single-seed flukes —
`is_bolt / cnn2d / cfdac_real` reached BA 0.71 on the default seed but
0.49 / 0.51 on the other two (mean 0.57 sd 0.12); same for
`is_hole / mlp / cfdac_all` (0.68 / 0.50 / 0.50). Do not use CFDAC-CNN
cells for one-vs-rest; use modal-MLP.

**Deployment implication:** build a **bank of modal-MLP per-damage-type
detectors**. `is_hole / mlp / modal` (BA 0.661 ± 0.004) is the
single most reproducibly transferring cell in the entire 244-cell sweep.
The aggregated trenchcoat (macro-F1 ≈ 0.29) discards this signal in the
aggregation step.

---

## 6. Goal 3 — Damage severity assessment

**Question:** how severe is the damage? Task `severity` — regression,
target normalised to [0, 1] per damage type, damage samples only.

### 6.1 Recommended training configuration

| element | value |
|---|---|
| training data | 8 000 synthetic damage samples (5 600 / 1 200 / 1 200) |
| feature | `cfdac_realimag` — 128×128×2 CFDAC real+imaginary stacked matrix |
| model | `MLP`, hidden (256, 128, 64), dropout 0.2 (flattened CFDAC input) |
| optimiser | AdamW, lr 1×10⁻³, weight decay 10⁻⁴ |
| schedule / epochs / batch | CosineAnnealing / 4 / 64 |
| loss | MSE; **sigmoid-bounded output head** (P0.2) so the prediction stays in [0, 1] |
| preprocessing | P0.1 experimental-Pristine CFDAC reference; P1.1 per-sample normalisation |
| HPO selection | grid hidden × lr (9 configs, `hpo_cfdac_allmodels.py`); best by validation R² |
| seed | 20260511 |

**Multi-seed rationale (P0.7).** The report-era recommendation was
`mlp/cfdac_imag` (single-seed exp R² +0.131). Across 3 seeds (42 / 101 /
202), `mlp/cfdac_realimag` is both **higher mean** (R² 0.132 ± 0.013)
and **tighter sd** (0.013 vs 0.026) — the right cell for the recommended
configuration. The report-era choice was within run-to-run noise of the
true best.

### 6.2 Result (seeded, zero-shot on 2 176 damage cases)

| cell | synth val R² | synth test R² | **exp R²** (3-seed) | exp MAE |
|---|---|---|---|---|
| mlp / cfdac_realimag (**recommended**) | 0.312 | 0.279 | **+0.132 ± 0.013** | 0.224 |
| mlp / cfdac_imag (report-era recommendation) | 0.311 | 0.281 | +0.101 ± 0.026 | 0.224 |

### 6.3 Comparison (3-seed mean ± sd, `results/multiseed_summary.json`)

| model / feature | exp R² (3-seed mean ± sd) | note |
|---|---|---|
| mlp / cfdac_realimag | **+0.132 ± 0.013** | **best & tightest** |
| mlp / cfdac_imag | +0.101 ± 0.026 | report-era recommendation; wider sd |
| xgb / cfdac_imag | +0.040 ± 0.116 | very wide sd — unreliable across seeds |
| transformer / frf_mag | +0.046 ± 0.037 | |
| mlp / cfdac_real | +0.018 ± 0.009 | |
| cnn / timeseries | +0.180 (single seed) | **synthesised `timeseries` feature — excluded** |

`timeseries` on experimental data is reconstructed from the FRF
(`H·F → IFFT`, P0.4); its 0.180 is not a genuine-feature result.

### 6.4 Verdict

**Weakly transfers.** Multi-seed (3 seeds) confirms **one robust
severity cell** — `mlp / cfdac_realimag`, R² 0.132 ± 0.013 — and a
second weaker cell (`mlp / cfdac_imag`, R² 0.101 ± 0.026) within its
noise band. Every other CFDAC variant either regresses to negative R²
or has sd > mean (`xgb / cfdac_imag` is a good cautionary example: mean
0.04, sd 0.12, range 0.26). That is well below a usable severity
estimator but distinctly above the ≈ 0 of every earlier genuine-feature
cell: severity carries a weak but consistent cross-domain signal in the
CFDAC real+imaginary representation. The earlier "does not transfer"
verdict was an artefact of the report-era sweep never having evaluated
these cells; the report-era +0.131 single-seed value sat at the upper
end of the noise band, which is why the recommended cell needed
correcting against the 3-seed mean.

---

## 7. Goal 4 — Damage location assessment

**Question:** *where* is the damage? Two distinct sub-tasks: which
**column-end** (`col_location`, 6 classes S1AD…S3BD) and which
**mass-plate** (`mass_location`, 4 classes Base/F1/F2/F3).

### 7.1 Column-end location (`col_location`)

**Recommended training configuration**

| element | value |
|---|---|
| training data | 10 000 synthetic samples (7 000 / 1 500 / 1 500) |
| feature | `modal` — 81-dim modal descriptor |
| model | `MLP`, hidden (512, 256, 128), dropout 0.2 |
| optimiser | AdamW, lr 3×10⁻³, weight decay 10⁻⁴ |
| schedule / epochs / batch | CosineAnnealing / 4 / 64 |
| loss | cross-entropy |
| preprocessing | P0.3 — `StandardScaler` on the experimental-Pristine subset |
| HPO selection | grid hidden × lr (9 configs); best by validation accuracy |
| seed | 20260511 |

**Result** (seeded, zero-shot on 2 638 real cases)

| synth val | synth test | exp accuracy | exp macro-F1 | exp balanced acc |
|---|---|---|---|---|
| 0.515 | 0.481 | 0.284 | **0.167** | 0.274 |

**Comparison** (seeded sweep, `_seeded.json`)

| model / feature | accuracy | macro-F1 | balanced acc |
|---|---|---|---|
| mlp / modal | 0.28 | 0.17 | 0.27 |
| cnn / frf_mag | 0.36 | 0.16 | 0.21 |
| mlp / cfdac_magphase | 0.41 | 0.15 | 0.18 |
| cnn2d / cfdac_mag | 0.02 | 0.01 | 0.17 |

**Verdict — does not transfer.** Best seeded macro-F1 0.17, balanced
accuracy 0.27 against the 0.167 six-class chance level — barely above
chance. The report-era headline cell `cnn2d/cfdac_mag` (`_basescore`:
accuracy 0.508 / macro-F1 0.19) **did not survive seeding** — re-run with a
fixed seed it collapses to macro-F1 0.007 and balanced accuracy exactly
0.167 (chance). It was a non-reproducible fluke; the seeded sweep confirms
column-end location does not transfer. The synthetic crack/hole model is
symmetric per storey, so the BD-vs-AD ends are nearly indistinguishable — a
property of the synthetic physics.

### 7.2 Mass-plate location (`mass_location`)

**Recommended training configuration**

| element | value |
|---|---|
| training data | 10 000 synthetic samples (7 000 / 1 500 / 1 500) |
| feature | `cfdac_imag` — 128×128 CFDAC imaginary-part matrix |
| model | `MLP`, hidden (256, 128, 64), dropout 0.2 (flattened CFDAC input) |
| optimiser | AdamW, lr 1×10⁻³, weight decay 10⁻⁴ |
| schedule / epochs / batch | CosineAnnealing / 4 / 64 |
| loss | cross-entropy |
| preprocessing | P0.1 experimental-Pristine CFDAC reference; P1.1 per-sample normalisation |
| HPO selection | grid hidden × lr (9 configs, `hpo_cfdac_allmodels.py`); best by validation accuracy |
| seed | 20260511 |

**Result** (seeded, zero-shot on 2 638 real cases)

| synth val | synth test | exp accuracy | exp macro-F1 | exp balanced acc |
|---|---|---|---|---|
| 0.97 | 0.973 | 0.441 | **0.452** | **0.512** |

**Comparison** (seeded sweep, `_seeded.json`)

| model / feature | accuracy | macro-F1 | balanced acc |
|---|---|---|---|
| mlp / cfdac_imag | 0.44 | **0.45** | 0.51 |
| cnn2d / cfdac_real | 0.53 | 0.37 | 0.45 |
| mlp / cfdac_mag | 0.45 | 0.27 | 0.31 |
| mlp / modal | 0.39 | 0.25 | 0.27 |

**Per-class breakdown — a partial success.** Macro-F1 0.45 is the best of
any goal; the confusion matrix (`mlp / cfdac_imag`, 238 Mass cases):

| true ↓ / pred → | Base | F1 | F2 | F3 | recall |
|---|---|---|---|---|---|
| Base | 27 | 0 | 32 | 38 | 0.28 |
| F1 | 22 | 21 | 0 | 18 | 0.34 |
| F2 | 0 | 0 | 26 | 14 | 0.65 |
| F3 | 0 | 0 | 9 | 31 | 0.78 |

Every plate has recall ≥ 0.28 — no class collapses to zero, so the signal
is genuine — but it is **partial and uneven**: the cell resolves the upper
plates (F2 0.65, F3 0.78) and the lower ones (Base 0.28, F1 0.34) poorly.
(The report-era cell `cnn2d/cfdac_real` had the *opposite* weakness — strong
Base, zero F3 — and a lower seeded balanced accuracy of 0.45; the two cells
locate different plates, neither uniformly.)

**Verdict — the best goal, but a partial success.** Balanced accuracy 0.51
(≈ 2× the 0.25 four-class chance) and macro-F1 0.45 — the strongest
synth-only signal of any goal. An added mass-plate shifts the floor-mode
amplitudes by a large, location-specific amount that partly survives the
domain gap; no single cell resolves all four plates uniformly. This is the
result to build on.

---

## 8. Cross-cutting — physics-aware augmentation A/B

The previous draft *recommended* an augmented-features retrain, *estimating*
"+0.05–0.10 on type". It was run as a seeded A/B: `hpo.py` on the plain
features (`features.h5`, 60 cells) and on a 20 000-sample augmented mix
(`features_mixed_aug.h5`, 50 cells — 10 000 original + 10 000 with
per-channel gain, input gain, 30 Hz shelf colouring, 30 dB noise), **same
seed**.

| goal | plain best macro-F1 | augmented best macro-F1 | Δ |
|---|---|---|---|
| detection (binary) | 0.488 | 0.472 | −0.016 |
| type | 0.250 | 0.291 | +0.041 |
| location — column | 0.167 | 0.124 | −0.043 |
| location — mass | 0.251 | 0.164 | −0.087 |
| severity (R²) | +0.006 | +0.075 | +0.069 |

Paired over the 20 main-task classification cells common to both arms:
**mean Δ macro-F1 −0.008, sd 0.054, ≈ 0.64σ from zero — not significant.**
The predicted +0.05–0.10 type lift did not materialise (observed type
cell-mean +0.004). The A/B is **single-seed** and **confounded** — the
augmented arm also has 2× the training data — so it shows the
predicted-magnitude benefit is absent without proving augmentation harmful.
Severity rises from R² +0.006 to +0.075, directionally positive but both
arms ≈ 0. To resolve properly: re-run over ≥ 3 seeds with a size-matched
control.

---

## 9. Limitations

1. **No goal transfers well enough to deploy.** The seeded sweep ranks them:
   mass-plate location is the strongest (macro-F1 0.45) but only a partial
   success (no cell resolves all four plates); type and severity transfer
   weakly (macro-F1 0.25; R² ≈ 0.12); detection is marginal (+0.04 over
   chance, within noise); column-end location does not transfer. This is the
   central finding — synthetic-only training yields, at best, weak
   cross-domain signal.
2. **Deep CFDAC models collapse to the class prior** for `type` and
   `binary` — the synth feature manifold projects to a near-constant on the
   experimental distribution, so argmax returns one class. Raw accuracy
   hides this; balanced accuracy reveals it.
3. **Synth Crack damage is anti-correlated with real Crack** — the
   binary-trenchcoat `is_Crack` classifier has cross-domain AUC 0.36, below
   chance (figure from [`REPORT_full.md` § 9](REPORT_full.md)). Synthetic
   Crack is symmetric across all 4 column corners; real Crack is per-corner
   asymmetric. The same symmetry is why column-end location fails.
4. **Multi-seed uncertainty measured (3 seeds, `multiseed_summary.json`).**
   Median macro-F1 sd 0.011, p90 sd 0.071 across 244 cells × 3 seeds — the
   earlier ≈ 0.05–0.07 estimate is now a measurement. **Noise band split by
   model family** (`balanced_acc_sd`, n cells):
   * torch (`cnn`, `cnn2d`, `mlp`, `transformer`): n = 120, median 0.019, p90 0.075.
   * sklearn (`rf`, `xgb`): n = 35, median 0.031, p90 0.079.

   Sklearn cells carry a slightly higher *median* sd than torch (the
   seeded train/val/test split is the dominant variance source — RF/XGB are
   otherwise deterministic per fold) but the **tails are comparable**
   (p90 ≈ 0.08 for both). CNN2D-on-CFDAC cells are the most seed-sensitive
   cluster individually (sd up to 0.2 for `is_bolt` / `is_hole` /
   `col_location`); a 5-seed re-run on those would tighten the bound.
   Conclusions are drawn only where the effect exceeds the band.
5. **The augmented arm is 50 of 60 cells** — the 10 `cnn2d/cfdac` cells
   (≈ 12 min each) were not completed under the ephemeral-container compute
   budget; they collapse to the class prior in the plain arm regardless.
6. **The IQS sampling is itself limiting** — zero AD-end Crack/Hole cases,
   every Mass case at one severity, only 80 balanced-cell Mass samples.

---

## 10. Recommendations

In cost / impact order, all synth-only.

1. **Augmented-chunks retrain — DONE; predicted lift not observed** (§ 8).
   The estimated type lift did not materialise (paired Δ −0.008 ± 0.054).
   To close it out properly: re-run over ≥ 3 seeds with a size-matched
   (10 000 augmented-only) control.
2. **Build on mass-plate location** — the strongest goal (§ 7.2).
   *Done:* the seeded `hpo_cfdac_*` re-run is complete; the best seeded cell
   is `mlp/cfdac_imag` (macro-F1 0.45). Its per-class breakdown shows it
   resolves the upper plates (F2/F3) but is weak on Base/F1 — while the
   report-era `cnn2d/cfdac_real` had the opposite weakness. Next: a model
   that combines both (e.g. an ensemble, or training on `cfdac_all`) should
   resolve all four plates; that is the concrete next experiment.
3. **Fix the synthetic damage physics — recommended next investment.**
   Promote `variation_v2.py` → `variation.py` and regenerate the chunk set
   (P2.1 + P2.2, ≈ 24 h CPU). Asymmetric per-corner Crack/Hole damage
   directly targets the two biggest failures — the `is_Crack` AUC-0.36
   anti-correlation (§ 9.3) and the column-end symmetry (§ 7.1). Expected:
   `is_Crack` AUC 0.36 → ≥ 0.5 and a non-degenerate `col_location`.
4. **SSL pretrain on unlabelled experimental data** (P2.3) — **withdrawn as
   stated.** The proposal pretrains on all 2 638 experimental cases, but
   2 176 of those are damaged-structure measurements that a genuine
   pre-deployment scenario does not have (§ 3.2); it is not a synth-only
   method. An assumption-respecting variant could pretrain only on synthetic
   data plus the single pristine reference — but that adds little over the
   existing synthetic training and is not the original recommendation.
5. **Full-data vision sweep** (≈ 14 h CPU) — not run; compute-bound.
6. **Nonlinear bolt model** (P2.4, Bouc-Wen, multi-day) — not started.

The cheap pipeline fixes (P0, P1.1) are real and kept, but the remaining
gap is **structural** — in the physics of the synthetic damage model, not
the ML pipeline. Post-hoc augmentation (rec 1) did not close it. The route
to deployable accuracy on the failing goals is either better synthetic
physics (rec 3) or training that uses experimental data — the joint
synth+exp fine-tune in [`REPORT_full.md` § 9.2](REPORT_full.md), out of
scope for a synth-only report.

---

## 11. Reproducibility

4-thread CPU, no GPU. Synth-only sweep ≈ 1 h; augmented A/B ≈ 2 h.

```bash
# 1. Build features (≈ 5 min)
cat experimental_frfs_chunks/experimental_frfs.h5.part_* > experimental_frfs.h5
python -m ml_pipeline.features
python -m ml_pipeline.cfdac
python -m ml_pipeline.cfdac_variants
python -m ml_pipeline.build_experimental_features

# 2. Seeded synth-only sweep + honest-metric evaluation
python -m ml_pipeline.hpo            --features dataset/features.h5
python -m ml_pipeline.hpo_cfdac_variants  --features dataset/features.h5
python -m ml_pipeline.hpo_cfdac_allmodels --features dataset/features.h5
python -m ml_pipeline.evaluate_full_experimental --skip-ind --out-suffix _plain

# 3. Augmentation A/B (controlled — same seed, only the data differs)
python -m ml_pipeline.build_augmented_chunks
python -m ml_pipeline.features --dataset dataset/aug_chunk --out dataset/features_aug.h5
python -m ml_pipeline.cfdac          --features dataset/features_aug.h5
python -m ml_pipeline.cfdac_variants --features dataset/features_aug.h5
python -m ml_pipeline.build_mixed_features \
    --sources dataset/features.h5 dataset/features_aug.h5 \
    --out dataset/features_mixed_aug.h5
python -m ml_pipeline.hpo --features dataset/features_mixed_aug.h5
python -m ml_pipeline.evaluate_full_experimental --skip-ind --out-suffix _aug
```

Evidence artefacts: `results/experimental_full_evaluation_plain.json`
(seeded synth-only, 60 cells), `_aug.json` (augmented arm, 50 cells),
`_basescore.json` (report-era models re-scored, 78 cells),
`ablation_log.json` (per-fix ablation table).
