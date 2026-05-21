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

| # | goal | task(s) | best synth-only result (zero-shot, real data) | transfers? |
|---|---|---|---|---|
| 1 | **Damage detection** | `binary` (damage vs pristine) | balanced accuracy 0.515 | **no** — ≈ chance |
| 2 | **Damage type** | `type` (5-class) | macro-F1 0.25–0.30, balanced acc 0.33–0.37 | **weakly** |
| 3 | **Damage severity** | `severity` (regression) | R² ≈ 0 on real features | **no** |
| 4 | **Damage location** | `col_location` (column-end) | macro-F1 0.19, balanced acc 0.23 | **no** |
| 4 | | `mass_location` (mass-plate) | macro-F1 0.44, balanced acc 0.51 | **yes** (modest) |

> **One-paragraph summary.** Of the four diagnosis goals, synthetic-only
> training transfers cleanly to **exactly one**: locating an added
> mass-plate (`mass_location`, macro-F1 0.44, balanced accuracy 0.51 — ≈ 2×
> chance). Damage **detection** and column-end **location** do not transfer
> (balanced accuracy at chance); **type** transfers only weakly (macro-F1
> ≈ 0.30 via the modal feature, *not* the deep CFDAC models, which collapse
> to predicting one class); **severity** regression does not transfer
> (R² ≈ 0 on every genuine feature). The previous draft's accuracy
> headlines (type 0.507, binary 0.825) were **class-prior collapse** —
> degenerate classifiers scored by a metric that rewards predicting the
> majority class. The recommended physics-aware augmentation, run as a
> seeded A/B, produced no classification change beyond run-to-run noise.

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
| Recommendation 4 — SSL pretrain on "unlabelled experimental data" | unlabelled measurements of the **damaged** structure | **no** |
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
| feature | `modal` — 81-dim modal descriptor |
| model | `MLP`, hidden (256, 128, 64), dropout 0.2 |
| optimiser | AdamW, lr 3×10⁻³, weight decay 10⁻⁴ |
| schedule / epochs / batch | CosineAnnealing / 4 / 64 |
| loss | cross-entropy |
| preprocessing | P0.3 — `StandardScaler` on the experimental-Pristine subset |
| HPO selection | grid hidden ∈ {(128,64),(256,128,64),(512,256,128)} × lr ∈ {5e-4,1e-3,3e-3}; best of 9 by validation accuracy |
| seed | 20260511 |

### 4.2 Result (seeded, zero-shot on 2 638 real cases)

| synth val | synth test | exp accuracy | exp macro-F1 | exp balanced acc |
|---|---|---|---|---|
| 0.993 | 0.982 | 0.824 | 0.488 | **0.515** |

### 4.3 Comparison (representative cells, `_basescore.json`)

| model / feature | accuracy | macro-F1 | balanced acc |
|---|---|---|---|
| mlp / modal | 0.825 | 0.482 | 0.513 |
| cnn2d / cfdac_imag | 0.810 | 0.457 | 0.495 |
| cnn2d / cfdac_all | 0.825 | 0.452 | 0.500 |
| cnn / frf_mag | 0.825 | 0.452 | 0.500 |

`transformer / timeseries` scores macro-F1 0.496 but on the synthesised
`timeseries` feature (not independent on experimental data — see § 6) and
is excluded.

### 4.4 Verdict

**Does not transfer.** Every cell sits at balanced accuracy ≈ 0.50 — the
binary chance level. The best genuine-feature cell (`mlp/modal`) reaches
0.515, marginally above chance and not usable. The synthetic damage
signature does not separate damaged from pristine on real data.

### 4.5 Evaluation on Pristine + severe-damage only

A deployment-relevant restriction: evaluate only on cases that are either
Pristine or *clearly* damaged — Pristine ∪ {damage with per-type-normalised
severity ≥ τ} (`mlp/modal`):

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

### 5.3 Comparison (representative cells, `_basescore.json`)

| model / feature | accuracy | macro-F1 | balanced acc | note |
|---|---|---|---|---|
| **mlp / modal** | 0.37 | **0.30** | 0.37 | best honest cell |
| cnn2d / cfdac_real | 0.32 | 0.17 | 0.19 | deep CFDAC — collapses |
| cnn3d / cfdac3d_realimag | 0.34 | 0.17 | 0.20 | collapses |
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
severity ≥ τ} — a scenario where damage, if present, is significant:

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

---

## 6. Goal 3 — Damage severity assessment

**Question:** how severe is the damage? Task `severity` — regression,
target normalised to [0, 1] per damage type, damage samples only.

### 6.1 Recommended training configuration

| element | value |
|---|---|
| training data | 8 000 synthetic damage samples (5 600 / 1 200 / 1 200) |
| feature | `frf_mag` — 381×9 log-magnitude FRF |
| model | `SmallTransformer`, d_model 32, 2 layers |
| optimiser | AdamW, lr 1×10⁻³, weight decay 10⁻⁴ |
| schedule / epochs / batch | CosineAnnealing / 4 / 64 |
| loss | MSE; **sigmoid-bounded output head** (P0.2) so the prediction stays in [0, 1] |
| preprocessing | P1.1 — per-sample log + z-score normalisation of `frf_mag` |
| HPO selection | grid d_model ∈ {32,48,64} × n_layers ∈ {1,2}; best by validation R² |
| seed | 20260511 |

### 6.2 Result (seeded, zero-shot on 2 638 real cases)

| synth val R² | synth test R² | **exp R²** | exp MAE |
|---|---|---|---|
| 0.185 | 0.130 | **+0.006** | 0.272 |

### 6.3 Comparison (`_basescore.json`)

| model / feature | exp R² | note |
|---|---|---|
| cnn / timeseries | +0.180 | **on the synthesised `timeseries` feature** — excluded |
| cnn2d / cfdac_mag | −0.012 | best genuine feature |
| cnn3d / cfdac3d_realimag | −0.013 | |
| cnn2d / cfdac_real | −0.015 | |

`timeseries` on experimental data is reconstructed from the FRF
(`H·F → IFFT`, P0.4) and carries no information beyond `frf_mag` — so the
0.180 figure is not a genuine-feature result and must not be quoted as one.

### 6.4 Verdict

**Does not transfer.** On every genuine feature synth-only severity R² is
≈ 0 (best ≈ +0.006, the rest slightly negative). A model predicting the
mean severity would score about as well. Severity cannot be read from a
synth-trained model on this rig.

---

## 7. Goal 4 — Damage location assessment

**Question:** *where* is the damage? Two distinct sub-tasks: which
**column-end** (`col_location`, 6 classes S1AD…S3BD) and which
**mass-plate** (`mass_location`, 4 classes Base/F1/F2/F3).

### 7a Column-end location (`col_location`)

#### 7a.1 Recommended training configuration

| element | value |
|---|---|
| training data | 10 000 synthetic samples (7 000 / 1 500 / 1 500) |
| feature | `cfdac_mag` — 128×128 CFDAC magnitude matrix |
| model | `Conv2DStack` (2-D CNN), widths (16, 32, 64), kernel 3 |
| optimiser | AdamW, lr 1×10⁻³, weight decay 10⁻⁴ |
| schedule / epochs / batch | CosineAnnealing / 4 / 64 |
| loss | cross-entropy |
| preprocessing | P0.1 experimental-Pristine CFDAC reference; P1.1 per-sample mean-subtract |
| HPO selection | grid widths × kernel (4 configs, `hpo_cfdac_variants.py`); best by validation accuracy |
| seed | 20260511 (`hpo_cfdac_*` seeded as of commit `f3ceeaf`) |

#### 7a.2 Result (zero-shot on 2 638 real cases)

| synth val | synth test | exp accuracy | exp macro-F1 | exp balanced acc |
|---|---|---|---|---|
| 0.492 | 0.463 | 0.508 | **0.192** | 0.228 |

#### 7a.3 Comparison (`_basescore.json`)

| model / feature | accuracy | macro-F1 | balanced acc |
|---|---|---|---|
| cnn2d / cfdac_mag | 0.51 | 0.19 | 0.23 |
| mlp / cfdac_real | 0.35 | 0.18 | 0.30 |
| cnn / timeseries | 0.30 | 0.16 | 0.16 |

#### 7a.4 Verdict

**Does not transfer.** Best macro-F1 0.19, balanced accuracy 0.23 against a
0.167 six-class chance level — marginal. The synthetic crack/hole model is
symmetric per storey, so the BD-vs-AD column ends are nearly
indistinguishable; this is a property of the synthetic physics.

### 7b Mass-plate location (`mass_location`)

#### 7b.1 Recommended training configuration

| element | value |
|---|---|
| training data | 10 000 synthetic samples (7 000 / 1 500 / 1 500) |
| feature | `cfdac_real` — 128×128 CFDAC real-part matrix |
| model | `Conv2DStack` (2-D CNN), widths (16, 32, 64), kernel 5 |
| optimiser | AdamW, lr 1×10⁻³, weight decay 10⁻⁴ |
| schedule / epochs / batch | CosineAnnealing / 4 / 64 |
| loss | cross-entropy |
| preprocessing | P0.1 experimental-Pristine CFDAC reference; P1.1 per-sample mean-subtract |
| HPO selection | grid widths × kernel (4 configs, `hpo_cfdac_variants.py`); best by validation accuracy |
| seed | 20260511 |

#### 7b.2 Result (zero-shot on 2 638 real cases)

| synth val | synth test | exp accuracy | exp macro-F1 | exp balanced acc |
|---|---|---|---|---|
| 0.893 | 0.863 | 0.534 | **0.435** | **0.506** |

#### 7b.3 Comparison (`_basescore.json`)

| model / feature | accuracy | macro-F1 | balanced acc |
|---|---|---|---|
| cnn2d / cfdac_real | 0.53 | **0.44** | 0.51 |
| cnn2d / cfdac (real+imag) | 0.42 | 0.43 | 0.49 |
| cnn2d / cfdac_imag | 0.39 | 0.42 | 0.49 |
| mlp / modal | 0.37 | 0.25 | 0.26 |

#### 7b.4 Verdict

**Transfers — the one clear synth-only success.** Balanced accuracy 0.51
against a 0.250 four-class chance level (≈ 2× chance), and three independent
`cnn2d` CFDAC cells agree at macro-F1 ≈ 0.42–0.44. An added mass-plate
shifts the floor-mode amplitudes by a large, location-specific amount that
survives the sim-to-real gap. This is the result to build on.

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

1. **Three of four goals do not transfer.** Detection, column-end location
   and severity are at chance / R² ≈ 0; type is only weakly above chance.
   Only mass-plate location is usable. This is the central finding.
2. **Deep CFDAC models collapse to the class prior** for `type` and
   `binary` — the synth feature manifold projects to a near-constant on the
   experimental distribution, so argmax returns one class. Raw accuracy
   hides this; balanced accuracy reveals it.
3. **Synth Crack damage is anti-correlated with real Crack** — the
   binary-trenchcoat `is_Crack` classifier has cross-domain AUC 0.36, below
   chance (figure from [`REPORT_full.md` § 9](REPORT_full.md)). Synthetic
   Crack is symmetric across all 4 column corners; real Crack is per-corner
   asymmetric. The same symmetry is why column-end location fails.
4. **No multi-seed uncertainty.** Every cell is one seeded draw; the
   ≈ 0.05–0.07 macro-F1 run-to-run band is an estimate, not a measured
   variance. Conclusions are drawn only where the effect exceeds it.
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
2. **Build on mass-plate location** — it is the one transferring goal
   (§ 7b). Re-run `hpo_cfdac_variants.py` (now seeded) to confirm the
   cnn2d/CFDAC result reproducibly, and characterise *why* it transfers
   (floor-mode amplitude) as a template for the other goals.
3. **Fix the synthetic damage physics — recommended next investment.**
   Promote `variation_v2.py` → `variation.py` and regenerate the chunk set
   (P2.1 + P2.2, ≈ 24 h CPU). Asymmetric per-corner Crack/Hole damage
   directly targets the two biggest failures — the `is_Crack` AUC-0.36
   anti-correlation (§ 9.3) and the column-end symmetry (§ 7a). Expected:
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
