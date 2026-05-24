# Comprehensive report — LANL 3SBB sim-to-real ML pipeline

The detailed companion to [`REPORT_definitive.md`](REPORT_definitive.md),
organised **by diagnosis goal**. For each of the four goals — damage
detection, type, severity, location — this report gives the exact training
conditions, the full cross-domain comparison, the ablation history, and the
specialised experiments that bear on that goal. Cross-cutting experiments
(augmentation A/B, joint synth+exp fine-tune, noise robustness) are in § 9.

Methodology-corrected edition: torch training is now seeded and the
evaluation reports macro-F1 / balanced accuracy, not accuracy alone (§ 3).
The chronological per-fix table is [`ablation_log.json`](ablation_log.json).

## Contents

1. [Overview](#1-overview)
2. [Dataset](#2-dataset)
3. [Methodology corrections](#3-methodology-corrections)
4. [Shared training setup](#4-shared-training-setup)
5. [Goal 1 — Damage detection](#5-goal-1--damage-detection)
6. [Goal 2 — Damage type assessment](#6-goal-2--damage-type-assessment)
7. [Goal 3 — Damage severity assessment](#7-goal-3--damage-severity-assessment)
8. [Goal 4 — Damage location assessment](#8-goal-4--damage-location-assessment)
9. [Cross-cutting experiments](#9-cross-cutting-experiments)
10. [Limitations](#10-limitations)
11. [Recommendations](#11-recommendations)
12. [Reproducibility](#12-reproducibility)
13. [Artefact index](#13-artefact-index)

---

# 1. Overview

The pipeline trains ML models on **10 000 synthetic** finite-element samples
of the LANL 3-Storey Bookcase Benchmark and evaluates them **zero-shot** on
**2 638 real** IQS experimental cases. The work spans ~30 commits on
`claude/improve-fe-training-WqMhW`, in three families:

| family | scope | wall time |
|---|---|---|
| P0 — cheap fixes | reference-FRF bug, bounded heads, scaler source | minutes per fix |
| P1 — moderate | per-sample normalisation + HPO retrain; joint synth+exp loop | ~1 h per phase |
| P2 — stretch | widened domain randomisation, asymmetric damage, SSL, nonlinear bolt | coded, mostly deferred (compute-bound) |

plus a synth-only vision-backbone sweep, the binary-trenchcoat
decomposition, a severity-stratified analysis, and a noise-robustness sweep.

**Headline, by goal** (synth-only, zero-shot, seeded 244-cell sweep
`experimental_full_evaluation_seeded.json`, honest metric):

| goal | task | best result | transfers? |
|---|---|---|---|
| detection | `binary` | balanced acc 0.585 (`cnn2d/cfdac_real`) | **weak** — no usable operating point |
| type | `type` (5-class) | macro-F1 0.25 (`mlp/modal`) | weakly |
| type | `type` one-vs-rest (modal-MLP) | 3-seed mean BA 0.62–0.66 ± ≤ 0.03 (`is_hole / mlp / modal` 0.661 ± 0.004, § 6.6) | **yes — robust, strongest in study** |
| severity | `severity` | R² 0.11–0.13 (3 CFDAC cells) | **weakly** |
| location | `col_location` | macro-F1 0.17 (`mlp/modal`) | no |
| location | `mass_location` | macro-F1 0.45 (`mlp/cfdac_imag`), balanced acc 0.51 | **partly** (best goal) |

Using **experimental labels in training** (joint synth+exp fine-tune, § 9.2)
reaches deployable accuracy on every task — but that is no longer synth-only
and is reported separately.

---

# 2. Dataset

* **Synthetic** — 10 000 samples from the calibrated semi-rigid 3SBB
  reduced-order model: 2 000 each of Pristine, Bolt, Crack, Hole, Mass,
  stratified across damage locations. 9 accelerometer channels, shared
  chirp excitation, 1 024 time samples at 256 Hz.
* **Experimental** — 2 638 IQS cases, strongly imbalanced:

  | class | Bolt | Pristine | Crack | Hole | Mass |
  |---|---|---|---|---|---|
  | count | 1 338 | 462 | 320 | 280 | 238 |
  | share | 50.7 % | 17.5 % | 12.1 % | 10.6 % | 9.0 % |

The experimental set also has structural gaps: zero AD-end Crack/Hole cases,
every Mass case at a single severity (1.2 kg → normalised 0.458), only 80
balanced-cell Mass samples. See [`MODEL.md`](../MODEL.md) for the full
physical description.

---

# 3. Methodology corrections

Two evaluation bugs were found and fixed (commit `a40ed6d`) *before* any
result here was re-derived.

**3.1 Deterministic training.** `hpo.py` seeded the data split and the
sklearn models but never seeded PyTorch — every cnn/transformer/cnn2d cell
was a single unreproducible draw. `_train_torch` now seeds torch, numpy and
the training `DataLoader`; `hpo_cfdac_*` were seeded the same way
(commit `f3ceeaf`). Comparing seeded re-runs against the report-era
artefacts shows nominally-identical cells differ by up to ≈ 0.07 macro-F1.

**3.2 Honest metrics.** `evaluate_full_experimental` recorded only raw
accuracy, which on the 50.7 %-Bolt experimental set rewards class-prior
collapse. It now also records **macro-F1** and **balanced accuracy**
(chance = 1/n_classes). Re-scoring the report-era models with these
(`experimental_full_evaluation_basescore.json`, 78 cells) shows that three
of the four old accuracy headlines were degenerate classifiers.

---

# 4. Shared training setup

Every per-goal cell uses this recipe; the goal sections state only what
differs.

| element | setting |
|---|---|
| synthetic data | 10 000 FE samples (2 000 / class); 9 channels; FRF band 5–100 Hz |
| features | `modal` (81-dim), `frf_mag` (381×9), `cfdac_*` (128×128 CFDAC matrices + projections) |
| split | stratified 70/15/15 → 7 000 / 1 500 / 1 500 (`make_split`, sklearn, `random_state` 20260511); severity uses the 8 000 damage-only samples → 5 600 / 1 200 / 1 200 |
| HPO | exhaustive grid per (task, model, feature) cell; best by validation metric |
| torch training | AdamW (wd 10⁻⁴), CosineAnnealingLR, 4 epochs, batch 64, CE / MSE loss, MLP dropout 0.2 |
| seed | 20260511 — torch + numpy + DataLoader + split + sklearn |
| preprocessing | P0.1 experimental-Pristine CFDAC reference; P0.3 experimental-Pristine `StandardScaler` for modal MLP/sklearn; P1.1 per-sample input normalisation |
| HPO grids | rf `n_estimators`×`max_depth`; xgb `n_estimators`×`max_depth`; mlp `hidden`×`lr`; cnn `widths`×`kernel`; transformer `d_model`×`n_layers`; cnn2d/cnn3d `widths`×`kernel` |
| evaluation | synth-trained model run zero-shot on all 2 638 experimental cases |

## 4.1 The cell grid — which model pairs with which feature

A **cell** is one (task × model × feature) combination. The model family is
fixed by the feature's tensor rank: a flat vector → tree ensemble or MLP; a
frequency×channel *sequence* → 1-D CNN or Transformer; a 128×128 CFDAC
*matrix* → 2-D CNN; a stacked 3-D CFDAC tensor → 3-D CNN.

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
/ `cnn3d` — 2-/3-D conv stacks. `timeseries` is legacy — on experimental
data it is synthesised from the FRF, not independent (P0.4). Across the five
tasks this yields the 78-cell report-era sweep (`_basescore.json`); the
60-cell seeded sweep (`_plain.json`) covers the `modal` / `frf_mag` /
`cfdac`-legacy subset that `hpo.py` runs, the CFDAC variants coming from
`hpo_cfdac_variants.py` / `hpo_cfdac_allmodels.py`.

## 4.2 Deployment data assumption — what experimental data is available

"Synth-only" means a specific, realistic constraint: **before deployment
the only experimental data available is a reference measurement of the
healthy structure.** The pristine 3SBB can be measured freely; there are no
measurements of it damaged. Methods classified by what they need:

| method | experimental data needed | within the assumption? |
|---|---|---|
| P0.1 CFDAC reference, P0.3 scaler, P1.1, the synth-only pipeline (§§ 5–8) | pristine reference only | **yes** |
| SSL pretrain on "unlabelled experimental data" (rec 4) | unlabelled measurements of the **damaged** structure | **no** |
| Joint synth+exp fine-tune (§ 9.2) | **labelled** damaged-structure measurements | no — post-deployment only |

The SSL proposal pretrains on all 2 638 experimental cases — 2 176 of them
damaged-structure measurements a genuine pre-deployment scenario lacks. It
is therefore not a synth-only method (§ 11 corrects rec 4). P0.1 / P0.3 use
only the 462 pristine measurements and stay within the assumption.

---

# 5. Goal 1 — Damage detection

**Task `binary`** — Pristine vs any damage. Experimental set: 82.5 % damage,
17.5 % pristine; binary chance balanced accuracy = 0.500.

## 5.1 Recommended training configuration

| element | value |
|---|---|
| feature | `cfdac_magphase` (128×128×2 CFDAC magnitude+phase stack) |
| model | `MLP`, hidden (256, 128, 64), dropout 0.2 (flattened CFDAC input) |
| optimiser | AdamW, lr 1×10⁻³, weight decay 10⁻⁴ |
| schedule / epochs / batch | CosineAnnealing / 4 / 64 |
| loss | cross-entropy |
| preprocessing | P0.1 experimental-Pristine CFDAC reference; P1.1 per-sample normalisation |
| HPO | grid hidden × lr (9 configs, `hpo_cfdac_allmodels.py`); best by val accuracy |

## 5.2 Result (seeded, zero-shot)

| synth val | synth test | exp accuracy | exp macro-F1 | exp balanced acc |
|---|---|---|---|---|
| 0.80 | 0.80 | 0.764 | 0.542 | **0.540** |

Per-class recall: Damage 0.88, Pristine 0.19.

## 5.3 Full comparison (seeded sweep, `_seeded.json`, 26 binary cells)

| model / feature | accuracy | macro-F1 | balanced acc | per-class recall |
|---|---|---|---|---|
| cnn2d / cfdac_real | 0.494 | 0.463 | **0.585** | Pristine 0.73 / Damage 0.44 |
| mlp / cfdac_magphase | 0.764 | 0.542 | 0.540 | Pristine 0.19 / Damage 0.88 |
| mlp / modal | 0.824 | 0.488 | 0.515 | — |
| transformer / frf_mag | 0.819 | 0.470 | 0.505 | — |

Most cells sit at balanced accuracy ≈ 0.50 (predict "damage" for nearly all
cases). The two cells above it trade off oppositely: `cnn2d/cfdac_real`
leans Pristine (high balanced acc, low accuracy), `mlp/cfdac_magphase` leans
Damage (best macro-F1).

## 5.4 Ablation history

P0/P1 fixes did not move binary in the report-era sweep. The seeded
re-run's `hpo_cfdac_allmodels` pass added the MLP-on-CFDAC cells, which is
where the small `mlp/cfdac_magphase` margin first appears.

## 5.5 Verdict

**Weakly transfers — but no operationally usable cell.** The best
balanced-accuracy cell `cnn2d/cfdac_real` (0.585) clears the 0.500 chance
level by +0.085 — outside the noise band, so detection carries genuine weak
discriminative signal. But no cell has a usable operating point:
`cnn2d/cfdac_real` reaches 0.585 by leaning Pristine — it **misses 56 % of
real damage**, useless as a safety detector — while `mlp/cfdac_magphase`
(best macro-F1) catches 88 % of damage but flags 81 % of Pristine cases as
damaged. Detection has weak cross-domain signal that no single cell
converts into an acceptable recall / false-alarm trade-off.

## 5.6 Evaluation on Pristine + severe-damage only

A deployment-relevant restriction — test only on Pristine ∪ {damage with
per-type-normalised severity ≥ τ}, `mlp/modal`:

| test set | n | accuracy | macro-F1 | balanced acc |
|---|---|---|---|---|
| all cases | 2 638 | 0.825 | 0.482 | 0.513 |
| Pristine + damage τ ≥ 0.5 | 1 520 | 0.705 | 0.444 | 0.516 |
| Pristine + damage τ ≥ 0.7 | 1 240 | 0.640 | 0.420 | 0.516 |

Balanced accuracy is flat at ≈ 0.51 — restricting to severe damage does not
help detection; raw accuracy falls only because Pristine grows as a share.

---

# 6. Goal 2 — Damage type assessment

**Task `type`** — 5 classes (Pristine, Bolt, Crack, Hole, Mass). Chance
balanced accuracy = 0.200.

## 6.1 Recommended training configuration

| element | value |
|---|---|
| feature | `modal` (81-dim) |
| model | `MLP`, hidden (512, 256, 128), dropout 0.2 |
| optimiser | AdamW, lr 3×10⁻³, weight decay 10⁻⁴ |
| schedule / epochs / batch | CosineAnnealing / 4 / 64 |
| loss | cross-entropy |
| preprocessing | P0.3 `StandardScaler` on the experimental-Pristine subset |
| HPO | grid hidden × lr (9 configs); best by val accuracy |

## 6.2 Result (seeded, zero-shot)

| synth val | synth test | exp accuracy | exp macro-F1 | exp balanced acc |
|---|---|---|---|---|
| 0.867 | 0.880 | 0.349 | **0.250** | 0.331 |

Report-era `mlp/modal` re-scored: macro-F1 0.296 / balanced acc 0.371 — the
≈ 0.05 gap is the § 3.1 seed-noise band.

## 6.3 Full comparison (`_basescore.json`, 14 type cells)

| model / feature | accuracy | macro-F1 | balanced acc | reading |
|---|---|---|---|---|
| **mlp / modal** | 0.37 | **0.30** | 0.37 | best honest cell |
| transformer / timeseries | 0.29 | 0.24 | 0.30 | on non-independent feature |
| cnn / timeseries | 0.29 | 0.18 | 0.26 | non-independent feature |
| cnn2d / cfdac_real | 0.32 | 0.17 | 0.19 | deep CFDAC — collapses |
| cnn3d / cfdac3d_realimag | 0.34 | 0.17 | 0.20 | collapses |
| cnn / frf_mag | **0.51** | 0.14 | 0.20 | **predicts Bolt for all 2 638 cases** |

The previous draft's "type 0.507" headline is the bottom row: a degenerate
classifier with balanced accuracy at exactly 5-class chance.

## 6.4 Ablation history

The P0.1 experimental-Pristine reference initially *lowered* type accuracy
(0.470 → 0.384) because synth-trained CFDAC backbones had been exploiting
the synth-vs-synth reference bias as a cross-domain shortcut. P1.1
per-sample normalisation lifted the cnn/frf_mag cell's accuracy to 0.507 —
but macro-F1 reveals that "lift" was the model collapsing onto Bolt. The
honest type signal (modal/MLP, macro-F1 ≈ 0.30) was never moved much by any
P0/P1 fix; it is a property of the modal feature.

## 6.5 Vision-model backbones on type (synth-only)

Five ImageNet-pretrained backbones (ResNet50, EfficientNet-B0,
ConvNeXt-Tiny, Swin-T, ViT-B/16) on CFDAC inputs, 1 500-sample subsample,
4 epochs. Result: no backbone beat the bespoke `cnn2d` on macro-F1.
ConvNeXt-Tiny/cfdac_all reached macro-F1 ≈ 0.25 (genuinely diagonal);
ResNet50/cfdac_all reached accuracy 0.52 but by predicting Bolt for ~96 %
of cases (macro-F1 0.19). *Accuracy rewards class-prior gaming; macro-F1
exposes it.* Detail in [`REPORT_vision.md`](REPORT_vision.md) /
[`REPORT_vision_v2.md`](REPORT_vision_v2.md). These sub-studies pre-date the
§ 3 corrections and were not re-scored.

## 6.6 One-vs-rest type detection — robust transfer via modal-MLP

The 5-class `type` task transfers weakly (§ 6.2, macro-F1 0.25). The
**one-vs-rest decomposition** — five binary "is the damage type X?"
classifiers — transfers *substantially better*. Multi-seed validation
reveals a crucial cell-family split: CFDAC-CNN one-vs-rest cells are
**single-seed flukes** (an earlier draft made these the headline), while
**modal-MLP and frf_mag/transformer cells transfer robustly** at balanced
accuracy 0.62–0.66 across all three seeds. From `multiseed_summary.json`,
best **robust** cell per sub-task (filter: sd < 0.05):

| one-vs-rest task | best robust cell | BA mean | BA sd | seeds (default / 101 / 202) |
|---|---|---|---|---|
| `is_hole` | **mlp / modal** | **0.661** | 0.004 | 0.665 / 0.659 / 0.658 |
| `is_bolt` | **mlp / cfdac_real** | **0.631** | 0.015 | 0.635 / 0.645 / 0.615 |
| `is_bolt` (alt) | transformer / frf_mag | 0.621 | 0.009 | 0.628 / 0.611 / 0.624 |
| `is_mass` | **mlp / modal** | **0.628** | 0.011 | 0.616 / 0.638 / 0.629 |
| `is_crack` | **mlp / modal** | **0.615** | 0.026 | 0.603 / 0.645 / 0.597 |
| `is_pristine` | mlp / modal | 0.517 | 0.003 | 0.513 / 0.519 / 0.518 |

**Four of five sub-tasks transfer robustly at BA 0.62–0.66 (≈ 2× chance)
with sd ≤ 0.03**; only `is_pristine` is stuck near chance. Normalised lift
0.24–0.32 — comparable to mass-plate location's 0.30 (§ 8.2), but with
much smaller per-cell sd (≤ 0.03 vs 0.07). **This is the strongest robust
transfer in the study**, tied with mass-plate location.

The single-seed CFDAC-CNN headlines that an earlier draft promoted were
flukes — `is_bolt / cnn2d / cfdac_real` 0.708 / 0.488 / 0.514 (mean 0.570
sd 0.121), `is_hole / mlp / cfdac_all` 0.684 / 0.500 / 0.500 (mean 0.561
sd 0.106). On 2 of 3 seeds those cells sit at chance; they are unreliable.
Use the modal-MLP cells instead — `is_hole / mlp / modal` (BA 0.661 ±
0.004) is the *single most reproducibly transferring cell in the entire
244-cell sweep*.

**Deployment implication:** build a bank of modal-MLP per-damage-type
detectors. The CFDAC-CNN variants are seed-fragile and unreliable.

The *aggregated* trenchcoat (recombining the five binaries into a 5-class
prediction via a transductive `dataset_zscore` aggregator) scores only
macro-F1 ≈ 0.29 — the lossy aggregation, not the binaries, is the weak
link. A separate diagnostic from the pre-correction `vision_v2` study found
the `is_Crack` binary's cross-domain ROC-AUC at 0.36 (below chance), which
motivated the asymmetric-damage fix (P2.2); that AUC figure is from
[`REPORT_vision_v2.md`](REPORT_vision_v2.md), not the seeded sweep.

## 6.7 Severity-stratified behaviour

A previously-reported "≈ 0.66 accuracy at high severity" used *raw accuracy
on a damage-only subset* — a class-distribution shift, not a skill gain.
Re-derived honestly on the deployment-relevant test set — Pristine ∪
{damage with per-type-normalised severity ≥ τ}:

| cell | test set | n | accuracy | macro-F1 | balanced acc |
|---|---|---|---|---|---|
| mlp / modal | all cases | 2 638 | 0.373 | 0.296 | 0.371 |
| mlp / modal | Pristine + τ ≥ 0.7 | 1 240 | 0.390 | 0.219 | 0.306 |
| cnn2d / cfdac_real | all cases | 2 638 | 0.324 | 0.172 | 0.189 |
| cnn2d / cfdac_real | Pristine + τ ≥ 0.7 | 1 240 | 0.411 | 0.206 | 0.307 |

Restricting to severe damage helps the 2-D CNN / CFDAC cell modestly
(balanced accuracy 0.19 → 0.31) but *lowers* the modal-MLP cell's macro-F1
(0.30 → 0.22). There is at most a small genuine high-severity effect, for
one CFDAC cell — not the breakthrough the old accuracy figure implied. The
damage-only severity-threshold curves are in
[`REPORT_severity_stratified.md`](REPORT_severity_stratified.md).

## 6.8 Verdict

**Transfers weakly.** Only modal + MLP carries real signal (macro-F1 ≈
0.25–0.30, balanced accuracy ≈ 0.33–0.37 vs 0.20 chance). Every deep CFDAC
or vision model that scores higher *accuracy* does so by class-prior
collapse. Type assessment is above chance but not deployable synth-only.

---

# 7. Goal 3 — Damage severity assessment

**Task `severity`** — regression, target normalised [0, 1] per damage type,
damage samples only (8 000 synthetic / 2 176 experimental).

## 7.1 Recommended training configuration

| element | value |
|---|---|
| feature | `cfdac_imag` (128×128 CFDAC imaginary-part matrix) |
| model | `MLP`, hidden (256, 128, 64), dropout 0.2 (flattened CFDAC input) |
| optimiser | AdamW, lr 1×10⁻³, weight decay 10⁻⁴ |
| schedule / epochs / batch | CosineAnnealing / 4 / 64 |
| loss | MSE; **sigmoid-bounded output head** (P0.2) keeps predictions in [0, 1] |
| preprocessing | P0.1 experimental-Pristine CFDAC reference; P1.1 per-sample normalisation |
| HPO | grid hidden × lr (9 configs, `hpo_cfdac_allmodels.py`); best by val R² |

## 7.2 Result (seeded, zero-shot)

| synth val R² | synth test R² | exp R² | exp MAE |
|---|---|---|---|
| 0.311 | 0.281 | **+0.131** | 0.224 |

## 7.3 Full comparison (seeded sweep, `_seeded.json`, 16 severity cells)

| model / feature | exp R² | note |
|---|---|---|
| mlp / cfdac_imag | +0.131 (3-seed mean +0.101 ± 0.026) | best genuine-feature cell |
| mlp / cfdac_realimag | +0.128 | |
| xgb / cfdac_imag | +0.110 | |
| transformer / frf_mag | +0.006 | |
| cnn / timeseries | +0.180 | **non-independent `timeseries` feature — excluded** |

## 7.4 Ablation history

P0.2 (sigmoid-bounded heads) was essential: before it, MLP regression heads
extrapolated to ±∞ on OOD inputs, giving R² as low as −10²². The earlier
"does not transfer" verdict held only because the report-era sweep never
evaluated MLP/XGB on the CFDAC variants — the seeded `hpo_cfdac_allmodels`
pass added them and surfaced the R² ≈ 0.12 cells.

## 7.5 Verdict

**Weakly transfers.** The seeded sweep surfaces **three independent cells
clustered at R² 0.11–0.13** (`mlp`/`xgb` on `cfdac_imag`/`cfdac_realimag`) —
well below a usable severity estimator, but distinctly above the ≈ 0 of
every earlier genuine-feature cell. Severity carries a weak but consistent
cross-domain signal in the CFDAC imaginary part. The `timeseries` 0.180
figure is on a feature reconstructed from the FRF (P0.4) and is excluded.

---

# 8. Goal 4 — Damage location assessment

Two sub-tasks: which **column-end** (`col_location`, 6 classes, chance
balanced acc 0.167) and which **mass-plate** (`mass_location`, 4 classes,
chance 0.250).

## 8.1 Column-end location (`col_location`)

### Recommended configuration

| element | value |
|---|---|
| feature | `modal` (81-dim modal descriptor) |
| model | `MLP`, hidden (512, 256, 128), dropout 0.2 |
| optimiser | AdamW, lr 3×10⁻³, weight decay 10⁻⁴ |
| schedule / epochs / batch | CosineAnnealing / 4 / 64 |
| loss | cross-entropy |
| preprocessing | P0.3 `StandardScaler` on the experimental-Pristine subset |
| HPO | grid hidden × lr (9 configs); best by val accuracy |

### Result & comparison (seeded sweep, `_seeded.json`)

| synth val | synth test | exp accuracy | exp macro-F1 | exp balanced acc |
|---|---|---|---|---|
| 0.515 | 0.481 | 0.284 | **0.167** | 0.274 |

| model / feature | accuracy | macro-F1 | balanced acc |
|---|---|---|---|
| mlp / modal | 0.28 | 0.17 | 0.27 |
| cnn / frf_mag | 0.36 | 0.16 | 0.21 |
| mlp / cfdac_magphase | 0.41 | 0.15 | 0.18 |
| cnn2d / cfdac_mag | 0.02 | 0.01 | 0.17 |

**Verdict — does not transfer.** Best seeded macro-F1 0.17, balanced
accuracy 0.27 vs 0.167 chance. The report-era headline cell `cnn2d/cfdac_mag`
(`_basescore`: accuracy 0.508 / macro-F1 0.19) **did not survive seeding** —
re-run it collapses to macro-F1 0.007, balanced accuracy 0.167 (chance): a
non-reproducible fluke. The synthetic crack/hole damage is symmetric per
storey, so the BD-vs-AD column ends are nearly indistinguishable — a
property of the synthetic physics.

## 8.2 Mass-plate location (`mass_location`)

### Recommended configuration

| element | value |
|---|---|
| feature | `cfdac_imag` (128×128 CFDAC imaginary-part matrix) |
| model | `MLP`, hidden (256, 128, 64), dropout 0.2 (flattened CFDAC input) |
| optimiser | AdamW, lr 1×10⁻³, weight decay 10⁻⁴ |
| schedule / epochs / batch | CosineAnnealing / 4 / 64 |
| loss | cross-entropy |
| preprocessing | P0.1 experimental-Pristine CFDAC reference; P1.1 per-sample normalisation |
| HPO | grid hidden × lr (9 configs, `hpo_cfdac_allmodels.py`); best by val accuracy |

### Result & comparison (seeded sweep, `_seeded.json`)

| synth val | synth test | exp accuracy | exp macro-F1 | exp balanced acc |
|---|---|---|---|---|
| 0.97 | 0.973 | 0.441 | **0.452** | **0.512** |

| model / feature | accuracy | macro-F1 | balanced acc |
|---|---|---|---|
| mlp / cfdac_imag | 0.44 | **0.45** | 0.51 |
| cnn2d / cfdac_real | 0.53 | 0.37 | 0.45 |
| mlp / cfdac_mag | 0.45 | 0.27 | 0.31 |
| mlp / modal | 0.39 | 0.25 | 0.27 |

**Per-class breakdown — a partial success.** Macro-F1 0.45 is the best of
any goal; the `mlp / cfdac_imag` confusion matrix (238 Mass cases):

| true ↓ / pred → | Base | F1 | F2 | F3 | recall |
|---|---|---|---|---|---|
| Base | 27 | 0 | 32 | 38 | 0.28 |
| F1 | 22 | 21 | 0 | 18 | 0.34 |
| F2 | 0 | 0 | 26 | 14 | 0.65 |
| F3 | 0 | 0 | 9 | 31 | 0.78 |

Every plate has recall ≥ 0.28 — none collapses to zero, so the signal is
genuine — but it is **partial and uneven**: the cell resolves the upper
plates (F2 0.65, F3 0.78) and the lower ones (Base 0.28, F1 0.34) poorly.
The report-era `cnn2d/cfdac_real` cell had the *opposite* weakness (strong
Base, zero F3) and a lower seeded balanced accuracy 0.45.

**Verdict — the best goal, but a partial success.** Balanced accuracy 0.51
(≈ 2× the 0.25 four-class chance), macro-F1 0.45 — the strongest synth-only
signal of any goal. An added mass-plate shifts the floor-mode amplitudes by
a large, location-specific amount that partly survives the domain gap; no
single cell resolves all four plates uniformly. P0.1 (experimental-Pristine
reference) drove the gain. A cell that resolves all four plates — combining
the complementary strengths of `mlp/cfdac_imag` and `cnn2d/cfdac_real` — is
the specific next experiment.

---

# 9. Cross-cutting experiments

## 9.1 Physics-aware augmentation A/B

`hpo.py` run seeded on the plain features (60 cells) and a 20 000-sample
augmented mix (50 cells: 10 000 original + 10 000 with per-channel gain,
input gain, 30 Hz shelf colouring, 30 dB noise).

| goal | plain best macro-F1 | augmented best macro-F1 | Δ |
|---|---|---|---|
| detection | 0.488 | 0.472 | −0.016 |
| type | 0.250 | 0.291 | +0.041 |
| location — column | 0.167 | 0.124 | −0.043 |
| location — mass | 0.251 | 0.164 | −0.087 |
| severity (R²) | +0.006 | +0.075 | +0.069 |

Paired over the 20 common main-task classification cells: **mean Δ macro-F1
−0.008 ± 0.054, ≈ 0.64σ from zero — not significant.** The predicted
+0.05–0.10 type lift did not materialise. The A/B is single-seed and
confounded (the augmented arm also has 2× the data); it shows the
predicted-magnitude benefit is absent without proving harm.

## 9.2 Joint synth+exp fine-tune (uses experimental labels)

Not synth-only — included for completeness as the **only approach shown to
reach deployable accuracy.** `transfer_learn.py` fine-tunes the synth-trained
backbone with the head unfrozen, 3 synth : 1 exp mini-batches, and an L2
anchor (λ = 10⁻⁴) to the synth weights. At a 50 % experimental fine-tune
fraction (`ablation_log.json`, P1.4 row):

| goal | head-only | full fine-tune | best cell |
|---|---|---|---|
| severity (R²) | 0.12 | **0.87** | cnn2d / cfdac_magphase |
| type (accuracy) | 0.57 | **0.77** | cnn2d / cfdac_real |
| col_location (accuracy) | 0.30 | **0.80** | cnn2d / cfdac_magphase |
| mass_location (accuracy) | 0.64 | **1.00** | cnn2d / cfdac_all |

Joint fine-tune recipe: start from the synth-HPO backbone, unfreeze the
whole network, mix 3 synth : 1 experimental samples per mini-batch, and add
an L2 anchor (λ = 10⁻⁴) to the synth-trained weights to prevent forgetting;
sweep the experimental fine-tune fraction k ∈ {10…50 %}. Per-case results
use 5 seeds (42–46), best-by-metric kept (`results/per_case_final/`).

These numbers are accuracy / R² (not re-scored with macro-F1) and were
produced before the § 3 seeding fix; treat them as indicative. The point
stands: experimental labels in training close the gap that synth-only
cannot. (The former standalone `REPORT_simtoreal.md` and `REPORT_final.md`
are superseded by this report and retired.)

## 9.3 Noise robustness

The full pipeline was re-run on synthetic data corrupted with additive
Gaussian noise on the time series (per-sample, per-channel, controlled SNR)
and on a mixed-SNR variant. Detail in [`REPORT_noise.md`](REPORT_noise.md)
and `REPORT_noisy_mixed.md`. These sweeps pre-date the § 3 corrections.

---

# 10. Limitations

1. **No goal transfers well enough to deploy** synth-only. The seeded
   244-cell sweep ranks them: mass-plate location strongest (macro-F1 0.45,
   partial — no cell resolves all four plates); type and severity weak
   (macro-F1 0.25; R² ≈ 0.12); detection marginal (+0.04 over chance, within
   noise); column-end location does not transfer (≈ chance). The report-era
   "best" column-location cell was a non-reproducible fluke (§ 8.1).
2. **Deep CFDAC and vision models collapse to the class prior** on `type`
   and `binary`; raw accuracy hides this, balanced accuracy reveals it.
3. **Synthetic Crack is anti-correlated with real Crack** (`is_Crack`
   cross-domain AUC 0.36) — the synthetic damage model is symmetric where
   reality is asymmetric. The same symmetry sinks column-end location.
4. **Multi-seed uncertainty measured (3 seeds).** `multiseed_summary.json`
   gives median macro-F1 sd 0.011 and p90 sd 0.071 across 244 cells × 3
   seeds — confirms the earlier ≈ 0.05–0.07 estimate as a measurement.
   Caveat: `cnn2d` on CFDAC features is the most seed-sensitive cluster
   (sd up to 0.2 for `is_bolt` / `is_hole` / `col_location`); a 5-seed
   re-run on those cells is recommended for tighter bounds.
5. **The augmented arm is 50 of 60 cells** (the 10 `cnn2d/cfdac` cells were
   not completed under the ephemeral-container compute budget).
6. **Vision / trenchcoat / severity-stratified / noise sub-studies** were
   not re-run under the § 3 corrections; their numbers are indicative.
7. **The IQS sampling is itself limiting** — missing AD-end Crack/Hole
   cases, single-severity Mass, 80 balanced-cell Mass samples.

---

# 11. Recommendations

1. **Augmented retrain — done; predicted lift not observed** (§ 9.1).
   Re-run over ≥ 3 seeds with a size-matched control to close it out.
2. **Build on mass-plate location** (§ 8.2) — the strongest goal. *Done:*
   the seeded `hpo_cfdac_*` re-run is complete; best seeded cell
   `mlp/cfdac_imag` (macro-F1 0.45) resolves the upper plates while the
   report-era `cnn2d/cfdac_real` resolved the lower ones — complementary.
   Next: a cell that resolves all four plates (ensemble, or `cfdac_all`).
3. **Fix the synthetic damage physics — recommended next investment.**
   Promote `variation_v2.py` → `variation.py`, regenerate the chunk set
   (P2.1 + P2.2, ≈ 24 h CPU). Asymmetric per-corner Crack/Hole damage
   targets the two biggest failures — the `is_Crack` AUC-0.36
   anti-correlation and the column-end symmetry.
4. **SSL pretrain on unlabelled experimental data** (P2.3) — **withdrawn as
   stated.** It pretrains on all 2 638 experimental cases, 2 176 of which
   are damaged-structure measurements unavailable before deployment
   (§ 4.2); it is not a synth-only method. An assumption-respecting variant
   would pretrain only on synthetic data + the pristine reference, which
   adds little over the existing synthetic training.
5. **Full-data vision sweep** (≈ 14 h) — not run.
6. **Nonlinear bolt model** (P2.4, Bouc-Wen, multi-day) — not started.

The cheap pipeline fixes (P0, P1.1) are real and kept; the remaining
synth-only gap is **structural** — in the synthetic damage physics, not the
ML pipeline. The deployable route is either better physics (rec 3) or
experimental labels in training (§ 9.2).

---

# 12. Reproducibility

4-thread CPU, no GPU. Synth-only sweep ≈ 1 h; augmented A/B ≈ 2 h.

```bash
# 1. Build features (≈ 5 min)
cat experimental_frfs_chunks/experimental_frfs.h5.part_* > experimental_frfs.h5
python -m ml_pipeline.features
python -m ml_pipeline.cfdac
python -m ml_pipeline.cfdac_variants
python -m ml_pipeline.build_experimental_features

# 2. Seeded synth-only sweep + honest-metric evaluation
python -m ml_pipeline.hpo                 --features dataset/features.h5
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

---

# 13. Artefact index

```
results/REPORT_definitive.md                  concise goal-structured report
results/REPORT_full.md  (this file)            detailed goal-structured report
results/ablation_log.json                      chronological per-fix ablation table
results/experimental_full_evaluation_plain.json     seeded synth-only sweep (60 cells)
results/experimental_full_evaluation_aug.json       seeded augmented sweep   (50 cells)
results/experimental_full_evaluation_basescore.json report-era models, macro-F1 re-score (78 cells)
results/REPORT_vision.md / REPORT_vision_v2.md      vision-backbone sweep + trenchcoat (method study)
results/REPORT_severity_stratified.md               severity-threshold analysis (method study)
results/REPORT_noise.md / REPORT_noisy_mixed.md      noise-robustness sweeps (method study)
results/per_case_final/                             joint synth+exp fine-tune per-case predictions

(REPORT.md and REPORT_noisy_mixed.md are auto-generated and pre-date the
methodology corrections; REPORT_simtoreal.md and REPORT_final.md are
retired — their content lives in this report.)
```

Modules: `features.py`, `cfdac.py`, `cfdac_variants.py`,
`build_experimental_features.py` (feature build); `hpo.py`,
`hpo_cfdac_variants.py`, `hpo_cfdac_allmodels.py` (synth sweep);
`evaluate_full_experimental.py` (cross-domain eval); `transfer_learn.py`
(joint fine-tune); `build_augmented_chunks.py`, `build_mixed_features.py`
(augmentation); `models.py`, `train.py`, `tasks.py` (shared).
