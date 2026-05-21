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

**Headline, by goal** (synth-only, zero-shot, honest metric):

| goal | task | best result | transfers? |
|---|---|---|---|
| detection | `binary` | balanced accuracy 0.515 | no |
| type | `type` | macro-F1 0.25–0.30 | weakly |
| severity | `severity` | R² ≈ 0 on real features | no |
| location | `col_location` | macro-F1 0.19 | no |
| location | `mass_location` | macro-F1 0.44, balanced acc 0.51 | **yes** |

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

The model menu: `rf`, `xgb` (sklearn), `mlp`, `cnn` (1-D), `transformer`,
`cnn2d`, `cnn3d` (torch). The feature menu: `modal`, `frf_mag`, `timeseries`
(legacy — synthesised on experimental data, not independent), and the CFDAC
family (`cfdac_real/imag/mag/phase`, `cfdac_realimag/magphase/all`, and 3-D
`cfdac3d_*`).

---

# 5. Goal 1 — Damage detection

**Task `binary`** — Pristine vs any damage. Experimental set: 82.5 % damage,
17.5 % pristine; binary chance balanced accuracy = 0.500.

## 5.1 Recommended training configuration

| element | value |
|---|---|
| feature | `modal` (81-dim) |
| model | `MLP`, hidden (256, 128, 64), dropout 0.2 |
| optimiser | AdamW, lr 3×10⁻³, weight decay 10⁻⁴ |
| schedule / epochs / batch | CosineAnnealing / 4 / 64 |
| loss | cross-entropy |
| preprocessing | P0.3 `StandardScaler` on the experimental-Pristine subset |
| HPO | grid hidden ∈ {(128,64),(256,128,64),(512,256,128)} × lr ∈ {5e-4,1e-3,3e-3}; best of 9 by val accuracy |

## 5.2 Result (seeded, zero-shot)

| synth val | synth test | exp accuracy | exp macro-F1 | exp balanced acc |
|---|---|---|---|---|
| 0.993 | 0.982 | 0.824 | 0.488 | **0.515** |

## 5.3 Full comparison (`_basescore.json`, 16 binary cells)

| model / feature | accuracy | macro-F1 | balanced acc |
|---|---|---|---|
| mlp / modal | 0.825 | 0.482 | 0.513 |
| cnn2d / cfdac_imag | 0.810 | 0.457 | 0.495 |
| cnn2d / cfdac_all | 0.825 | 0.452 | 0.500 |
| cnn / frf_mag, transformer / frf_mag, … | 0.825 | 0.452 | 0.500 |

Every CFDAC cnn2d cell sits at balanced accuracy exactly 0.500 — they
predict "damage" for all 2 638 cases. `transformer/timeseries` reaches
macro-F1 0.496 but on the non-independent `timeseries` feature.

## 5.4 Ablation history

P0/P1 fixes did not move binary: it was at the class-prior floor before and
after. The synthetic damage signature does not separate damaged from
pristine on real data at all.

## 5.5 Verdict

**Does not transfer.** Best genuine-feature balanced accuracy 0.515 ≈
chance. Detection is the goal the synthetic model is *least* able to do —
ironic, since it is nominally the easiest task.

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

## 6.6 Binary-trenchcoat decomposition

Five one-vs-rest binaries (`is_pristine/bolt/crack/hole/mass`) aggregated
into a 5-class prediction via a transductive `dataset_zscore` aggregator.
Best aggregator: macro-F1 ≈ 0.29, above the multi-class baseline. The key
diagnostic: the **`is_Crack` binary has cross-domain AUC 0.36 — below
chance** — synthetic Crack damage is *anti-correlated* with real Crack
(symmetric 4-corner synth model vs per-corner-asymmetric reality). Detail in
[`REPORT_vision_v2.md` § trenchcoat](REPORT_vision_v2.md).

## 6.7 Severity-stratified behaviour

Restricting evaluation to high-severity damage (τ ≥ 0.7) lifts several
cells' accuracy to ≈ 0.66 — but the per-class breakdown shows this is partly
class-distribution shift (the surviving population is Bolt-heavy), not
uniform per-class gain. Detail in
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
| feature | `frf_mag` (381×9 log-magnitude FRF) |
| model | `SmallTransformer`, d_model 32, 2 layers |
| optimiser | AdamW, lr 1×10⁻³, weight decay 10⁻⁴ |
| schedule / epochs / batch | CosineAnnealing / 4 / 64 |
| loss | MSE; **sigmoid-bounded output head** (P0.2) keeps predictions in [0, 1] |
| preprocessing | P1.1 per-sample log + z-score of `frf_mag` |
| HPO | grid d_model ∈ {32,48,64} × n_layers ∈ {1,2}; best by val R² |

## 7.2 Result (seeded, zero-shot)

| synth val R² | synth test R² | exp R² | exp MAE |
|---|---|---|---|
| 0.185 | 0.130 | **+0.006** | 0.272 |

## 7.3 Full comparison (`_basescore.json`, 16 severity cells)

| model / feature | exp R² | note |
|---|---|---|
| cnn / timeseries | +0.180 | **non-independent `timeseries` feature — excluded** |
| cnn2d / cfdac_mag | −0.012 | best genuine feature |
| cnn3d / cfdac3d_realimag | −0.013 | |
| cnn2d / cfdac_real | −0.015 | |
| mlp / modal | −0.027 | |

## 7.4 Ablation history

P0.2 (sigmoid-bounded heads) was essential: before it, MLP regression heads
extrapolated to ±∞ on OOD inputs, giving R² as low as −10²². P0.3
(experimental-Pristine scaler) lifted the modal MLP cell from R² −1.17 to
+0.06 *in-cell*. But the best **genuine-feature** cell still lands at R² ≈ 0:
the fixes made severity *finite and well-posed*, not *transferable*.

## 7.5 Verdict

**Does not transfer.** On every genuine feature synth-only severity R² ≈ 0.
The `timeseries` 0.180 figure is on a feature reconstructed from the FRF
(P0.4) and is not a genuine-feature result.

---

# 8. Goal 4 — Damage location assessment

Two sub-tasks: which **column-end** (`col_location`, 6 classes, chance
balanced acc 0.167) and which **mass-plate** (`mass_location`, 4 classes,
chance 0.250).

## 8.1 Column-end location (`col_location`)

### Recommended configuration

| element | value |
|---|---|
| feature | `cfdac_mag` (128×128 CFDAC magnitude) |
| model | `Conv2DStack` (2-D CNN), widths (16, 32, 64), kernel 3 |
| optimiser | AdamW, lr 1×10⁻³, weight decay 10⁻⁴ |
| schedule / epochs / batch | CosineAnnealing / 4 / 64 |
| loss | cross-entropy |
| preprocessing | P0.1 experimental-Pristine CFDAC reference; P1.1 per-sample mean-subtract |
| HPO | `hpo_cfdac_variants.py` grid widths × kernel (4 configs); best by val accuracy |

### Result & comparison

| synth val | synth test | exp accuracy | exp macro-F1 | exp balanced acc |
|---|---|---|---|---|
| 0.492 | 0.463 | 0.508 | **0.192** | 0.228 |

| model / feature | accuracy | macro-F1 | balanced acc |
|---|---|---|---|
| cnn2d / cfdac_mag | 0.51 | 0.19 | 0.23 |
| mlp / cfdac_real | 0.35 | 0.18 | 0.30 |
| cnn / timeseries | 0.30 | 0.16 | 0.16 |

**Verdict — does not transfer.** Best macro-F1 0.19, balanced accuracy 0.23
vs 0.167 chance. The synthetic crack/hole damage is symmetric per storey, so
the BD-vs-AD column ends are nearly information-theoretically
indistinguishable — a property of the synthetic physics, not the model.

## 8.2 Mass-plate location (`mass_location`)

### Recommended configuration

| element | value |
|---|---|
| feature | `cfdac_real` (128×128 CFDAC real part) |
| model | `Conv2DStack` (2-D CNN), widths (16, 32, 64), kernel 5 |
| optimiser | AdamW, lr 1×10⁻³, weight decay 10⁻⁴ |
| schedule / epochs / batch | CosineAnnealing / 4 / 64 |
| loss | cross-entropy |
| preprocessing | P0.1 experimental-Pristine CFDAC reference; P1.1 per-sample mean-subtract |
| HPO | `hpo_cfdac_variants.py` grid widths × kernel (4 configs); best by val accuracy |

### Result & comparison

| synth val | synth test | exp accuracy | exp macro-F1 | exp balanced acc |
|---|---|---|---|---|
| 0.893 | 0.863 | 0.534 | **0.435** | **0.506** |

| model / feature | accuracy | macro-F1 | balanced acc |
|---|---|---|---|
| cnn2d / cfdac_real | 0.53 | **0.44** | 0.51 |
| cnn2d / cfdac (real+imag) | 0.42 | 0.43 | 0.49 |
| cnn2d / cfdac_imag | 0.39 | 0.42 | 0.49 |
| mlp / modal | 0.37 | 0.25 | 0.26 |

**Verdict — transfers; the one clear synth-only success.** Balanced accuracy
0.51 vs 0.250 chance (≈ 2× chance), and three independent `cnn2d` CFDAC
cells agree at macro-F1 ≈ 0.42–0.44. An added mass-plate shifts the
floor-mode amplitudes by a large, location-specific amount that survives the
sim-to-real gap. P0.1 (experimental-Pristine reference) drove this:
mass_location accuracy 0.282 → 0.534.

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

These numbers are accuracy / R² (not re-scored with macro-F1) and were
produced before the § 3 seeding fix; treat them as indicative. The point
stands: experimental labels in training close the gap that synth-only
cannot. Detail in [`REPORT_simtoreal.md`](REPORT_simtoreal.md) /
[`REPORT_final.md`](REPORT_final.md).

## 9.3 Noise robustness

The full pipeline was re-run on synthetic data corrupted with additive
Gaussian noise on the time series (per-sample, per-channel, controlled SNR)
and on a mixed-SNR variant. Detail in [`REPORT_noise.md`](REPORT_noise.md)
and `REPORT_noisy_mixed.md`. These sweeps pre-date the § 3 corrections.

---

# 10. Limitations

1. **Three of four goals do not transfer** synth-only — detection,
   column-end location and severity are at chance / R² ≈ 0; type only
   weakly above chance. Only mass-plate location is usable.
2. **Deep CFDAC and vision models collapse to the class prior** on `type`
   and `binary`; raw accuracy hides this, balanced accuracy reveals it.
3. **Synthetic Crack is anti-correlated with real Crack** (`is_Crack`
   cross-domain AUC 0.36) — the synthetic damage model is symmetric where
   reality is asymmetric. The same symmetry sinks column-end location.
4. **No multi-seed uncertainty** — every cell is one seeded draw; the
   ≈ 0.05–0.07 macro-F1 run-to-run band is an estimate, not a measurement.
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
2. **Build on mass-plate location** (§ 8.2) — the one transferring goal.
   Confirm reproducibly with the now-seeded `hpo_cfdac_variants.py` and
   characterise *why* it transfers (floor-mode amplitude) as a template.
3. **Fix the synthetic damage physics — recommended next investment.**
   Promote `variation_v2.py` → `variation.py`, regenerate the chunk set
   (P2.1 + P2.2, ≈ 24 h CPU). Asymmetric per-corner Crack/Hole damage
   targets the two biggest failures — the `is_Crack` AUC-0.36
   anti-correlation and the column-end symmetry.
4. **SSL pretrain on unlabelled experimental data** (P2.3, ≈ 6 h) — not run.
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
results/REPORT_simtoreal.md / REPORT_final.md       joint synth+exp fine-tune (uses exp labels)
results/REPORT_vision.md / REPORT_vision_v2.md      vision-backbone sweep + trenchcoat
results/REPORT_severity_stratified.md               severity-threshold analysis
results/REPORT_noise.md / REPORT_noisy_mixed.md      noise-robustness sweeps
```

Modules: `features.py`, `cfdac.py`, `cfdac_variants.py`,
`build_experimental_features.py` (feature build); `hpo.py`,
`hpo_cfdac_variants.py`, `hpo_cfdac_allmodels.py` (synth sweep);
`evaluate_full_experimental.py` (cross-domain eval); `transfer_learn.py`
(joint fine-tune); `build_augmented_chunks.py`, `build_mixed_features.py`
(augmentation); `models.py`, `train.py`, `tasks.py` (shared).
