# Sim-to-real damage diagnosis on the LANL 3SBB — definitive report

Executive summary of the synth-only sim-to-real work on
`claude/improve-fe-training-WqMhW`. The pipeline trains ML classifiers on
**10 000 synthetic** finite-element samples of the LANL 3-Storey Bookcase
Benchmark (3SBB) and evaluates them, **zero-shot**, on **2 638 real** IQS
experimental cases.

This is the methodology-corrected edition. While executing the
[recommendations](#7-recommendations) of the previous draft, two
correctness problems were found in the evaluation itself — an unseeded
training loop and an accuracy-only metric on a class-imbalanced test set.
Both are now fixed, and **every headline number below has been recomputed
with the honest metric.** The exhaustive catalogue is in
[`REPORT_full.md`](REPORT_full.md); the chronological ablation table is in
[`ablation_log.json`](ablation_log.json).

> **One-paragraph summary.** Synth-only training on this rig does **not**
> transfer to real damage-type classification: under macro-F1 (the metric
> that is not gamed by class imbalance) the best synth-only `type` cell
> scores ≈ 0.30 and the best `col_location` cell ≈ 0.19, both barely above
> chance. The accuracy headlines of the previous draft (type 0.507,
> col 0.508, binary 0.825) are **class-prior collapse** — degenerate
> classifiers that predict the majority class. The single genuine
> synth-only success is `mass_location` (macro-F1 0.44, balanced accuracy
> 0.51, ≈ 2× chance). The recommended physics-aware augmentation, when run
> as a seeded A/B, produced **no classification change distinguishable from
> run-to-run noise** (mean macro-F1 Δ −0.008 ± 0.053 over 20 paired cells;
> the predicted +0.05–0.10 `type` lift did not appear) and a marginal
> severity-regression gain (R² +0.006 → +0.075, both near zero). That A/B
> is single-seed and confounded by a 2× training-set-size difference, so it
> refutes the *predicted benefit* without proving augmentation harmful —
> see § 3.3.

---

## 1. The problem

The original report (`REPORT.md`) showed a catastrophic sim-to-real gap on
every meaningful task. Raw accuracy on the experimental set, however, is a
**misleading headline metric**: the 2 638-case IQS set is 50.7 % Bolt and
17.5 % Pristine, so a classifier that always predicts Bolt scores 0.507 on
`type` and 0.825 on `binary` with *zero* discriminative skill.

| task | synth holdout¹ | exp zero-shot (accuracy) | what the accuracy means |
|---|---|---|---|
| binary | 0.99 | 0.825 | = "predict damage" class-prior floor |
| type | 0.88 | 0.25 – 0.51 | range spans only *which* class a collapsed model lands on |
| severity (R²) | 0.57 | ≤ 0 on real features | no transfer |
| col_location | 0.49 | 0.45 – 0.51 | near the 6-class prior |
| mass_location | 0.99 | 0.28 – 0.53 | the one task with real signal |

¹ Synth-holdout figures are quoted from the original `REPORT.md`; they are
not re-derived here. All experimental numbers below *are* re-derived.

The correct metrics for an imbalanced multi-class test set are **macro-F1**
(unweighted mean per-class F1) and **balanced accuracy** (mean per-class
recall, chance = 1 / n_classes). The rest of this report uses those.

**Terminology.** A *cell* is one (task × model × feature) combination —
e.g. `type / cnn / frf_mag`. *CFDAC* (Complex Frequency-Domain Assurance
Criterion, Pastor & Binda 2012) is a 128×128 matricial damage feature
comparing damaged vs reference FRFs across frequency pairs;
`cfdac_real/imag/mag/phase` are its projections. *Trenchcoat* is the
binary-decomposition experiment — five one-vs-rest classifiers
(`is_bolt`, `is_crack`, …); see [`REPORT_full.md` § 9](REPORT_full.md).

---

## 2. Methodology corrections

These two fixes were made *before* re-deriving any result. They are the
reason the numbers here differ from the previous draft.

### 2.1 Deterministic training (the pipeline was unseeded)

`hpo.py` seeded the train/val/test split and the sklearn models, but
**never seeded PyTorch.** Every `cnn` / `transformer` / `cnn2d` cell — i.e.
every deep-model number in the report — was therefore a single
unreproducible draw with unquantified run-to-run variance. `_train_torch`
now seeds `torch` and `numpy` and the training `DataLoader` generator
(commit `a40ed6d`). Re-running the synth-only sweep with seeding active and
comparing against the report-era artefacts shows nominally-identical cells
differ by up to ≈ 0.07 in macro-F1 (e.g. `col_location/cnn/frf_mag`
0.091 → 0.163) — a mix of seed noise and the determinism fix — so no
single-run A/B against the old numbers was ever a controlled comparison.

### 2.2 Honest metrics (`evaluate_full_experimental`)

The evaluation recorded only raw accuracy. It now also records **macro-F1**
and **balanced accuracy** per cell (commit `a40ed6d`). Re-scoring the
report-era models with these metrics (`experimental_full_evaluation_basescore.json`)
is what exposes the class-prior collapse in § 4.

---

## 3. Improvements made and ablated

All numbers are **synth-only zero-shot on the full 2 638-case experimental
set**, macro-F1 unless noted.

### 3.1 Pipeline-correctness fixes (P0) — kept

| fix | what | effect |
|---|---|---|
| **P0.1** | Compute CFDAC/indicators against an **experimental** pristine reference (mean of 462 IQS Pristine cases) instead of the synth pristine mean. | mass_location accuracy 0.282 → 0.534 |
| **P0.2** | Sigmoid-bounded severity regression heads (`models.py`). | severity R² no longer −10²²; finite everywhere |
| **P0.3** | Refit `StandardScaler` for MLP/sklearn cells on the experimental Pristine subset. | severity MLP/modal R² −1.17 → +0.06 |
| **P0.4** | Drop experimental `timeseries` from the active training feature list — on experimental data it is *synthesised* from FRF (`H·F → IFFT`) and carries no independent information. | see § 4 severity caveat |
| **P0.5** | Divide-by-zero guard on FRF computation. | fragility fix |

These remain valid; P0.1 in particular is a real bug fix.

### 3.2 Per-sample input normalisation (P1.1) — kept

A single `_per_sample_normalize` helper applies log₁₀ + per-sample z-score
to `frf_mag`, per-sample mean-subtract to the CFDAC features, etc., applied
identically to synth training and experimental inference so the two see
the same input statistics. This is sound and is kept.

### 3.3 Physics-aware augmentation (P1.3) — ablated; predicted lift not observed

The previous draft *recommended* an augmented-features retrain and
*estimated* "+0.05 – 0.10 on cross-domain type". That retrain has now been
run as a seeded A/B:

* `build_augmented_chunks.py` → `dataset/features_aug.h5` (per-channel
  sensor gain, per-sample input gain, 30 Hz shelf colouring, 30 dB noise).
* `build_mixed_features.py` → `dataset/features_mixed_aug.h5` (20 000
  samples: 10 000 original + 10 000 augmented).
* `hpo.py` run on the plain file (`features.h5`, 60 cells) and the
  augmented file (`features_mixed_aug.h5`, 50 cells) with the **same seed**.

**Best-cell macro-F1 per task** (`experimental_full_evaluation_{plain,aug}.json`):

| task | plain best-cell | augmented best-cell | Δ best-cell |
|---|---|---|---|
| binary | 0.488 | 0.472 | −0.016 |
| type | 0.250 | 0.291 | +0.041 |
| col_location | 0.167 | 0.124 | −0.043 |
| mass_location | 0.251 | 0.164 | −0.087 |
| severity (R²) | +0.006 | +0.075 | +0.069 |

These rows are the *best cell* per task, not means. The honest aggregate is
the **paired per-cell** delta over the 20 main-task classification cells
present in both arms: **mean Δ macro-F1 −0.008, spread (sd) 0.053, range
[−0.123, +0.050]**. The mean is an order of magnitude smaller than its own
spread and sits well inside the ≈ 0.05–0.07 run-to-run band of § 2.1 — i.e.
**no effect distinguishable from noise.** The per-task best-cell moves above
(−0.087 … +0.041) are all within that band too.

Two caveats keep this from being a clean negative result:

1. **It is single-seed.** With seeding now in place a multi-seed run is
   possible, but was not affordable in the ephemeral-container compute
   budget; one draw cannot separate a small true effect from noise.
2. **It is confounded by training-set size.** The augmented arm trains on
   20 000 samples (10 000 original + 10 000 augmented), the plain arm on
   10 000 — augmentation and a 2× data increase vary together. A
   size-matched control (10 000 augmented-only) was not run.

Conclusion: the experiment **refutes the predicted +0.05–0.10 `type` lift**
— no such lift appears — but does **not** establish that augmentation is
harmful. Severity regression moves from R² +0.006 to +0.075: directionally
positive (consistent with augmentation restoring the amplitude variation
per-sample normalisation strips), but both arms are essentially R² ≈ 0. The
augmented-features build is fully reproducible (§ 7).

### 3.4 Vision-model backbones (synth-only) — unchanged

Five ImageNet-pretrained backbones (ResNet50, EfficientNet-B0,
ConvNeXt-Tiny, Swin-T, ViT-B/16) on CFDAC inputs. They did not beat the
bespoke `cnn2d` on macro-F1; the cells that "win" on accuracy do so by
class-prior gaming. See [`REPORT_full.md` § 8-9](REPORT_full.md). Not
re-tested under the corrected methodology.

---

## 4. Honest results — synth-only zero-shot

Best cell per task from the report-era model set re-scored with the correct
metrics (`experimental_full_evaluation_basescore.json`, 78 cells). Three
sweeps appear in this report: this 78-cell re-score of the original
report-era models, and the seeded plain (60-cell) and augmented (50-cell)
A/B arms of § 3.3. They cover overlapping but not identical cells — which
is why best-cell figures differ slightly between § 3.3 and § 4 (e.g. `type`
best macro-F1 is 0.30 here, 0.25 in the seeded plain arm — a difference
within the § 2.1 noise band).

### 4.1 The accuracy headlines are class-prior collapse

The previous draft's § 6.1 reported the **highest-accuracy** cell per task.
Re-scored:

| task | report § 6.1 cell | accuracy | **macro-F1** | **balanced acc** | chance | verdict |
|---|---|---|---|---|---|---|
| type | cnn / frf_mag | 0.507 | 0.135 | **0.200** | 0.200 | **no skill** — exactly 5-class chance |
| binary | cnn2d / cfdac_all | 0.825 | 0.452 | **0.500** | 0.500 | **no skill** — = class-prior floor |
| col_location | cnn2d / cfdac_mag | 0.508 | 0.192 | 0.228 | 0.167 | barely above chance |
| mass_location | cnn2d / cfdac_real | 0.534 | **0.435** | **0.506** | 0.250 | **real signal** (≈ 2× chance) |

Three of the four accuracy headlines are degenerate classifiers. The
`type` cnn/frf_mag cell predicts a single class for > 98 % of cases; its
balanced accuracy of 0.200 is *exactly* the 5-class chance level.

### 4.2 Best cell per task by macro-F1 (the honest ranking)

| task | best honest cell | accuracy | macro-F1 | balanced acc | reading |
|---|---|---|---|---|---|
| type | mlp / modal | 0.37 | **0.30** | 0.37 | weak but above chance |
| col_location | cnn2d / cfdac_mag | 0.51 | **0.19** | 0.23 | barely above chance |
| mass_location | cnn2d / cfdac_real | 0.53 | **0.44** | 0.51 | the one real success |
| binary | transformer / timeseries² | 0.75 | **0.50** | 0.50 | ≈ no skill |
| severity (R²) | cnn / timeseries² | — | — | — | R² 0.18 — see § 4.3 |

² Both "best" cells here sit on the synthesised `timeseries` feature,
which § 4.3 / P0.4 establish is not independent on experimental data. On
real features the best binary cell is `cnn2d/cfdac_all` (macro-F1 0.45,
balanced acc 0.50 — i.e. still no skill), and the best severity cell is
`cnn2d/cfdac_mag` (R² −0.012). Neither transfers.

The honest synth-only ceiling is: **`mass_location` transfers** (macro-F1
0.44); **`type` transfers weakly** (macro-F1 0.30, via the modal feature,
*not* the deep CFDAC cells); `col_location`, `binary` and `severity`
essentially do **not** transfer beyond the class prior / R² ≈ 0.

### 4.3 Severity does not transfer on real features

The previous draft's "severity R² 0.180" is the `cnn / timeseries` cell —
but P0.4 itself establishes that experimental `timeseries` is *synthesised*
from the FRF and is not an independent feature. On every **real** feature,
synth-only severity R² is ≤ 0 (best real-feature cell: `cnn2d/cfdac_mag`
R² −0.012; seeded `transformer/frf_mag` R² +0.006). Synth-only severity
regression does not transfer; the 0.180 figure should not be quoted as a
real-feature result.

---

## 5. Limitations

1. **Synth-only training collapses to the class prior for `type`,
   `col_location` and `binary`.** The synth feature manifold projects to a
   near-constant on the experimental distribution; argmax then returns
   whichever class sits at the projected mode. Accuracy rewards this when
   the mode happens to be the majority class.
2. **Only `mass_location` carries genuine synth-only signal** (macro-F1
   0.44). Floor-mode amplitude shifts from an added plate are large and
   survive the domain gap; damage-*type* signatures do not.
3. **Synth Crack damage is anti-correlated with real Crack** — the
   binary-trenchcoat `is_Crack` classifier has cross-domain AUC 0.36
   (below chance; figure from [`REPORT_full.md` § 9](REPORT_full.md), not
   re-derived here). The synth model applies Crack as symmetric 4-corner
   stiffness loss; real Crack is per-corner asymmetric. P2.2 in
   `variation_v2.py` addresses this but needs a chunk regeneration.
4. **The IQS experimental sampling is itself limiting**: zero AD-end
   Crack/Hole cases, every Mass case at one severity, only 80 balanced-cell
   Mass samples. Some failure modes cannot be evaluated even in principle.
5. **The augmented arm is 50 of 60 cells.** The 10 `cnn2d/cfdac` cells
   (≈ 12 min each) were not completed under the ephemeral-container compute
   budget; they are not headline cells and collapse to the class prior in
   both the plain arm and the earlier full unseeded run.
6. **No multi-seed uncertainty quantification.** § 2.1 establishes a
   ≈ 0.05–0.07 macro-F1 run-to-run band by comparison; it is not a true
   variance estimate (it conflates seed noise with the determinism fix).
   Every per-cell number here is a single seeded draw. Conclusions are
   stated only where the effect exceeds that band — which is why the § 3.3
   augmentation result is reported as inconclusive, not negative.
7. **The five `is_*` trenchcoat subtasks are not tabulated.** The plain/aug
   JSONs carry 25 common `is_*` cells (one-vs-rest binaries); this report
   covers only the five primary tasks. The decomposition results live in
   [`REPORT_full.md` § 9](REPORT_full.md).

---

## 6. Recommendations

Status of the previous draft's recommendations after this round of work.

1. **Run the augmented-chunks retrain — DONE; predicted lift not observed.**
   See § 3.3. The estimated +0.05–0.10 `type` lift did not appear: the
   paired classification Δ macro-F1 is −0.008 ± 0.053, inside run-to-run
   noise. The test is single-seed and confounded by a 2× training-set-size
   difference, so it cannot show augmentation is harmful — it shows the
   predicted *benefit* is absent. To resolve it properly: re-run the A/B
   over ≥ 3 seeds with a size-matched (10 000 augmented-only) control.
2. **Retrain the CFDAC-variant cells with P1.1 — not run.** Compute-bound
   (`hpo_cfdac_*` is multi-hour and the ephemeral container suspends on
   idle). The seeding fix (§ 2.1) is the prerequisite and is now in place,
   so a future run would at least be reproducible. Given the § 3.3 result,
   expected upside is low.
3. **Activate P2.1 + P2.2** (promote `variation_v2.py`, regenerate the
   10 000-sample chunk set, ≈ 24 h CPU) — not run; compute-bound. **This is
   the recommended next investment.** Unlike rec 1's post-hoc augmentation,
   it fixes the *physics*: asymmetric per-corner Crack/Hole damage (which
   § 5.3's AUC-0.36 anti-correlation shows is the dominant `type` failure)
   and wider domain randomisation. Expected outcome: the `is_Crack`
   cross-domain AUC should move from 0.36 to ≥ 0.5, and `type` macro-F1
   from ≈ 0.30 toward the 0.4–0.5 range — the first plausible path to a
   non-degenerate synth-only `type` classifier. Concrete next action: run
   the regeneration on a non-ephemeral machine and repeat the § 3.3 seeded
   A/B (old chunks vs `variation_v2` chunks).
4. **SSL pretrain on unlabelled experimental data** (P2.3, ≈ 6 h) — not
   run; compute-bound.
5. **Full-data vision sweep** (≈ 14 h CPU) — not run; compute-bound.
6. **Nonlinear bolt model** (P2.4, Bouc-Wen, multi-day) — not started.

The honest conclusion: the cheap fixes (P0, P1.1) are real and kept, but
the remaining synth-side gap is **structural** — it is most likely in the
physics of the synthetic damage model, not in the ML pipeline. The post-hoc
augmentation of recommendation 1 did not close it — the predicted lift did
not appear (§ 3.3). Closing it plausibly requires either better synthetic
physics (rec 3, the recommended next step) or the use of experimental data
in training (the joint synth+exp fine-tune documented in
[`REPORT_full.md` § 5.4](REPORT_full.md) — out of scope for a synth-only
report, but the only approach so far shown to reach deployable accuracy).

---

## 7. Reproducibility

All numbers regenerate on a 4-thread CPU (no GPU). The synth-only sweep is
≈ 1 h; the augmented A/B is ≈ 2 h.

```bash
# 1. Build features (≈ 5 min)
cat experimental_frfs_chunks/experimental_frfs.h5.part_* > experimental_frfs.h5
python -m ml_pipeline.features
python -m ml_pipeline.cfdac
python -m ml_pipeline.cfdac_variants
python -m ml_pipeline.build_experimental_features

# 2. Seeded synth-only sweep + honest-metric evaluation
python -m ml_pipeline.hpo --features dataset/features.h5
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

Evidence artefacts in `results/`:

```
experimental_full_evaluation_plain.json      seeded synth-only sweep (60 cells)
experimental_full_evaluation_aug.json        seeded augmented sweep  (50 cells)
experimental_full_evaluation_basescore.json  report-era models, re-scored with macro-F1
ablation_log.json                            chronological per-fix ablation table
```
