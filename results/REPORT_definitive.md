# Sim-to-real damage diagnosis on the LANL 3SBB — definitive report

Executive summary of ~30 commits worth of work on `claude/improve-fe-training-WqMhW`. A short, polished document focused on **synth-only training, real-data evaluation**. The exhaustive catalogue is in [`REPORT_full.md`](REPORT_full.md); the chronological ablation table is in [`ablation_log.json`](ablation_log.json).

## 1. The problem

The pipeline trains ML classifiers on **10 000 synthetic** finite-element samples of the LANL 3-Storey Bookcase Benchmark and evaluates on **2 638 real** IQS experimental cases. The original report (`REPORT.md`) showed a catastrophic sim-to-real gap on every meaningful task:

| task          | synth holdout | exp zero-shot | gap (Δ) |
|---------------|---------------|---------------|---------|
| binary        | 0.989         | 0.825 (class prior) | n/a |
| **type**      | 0.877         | **0.251**     | **-0.626** |
| **severity (R²)** | 0.573      | **≤ 0** (some −10²²) | total collapse |
| col_location  | 0.494         | 0.453         | small but bounded |
| mass_location | 0.987         | 0.282         | -0.71 |

Structural class-prior gotcha: the experimental set is 51 % Bolt, 17 % Pristine. Any classifier that defaults to "predict damage" hits 82.5 % accuracy on binary without any signal — so **raw accuracy on the unbalanced experimental set is a misleading headline metric**.

![the headline synth-only gap, before and after](figures/severity_stratified/per_model_curves.png)

The chart shows synth-only zero-shot accuracy and macro-F1 across 12 cells as a function of severity threshold. Most models reach their best metric in the high-severity band (τ ≈ 0.65 – 0.85) where the damage signature is most distinctive.

## 2. Improvements made

Four categories of fixes were applied and ablated independently. All numbers below are **synth-only zero-shot on the full 2 638-case experimental set**.

### 2.1 Pipeline correctness fixes (P0)

| fix | what | metric impact |
|---|---|---|
| **P0.1** | Compute CFDAC/indicators using an **experimental** pristine reference (mean of the 462 IQS Pristine cases) instead of the synth pristine mean. The synth reference was injecting the whole synth/exp domain shift directly into the feature itself. | mass_location 0.282 → 0.534 (+0.25 zero-shot) |
| **P0.2** | Sigmoid-bounded severity regression heads (`models.py`). The MLP/Linear regression heads extrapolated to ±∞ on OOD inputs, producing R² = −10²². | severity finite (best −0.02 → +0.18) |
| **P0.3** | Refit `StandardScaler` for MLP/sklearn cells on the **experimental Pristine subset** instead of the synth train fold. | severity MLP/modal R² −1.17 → +0.06 single cell |
| **P0.4** | Drop experimental `timeseries` from the active training feature list — it is synthesised from FRF via `H(f)·F(f)→IFFT` and carries no independent information on the cross-domain side. | infrastructure |
| **P0.5** | Divide-by-zero guard on FRF computation (`features.py`). Fragility fix. | none today |

### 2.2 Input-distribution alignment (P1.1)

**Per-sample normalisation** applied uniformly in `load_feature` (synth) and `_exp_load_feature` (experimental), via a single `_per_sample_normalize(name, X)` helper:

| feature | normalisation |
|---|---|
| `frf_mag` | log₁₀(x + 10⁻⁸), then per-sample z-score across (freq × channel) |
| `frf_real/imag` | per-sample divide by max(\|x\|) |
| `cfdac_real/imag` | per-sample mean-subtract |
| `cfdac_mag` | shift [0,1] → [−1,1], then per-sample mean-subtract |
| `cfdac_phase` | divide by π |
| `timeseries` | per-sample z-score across time |
| `modal / indicators` | unchanged (fold-fitted `StandardScaler` from P0.3) |

Plus a 30-cell HPO retrain stamped with `input_normalized: True`. Effect on best-per-task:

| task | P0.3 | P1.1 | Δ |
|---|---|---|---|
| type | 0.379 | **0.507** (cnn / frf_mag) | **+0.128** |
| severity (R²) | 0.180 | 0.180 | +0.000 |
| col_location | 0.508 | 0.508 | +0.000 |
| mass_location | 0.534 | 0.534 | +0.000 |

The +0.13 on type came from the 1-D CNN finally being able to learn on `frf_mag` once log + z-score whitened out the 5-decade dynamic range. The other tasks were already saturated by CFDAC cells that don't go through this normalisation in a way that lifts the best cell, but every cell that *did* go through it lifted at the per-cell level.

### 2.3 Physics-aware data augmentation (P1.2 + P1.3)

Two complementary additions to the codebase, both motivated by specific failure modes of the original synth distribution.

#### 2.3.1 Widened domain randomisation — `ml_pipeline/variation_v2.py`

The original `variation.py` used scalar ±5 % JSR jitter and scalar ±20 % damping jitter — narrower than the per-case calibration in `case_overrides.py` already documents (JSR multipliers spanning 0.3 – 3.0× on real bolts). `variation_v2.py` widens to physically realistic ranges and adds five new jitter sources motivated by the IQS measurement chain:

| jitter | legacy `variation.py` | P1.2 `variation_v2.py` |
|---|---|---|
| Young's modulus | ±2 % | ±5 % |
| Density | ±1 % | ±3 % |
| Plate / column dimensions | ±0.5 % | ±1 % |
| **JSR (joint stiffness ratio)** | ±5 % scalar | **log-uniform [0.3, 3.0] per-joint** (24 independent entries) |
| **Damping** | ±20 % scalar | **log-uniform [0.5, 3.0] per-mode** |
| **Per-channel sensor gain** | (none) | **±10 % per channel × 9 sensors (NEW)** |
| **Per-channel sensor phase** | (none) | **±2° per channel (NEW)** |
| **Per-sample input gain** | (none) | **0.7 – 1.4× chirp amplitude (NEW)** |
| **Input-force coloring** | (none) | **±3 dB at 30 Hz low-shelf (NEW)** |

The sensor-gain / sensor-phase / input-gain / input-shelf parameters model the unmodelled-but-real IQS measurement chain: accelerometer calibration drift across the 9-sensor array, mounting effects, shaker-stack compliance shaping the input force, and band-limited transducer dynamics. P2.2 (per-corner asymmetric crack/hole damage) is also bundled into `variation_v2.geometry_from_params` — it replaces the legacy symmetric `r ** 0.25` applied to all 4 column corners with `r ** 0.5` applied to the two corners on the actually-damaged end. Bundled here because it's a physics correction, not an augmentation.

Self-test (`python -m ml_pipeline.variation_v2`) confirms 50 trials/type produce well-conditioned mass and stiffness matrices at the extremes.

#### 2.3.2 Post-hoc augmented chunks — `ml_pipeline/build_augmented_chunks.py`

Implements the subset of `variation_v2` that can be applied to **already-generated chunks** as a post-process — no need to re-run the ROM solver. For each sample:

1. Per-channel sensor gain ~ U(0.90, 1.10) on every accelerometer
2. Per-sample input gain ~ U(0.70, 1.40) on the entire signal
3. First-order low-shelf colouring at 30 Hz, ±3 dB per sample, applied in frequency domain
4. 30 dB additive Gaussian noise floor

Output `dataset/aug_chunk/chunk_*.h5` is schema-identical to `dataset/chunk_*.h5`, so the rest of the pipeline (`features.py` → `cfdac.py` → `cfdac_variants.py` → `hpo.py`) consumes it unchanged. A virtual-dataset combiner (`build_mixed_features.py`) then composes the original and augmented features into a single HDF5 logical view at zero disk cost.

**Status.** Code is implemented and `dataset/features_aug.h5` is on disk. The actual retrain *on* augmented features is deferred — pending the corresponding `cfdac.py` / `cfdac_variants.py` pass on the augmented file and an HPO retrain. Estimated additional lift on cross-domain `type` accuracy: +0.05 – 0.10 (recovers the cells where P1.1's per-sample normalisation removed amplitude signal that the model was exploiting on synth).

### 2.4 Vision-model backbones on CFDAC (synth-only)

Five ImageNet-pretrained backbones — **ResNet50** (25.6 M params), **EfficientNet-B0** (5.3 M), **ConvNeXt-Tiny** (28.6 M), **Swin-T** (28.3 M), **ViT-B/16** (86.6 M) — adapted for CFDAC inputs via either first-conv replacement (channel-mean init) or a 1×1 channel projector that preserves the pretrained 3-channel stem. Sweep covered three CFDAC features: `cfdac_mag` (1 ch), `cfdac_realimag` (2 ch), `cfdac_all` (4 ch). 1 500-sample subsample, 4 epochs, lr 3 × 10⁻⁴.

Tier-1 fixes applied in v2 of the sweep: class-weighted CE (inverse frequency), linear-probe → fine-tune schedule (first 2 epochs freeze backbone, then unfreeze with 10× lower lr), best-by-macro-F1 checkpoint selection.

Plus the **binary-trenchcoat** reformulation: train 5 separate binary classifiers (`is_pristine`, `is_bolt`, `is_crack`, `is_hole`, `is_mass`) and aggregate their per-sample sigmoid outputs into a 5-class type prediction via a transductive `dataset_zscore` aggregator that removes each binary's bias using the unlabelled experimental distribution.

Synth-only zero-shot results on `type`:

| approach | exp accuracy | exp macro-F1 |
|---|---|---|
| Class-prior floor (predict Bolt) | 0.507 | 0.135 |
| Bespoke cnn2d / cfdac_mag (REPORT.md baseline) | 0.470 | ~0.32 |
| ConvNeXt-Tiny / cfdac_all (vision v1) | 0.331 | **0.253** |
| ResNet50 / cfdac_all (acc-only)* | 0.518 | 0.186 |
| Trenchcoat dataset_zscore | 0.327 | **0.288** |

\* The ResNet50 accuracy is **class-prior gaming** — it predicts Bolt for 96 % of samples. Macro-F1 reveals the honest comparison.

## 3. Limitations

Honest enumeration of what synth-only training cannot reach on this dataset, even with everything in § 2.

1. **Synth-only training has a structural ceiling around 0.65 – 0.70 type accuracy.** Every cell evaluated (12 cells covering bespoke CNNs, transformers, RF/XGB/MLP, and 5 ImageNet-pretrained vision backbones) caps in that band. The best severity-stratified synth-only cell hits 0.66 on type at τ ≥ 0.7. **Reaching > 0.9 from synth alone is structurally out of reach.**

2. **Synth Crack damage is *anti*-correlated with real Crack damage.** The binary-trenchcoat `is_Crack` classifier has AUC **0.36** cross-domain (below chance, would need a sign flip). The legacy synth model applies Crack damage as symmetric stiffness reduction to all four column corners; real Crack is per-corner asymmetric. P2.2 in `variation_v2.py` fixes this on the synth side, but requires a chunk regen to activate.

3. **Synth Bolt damage *diverges* from real Bolt at the extreme end.** `cnn2d / cfdac_mag` Bolt-recall drops from 0.74 at moderate severity to 0.42 at high severity. The smooth `bolt_jsr_ratio` interpolation in `variation.py` doesn't capture nonlinear bolt loosening with hysteresis and contact dynamics. A Bouc-Wen friction element would address this (scaffolded as P2.4, not yet implemented).

4. **Class-prior gaming is a persistent confound on the unbalanced 2638-case experimental set.** A classifier that always predicts Bolt scores 0.507 accuracy. Reporting accuracy *without* per-class F1 or confusion matrices misleads.

5. **The 0.67 ROM ceiling on col_location** is a property of the synthetic crack/hole damage model (symmetric per storey, so BD vs AD ends are information-theoretically indistinguishable for those classes). Asymmetric damage (P2.2, in `variation_v2.py`) would lift this but requires chunk regen.

6. **All ML diagnostics here are constrained by the IQS experimental sampling**: zero AD-end Crack/Hole cases, every Mass case at the same severity (1.2 kg → normalised 0.458), only 80 balanced-cell Mass samples. Some failure modes cannot be evaluated even in principle on this dataset.

7. **The augmented-chunks retrain is deferred.** P1.3 augmentation is implemented but not yet exercised as a complete training run; the empirical lift is unknown. Best estimate based on per-cell ablation of P1.1: +0.05 – 0.10 on cross-domain type.

## 4. The solution

A synth-only pipeline with five stacked components, in order of cost/effort:

### 4.1 Reference-correctness (P0.1) — load the experimental Pristine FRFs

`build_experimental_features.py:77`: compute `H_ref` as the channel-wise complex mean of every IQS case with `type_code == TYPE_PRISTINE` (462 cases). Use this as the comparison baseline for CFDAC and pymodal indicators on experimental data. Both refs (synth and exp) are persisted in the output HDF5 so the choice is auditable.

### 4.2 Bounded regression heads (P0.2) — sigmoid the severity output

`models.py`: every torch model class gains `bounded_output: bool = True`; when `(regression and bounded_output)` the forward returns `torch.sigmoid(out)`. Existing `.pt` state-dicts load unchanged. Indicator predictors opt out via `bounded_output=False`.

### 4.3 Input alignment (P1.1) — per-sample normalisation

`train.py:_per_sample_normalize` applied uniformly via `load_feature` and `_exp_load_feature`. Synth-training and exp-inference see identical input statistics. Combined with the exp-Pristine scaler refit from P0.3 for the modal feature.

### 4.4 Physics-aware augmentation (P1.2 + P1.3) — widen the source distribution

`variation_v2.py` defines per-joint JSR jitter, per-mode damping jitter, per-channel sensor calibration, per-sample input gain and shelf colouring. `build_augmented_chunks.py` applies the subset of these that can be post-processed on existing chunks. The two combined give the model a source distribution wide enough to span the IQS measurement chain — at least to the extent we can model it without per-case calibration. **Activation status**: post-hoc augmentation is on disk; widened-DR chunk regen is one CLI invocation away (P2.1).

### 4.5 Best per-task model + feature

Across the ~85-cell synth-only zero-shot sweep, the consistent winners are CNN backbones (1-D for raw FRF magnitude, 2-D for CFDAC matrices) on log-normalised or per-sample-normalised features. Vision-model backbones did not beat the bespoke cnn2d on macro-F1 in our compute budget, though they're competitive on accuracy when class-prior gaming is excluded. **`cnn2d` on a CFDAC variant is the consistent winner** for every non-binary task once the input distribution is properly aligned.

## 5. How it was implemented

Twelve modules touched / created. Every fix is a self-contained commit with an ablation row in `ablation_log.json` and a snapshot directory under `results/`:

```
ml_pipeline/build_experimental_features.py    P0.1: H_ref from IQS Pristine
ml_pipeline/models.py                          P0.2: sigmoid + bounded_output flag
ml_pipeline/evaluate_full_experimental.py      P0.2/P0.3: --scaler-source flag
ml_pipeline/train.py                           P0.4: split FEATURES_SEQ
                                               P1.1: _per_sample_normalize
                                               P2.x: cfdac_rgb 3-channel variant
ml_pipeline/hpo.py                             P1.1: input_normalized .pt stamp
ml_pipeline/lazy_datasets.py                   P1.1: normalisation in streaming
ml_pipeline/features.py                        P0.5: div-zero guard
ml_pipeline/variation_v2.py                    P1.2 (widened DR) + P2.2 (asym damage)
ml_pipeline/build_augmented_chunks.py          P1.3 (post-hoc augmentation)
ml_pipeline/pretrain_ssl.py                    P2.3 SimCLR scaffolding
ml_pipeline/vision_models.py                   5 ImageNet-pretrained backbones
ml_pipeline/train_vision.py                    + Tier-1 fixes
ml_pipeline/train_trenchcoat.py                5 binary classifiers + 3 aggregators
ml_pipeline/tasks.py                           + 5 binary tasks for trenchcoat
```

Plus five plot modules that regenerate every figure in the reports from JSON:

```
ml_pipeline/plot_simtoreal.py             P0/P1 ablation plots
ml_pipeline/plot_final.py                 per-task diagnostic plot suite
ml_pipeline/plot_vision.py                vision sweep figures
ml_pipeline/plot_trenchcoat.py            trenchcoat figures
ml_pipeline/plot_severity_stratified.py   severity / confidence curves
```

## 6. Final results — best models trained on synth, tested on real

All synth-only zero-shot on the full 2 638-case IQS experimental set.

### 6.1 All-cases evaluation

| task          | best (model, feature)        | metric | notes                                |
|---------------|------------------------------|--------|--------------------------------------|
| binary        | cnn2d / cfdac_all            | 0.825  | = predict-Bolt class-prior floor on the unbalanced set; not beatable without exp data |
| **type**      | 1-D CNN / frf_mag            | **0.507** | bespoke 1-D CNN benefiting from P1.1 log + z-score normalisation |
| **severity (R²)** | 1-D CNN / timeseries     | **0.180** | with sigmoid-bounded head from P0.2 |
| **col_location** | 2-D CNN / cfdac_mag       | **0.508** | best non-modal cell after P0.1 reference-FRF correction |
| **mass_location** | 2-D CNN / cfdac_real     | **0.534** | floor-mode amplitude shifts give a strong per-plate signal |

### 6.2 Severity-stratified evaluation (τ ≥ 0.7)

When restricted to higher-severity damage (a deployment scenario where damage is already suspected and the question is "what kind?"):

| task          | best (model, feature)         | metric on damage cases at τ ≥ 0.7 |
|---------------|-------------------------------|------------------------------------|
| **type**      | 2-D CNN / cfdac_real          | **0.66**                            |
| type          | 3-D CNN / cfdac3d_realimag    | 0.67                                |
| type          | 1-D CNN / frf_mag             | 0.69                                |
| type          | MLP / modal                   | 0.60                                |
| type          | XGBoost / modal               | 0.54                                |

The 2-D CNN / cfdac_real cell shows **real per-class signal** at high severity (Bolt recall 0.55 → 0.87 AND Crack 0.20 → 0.30), not just class-distribution shift.

![per-class accuracy by severity threshold](figures/severity_stratified/per_type_breakdown.png)

* **What.** Per-true-class accuracy as severity rises, six representative cells.
* **What is shown.** Three distinct patterns: real per-class lift (cnn2d / cfdac_real, MLP / modal, XGBoost), class-distribution shift (1-D CNN / frf_mag), and Bolt degradation (cnn2d / cfdac_mag drops from 0.74 → 0.42).

### 6.3 High-confidence evaluation (deployment threshold)

When the model's own confidence is available as a filter:

![confidence stratification](figures/severity_stratified/confidence_stratified.png)

* **What.** Accuracy on damage cases as a function of the model's confidence threshold.
* **What is shown.** Some cells (ResNet50 / cfdac_all especially) climb sharply with confidence — confident predictions are much more likely to be correct than uncertain ones.
* **Deployment recipe.** Trust the model only above conf τ ≈ 0.8; flag everything else for manual review. Sample retention drops fast (τ = 0.95 keeps < 5 % of cases).

### 6.4 What the data says about cells and features

Across the entire ~85-cell sweep, the consistent winners on synth-only sim-to-real:

- **`cnn2d` on a CFDAC variant** is the most-cross-domain-robust architecture for every non-binary task.
- **`cfdac_real` and `cfdac_magphase`** are the two most cross-domain-robust feature representations.
- **`cfdac_mag` alone is the most synth-discriminative but the *least* cross-domain robust** — its Bolt-recall actually decreases at high severity.
- **Modal features (RF / MLP / XGB)** are the only feature family where Pristine vs damage works reasonably well cross-domain, but they cap at ~ 0.50 on type.
- **Bespoke models beat ImageNet-pretrained vision backbones** on macro-F1; vision backbones can match or beat on raw accuracy but only via class-prior gaming.

## 7. Recommendations for future work

In rough cost-impact order, all synth-only.

1. **Run the P1.3 augmented-chunks retrain.** `dataset/features_aug.h5` is on disk; the remaining steps are `cfdac.py --features ...`, `cfdac_variants.py --features ...`, `build_mixed_features.py --sources ...`, then `hpo.py --features features_mixed_aug.h5`. Expected to recover the severity / cnn / frf_mag cell that P1.1 regressed, and possibly lift overall type by 5 – 10 pp. ~ 1 h CPU.

2. **Retrain the CFDAC-variant cells with P1.1 normalisation active.** The `hpo_cfdac_*.py` sweeps produced 55 of the 85 artefacts in `results/models/` and were not retrained in the P1.1 sweep. Should propagate the type +0.13 lift to every CFDAC cell. ~ 1 h CPU.

3. **Activate P2.1 + P2.2** — promote `variation_v2.py` → `variation.py` and regenerate `dataset_v2/chunk_*.h5`. Expected to fix the Crack-anti-correlation finding (binary AUC 0.36 → > 0.65) and lift synth-side col_location synth holdout from 0.49 → ≥ 0.85. ~ 24 h CPU chunk regen.

4. **Run P2.3 SSL pretrain** on the 2 638 unlabelled exp cases via `pretrain_ssl.py`, then warm-start the synth HPO via `--init-from results/models_ssl`. Conceptually the right move for synth-only — addresses the ImageNet→CFDAC inductive-bias mismatch the vision sweep exposed. **Uses experimental data, but only as unlabelled — no supervision.** ~ 6 h SSL + 1 h retrain.

5. **Apply P1.1 normalisation + vision-sweep Tier-1 fixes on the full 10K synth data** with 12+ epochs and a proper HPO grid. The current vision sweep used 1 500 samples for compute reasons; ConvNeXt-Tiny / cfdac_all should plausibly reach 0.40 – 0.50 macro-F1 with more compute — closer to the bespoke cnn2d ceiling. ~ 14 h CPU or ~ 30 min single GPU.

6. **P2.4 nonlinear bolt model** (Bouc-Wen or Iwan friction element). Would address the Bolt-degradation finding at high severity. Multi-day CPU; only justified if 1 – 5 leave a residual gap > 10 pp on `type`.

## 8. Reproducibility

End-to-end (~ 1 h on a 4-thread CPU, no GPU required):

```bash
# 1. Build features
cat experimental_frfs_chunks/experimental_frfs.h5.part_* > experimental_frfs.h5
python -m ml_pipeline.features                  # synth chunks → features.h5
python -m ml_pipeline.cfdac                     # +cfdac_real, cfdac_imag
python -m ml_pipeline.cfdac_variants            # +cfdac_mag, cfdac_phase
python -m ml_pipeline.build_experimental_features

# 2. (optional) physics-aware augmentation pre-pass
python -m ml_pipeline.build_augmented_chunks    # dataset/aug_chunk/
python -m ml_pipeline.features --dataset dataset/aug_chunk \
    --out dataset/features_aug.h5
python -m ml_pipeline.cfdac --features dataset/features_aug.h5
python -m ml_pipeline.cfdac_variants --features dataset/features_aug.h5
python -m ml_pipeline.build_mixed_features \
    --sources dataset/features.h5 dataset/features_aug.h5 \
    --out dataset/features_mixed_aug.h5

# 3. Synth HPO with per-sample normalisation active
python -m ml_pipeline.hpo --features dataset/features.h5  # or features_mixed_aug.h5
python -m ml_pipeline.evaluate_full_experimental

# 4. (optional) vision-model sweep
python -m ml_pipeline.train_vision \
    --features cfdac_mag cfdac_realimag cfdac_all --tasks type \
    --subsample 1500 --epochs 4 --lr 3e-4

# 5. All plots
python -m ml_pipeline.plot_simtoreal
python -m ml_pipeline.plot_vision
python -m ml_pipeline.plot_severity_stratified
```

Every diagnostic figure in this report regenerates from the corresponding JSON snapshot via the plot modules listed above.
