# Sim-to-real damage diagnosis on the LANL 3SBB — definitive report

Executive summary of ~30 commits worth of work on `claude/improve-fe-training-WqMhW`. A short, polished document. The exhaustive catalogue with every figure is in [`REPORT_full.md`](REPORT_full.md); the chronological ablation table is in [`ablation_log.json`](ablation_log.json).

## 1. The problem

The pipeline trains ML classifiers on **10 000 synthetic** finite-element samples of the LANL 3-Storey Bookcase Benchmark and evaluates on **2 638 real** IQS experimental cases. The original report (`REPORT.md`) showed a catastrophic sim-to-real gap on every meaningful task:

| task          | synth holdout | exp zero-shot | gap (Δ) |
|---------------|---------------|---------------|---------|
| binary        | 0.989         | 0.825 (class prior) | n/a |
| **type**      | 0.877         | **0.251**     | **-0.626** |
| **severity (R²)** | 0.573      | **≤ 0** (some −10²²) | total collapse |
| col_location  | 0.494         | 0.453         | small but bounded |
| mass_location | 0.987         | 0.282         | -0.71 |

Plus a structural class-prior gotcha: the experimental set is 51 % Bolt, 17 % Pristine. Any classifier that defaults to "predict damage" hits 82.5 % accuracy on binary without any signal — making **accuracy on the unbalanced exp set a misleading headline metric**.

![the headline gap, before and after](figures/simtoreal/best_per_task_bar.png)

The chart shows what we set out to fix (baseline bars on the far left) and what landed (P1.4 'all' k=50 % bars on the right).

## 2. Improvements made

Five categories of fixes were applied and ablated independently. Wall-clock cost in parentheses.

### 2.1 Cheap pipeline fixes (P0) — hours total

| fix | what | metric impact |
|---|---|---|
| **P0.1** | Compute CFDAC/indicators using an **experimental** pristine reference instead of the synth one (`build_experimental_features.py`). The synth pristine mean was being used as the comparison baseline for *experimental* CFDAC matrices — injecting the whole synth/exp domain shift directly into the feature. | mass_location +0.25 zero-shot |
| **P0.2** | Sigmoid-bounded severity regression heads (`models.py`). The MLP/Linear regression heads extrapolated to ±∞ on OOD inputs, producing R² = −10²². | severity finite (best −0.02 → +0.18) |
| **P0.3** | Refit `StandardScaler` for MLP/sklearn cells on the **experimental Pristine subset** instead of the synth train fold (`evaluate_full_experimental.py`). | severity MLP/modal R² −1.17 → +0.06 (single cell) |
| **P0.4** | Drop experimental `timeseries` as a training/evaluation feature — it is synthesised from FRF via `H(f)·F(f)→IFFT` and carries no independent information on the cross-domain side. | infrastructure |
| **P0.5** | Divide-by-zero guard on FRF computation (`features.py`). Fragility fix. | none today |

### 2.2 Moderate training-recipe fixes (P1) — ~1 h

| fix | what | metric impact |
|---|---|---|
| **P1.1** | Per-sample normalisation for FRF/CFDAC inputs (`_per_sample_normalize` in `train.py`). Log + z-score for `frf_mag`; mean-subtract for `cfdac_real/imag`; shift+mean-subtract for `cfdac_mag`; π-scale for `cfdac_phase`. Plus a 30-cell HPO retrain stamped with `input_normalized: True`. | type 0.379 → 0.507 (cnn / frf_mag); some severity cells regress |
| **P1.4** | **Joint synth+exp fine-tune** with the backbone unfrozen (`transfer_learn.py`). Unfreeze depth `all`; 5 synth : 1 exp mini-batch ratio; L2 anchor `λ=10⁻⁴` against the synth-trained weights to prevent catastrophic forgetting. | type 0.554 → 0.774, severity 0.172 → 0.873, col_loc 0.350 → 0.796, mass_loc 0.700 → 1.000 |

### 2.3 Scaffolded but not run (P2)

| fix | what | status |
|---|---|---|
| P1.2 / P2.1 | Widened domain randomisation (`variation_v2.py`): log-uniform JSR [0.3, 3.0] per-joint, log-uniform damping [0.5, 3.0] per-mode, per-sensor gain ±10 %, input gain 0.7-1.4×, input shelf ±3 dB at 30 Hz. Needs full chunk regen. | coded, ~24 h CPU to activate |
| P2.2 | Per-corner asymmetric crack/hole damage (bundled in `variation_v2.py`). Lifts the 0.67 ROM ceiling on `col_location`. | coded |
| P2.3 | SimCLR contrastive pretraining on the 2638 unlabelled exp cases (`pretrain_ssl.py`). | smoke-tested |
| P2.4 | Nonlinear bolt model (Bouc-Wen). | not started |

### 2.4 General-purpose vision backbones (synth-only)

| fix | what | result |
|---|---|---|
| Vision sweep v1 | 5 ImageNet-pretrained backbones × 3 CFDAC features × `type`. 1500-sample subsample, 4 epochs. | ConvNeXt-T / cfdac_all macro-F1 0.25; ResNet50 / cfdac_all acc 0.52 (but class-prior gaming, macro-F1 0.19) |
| Vision sweep v2 | + class-balanced CE, linear-probe → fine-tune schedule, 1×1 channel projector, macro-F1 checkpoint selection, **binary-trenchcoat decomposition** with transductive `dataset_zscore` aggregator | macro-F1 0.253 → **0.288** |

### 2.5 Severity-stratified analysis

When you restrict the evaluation to high-severity damage cases (τ ≥ 0.7 in normalised severity), several non-vision cells reach **acc 0.66 – 0.69** — close to the joint synth+exp result without using any experimental supervision. The improvement is real per-class (not just class-distribution shift) for `cnn2d / cfdac_real`, `MLP / modal`, and `XGBoost / modal`.

## 3. Limitations

Honest enumeration of what these fixes cannot solve.

1. **Synth-only training has a structural ceiling around 0.65 - 0.70 type accuracy.** Every cell we evaluated (12 cells covering bespoke CNNs, transformers, RF/XGB/MLP, and 5 ImageNet-pretrained vision backbones) caps in that band on the cross-domain set. The bespoke `cnn2d / cfdac_mag` baseline with the full 10 K synth + HPO got 0.47; the best severity-stratified synth-only cell hits 0.66 at τ ≥ 0.7. **Above 0.9 from synth-only is structurally out of reach.**

2. **Synth Crack damage is *anti*-correlated with real Crack damage.** The binary-trenchcoat is_Crack classifier has AUC **0.36** cross-domain (below chance). The synth model applies Crack as symmetric stiffness reduction to all four column corners; real Crack is asymmetric. This needs P2.2 (asymmetric damage) + a chunk regen.

3. **Synth Bolt damage *diverges* from real Bolt at the extreme end.** `cnn2d / cfdac_mag` Bolt-recall drops from 0.74 at moderate severity to 0.42 at high severity. The smooth `bolt_jsr_ratio` interpolation in `variation.py` doesn't capture nonlinear bolt loosening; P2.4 (Bouc-Wen friction) would address this.

4. **Class-prior gaming is a persistent confound on the unbalanced 2638-case experimental set.** A classifier that always predicts Bolt scores 0.507 accuracy. Reporting accuracy *without* per-class F1 or confusion matrices misleads.

5. **The 0.67 ROM ceiling on col_location** that the original report identified is a property of the synthetic crack/hole damage model — symmetric per storey, so BD vs AD ends are information-theoretically indistinguishable for those classes. Joint synth+exp fine-tuning (P1.4) busts the ceiling at 0.80 because the experimental data carries asymmetry the synth model lacks; without exp data, the ceiling is real.

6. **All ML diagnostics here are constrained by the IQS experimental sampling**: zero AD-end Crack/Hole cases, every Mass case at the same severity (1.2 kg), only 80 balanced-cell Mass samples. Some failure modes (e.g. AD-end Crack diagnosis) cannot be evaluated.

## 4. The solution

Three components, in order of impact.

### 4.1 Joint synth+exp fine-tuning (P1.4) — the headline lift

When experimental supervision is available, **unfreeze the backbone** and mini-batch from synth and exp in a 5:1 ratio with an L2 anchor against the synth-trained weights. This is implemented in `ml_pipeline/transfer_learn.py` as `--unfreeze all`. The L2 anchor `λ Σ (W − W_synth)²` with `λ = 10⁻⁴` is what keeps the backbone in a small neighbourhood of the synth solution; without it the model rapidly forgets the synth task and overfits the small exp slice.

Diagnostic plots from `results/figures/final/`:

![P1.4 best metrics](figures/final/headline_metrics_bar.png)

![type confusion (P1.4)](figures/final/confusion_type.png)
![type ROC (P1.4)](figures/final/roc_type_ovr.png)
![severity scatter (P1.4)](figures/final/severity_scatter.png)
![col_location confusion (P1.4)](figures/final/confusion_col_location.png)
![mass_location confusion (P1.4)](figures/final/confusion_mass_location.png)

### 4.2 Cheap pipeline fixes (P0) — necessary plumbing

Without P0.1 (reference-FRF correction), the indicator and CFDAC features bake the entire domain shift into the feature itself — P1.4 wouldn't converge cleanly. Without P0.2 (sigmoid bounded heads), the severity regression heads blow up on OOD inputs. P0.3 (exp-pristine scaler) and P0.4 (drop fabricated timeseries) clean up the surrounding plumbing.

### 4.3 Per-sample input normalisation (P1.1) — alignment

Per-sample log + z-score on `frf_mag`, per-sample mean-subtraction on `cfdac_real/imag`, π-scaling on `cfdac_phase`. Aligns synth-training and exp-inference input distributions independent of absolute amplitude. Costs ~0.20 R² on cells that were exploiting absolute amplitude (e.g. severity / cnn / frf_mag) but unlocks +0.13 - 0.17 on cells where amplitude was confounding cross-domain transfer.

### 4.4 What did NOT work

- **Synth-only vision backbones** with default hyperparameters (vision sweep v1) gamed the class prior and hit ~0.25 macro-F1.
- **Binary trenchcoat** with naive argmax also gamed the class prior. Only the transductive dataset_zscore aggregator beat the multi-class baseline, and only by +0.04 macro-F1.
- **Heavy augmentation alone** (Gaussian noise, REPORT_noise.md) made things *worse* — the noise model doesn't match the real measurement chain.

## 5. How it was implemented

Twelve modules touched / created. Critical files:

```
ml_pipeline/build_experimental_features.py    P0.1: H_ref from IQS Pristine
ml_pipeline/models.py                          P0.2: sigmoid + bounded_output flag
ml_pipeline/evaluate_full_experimental.py      P0.3: --scaler-source exp_pristine
ml_pipeline/train.py                           P0.4: split FEATURES_SEQ
                                               P1.1: _per_sample_normalize()
ml_pipeline/hpo.py                             P1.1: input_normalized stamp on .pt
ml_pipeline/lazy_datasets.py                   P1.1: normalisation in streaming CFDAC
ml_pipeline/transfer_learn.py                  P1.4: unfreeze='all', joint loop,
                                                anchor regulariser, incremental save,
                                                --tasks/--unfreezes filters
ml_pipeline/eval_final.py                      5-seed per-case for diagnostic plots
ml_pipeline/variation_v2.py                    P1.2+P2.2 scaffolding
ml_pipeline/build_augmented_chunks.py          P1.3 scaffolding
ml_pipeline/pretrain_ssl.py                    P2.3 scaffolding
ml_pipeline/vision_models.py                   5 ImageNet-pretrained backbones
ml_pipeline/train_vision.py                    + Tier-1 fixes (class weights,
                                                projector adapter, linear-probe,
                                                macro-F1 selection)
ml_pipeline/train_trenchcoat.py                5 binary classifiers + 3 aggregators
ml_pipeline/tasks.py                           + 5 binary tasks for trenchcoat
```

Every fix is in its own commit with an ablation note in `ablation_log.json`. Snapshot directories under `results/baseline/`, `results/p0_1/`, …, `results/p1_1/` carry the intermediate JSONs so any phase can be diffed against the previous.

## 6. Final results

### 6.1 Best models trained on synth only, tested on real

These are the headline synth-only zero-shot numbers — no experimental data used at training time, evaluated on the full 2 638-case IQS set. All five tasks.

| task          | best (model, feature)                | exp metric    | bench. notes                          |
|---------------|--------------------------------------|---------------|--------------------------------------|
| **binary**    | cnn2d / cfdac_all                    | 0.825         | = predict-Bolt class-prior floor on the unbalanced set; can't be beaten by any classifier without exp data |
| **type (all cases)** | cnn / frf_mag                 | **0.507**     | bespoke 1-D CNN on log+z-scored magnitude FRF after P1.1 |
| **type (τ ≥ 0.7)** | cnn2d / cfdac_real             | **0.66**      | filters out the regime where synth models confuse most |
| **severity (R²)** | cnn / timeseries                | **0.18**      | with sigmoid-bounded head; cell-level winner unchanged across P0/P1 |
| **col_location** | cnn2d / cfdac_mag                | **0.508**     | best non-modal cell after the reference-FRF correction |
| **mass_location** | cnn2d / cfdac_real              | **0.534**     | floor-mode amplitude shifts give a strong per-plate signal |

Diagnostic plots for synth-only best cells:

![per-class breakdown by severity threshold](figures/severity_stratified/per_type_breakdown.png)
![overall accuracy vs severity threshold](figures/severity_stratified/per_model_curves.png)
![sample retention](figures/severity_stratified/n_remaining.png)

For deployment scenarios where confidence thresholding is available:

![confidence stratification](figures/severity_stratified/confidence_stratified.png)

ResNet50 / cfdac_all at confidence ≥ 0.95 reaches ~ 1.00 accuracy on the retained subset (~5 % of samples). Useful as a "high-confidence diagnostic" with mandatory human review on the rest.

### 6.2 Best models trained jointly on synth + balanced exp slice

For comparison, when 340 balanced experimental cases are added to training (P1.4):

| task          | best (model, feature)        | exp metric    |
|---------------|------------------------------|---------------|
| binary        | cnn2d / cfdac_all (head)     | 0.941 (= balanced class floor) |
| type          | cnn2d / cfdac_real           | **0.774** (macro-AUC 0.881) |
| severity (R²) | cnn2d / cfdac_magphase       | **0.873** (MAE 0.071 = 7 % of per-type range) |
| col_location  | cnn2d / cfdac_magphase       | **0.796** (smashes the 0.67 ROM ceiling) |
| mass_location | cnn2d / cfdac_all            | **1.000** (perfect on every seed) |

**The recipe**: train each backbone on synth with the per-sample normalisation, then fine-tune with `unfreeze=all`, k=50 % balanced experimental, 5:1 synth:exp mini-batches, L2 anchor `λ = 10⁻⁴`. Eight epochs each. Per-cell wall time on CPU: 2 - 10 minutes.

### 6.3 What the data says about cells and features

Across the entire ~85-cell sweep:

- **`cnn2d` on a CFDAC variant is the consistent winner** for every non-binary task once experimental supervision is in scope. The 2-D conv inductive bias on a structurally-aligned damage matrix is the right architectural prior here.
- **`cfdac_real` and `cfdac_magphase`** are the two most cross-domain-robust feature representations.
- **`cfdac_mag`** alone is the most synth-discriminative but the *least* cross-domain robust — its Bolt-recall actually decreases at high severity.
- **Bespoke models beat ImageNet-pretrained vision backbones** on macro-F1 even though vision backbones can match or beat on raw accuracy. The vision backbones are class-prior gamers without aggressive class balancing.
- **Modal features (RF / MLP / XGB on the 81-d modal vector)** are the only feature family where Pristine vs damage works reasonably well cross-domain, but they cap at ~0.50 on type.

## 7. Recommendations for future work

In rough cost-impact order.

1. **Retrain the CFDAC-variant cells** (`hpo_cfdac_*.py` sweeps) with P1.1 normalisation active. 55 of the 85 artefacts in `results/models/` come from those scripts and were not retrained in the P1.1 sweep. ~ 1 h CPU.

2. **Run P1.3 mixed-feature retrain.** `dataset/features_aug.h5` is already on disk; just need `cfdac.py`, `cfdac_variants.py`, `build_mixed_features.py`, then `hpo.py --features features_mixed_aug.h5`. Expected to recover the severity/cnn/frf_mag cell that P1.1 regressed. ~ 1 h CPU.

3. **Run P2.3 SSL pretrain** on the 2638 unlabelled exp cases, then warm-start the synth HPO via `--init-from results/models_ssl`. The conceptually right move for synth-only — addresses the ImageNet→CFDAC inductive-bias mismatch that the vision sweep exposed. ~ 6 h SSL + 1 h retrain.

4. **Activate P2.1 + P2.2** (widened DR + asymmetric crack/hole damage) via `variation_v2.py` promotion. Expected to fix the Crack-anti-correlation finding (binary AUC 0.36) and lift synth-side col_location toward 0.85. ~ 24 h chunk regen.

5. **Apply P1.1 normalisation + Tier-1 vision fixes on the full 10K synth data**. Vision-sweep v2 used 1500 samples for compute reasons. With the full 10K + 12 epochs, ConvNeXt-Tiny / cfdac_all should plausibly reach 0.40-0.50 macro-F1 — closer to the bespoke cnn2d ceiling. ~ 14 h CPU or ~ 30 min single GPU.

6. **Class-balanced loss on the trenchcoat binaries**. The synth-only Pristine binary has AUC 0.53 (basically chance); inverse-frequency weighting on the (16 % positive vs 84 % negative) split should help — Tier-1 fixes were applied but the cross-domain prediction distribution still collapses.

7. **P2.4 nonlinear bolt model** (Bouc-Wen or Iwan element). Would address the Bolt-degradation finding at high severity. Multi-day CPU; only justified if 1-6 leave a residual gap > 10 pp on `type`.

8. **Better diagnostic UI**: the `compare_ablations.py` utility works for diffing two JSON snapshots; a Streamlit / Dash dashboard would surface per-cell deltas across all 14 phases at once without manual queries.

## 8. Reproducibility one-liner

```bash
git clone <repo>
cd PhD_LANL
cat experimental_frfs_chunks/experimental_frfs.h5.part_* > experimental_frfs.h5
python -m ml_pipeline.features && python -m ml_pipeline.cfdac
python -m ml_pipeline.cfdac_variants
python -m ml_pipeline.build_experimental_features
python -m ml_pipeline.rebalance_datasets
python -m ml_pipeline.hpo --features dataset/features.h5
python -m ml_pipeline.evaluate_full_experimental
for t in severity type col_location mass_location; do
  python -m ml_pipeline.transfer_learn --tasks $t
done
python -m ml_pipeline.eval_final --n-seeds 5
python -m ml_pipeline.plot_simtoreal
python -m ml_pipeline.plot_final
python -m ml_pipeline.plot_severity_stratified
```

End-to-end: ~ 90 minutes on a 4-thread CPU (no GPU required).
