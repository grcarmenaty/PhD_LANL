> **Method study — bears on the *damage type* and *detection* goals.** A
> focused study, not a canonical report; the methodology-corrected,
> goal-structured results are in [`REPORT_definitive.md`](REPORT_definitive.md)
> and [`REPORT_full.md`](REPORT_full.md). It asks how cross-domain accuracy
> behaves when the evaluation is restricted to high-severity damage; the
> finding (see `REPORT_full.md` § 6.7) is that the apparent high-severity
> lift is partly a class-distribution shift, not uniform per-class gain.
> Numbers below pre-date the seeding / macro-F1 corrections.

# Does accuracy improve when restricted to extreme cases?

This is **v2** — the first version had only one non-vision model and produced a misleading conclusion. The wider sweep here covers six bespoke non-vision cells (1-D CNN, 2-D CNN with three CFDAC variants, 3-D CNN, Transformer, MLP, RF, XGBoost) plus two vision cells and the trenchcoat aggregator. **The picture is more nuanced than "no, severity doesn't help"** — most bespoke models *do* improve on high-severity cases, sometimes substantially. Whether the improvement is real per-class gain or class-distribution shift depends on the model.

## Setup

Severity is normalised per damage type so `0` = least extreme of that type and `1` = most extreme:

| type  | physical range              | what `τ ≥ 0.99` selects        |
|-------|------------------------------|--------------------------------|
| Bolt  | 5 – 95 % loosening           | ≥ 94.1 % loosening              |
| Crack | 1 – 8 mm                     | ≥ 7.9 mm                        |
| Hole  | 1 – 6 mm                     | ≥ 5.95 mm                       |
| Mass  | 0.1 – 2.5 kg                 | (none — every Mass is 0.458 normalised) |

Experimental severity values are quantised: Bolt 11/20/50/85 %, Crack {5, 8} mm, Hole {4, 6} mm, Mass {1.2} kg. That quantisation produces step-shaped sample-retention curves:

![n_remaining](figures/severity_stratified/n_remaining.png)

Damage cases dropped per threshold band:

| band            | cause                                            |
|-----------------|--------------------------------------------------|
| τ ≈ 0.07        | low-end Bolt at 11 % (normalised 0.067) drops out |
| τ ≈ 0.46        | **Mass disappears entirely** + low Crack (5 mm = 0.571 stays, anything lower drops) |
| τ ≈ 0.55        | mid Bolt drops                                    |
| τ ≈ 0.86        | almost everything except the highest Bolt        |
| τ ≥ 0.99        | only 240 cases left — extreme Bolt + a handful of Crack at 8 mm + Hole at 6 mm |

## Overall accuracy / macro-F1 vs severity

![per-model curves](figures/severity_stratified/per_model_curves.png)

* **What.** Damage-case accuracy and macro-F1 as a function of severity threshold for 12 zero-shot synth-only cells. Solid lines = bespoke (non-vision) models; dashed lines = ImageNet-pretrained vision backbones; the trenchcoat aggregator is also dashed. Grey dotted reference at 0.25 = chance for the 4-class damage subset.
* **What is shown.** Many models climb. A few representative cells reach a clear peak between τ ≈ 0.65 and τ ≈ 0.85:
  * **ResNet50 / cfdac_all (vision)**: 0.63 → 0.69 (+0.06 peak)
  * **1-D CNN / frf_mag**: 0.61 → 0.69 (+0.08 peak)
  * **2-D CNN / cfdac_real**: 0.39 → 0.66 (**+0.27 peak**)
  * **3-D CNN / cfdac3d_realimag**: 0.42 → 0.67 (**+0.25 peak**)
  * **MLP / modal**: 0.45 → 0.60 (+0.15 peak)
  * **Transformer / frf_mag**: 0.28 → 0.52 (+0.24 peak)
  Others stay flat or decline:
  * **2-D CNN / cfdac_mag** (the previously cherry-picked baseline): 0.46 → 0.30 — *gets worse*.
  * **Random Forest / modal**: 0.51 → 0.53 — barely moves.
* **Conclusion.** Severity *does* lift accuracy for ~7 of the 12 cells. The improvement is concentrated in the **τ = 0.55 – 0.85** band. Past τ ≈ 0.86 most models drop because only extreme-Bolt cases survive — a regime that's actually harder for synth-trained models (synth Bolt loosening uses a smooth JSR interpolation that diverges from real bolts at the extremes).

## Per-class breakdown — is the lift real or class-shift?

![per-class breakdown](figures/severity_stratified/per_type_breakdown.png)

Six panels, one per representative cell. Per-true-class accuracy as severity rises.

* **2-D CNN / cfdac_mag** (top-left): the previously cherry-picked baseline. Bolt-recall *decreases* monotonically from 0.74 → 0.42. Crack/Hole/Mass at zero throughout. This was the misleading exemplar in my first version of this analysis.
* **2-D CNN / cfdac_real** (top-middle): the most interesting cell. **Bolt jumps 0.55 → 0.87** between τ = 0.5 and 0.6 — real per-class lift, not population shift. **Crack also lifts 0.20 → 0.30** at the same threshold. Hole stays at 0; Mass disappears past τ = 0.46. *Two different classes show real signal at higher severity.*
* **1-D CNN / frf_mag** (top-right): Bolt at 1.00 throughout, everything else at 0. The model only predicts Bolt and gets nothing else right. Overall accuracy rises only because the surviving population is Bolt-heavy.
* **MLP / modal** (bottom-left): Bolt 0.44 → 0.78 at τ = 0.6 (real lift). Crack 0.48 → 0.35 (declines). Hole drops sharply past τ = 0.6. So MLP gains on Bolt and loses on Crack/Hole — net positive on overall accuracy but split per-class.
* **Random Forest / modal** (bottom-middle): Bolt 0.79 → 0.70 (mild decline). Crack 0.11 → 0.27 (mild lift). RF is roughly insensitive to severity per class.
* **XGBoost / modal** (bottom-right): Bolt 0.50 → 0.70 (real lift at τ = 0.5). Crack 0.58 → 0.40 (declines past τ = 0.55). Mirror of MLP/modal — Bolt up, Crack down.

**Conclusion.** Four distinct patterns emerge:

  1. **Real per-class improvement** (cnn2d / cfdac_real, MLP / modal, XGB / modal): Bolt-recall genuinely rises at higher severity. Plausible mechanism — high-severity Bolt loosening creates a stronger, more distinctive FRF shift that's closer to what synth modelled.
  2. **Mass-dropout effect** (ConvNeXt-T, trenchcoat, 1-D CNN): per-class accuracy is essentially flat; overall accuracy moves only because the class distribution shifts when Mass cases all drop out at τ = 0.46.
  3. **Bolt-degradation** (cnn2d / cfdac_mag): per-class Bolt-recall actually *decreases* with severity. High-severity Bolt FRF diverges from synth-trained features.
  4. **Mixed** (RF / modal): roughly insensitive to severity per class — neither gains nor loses meaningfully.

## Updated answers

**"Does accuracy improve if I judge models only against the most extreme cases?"**

**Yes — for several models, by ≥ 0.10 in overall accuracy.** The biggest gains:

| cell                              | τ = 0   | peak (τ band)    | Δ     |
|-----------------------------------|---------|------------------|-------|
| 2-D CNN / cfdac_real              | 0.39    | **0.66** (τ ≈ 0.7) | +0.27 |
| 3-D CNN / cfdac3d_realimag        | 0.42    | **0.67** (τ ≈ 0.7) | +0.25 |
| Transformer / frf_mag             | 0.28    | **0.52** (τ ≈ 0.7) | +0.24 |
| MLP / modal                       | 0.45    | **0.60** (τ ≈ 0.7) | +0.15 |
| XGBoost / modal                   | 0.40    | **0.54** (τ ≈ 0.7) | +0.14 |
| 1-D CNN / frf_mag                 | 0.61    | **0.69** (τ ≈ 0.7) | +0.08 |
| ResNet50 / cfdac_all (vision)     | 0.63    | **0.69** (τ ≈ 0.7) | +0.06 |
| ConvNeXt-T / cfdac_all (vision)   | 0.39    | 0.48 (τ ≈ 0.8)   | +0.09 |

Three counter-examples:

| cell                  | τ = 0 | far τ  | Δ     |
|-----------------------|-------|--------|-------|
| 2-D CNN / cfdac_mag   | 0.46  | 0.30   | −0.16 |
| Random Forest / modal | 0.51  | 0.53   | +0.02 |
| trenchcoat            | 0.34  | 0.34   | 0.00  |

**"Beyond what point does the accuracy start to improve?"**

For the models that *do* improve, the inflection is at **τ ≈ 0.46** (where Mass cases drop out and the surviving Bolt/Crack/Hole population skews more discriminable). The peak is at **τ ≈ 0.65 – 0.85**. Past τ ≈ 0.86 the surviving population is almost entirely extreme-Bolt cases that the synth-trained models handle worse, so accuracy drops back.

**"Is the lift real or class-distribution shift?"**

Both, depending on the model:
* **Real per-class lift**: 2-D CNN / cfdac_real (Bolt 0.55 → 0.87 *and* Crack 0.20 → 0.30), MLP / modal, XGB / modal — these models actually become better discriminators on severe damage.
* **Pure class-distribution shift**: 1-D CNN / frf_mag, ConvNeXt-T / cfdac_all — per-class accuracy is flat; overall gain is just the Mass-dropout effect.
* **Negative** for cnn2d / cfdac_mag — Bolt recall *decreases* with severity. Worth investigating: probably the synth Bolt-loosening model (a smooth `bolt_jsr_ratio` curve) diverges from real Bolt at high severity.

**"What if I want a usable per-sample threshold?"**

Severity isn't a knob you can apply at inference time (you don't *know* the severity until you've classified the case). The actually-deployable knob is **model confidence**:

![confidence-stratified](figures/severity_stratified/confidence_stratified.png)

Some vision cells (ResNet50/cfdac_all especially) climb in accuracy as you require higher confidence. Useful for deployment thresholding ("trust above conf 0.8, flag below") with the trade-off that sample retention drops fast.

## What's in the repository

```
ml_pipeline/plot_severity_stratified.py        sweep + 4 figures + JSON dump
results/severity_stratified.json               12 models × 21 thresholds
results/figures/severity_stratified/
    per_model_curves.png       12 zero-shot cells, accuracy + macro-F1 vs τ
    per_type_breakdown.png     6 cells, per-true-class accuracy
    n_remaining.png            sample retention vs τ
    confidence_stratified.png  alternative: accuracy vs model confidence
```

Reproducibility:

```bash
python -m ml_pipeline.plot_severity_stratified
```
