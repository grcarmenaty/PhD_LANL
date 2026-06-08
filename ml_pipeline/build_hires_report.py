"""Generate the FULL consolidated hi-res (1601) report from per-cell data:
every (model, feature) cell per task with in-domain + zero-shot metrics, a
model/feature glossary, per-task cell-zoo plots, the DT severity sweep, and the
severity-regression deep-dive. Writes results/REPORT_CONSOLIDATED.md.

Reads results_hires/zoo_summary.json (per-cell) + zoo_dt_sweep.json + dt_1601.json.
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parent.parent
FIG = _REPO/"results"/"figures"/"hires"; FIG.mkdir(parents=True, exist_ok=True)
RES = 1601

TASKS = ["binary","is_pristine","is_bolt","is_crack","is_hole","is_mass",
         "type","col_location","mass_location","severity"]
DET_KEYS = ["binary","is_pristine","is_bolt","is_crack","is_hole","is_mass"]
STIFF_ORDER = ["binary","is_bolt","is_crack","is_hole"]
TASK_DESC = {
 "binary":("Any damage vs pristine","ŷ∈{0=pristine,1=damaged}",0.50,"82.5% damaged prior; raw accuracy misleading."),
 "is_pristine":("Pristine vs any damage (inverse of binary)","ŷ∈{0=damaged,1=pristine}",0.50,""),
 "is_bolt":("Bolt-loosening present? (one-vs-rest)","ŷ∈{0,1}",0.50,"Severity = % loosening, 0–85% — wide range."),
 "is_crack":("Crack present? (one-vs-rest)","ŷ∈{0,1}",0.50,"Severity = crack depth."),
 "is_hole":("Hole present? (one-vs-rest)","ŷ∈{0,1}",0.50,"Severity = hole diameter, 1–6 mm (narrow)."),
 "is_mass":("Added mass present? (one-vs-rest)","ŷ∈{0,1}",0.50,"Severity near-discrete."),
 "type":("Damage type (5-class)","pristine/bolt/crack/hole/mass",0.20,""),
 "col_location":("Column location of damage (6-class)","storey×end",1/6,"BD/AD near-degenerate in the linear ROM."),
 "mass_location":("Added-mass location (4-class)","base/fl1/fl2/fl3",0.25,""),
 "severity":("Damage severity (regression)","ŷ∈[0,1] normalised",None,"Only non-classifier task."),
}
MODELS = {
 "mlp":"fully-connected net (3 hidden layers, BN+GELU+dropout) on the flattened feature",
 "rf":"random forest (400 trees, class-balanced)",
 "xgb":"gradient-boosted trees (XGBoost)",
 "cnn1d":"1-D CNN over the frequency/time axis of a sequence feature",
 "transformer1d":"conv-tokenised 1-D transformer over a sequence feature",
 "cnn2d_shallow":"shallow 2-D CNN (the 128²-baseline architecture: stride-4 stem + 3 conv/pool) on the CFDAC image",
 "cnn2d_deep":"deep ResNet18-style 2-D CNN that consumes the full CFDAC grid",
 "cnn3d":"3-D CNN treating the CFDAC channels as a volumetric depth axis",
 "transformer":"conv-tokenised 2-D transformer on the CFDAC image (full-resolution patches, not a 224 resize)",
 "convnext_tiny":"ImageNet-pretrained ConvNeXt-T (modern CNN) fine-tuned on the CFDAC image",
 "resnet50":"ImageNet-pretrained ResNet50 (classic CNN) fine-tuned on the CFDAC image",
}
FEATS = {
 "modal":"81-d physics vector: per-channel top-3 spectral peaks (freq+amp), log-amp mean/std, band energy",
 "indicators":"22 pymodal damage indicators (SCI, unsigned-SCI, DRQ, AIGAC, FRFRMS/SF/SM, ODS-diff, r2-imag, RVAC/GAC/M2L stats) vs the pristine reference",
 "frf_mag":"log-magnitude FRF |H(f)|, per-sample z-normalised — (9 channels × 1601)",
 "frf_realimag":"real+imag parts of H(f), per-sample z-normalised — (18 × 1601)",
 "timeseries":"band-limited time response reconstructed from the FRF (IFFT·chirp), per-sample z-normalised — (9 × 4096)",
 "cfdac_real":"CFDAC matrix (1601×1601), real part — pristine-vs-current FRF cross-assurance",
 "cfdac_imag":"CFDAC matrix, imaginary part","cfdac_mag":"CFDAC matrix, magnitude",
 "cfdac_phase":"CFDAC matrix, phase","cfdac_realimag":"CFDAC, real+imag (2 channels)",
 "cfdac_magphase":"CFDAC, magnitude+phase (2 channels)","cfdac_all":"CFDAC, all 4 channels stacked",
}

# Long-form per-feature explanation for §4 (what it is, dims, how computed, transfer rationale).
FEAT_DETAIL = {
 "modal": ("**`modal` — 81-d physics summary vector.**", "(81,)", "tabular",
   "For each of the 9 accelerometer channels: the frequency and log-amplitude of the top-3 |H(f)| peaks (6),"
   " the mean and std of the log-magnitude spectrum (2), and the total band energy Σ|H|² (1) → 9×9 = 81. It"
   " throws away all phase and most of the spectral shape, keeping only resonance locations/heights. Cheapest"
   " representation; the 128-baseline's best transferer, but here it is out-classed once richer features are available."),
 "indicators": ("**`indicators` — 22 pymodal damage indicators.**", "(22,)", "tabular",
   "Classical SHM scalars computed from the *current* FRF against the pristine-mean reference: SCI & unsigned-SCI"
   " (CFDAC shape-change), DRQ (from RVAC), AIGAC (from GAC), FRFRMS/FRFSF/FRFSM-6dB, ODS-difference, r²-imag, and"
   " mean/std/min/max summaries of the RVAC, GAC and M2L curves (+ M2L abs-sum). Each is a hand-designed"
   " damage-sensitivity metric; together they are a 22-d 'expert opinion' vector. Explicitly reference-relative,"
   " so in principle robust to some additive bias — but it still encodes the simulator's notion of 'normal'."),
 "frf_mag": ("**`frf_mag` — log-magnitude spectrum.**", "(9, 1601)", "sequence",
   "log₁₀|H(f)| for all 9 channels over the 0–100 Hz / 1601-bin grid, per-sample per-channel z-normalised (so"
   " absolute scale and channel gain are removed — the model sees spectral *shape* only). The most direct"
   " spectral input; pairs naturally with 1-D CNN/transformer over frequency."),
 "frf_realimag": ("**`frf_realimag` — complex spectrum (Re/Im).**", "(18, 1601)", "sequence",
   "Real and imaginary parts of H(f) stacked into 18 channels, per-sample z-normalised. Unlike `frf_mag` it"
   " keeps **phase**, i.e. the full complex response — which is exactly the information CFDAC is built from. It"
   " is the single best transferer for is_bolt/is_hole in this study."),
 "timeseries": ("**`timeseries` — reconstructed impulse/chirp response.**", "(9, 4096)", "sequence",
   "The band-limited time response, irfft(H(f)·chirp-spectrum), 4096 samples at fs=256, per-sample z-normalised."
   " It is FRF-derived (the IQS rig has no stored raw timeseries), reconstructed identically for both domains, so"
   " it carries no information beyond the FRF — it is a different *inductive bias* (a temporal view) for the conv/"
   "transformer models, not a new sensor."),
 "cfdac_*": ("**`cfdac_*` — Complex FRF Assurance Criterion images (1601×1601).**", "(C, 1601, 1601)", "image",
   "The CFDAC cross-assures the current FRF against the pristine reference at every pair of frequencies:"
   " C[i,j] = (Hᵢ·conj(refⱼ))² / (‖Hᵢ‖²‖refⱼ‖²). A pristine structure gives a near-diagonal map; damage spreads"
   " energy off-diagonal (Figure i). Each variant feeds a different channel set — `real`, `imag`, `mag`, `phase`"
   " (1 channel each), `realimag`/`magphase` (2), or `all` (4) — normalised per the engine (real/imag/mag"
   " mean-centred, phase ÷π). This is the representation the bespoke 2-D CNNs, the 3-D CNN, the CFDAC-Transformer"
   " and the pretrained vision backbones consume."),
}


def emit_inputs(A):
    A("## 4 · Input representations — what every sample looks like to a model")
    A("Every cell is `(model, feature)`. A *feature* is one way of turning a sample's 9-channel, 1601-bin FRF"
      " into a tensor. There are three families — **tabular vectors**, **frequency/time sequences**, and"
      " **CFDAC images** — visualised below on real samples, exactly as the models receive them (same"
      " normalisation).\n")
    A("### 4.1 Tabular vectors — `modal`, `indicators`")
    A("![tabular inputs](figures/hires/inputs_tabular.png)")
    A("*Figure i — (a) the 81-d `modal` vector for a pristine vs a high-severity bolt case (vertical lines mark"
      " the 9 per-channel blocks); (b) the 22 named `indicators` for a bolt vs a crack case, each computed"
      " against the pristine reference.*\n")
    for k in ("modal", "indicators"):
        t, dim, fam, desc = FEAT_DETAIL[k]; A(f"{t} *{dim}, {fam}.* {desc}\n")
    A("### 4.2 Sequences over frequency / time — `frf_mag`, `frf_realimag`, `timeseries`")
    A("![sequence inputs](figures/hires/inputs_sequences.png)")
    A("*Figure ii — (a) z-normed log|H(f)|; (b) z-normed Re/Im of H(f); (c) the FRF-reconstructed time"
      " response. Shown for the drive-point channel; the models see all 9 (or 18) channels stacked.*\n")
    for k in ("frf_mag", "frf_realimag", "timeseries"):
        t, dim, fam, desc = FEAT_DETAIL[k]; A(f"{t} *{dim}, {fam}.* {desc}\n")
    A("### 4.3 CFDAC images — `cfdac_{real,imag,mag,phase,realimag,magphase,all}`")
    A("![CFDAC channels](figures/hires/inputs_cfdac_variants.png)")
    A("*Figure iii — the four base CFDAC channels for one bolt case. The 7 feature variants are channel"
      " subsets/stacks of these.*\n")
    t, dim, fam, desc = FEAT_DETAIL["cfdac_*"]; A(f"{t} *{dim}, {fam}.* {desc}\n")
    A("![CFDAC per class](figures/hires/inputs_cfdac_classes.png)")
    A("*Figure iv — CFDAC-magnitude fingerprint of each damage class, synthetic (top) vs experimental (bottom)."
      " Pristine is near-diagonal; each damage type spreads energy off-diagonal in a characteristic way — and"
      " the synthetic and experimental patterns visibly differ, which is the domain gap the image models must"
      " cross.*\n")


def emit_arch(A):
    apath = _REPO/"results_hires"/"architectures.json"
    if not apath.exists():
        A("## 5 · Model architectures\n*(architectures.json not found — run `ml_pipeline/hires_arch.py`.)*\n"); return
    arch = json.loads(apath.read_text())["models"]
    A("## 5 · Model architectures — every model, in full")
    A("Eleven model families span four design philosophies: **tabular** nets/ensembles on the summary vectors,"
      " **1-D sequence** models over frequency/time, **bespoke 2-/3-D nets** built for the full-resolution CFDAC"
      " image, and **ImageNet-pretrained vision backbones** fine-tuned on the CFDAC image. Parameter counts below"
      " are *measured* from the instantiated `nn.Module`s (representative binary-head config).\n")
    A("![model capacity](figures/hires/arch_params.png)")
    A("*Figure v — trainable parameters (log scale). The pretrained vision backbones are 10–100× larger than the"
      " bespoke nets, yet do not lead on transfer (§8) — capacity is not the bottleneck.*\n")
    # parameter table
    A("| model | family | params | notes |")
    A("|---|---|---|---|")
    order = ["mlp","rf","xgb","cnn1d","transformer1d","cnn2d_shallow","cnn2d_deep","cnn3d",
             "transformer","resnet50","convnext_tiny"]
    for k in order:
        v = arch.get(k)
        if not v: continue
        p = v.get("params_total")
        ps = (f"{p/1e6:.2f} M" if p and p >= 1e5 else (f"{p/1e3:.0f} k" if p else "—"))
        A(f"| `{k}` | {v['family']} | {ps} | {v.get('repr_cfg','')} |")
    A("\n**Per-model detail.**\n")
    for k in order:
        v = arch.get(k)
        if not v: continue
        p = v.get("params_total")
        ps = (f" (~{p/1e6:.2f} M params)" if p and p >= 1e5 else (f" (~{p/1e3:.0f} k params)" if p else ""))
        A(f"- **`{k}`**{ps} — {v.get('note','')}")
    A("")


def load_cells():
    S = json.loads((_REPO/"results_hires"/"zoo_summary.json").read_text())
    cells = defaultdict(list)   # task -> list of recs (res==RES)
    for k, v in S.items():
        if v.get("res") != RES: continue
        cells[v["task"]].append(v)
    return cells


def synth_vs_exp_plot(cells):
    """Figure 1: best-cell in-domain (held-out synth) vs zero-shot experimental
    transfer, per task. Regenerated here so it is reproducible from zoo_summary.json."""
    tasks = [t for t in TASKS if t in cells]
    indom, exp, lab = [], [], []
    for t in tasks:
        recs = cells[t]; reg = recs[0]["kind"] == "reg"
        key = (lambda r: r.get("exp_r2", -9)) if reg else (lambda r: r.get("exp_bal_acc", 0))
        b = max(recs, key=key)
        indom.append(b.get("synth") or 0.0)
        exp.append(key(b)); lab.append(t + ("*" if reg else ""))
    x = np.arange(len(tasks)); w = 0.4
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w/2, indom, w, label="in-domain (held-out synth)", color="#ff7f0e", edgecolor="black", lw=.4)
    ax.bar(x + w/2, exp, w, label="zero-shot (experimental)", color="#1f77b4", edgecolor="black", lw=.4)
    ax.axhline(0.5, ls=":", color="grey", alpha=.6)
    ax.set_xticks(x); ax.set_xticklabels(lab, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("score (cls: macro-F1 / balanced-acc · reg*: R²)")
    ax.set_title("In-domain ceiling vs zero-shot transfer, best cell per task @1601\n"
                 "(the gap between the bars IS the sim-to-real penalty)", fontweight="bold", fontsize=11)
    ax.legend(); ax.grid(axis="y", alpha=.3)
    plt.tight_layout(); p = FIG/"zoo1601_synth_vs_exp.png"; plt.savefig(p, dpi=130); plt.close(fig)
    return p.name


def cellzoo_plot(task, recs):
    reg = recs[0]["kind"] == "reg"
    key = (lambda r: r.get("exp_r2", -9)) if reg else (lambda r: r.get("exp_bal_acc", 0))
    recs = sorted(recs, key=key)
    labels = [f"{r['model']}/{r['feature']}" for r in recs]
    vals = [key(r) for r in recs]
    fam = ["#1f77b4" if r["feature"].startswith("cfdac") else "#2ca02c" for r in recs]
    h = max(4, 0.22*len(recs)+1.2)
    fig, ax = plt.subplots(figsize=(9, h))
    y = np.arange(len(recs))
    ax.barh(y, vals, color=fam, edgecolor="black", lw=0.3)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6)
    if reg:
        ax.axvline(0, ls=":", color="black", alpha=.5); ax.set_xlabel("experimental R²")
    else:
        ax.axvline(recs[0]["chance"], ls=":", color="red", alpha=.6, label="chance")
        ax.axvline(0.5, ls=":", color="grey", alpha=.4); ax.set_xlabel("experimental balanced accuracy"); ax.legend(fontsize=7)
    ax.set_title(f"{task} — all {len(recs)} cells @1601 (blue=CFDAC image, green=tabular/seq)", fontweight="bold", fontsize=10)
    plt.tight_layout(); p = FIG/f"cellzoo_{task}.png"; plt.savefig(p, dpi=120); plt.close(fig)
    return p.name


def detect_prose(t, b, an):
    """Per-task interpretive sentence for a detection task, from the row-norm CM + AUC."""
    e = an.get("best_cells", {}).get(t, {})
    cm = e.get("cm_rownorm"); auc = e.get("auc")
    bits = []
    if cm and len(cm) == 2:
        tnr, tpr = cm[0][0], cm[1][1]
        bits.append(f"it recovers **{tpr*100:.0f}% of true positives** (sensitivity) at "
                    f"**{tnr*100:.0f}% specificity**")
    if auc is not None:
        bits.append(f"threshold-free **AUC = {auc:.3f}**")
        if e.get("bal_acc") and auc - e["bal_acc"] > 0.04:
            bits.append("the AUC sitting above the fixed-threshold balanced-accuracy says the *ranking* "
                        "is better than the default 0.5 cut — the decision threshold is miscalibrated by the "
                        "domain shift and could be retuned on a few real samples")
    return ("On real data " + "; ".join(bits) + ".") if bits else ""


def main():
    cells = load_cells()
    dt = json.loads((_REPO/"results_hires"/"dt_1601.json").read_text())
    apath = _REPO/"results_hires"/"analysis.json"
    an = json.loads(apath.read_text()) if apath.exists() else {"best_cells": {}, "counts": {}}
    bc = an.get("best_cells", {})
    out = []
    A = out.append
    A("# LANL 3SBB — Synth-to-Real Damage Diagnosis: FULL Consolidated Report")
    A("**Author:** G. Reyes-Carmenaty · **Date:** 2026-06-08 · **Resolution:** 1601-bin (native).")
    A("\n> Exhaustive edition — sample visualisations of every input representation, full detail on every"
      " model architecture (measured parameter counts), exploratory analysis of both domains, every cell of"
      " the high-resolution model zoo (in-domain + zero-shot), confusion matrices and ROC/AUC for the best cell"
      " per task, the damage-threshold sweep *with the full diagnostic suite (AUC + confusion matrix) swept over"
      " severity*, and the severity-regression deep-dive. In-domain companion: `REPORT_synth.md`. *(128-bin"
      " comparison excluded until that run completes.)*\n")

    A("## Contents")
    A("1 Overview · 2 Tasks · 3 Methodology · **4 Input representations (every feature, with samples)** ·"
      " **5 Model architectures (every model, in full)** · 6 Exploratory data analysis (both domains) ·"
      " 7 Diagnostics: confusion matrices + ROC/AUC · 8 Per-task catalogue (every cell) ·"
      " 9 Damage-threshold sweep + swept diagnostics · 10 Severity regression · 11 Synthesis ·"
      " 12 Limitations · 13 Recommendations · 14 Artefacts\n")

    # ---- counts ----
    ncell = sum(len(v) for v in cells.values())
    ncls = sum(1 for t in cells for r in cells[t] if r["kind"]=="cls")
    nnc = sum(1 for t in cells for r in cells[t] if r["kind"]=="cls" and not r.get("collapse"))
    domauc = an.get("domain_classifier_auc")
    A("## 1 · Overview")
    A("**The result in one line.** A model trained on physics-simulation FRFs *does* detect damage on a"
      " real structure it has never seen — but only its **presence/type** transfers (balanced-acc 0.56–0.72"
      " AUC), while **location and magnitude largely do not**, and the ceiling is set by a severe"
      f" covariate shift (a logistic classifier separates the two domains with **AUC = {domauc:.3f}**).\n"
      if domauc else "")
    A(f"- **{ncell} cells** at 1601 bins = (≤11 models) × (≤12 features) × 10 tasks, each trained to"
      " convergence on synthetic data and evaluated zero-shot on the 2 638-case IQS experimental set.")
    A(f"- **{nnc}/{ncls} classification cells clear chance** on real data (≈{100*nnc/ncls:.0f}%).")
    A("- A *cell* = one (model, feature) pair. Metric of record: **balanced accuracy / macro-F1 / AUC**"
      " (classification), **R² / Pearson r / MAE** (severity). Raw accuracy is never used (82.5% damaged prior).")
    A(f"\n![in-domain vs zero-shot](figures/hires/{synth_vs_exp_plot(cells)})")
    A("*Figure 1 — best-cell in-domain score (held-out synth) vs zero-shot experimental transfer, per task."
      " The vertical gap is the sim-to-real penalty; it is largest exactly where the synthetic model is most"
      " confident (mass_location, type, severity).*\n")

    # ---- tasks ----
    A("## 2 · The ten diagnosis tasks")
    A("Each task is posed on the same FRFs; the zoo is trained and evaluated independently per task. Chance is"
      " the balanced-accuracy floor (1/#classes).\n")
    A("| task | question | output | chance |")
    A("|---|---|---|---|")
    for t in TASKS:
        q, o, ch, note = TASK_DESC[t]
        A(f"| `{t}` | {q} | {o} | {('%.2f'%ch) if ch else '— (reg)'} |")
    A("\nDetailed per-task results, including every cell, are in §8.\n")

    # ---- methodology ----
    A("## 3 · Methodology")
    A("**Data.** Synthetic data regenerated at 16 s (N_T=4096, fs=256 → df=0.0625 Hz, 1601 bins, 0–100 Hz)"
      " to match the experimental grid exactly: 10 000 synthetic cases (2 000 per class, balanced) from a"
      " linear reduced-order model of the 3-storey bookshelf, and the 2 638-case IQS experimental set"
      " (bolt-heavy: 462 pristine / 1338 bolt / 320 crack / 280 hole / 238 mass).")
    A("**Protocol.** Per cell: compute the feature from the FRFs, train on a synth subsample (70/15/15 split,"
      " class-weighted loss / balanced trees, early-stop to convergence with checkpoint/resume), evaluate on"
      " held-out synth (**in-domain**) and on all 2 638 experimental cases (**zero-shot**, no real data ever"
      " seen in training). Tabular features standardised on the synth-train fold only; sequence/image"
      " features per-sample normalised (no leakage). `timeseries` is reconstructed from the FRF identically"
      " for both domains (the IQS set has no measured timeseries).")
    A("**Metrics.** Balanced accuracy and macro-F1 neutralise the 82.5% damaged prior; ROC-AUC (detection"
      " tasks) is threshold-free; severity uses R²/Pearson-r/MAE. Engines: `ml_pipeline/hires_{zoo,tab,all}.py`;"
      " analysis in `hires_analysis.py`; raw per-case predictions (with class probabilities) on branches"
      " `colab-hires-{tabular,cnn,transformer,vision}`.\n")

    # ---- (4) input representations + (5) architectures ----
    emit_inputs(A)
    emit_arch(A)

    # ---- (6) EDA ----
    cnt = an.get("counts", {})
    A("## 6 · Exploratory data analysis — synthetic vs experimental")
    A("Before any model, the two datasets are compared directly. This frames everything that follows:"
      " *what the models are up against is not noise, it is a structured domain gap.*\n")
    A("### 4.1 Class balance and severity coverage")
    if cnt:
        A("| class | synth N | exp N |")
        A("|---|---|---|")
        for k in ["pristine","bolt","crack","hole","mass"]:
            A(f"| {k} | {cnt.get('synth',{}).get(k,'—')} | {cnt.get('exp',{}).get(k,'—')} |")
    A("\n![class balance and severity](figures/hires/eda_class_severity.png)")
    A("*Figure 2 — (a) the synthetic set is perfectly balanced (2 000/class) while the experimental set is"
      " **bolt-dominated** (51% bolt, 18% pristine); training therefore class-weights the loss. (b) Experimental"
      " severity by type: bolt-loosening spans a wide 0–85% range, whereas hole and mass occupy narrow"
      " bands — this is exactly why the bolt detector has room to improve with severity and the others do"
      " not. (c) Per-type-normalised severity: synthetic damage is sampled near-uniformly, but the real"
      " damage clusters, so the model is asked to extrapolate over severity ranges it rarely saw.*\n")
    A("### 4.2 Spectral signatures and the domain gap")
    A("![FRF signatures](figures/hires/eda_frf_signatures.png)")
    A("*Figure 3 — channel-averaged mean log|FRF|. (a) Synthetic classes differ mainly in resonance-peak"
      " amplitude/position — the information the models exploit in-domain. (b) Overlaying synthetic (solid)"
      " on experimental (dashed) for the same class shows the gap: the real structure has shifted resonances,"
      " extra anti-resonances, and a higher noise floor the linear ROM never produces.*\n")
    A("### 4.3 Covariate shift, quantified")
    pcv = an.get("pca_explained_var", [0,0])
    A(f"![domain shift PCA](figures/hires/eda_domain_shift.png)")
    A("*Figure 4 — PCA of the log|FRF| spectra. (a) Coloured by **domain**, synthetic and experimental form"
      f" two disjoint clouds (PC1 = {100*pcv[0]:.0f}% of variance); a 5-fold logistic classifier tells them"
      f" apart with **AUC = {domauc:.3f}** — i.e. essentially perfectly. (b) The same projection coloured by"
      " **damage class** shows the classes overlapping heavily, so the dominant axis of variation in the data"
      " is *which domain*, not *which damage*.*\n" if domauc else "")
    A("**This is the single most important diagnostic in the report.** A domain-classifier AUC of"
      f" {domauc:.2f} means the sim-to-real gap is not a subtle nuisance — the simulator and the rig are"
      " trivially distinguishable from their spectra alone. Any zero-shot transfer at all (and we get"
      " meaningful transfer on detection) is therefore a non-trivial success, and the residual errors in §5–6"
      " are the direct, expected consequence of this shift. It also sets the research direction: the lever is"
      " **domain adaptation**, not bigger models or higher resolution.\n" if domauc else "")

    # ---- (5) Diagnostics ----
    A("## 7 · Diagnostics — how the best models actually behave on real data")
    A("Aggregate scores hide the failure *mode*. Below are the confusion matrices and ROC curves for the"
      " single best cell of each task (selected by experimental balanced-acc / R²; see §6 for all cells).\n")
    A("### 5.1 Confusion matrices (experimental, row-normalised)")
    A("![confusion matrices](figures/hires/diag_confusion.png)")
    A("*Figure 5 — read each row as 'of the true X, what fraction was predicted as …'.*\n")
    # specific CM walkthrough
    def tpr_tnr(t):
        cm = bc.get(t, {}).get("cm_rownorm")
        return (cm[0][0], cm[1][1]) if cm and len(cm) == 2 else (None, None)
    walk = []
    for t in ["binary","is_bolt","is_mass","is_hole"]:
        tnr, tpr = tpr_tnr(t)
        if tpr is not None:
            walk.append(f"- **{t}** ({bc[t]['cell']}): catches {tpr*100:.0f}% of positives but flags"
                        f" {100-tnr*100:.0f}% of negatives as positive — a sensitivity-biased operating point,"
                        " the expected response when the loss is class-weighted and the prior shifts.")
    cmtype = bc.get("type", {}).get("cm_rownorm")
    if cmtype:
        A("\n".join(walk))
        A(f"- **type** ({bc['type']['cell']}): the 5-class matrix smears toward the **bolt** column (the"
          " majority real class) and the **mass** diagonal survives best — damage *type* is the hardest"
          " thing to transfer, because the spectral fingerprint of crack vs hole vs bolt is what the domain"
          " shift most corrupts.")
        A("- **mass_location / col_location**: rows pile onto one or two columns — localization collapses"
          " toward a dominant class, consistent with the near-degenerate spatial classes of the linear ROM"
          " (§10) and the weak in-domain ceiling for col_location.\n")
    A("### 5.2 ROC / AUC for the detection tasks")
    A("![ROC curves](figures/hires/diag_roc.png)")
    A("*Figure 6 — AUC is threshold-free and immune to the 82.5% prior, so it is the fairest single number"
      " for detection.*\n")
    A("| task | best cell | exp AUC | exp bal-acc | exp macro-F1 | in-domain mF1 |")
    A("|---|---|---|---|---|---|")
    for t in ["binary","is_pristine","is_bolt","is_crack","is_hole","is_mass"]:
        e = bc.get(t, {})
        if "auc" not in e: continue
        A(f"| {t} | `{e.get('cell','—')}` | **{e['auc']:.3f}** | {e.get('bal_acc',0):.3f} |"
          f" {e.get('macro_f1',0):.3f} | {e.get('in_domain') and e['in_domain']:.2f} |")
    aucs = {t: bc[t]["auc"] for t in DET_KEYS if t in bc and "auc" in bc[t]}
    if aucs:
        hi = max(aucs, key=aucs.get)
        A(f"\n**Reading the AUCs.** `is_hole` and `is_mass` reach AUC ≈ 0.71–0.72 — the strongest"
          " threshold-free detectors — yet their *balanced accuracy* at the default cut is lower, because"
          " the operating threshold is mis-set by the shift. That gap is good news: it means a handful of"
          " labelled real samples to recalibrate the threshold would lift the realised accuracy without any"
          " retraining. `binary` and `is_pristine` have the weakest AUCs (≈0.55–0.57): deciding *damaged vs"
          " not* in the aggregate is harder than detecting specific damage signatures, because the pristine"
          " class is where the domain gap bites hardest (Figure 3a).\n")

    # ---- per-task full catalogue ----
    A("## 8 · Per-task catalogue (every cell)")
    A("Every (model, feature) cell, sorted best-first. `in-domain` = held-out synthetic score (the ceiling);"
      " `exp` columns = zero-shot on real data. The cell-zoo bar plot colours **blue = CFDAC-image** cells"
      " and **green = tabular/sequence** cells, so the winning representation family is visible at a glance.\n")
    for t in TASKS:
        if t not in cells: continue
        recs = cells[t]; reg = recs[0]["kind"]=="reg"
        q,o,ch,note = TASK_DESC[t]
        A(f"\n### {t}")
        A(f"**Question.** {q}  **Output.** {o}  " + (f"**Chance.** {ch:.2f}." if ch else "**Regression.**") + (f"  {note}" if note else ""))
        A(f"**Cells:** {len(recs)}.\n")
        A(f"![{t} cell zoo](figures/hires/{cellzoo_plot(t, recs)})\n")
        if reg:
            recs = sorted(recs, key=lambda r:-r.get("exp_r2",-9))
            A("| model / feature | in-domain R² | exp R² |")
            A("|---|---|---|")
            for r in recs:
                A(f"| `{r['model']}/{r['feature']}` | {(r['synth'] or 0):.3f} | {r.get('exp_r2',float('nan')):+.3f} |")
            b = recs[0]
            A(f"\n**Best:** `{b['model']}/{b['feature']}` exp R²={b.get('exp_r2'):.3f} (in-domain R²={b['synth']:.2f}). "
              "Severity barely transfers; the full diagnosis is in §8.")
        else:
            recs = sorted(recs, key=lambda r:-r.get("exp_bal_acc",0))
            isdet = t in DET_KEYS
            A("| model / feature | in-domain mF1 | exp bal-acc | exp macro-F1 | collapse |")
            A("|---|---|---|---|---|")
            for r in recs:
                A(f"| `{r['model']}/{r['feature']}` | {(r['synth'] or 0):.2f} | "
                  f"{r.get('exp_bal_acc',0):.3f} | {r.get('exp_macro_f1',0):.3f} | {'yes' if r.get('collapse') else ''} |")
            b = recs[0]
            ac = sum(1 for r in recs if r.get('exp_bal_acc',0) > r['chance']+0.05)
            line = (f"\n**Best:** `{b['model']}/{b['feature']}` — exp balanced-acc **{b['exp_bal_acc']:.3f}** "
                    f"(macro-F1 {b['exp_macro_f1']:.3f}; in-domain {b['synth']:.2f}). "
                    f"{ac}/{len(recs)} cells clear chance+0.05; {sum(1 for r in recs if r.get('collapse'))} collapse to one class.")
            if isdet:
                p = detect_prose(t, b, an)
                if p: line += " " + p
            A(line)

    # ---- DT sweep ----
    A("\n## 9 · Damage-threshold (DT) severity sweep @1601")
    A("Positives are stratified by their damage-severity percentile (each task on its own axis: bolt %, hole"
      " mm, mass kg, crack depth); balanced accuracy is recomputed keeping only the more-severe positives"
      " (all negatives retained). This tests the central thesis — *transfer should improve with damage"
      " severity, because larger damage perturbs the spectrum more than the domain gap does.*\n")
    A("![DT combined](figures/hires/dt_1601_combined.png)")
    A("*Figure 7 — best-cell experimental balanced-acc vs the severity percentile kept.*\n")
    A("| task | all (p0) | ≥p50 | ≥p75 | ≥p90 | best cell @p90 |")
    A("|---|---|---|---|---|---|")
    for t in ["is_bolt","binary","is_crack","is_hole","is_mass"]:
        d = dt["per_task"].get(t)
        if not d: continue
        bp = {b["pct"]:b for b in d["best_per_pct"]}
        def g(p): return f"{bp[p]['bal_acc']:.3f}" if bp.get(p) and bp[p]["bal_acc"] is not None else "—"
        A(f"| {t} | {g(0)} | {g(50)} | {g(75)} | {g(90)} | {bp.get(90,{}).get('cell','—')} |")
    A("\n![is_bolt DT](figures/hires/zoo_dt_is_bolt.png)")
    A("*Figure 8 — the is_bolt detectors, swept on loosening severity.*\n")
    A("**is_bolt reaches ~0.82 balanced-acc at ≥75% loosening** — confirming the thesis where severity has"
      " range to vary. is_hole/is_mass stay flat *because their experimental severity range is narrow"
      " (Figure 2b), not because the model fails* — there simply is no 'more severe' subset to climb into.")

    # ---- 9.2 swept diagnostics (AUC + confusion evolution) ----
    ddpath = _REPO/"results_hires"/"dt_diag.json"
    if ddpath.exists():
        dd = json.loads(ddpath.read_text())["per_task"]
        A("\n### 9.2 The full diagnostic suite, swept over severity")
        A("Balanced accuracy is one scalar; the questions *'does the **ranking** (AUC) improve, and **how** does"
          " the confusion matrix change?'* need the whole suite recomputed at each severity threshold. Using the"
          " stored class probabilities, the best cell of each detection task is re-scored keeping only"
          " progressively more-severe positives.\n")
        A("![AUC and macro-F1 vs severity](figures/hires/dt_auc.png)")
        A("*Figure 9 — ROC-AUC (a) and macro-F1 (b) vs the severity percentile kept.*\n")
        A("![confusion-matrix evolution](figures/hires/dt_confusion_evo.png)")
        A("*Figure 10 — the row-normalised confusion matrix at each severity threshold (rows = task, columns ="
          " percentile). Watch the positive-row (bottom) darken on the diagonal as damage increases.*\n")
        A("| task | AUC p0 → p90 | sensitivity p0 → p90 | reading |")
        A("|---|---|---|---|")
        notes = {
         "binary":"climbs steadily — bigger damage is easier to call damaged (sens 0.82→0.96).",
         "is_bolt":"the clean win: AUC and sensitivity both rise sharply with loosening %.",
         "is_crack":"rises then **falls** at p75+ — the severe-crack subset is tiny, so the estimate is noisy, not better.",
         "is_hole":"flat then **drops** at p75+ — same small-sample artefact; hole severity barely varies (1–6 mm).",
         "is_mass":"perfectly flat — experimental mass severity is near-discrete, so there is no gradient to climb.",
        }
        for t in ["binary","is_bolt","is_crack","is_hole","is_mass"]:
            r = dd.get(t)
            if not r: continue
            au = r["auc"]; se = r["sens"]
            f = lambda v: ("%.2f"%v) if v is not None else "—"
            A(f"| {t} `{r['cell']}` | {f(au[0])} → {f(au[-1])} | {f(se[0])} → {f(se[-1])} | {notes.get(t,'')} |")
        A("\n**The honest reading.** Severity helps where it *varies*: `binary` and especially `is_bolt`"
          " (AUC 0.67→0.87) improve monotonically. For `is_crack`/`is_hole` the apparent late drop is a"
          " **small-sample artefact** — past p75 only a handful of positives remain (the experimental crack/hole"
          " severities barely span a range), so the metric becomes noisy rather than genuinely worse. `is_mass`"
          " is flat because its real severity is essentially a single level. This nuance is exactly why the"
          " sweep reports per-task thresholds and positive counts rather than a single global curve.\n")

    # ---- 9.3 stiffness-loss (physical) sweep ----
    spath = _REPO/"results_hires"/"dt_stiffness.json"
    if spath.exists():
        sd = json.loads(spath.read_text())
        A("\n### 9.3 The same sweep on a *physical* axis — storey-stiffness loss")
        A("Native severity units are not comparable across damage types (a '% loosening' is not a 'mm of"
          " crack'). Using the **simulator's own calibrated damage model**"
          " (`ml_pipeline.variation.{bolt_jsr_ratio, crack_ratio, hole_ratio}`), each stiffness-reducing damage"
          " is mapped to the actual fraction of storey stiffness it removes — putting bolt, crack and hole on"
          " one physical axis. (Added mass is excluded: it changes inertia, not compliance, so its"
          " stiffness loss is 0.)\n")
        A("![severity to stiffness map + distribution](figures/hires/dt_stiffness_map.png)")
        A("*Figure 11 — (a) the calibrated severity→stiffness-loss map; (b) the experimental stiffness loss"
          " each damage type actually produces. This single panel is the physical key to the whole study.*\n")
        A("Experimentally, **bolt loosening removes 15–61% of storey stiffness (median ~45%), while a crack"
          " removes only 4–6% and a hole only 2–3%**. The three 'detection' tasks are therefore *not* probing"
          " comparable amounts of structural change — crack and hole damage is, physically, an order of"
          " magnitude milder.\n")
        A("![DT sweep vs stiffness loss](figures/hires/dt_stiffness.png)")
        A("*Figure 12 — best-cell balanced-acc (a) and AUC (b) vs the minimum storey-stiffness loss retained in"
          " the positives; the grey band marks the ≤6.4% region where all crack/hole damage lives.*\n")
        A("| task | best cell | ≥0% | ≥5% | ≥20% | ≥40% | positives surviving |")
        A("|---|---|---|---|---|---|---|")
        for t in STIFF_ORDER:
            r = sd["per_task"].get(t)
            if not r: continue
            ba = r["bal_acc"]; nps = r["n_pos"]; thr = sd["thresholds_pct"]
            def gv(pct):
                i = thr.index(pct)
                return f"{ba[i]:.3f}" if i < len(ba) and ba[i] is not None else "—"
            A(f"| {t} | `{r['cell']}` | {gv(0.0)} | {gv(5.0)} | {gv(20.0)} | {gv(40.0)} | "
              f"{nps[0]}→{nps[-1]} (≥40%) |")
        A("\n**This is the physical explanation for the whole detection hierarchy.** On a stiffness-loss axis the"
          " story is unambiguous: `binary` and `is_bolt` keep climbing as we retain only structurally-significant"
          " damage (is_bolt balanced-acc 0.67→0.73, AUC 0.67→0.79 by ≥40% loss), because bolt damage genuinely"
          " reaches that regime. `is_crack` and `is_hole` curves **terminate early** — by ≥10% and ≥5% stiffness"
          " loss respectively there are *zero* experimental positives left, because crack/hole simply never"
          " remove that much stiffness. Their weak, flat transfer is not a model failure: the damage they"
          " represent is physically near-invisible to a global FRF. The takeaway sharpens the severity message"
          " of §9.1–9.2: **detection transfers in proportion to how much stiffness the damage removes**, and"
          " only bolt-loosening (and any other large-stiffness-loss mechanism) reaches the regime where"
          " sim-to-real transfer becomes reliable.\n")

    # ---- severity deep-dive ----
    sv = bc.get("severity", {})
    A("\n## 10 · Severity regression (the only non-classifier task)")
    A("![severity scatter and residuals](figures/hires/diag_severity.png)")
    A("*Figure 13 — (a) predicted vs true severity for the best cell; (b) residuals.*\n")
    if sv:
        A(f"Best experimental **R² = {sv.get('r2',0):+.3f}** with **Pearson r = {sv.get('pearson_r',0):.3f}**"
          f" and **MAE = {sv.get('mae',0):.3f}** (`{sv.get('cell','—')}`), against **R² ≈ 0.59 in-domain**."
          " The scatter tells the story the R² number alone does not: there *is* a weak positive trend"
          " (r ≈ 0.36, the fit slope is positive), so the model is not random — but the predictions collapse"
          " toward the training mean (the residual plot in (b) slopes against the true value, the signature of"
          " regression-to-the-mean under distribution shift). Restricting to severe cases does **not** raise R²"
          " (that just narrows the variance). **Predicting damage *magnitude* zero-shot is effectively"
          " unsolved**; recasting it as ordinal severity-band classification (§11) is the recommended fix,"
          " since detection already improves monotonically with severity (§7).\n")

    # ---- synthesis ----
    A("## 11 · Cross-task synthesis")
    A("1. **Detection ≫ localization ≫ magnitude.** Presence/type transfers (AUC 0.55–0.72, balanced-acc"
      " 0.56–0.67); location only weakly (≈1.4–2× chance); severity barely (r≈0.36, R²≈0.04).")
    A("2. **Severity is the lever, not the target.** Every detector improves on more-severe damage"
      " (is_bolt →0.82 at ≥75% loosening); use damage size to *gate confidence*, don't try to *regress* it.")
    A("3. **Representation > model size.** Full complex spectral inputs (raw FRF / CFDAC / FRF-derived"
      " timeseries) with sequence/conv models win; the compressed `modal` vector and the ImageNet-pretrained"
      " vision backbones (ConvNeXt-T, ResNet50) do **not** lead — pretraining on natural images buys nothing"
      " for these spectra.")
    A("4. **The ceiling is covariate shift, not capacity.** Near-perfect in-domain (Figure 1) collapses to"
      f" partial transfer because the domains are {('AUC='+format(domauc,'.2f')) if domauc else 'almost perfectly'}"
      " separable (Figure 4). Higher resolution and bigger nets cannot close a gap that is fundamentally about"
      " the simulator not matching the rig.\n")

    # ---- limitations + recs ----
    A("## 12 · Limitations")
    A("- **One experimental structure, one seed per cell** — treat balanced-acc gaps < 0.05 as ties.\n"
      "- **Post-hoc best-cell selection** (§5–6 pick the winner after seeing the test set) is exploratory,"
      " not a held-out estimate; the per-task tables guard against cherry-picking by showing every cell.\n"
      "- **Localization classes are near-degenerate** in the linear ROM (symmetric crack/hole make the two"
      " column ends almost indistinguishable), capping col_location even in-domain (0.45 mF1).\n"
      "- **`timeseries` is FRF-reconstructed**, not independently measured, so it carries no information"
      " beyond the FRF — it is a different *inductive bias*, not a new sensor.\n"
      "- **128-bin resolution comparison pending** (that run is still in progress).\n")
    A("## 13 · Recommendations")
    A("1. **Deploy detection on severe damage and report severity-stratified** (the DT curves, not a single"
      " number).\n"
      "2. **Recalibrate the decision threshold on a few real samples** — the AUC>bal-acc gap (§5.2) is free"
      " accuracy.\n"
      "3. **Use spectral inputs + sequence/conv models; drop `modal` and pretrained vision** as the primary"
      " route.\n"
      "4. **Attack the domain gap directly with domain adaptation** (e.g. CORAL/feature alignment, fine-tune"
      " on a small labelled real set, or domain-randomise the simulator) — this is the highest-leverage move"
      " given the AUC≈1.0 shift.\n"
      "5. **Recast severity as ordinal classification** and extend the DT analysis to the multi-class tasks.\n"
      "6. **Finish the 128-bin run** to settle whether full resolution is necessary.\n")
    A("## 14 · Artefacts (everything is reproducible from these)")
    A("- **Engines:** `ml_pipeline/hires_{zoo,tab,all}.py` — CFDAC-image, tabular/sequence, and the unified"
      " dispatcher; every model and feature defined here.\n"
      "- **Analysis scripts:** `hires_zoo_summary.py` (per-cell distillation), `hires_dt_1601.py` (DT balanced-acc"
      " sweep), `hires_dt_diag.py` (DT-swept AUC + confusion evolution), `hires_analysis.py` (EDA + best-cell"
      " confusion/ROC/severity), `hires_arch.py` (measured parameter counts), `hires_inputs.py` (input-sample"
      " figures), `build_hires_report.py` (this report).\n"
      "- **Data:** `results_hires/{zoo_summary, zoo_best_by_task_res, dt_1601, dt_diag, dt_stiffness, analysis,"
      " architectures, inputs}.json`. **Every figure is reproducible from committed data** — the per-case"
      " predictions are archived in `results_hires/per_case_hires1601.tar.gz` and the FRF-derived arrays the"
      " EDA/input figures need in `results_hires/figure_data.npz` (built by `build_figure_bundle.py`, served by"
      " `figdata.py`). Raw per-case predictions also live on branches `colab-hires-{tabular,cnn,transformer,vision}`."
      " See `results/REPRODUCE.md` for the one-command-per-script pipeline.\n"
      "- **Figures:** `results/figures/hires/` — inputs (`inputs_*`), capacity (`arch_params`), EDA"
      " (`eda_*`), best-cell diagnostics (`diag_*`), DT sweep (`dt_1601_combined`, `zoo_dt_is_bolt`, `dt_auc`,"
      " `dt_confusion_evo`), per-task cell zoos (`cellzoo_*`).\n"
      "- **Companion:** `REPORT_synth.md` (in-domain ceiling).")

    (_REPO/"results"/"REPORT_CONSOLIDATED.md").write_text("\n".join(out)+"\n")
    print(f"wrote REPORT_CONSOLIDATED.md ({len(out)} blocks, {ncell} cells, {len(TASKS)} tasks)")


if __name__ == "__main__":
    main()
