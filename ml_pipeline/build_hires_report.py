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


def load_cells():
    S = json.loads((_REPO/"results_hires"/"zoo_summary.json").read_text())
    cells = defaultdict(list)   # task -> list of recs (res==RES)
    for k, v in S.items():
        if v.get("res") != RES: continue
        cells[v["task"]].append(v)
    return cells


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


def main():
    cells = load_cells()
    dt = json.loads((_REPO/"results_hires"/"dt_1601.json").read_text())
    out = []
    A = out.append
    A("# LANL 3SBB — Synth-to-Real Damage Diagnosis: FULL Consolidated Report")
    A("**Author:** G. Reyes-Carmenaty · **Date:** 2026-06-08 · **Resolution:** 1601-bin (native).")
    A("\n> Exhaustive edition — every cell of the high-resolution model zoo, with in-domain"
      " and zero-shot metrics, per-task cell-zoo plots, the damage-threshold sweep, and the"
      " severity deep-dive. Executive summary of the same results: see the top sections;"
      " the in-domain companion is `REPORT_synth.md`. *(128-bin comparison excluded until that run completes.)*\n")

    # ---- counts ----
    ncell = sum(len(v) for v in cells.values())
    ncls = sum(1 for t in cells for r in cells[t] if r["kind"]=="cls")
    nnc = sum(1 for t in cells for r in cells[t] if r["kind"]=="cls" and not r.get("collapse"))
    A("## 1 · Overview")
    A(f"- **{ncell} cells** at 1601 bins = (≤11 models) × (≤12 features) × 10 tasks, each trained to"
      " convergence on synthetic data and evaluated zero-shot on the 2 638-case IQS experimental set.")
    A(f"- **{nnc}/{ncls} classification cells clear chance** on real data (≈{100*nnc/ncls:.0f}%).")
    A("- A *cell* = one (model, feature) pair. Metric of record: **balanced accuracy / macro-F1**"
      " (classification), **R² / Pearson r / MAE** (severity). Raw accuracy is never used (82.5% damaged prior).")
    A("\n![in-domain vs zero-shot](figures/hires/zoo1601_synth_vs_exp.png)\n")

    # ---- glossary ----
    A("## 2 · Model & feature glossary (what every cell is)")
    A("**Models**\n")
    for m, d in MODELS.items(): A(f"- `{m}` — {d}")
    A("\n**Features** (all computed from the native 1601-bin FRFs)\n")
    for f, d in FEATS.items(): A(f"- `{f}` — {d}")
    A("\nEach per-task table below lists **every** cell; read `model/feature` against this glossary.\n")

    # ---- methodology ----
    A("## 3 · Methodology")
    A("Synthetic data regenerated at 16 s (N_T=4096, fs=256 → df=0.0625 Hz, 1601 bins, 0–100 Hz) to"
      " match the experimental grid exactly. Per cell: compute the feature from the FRFs, train on a"
      " synth subsample (70/15/15 split, class-weighted loss / balanced trees, early-stop to"
      " convergence with checkpoint/resume), evaluate on held-out synth (in-domain) and all 2 638"
      " experimental cases (zero-shot). Tabular features standardised on the synth-train fold only;"
      " sequence/image features per-sample normalised (no leakage). `timeseries` is reconstructed from"
      " the FRF identically for both domains (the IQS set has no measured timeseries).")
    A("Engines: `ml_pipeline/hires_zoo.py`, `hires_tab.py`, `hires_all.py`. Raw per-case predictions on"
      " branches `colab-hires-{tabular,cnn,transformer,vision}`.\n")

    # ---- per-task full catalogue ----
    A("## 4 · Per-task catalogue (every cell)")
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
              "Severity barely transfers; see §6.")
        else:
            recs = sorted(recs, key=lambda r:-r.get("exp_bal_acc",0))
            A("| model / feature | in-domain mF1 | exp bal-acc | exp macro-F1 | collapse |")
            A("|---|---|---|---|---|")
            for r in recs:
                A(f"| `{r['model']}/{r['feature']}` | {(r['synth'] or 0):.2f} | "
                  f"{r.get('exp_bal_acc',0):.3f} | {r.get('exp_macro_f1',0):.3f} | {'yes' if r.get('collapse') else ''} |")
            b = recs[0]
            ac = sum(1 for r in recs if r.get('exp_bal_acc',0) > r['chance']+0.05)
            A(f"\n**Best:** `{b['model']}/{b['feature']}` — exp balanced-acc **{b['exp_bal_acc']:.3f}** "
              f"(macro-F1 {b['exp_macro_f1']:.3f}; in-domain {b['synth']:.2f}). "
              f"{ac}/{len(recs)} cells clear chance+0.05; {sum(1 for r in recs if r.get('collapse'))} collapse to one class.")

    # ---- DT sweep ----
    A("\n## 5 · Damage-threshold (DT) severity sweep @1601")
    A("Positives stratified by their damage-severity percentile (each task on its own axis); balanced"
      " accuracy recomputed keeping only the more-severe positives. Confirms transfer improves with severity.\n")
    A("![DT combined](figures/hires/dt_1601_combined.png)\n")
    A("| task | all (p0) | ≥p50 | ≥p75 | ≥p90 | best cell @p90 |")
    A("|---|---|---|---|---|---|")
    for t in ["is_bolt","binary","is_crack","is_hole","is_mass"]:
        d = dt["per_task"].get(t)
        if not d: continue
        bp = {b["pct"]:b for b in d["best_per_pct"]}
        def g(p): return f"{bp[p]['bal_acc']:.3f}" if bp.get(p) and bp[p]["bal_acc"] is not None else "—"
        A(f"| {t} | {g(0)} | {g(50)} | {g(75)} | {g(90)} | {bp.get(90,{}).get('cell','—')} |")
    A("\n![is_bolt DT](figures/hires/zoo_dt_is_bolt.png)\n")
    A("**is_bolt reaches ~0.82 balanced-acc at ≥75% loosening.** is_hole/is_mass stay flat because"
      " their experimental severity range is narrow, not because the model fails.")

    # ---- severity deep-dive ----
    A("\n## 6 · Severity regression (the non-classifier)")
    A("Best experimental **R² ≈ 0.04** (most cells negative), weak monotonic signal (**Pearson r ≈ 0.36**"
      " for frf/timeseries cells), vs **R² ≈ 0.59 in-domain**. Restricting to severe cases does not raise R²"
      " (variance-narrowing artefact); MAE ≈ 0.25 on a 0.07–1.0 scale. Predicting damage *magnitude*"
      " zero-shot is unsolved — recast as ordinal classification is the recommended next step.\n")

    # ---- synthesis ----
    A("## 7 · Cross-task synthesis")
    A("1. **Detection ≫ localization ≫ magnitude.** Presence/type transfers (0.56–0.67 balanced-acc);"
      " location weakly (≈1.4–2× chance); severity barely.")
    A("2. **Severity is the lever** — every detector improves on more-severe damage (is_bolt →0.82).")
    A("3. **Representation > model size** — full complex spectral inputs (CFDAC / raw FRF / timeseries)"
      " with sequence/conv models beat the compressed `modal` baseline and the pretrained vision backbones.")
    A("4. **The gap is covariate shift, not capacity** — near-perfect in-domain, partial transfer.\n")

    # ---- limitations + recs ----
    A("## 8 · Limitations\n- Single experimental structure; one seed per cell (treat <0.05 gaps as ties);"
      " post-hoc best-cell selection is exploratory; localization classes near-degenerate in the linear ROM;"
      " `timeseries` is FRF-derived; 128-bin comparison pending.\n")
    A("## 9 · Recommendations\n1. Deploy detection on severe damage (report severity-stratified).\n"
      "2. Use spectral inputs + sequence/conv models; drop `modal` and pretrained vision as the primary route.\n"
      "3. Attack severity & localization with domain adaptation, not bigger nets.\n"
      "4. Recast severity as ordinal classification.\n5. Finish the 128-bin run to settle the resolution question.\n")
    A("## 10 · Artefacts\n- Code: `ml_pipeline/hires_{zoo,tab,all,zoo_summary,dt_1601}.py`\n"
      "- Data: `results_hires/{zoo_summary,zoo_best_by_task_res,dt_1601}.json`; per-case on `colab-hires-*` branches\n"
      "- Figures: `results/figures/hires/{zoo1601_synth_vs_exp,dt_1601_combined,zoo_dt_is_bolt,cellzoo_*}.png`\n"
      "- Companion: `REPORT_synth.md` (in-domain).")

    (_REPO/"results"/"REPORT_CONSOLIDATED.md").write_text("\n".join(out)+"\n")
    print(f"wrote REPORT_CONSOLIDATED.md ({len(out)} blocks, {ncell} cells, {len(TASKS)} tasks)")


if __name__ == "__main__":
    main()
