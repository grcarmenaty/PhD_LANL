"""Build REPORT_synth.md — the *synthetic-domain* (in-distribution)
training results, BEFORE any zero-shot test against the real LANL data.

Deep companion to REPORT_CONSOLIDATED.md: per-task explanations, embedded
plots, cell-by-cell tables with the HPO-selected hyper-parameters, and the
sim-to-real gap (synthetic test vs experimental zero-shot, cell by cell).

Reads:
  results/training_metrics.json                          (bespoke synth val/test)
  results/cells_v1_v2_v2a.json                           (experimental, for the gap)
  results_vision/<variant>_seed<seed>/per_case_vision/*  (vision synth_test)
  results/figures/synth/fig{1..5}*.png                   (built by plot_synth.py)
Writes:
  results/REPORT_synth.md
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
TM = _REPO / "results" / "training_metrics.json"
CELLS = _REPO / "results" / "cells_v1_v2_v2a.json"
VIS_ROOT = _REPO / "results_vision"
OUT = _REPO / "results" / "REPORT_synth.md"

TASK_ORDER = ["binary", "type", "col_location", "mass_location", "severity"]

# Per-task: chance level + what the in-domain number means.
TASK_INFO = {
    "binary": dict(
        chance="0.5 balanced; 0.825 by majority-class on the real set",
        metric="accuracy",
        blurb=(
            "Damaged-vs-pristine on the synthetic data. This is the easiest "
            "in-domain task: the synthetic generator stamps a clear modal "
            "signature on every damaged case, so a tiny MLP on the extracted "
            "modal features separates the two classes almost perfectly. The "
            "interesting part is §4 — almost none of this separability "
            "survives transfer to the real structure.")),
    "type": dict(
        chance="0.20 (5-class)",
        metric="accuracy",
        blurb=(
            "Five-way {pristine, bolt, crack, hole, mass}. Modal features "
            "carry most of the signal in-domain; the difficulty is that crack "
            "and hole produce similar global modal changes, so the confusion "
            "concentrates there.")),
    "col_location": dict(
        chance="0.111 (9-class)",
        metric="accuracy",
        blurb=(
            "Which column section is damaged. This is the hard task even "
            "in-domain: best cells sit near 0.49 — the global modal/FRF "
            "response simply does not localise 'where' very well, regardless "
            "of model. The ceiling here is low before transfer even enters "
            "the picture.")),
    "mass_location": dict(
        chance="0.333 (3-class)",
        metric="accuracy",
        blurb=(
            "Which tier carries the added mass. Added mass shifts global "
            "natural frequencies in a tier-specific way, so modal features "
            "nail this in-domain (~0.99). Contrast with the cross-domain "
            "number in the consolidated report, where it falls to ~0.43 "
            "macro-F1.")),
    "severity": dict(
        chance="R²=0 ≡ predict-the-mean",
        metric="R²",
        blurb=(
            "Continuous bolt-loosening fraction (regression). In-domain R² "
            "tops out around 0.57 for the modal random-forest — the synthetic "
            "severity axis is only moderately encoded even before transfer.")),
}

MODEL_DESC = {
    "mlp": "MLP", "cnn": "1-D CNN", "cnn1d": "1-D CNN (deep)",
    "cnn2d": "2-D CNN", "rf": "random forest", "xgb": "gradient boosting",
    "transformer": "transformer",
    "convnext_tiny": "ConvNeXt-Tiny", "resnet50": "ResNet50",
    "vit_b_16": "ViT-B/16",
}
FEATURE_DESC = {
    "modal": "natural freqs + damping + mode shapes",
    "frf_mag": "FRF magnitude spectrum",
    "cfdac": "CFDAC image (4-ch)", "indicators": "scalar damage indicators",
    "timeseries": "raw acceleration time series",
    "cfdac_all": "CFDAC 4-ch", "cfdac_mag": "CFDAC magnitude",
    "cfdac_realimag": "CFDAC real+imag",
}


def fmt(x, k=3):
    return "—" if x is None else (f"{x:.{k}f}" if isinstance(x, float) else str(x))


def hp_str(hp: dict) -> str:
    if not hp:
        return "—"
    parts = []
    for kk, vv in hp.items():
        parts.append(f"{kk}={vv}")
    s = ", ".join(parts)
    return s if len(s) <= 60 else s[:57] + "…"


def load_vision():
    rows = []
    if not VIS_ROOT.exists():
        return rows
    for d in sorted(VIS_ROOT.iterdir()):
        if not d.is_dir() or "_seed" not in d.name:
            continue
        variant, seed = d.name.split("_seed")
        pc = d / "per_case_vision"
        if not pc.exists():
            continue
        for jf in sorted(pc.glob("*.json")):
            try:
                meta = json.load(open(jf)).get("meta", {})
            except Exception:
                continue
            rows.append(dict(variant=variant, seed=seed,
                             task=meta.get("task"), backbone=meta.get("backbone"),
                             feature=meta.get("feature"), kind=meta.get("kind"),
                             synth_test=meta.get("synth_test")))
    return rows


def build():
    tm = json.load(open(TM))
    cells = json.load(open(CELLS))["cells"]
    vision = load_vision()
    by_task = defaultdict(list)
    for r in tm:
        by_task[r["task"]].append(r)

    out = []
    w = out.append

    # ---- header ----
    w("# LANL 3SBB — Synthetic-domain training results (pre-transfer)\n")
    w("**Companion to** [`REPORT_CONSOLIDATED.md`](REPORT_CONSOLIDATED.md) "
      "(the full cross-domain/experimental study). Date: 2026-06-02.\n\n")
    w("This report answers one question: **how well does each model fit the "
      "task on the synthetic data it was trained on**, scored on a held-out "
      "synthetic test fold, *before any contact with the real experimental "
      "structure*? These are the in-distribution ceilings. The difference "
      "between a number here and its counterpart in the consolidated report "
      "is, cell for cell, the **sim-to-real gap** — quantified in §4.\n")
    w("\n---\n")

    # ---- methodology ----
    w("## 1. Method\n")
    w("- **Data.** Synthetic 3SBB cases from the physics generator. A "
      "70/15/15 train/validation/test split (stratified for classification, "
      "random for regression). The *test* fold is synthetic and drawn from "
      "the same generator as training — so these numbers are optimistic by "
      "construction; they are a learnability ceiling, not a deployment "
      "estimate.\n")
    w("- **Metrics.** Accuracy on the synthetic test fold for classification; "
      "R² for `severity` regression.\n")
    w("- **Two model families.**\n")
    w("  - *Bespoke* (mlp, cnn, cnn1d, cnn2d, rf, xgb, transformer) on the "
      "tabular/spectral features. Numbers are HPO-tuned on the **v1** baseline "
      "physics and cover the 5 original tasks "
      "(`results/training_metrics.json`).\n")
    w("  - *Vision* (ConvNeXt-Tiny, ResNet50, ViT-B/16; ImageNet-pretrained "
      "via timm) on CFDAC images. Synthetic test scores are read from each "
      "per-case JSON's `meta.synth_test`; this family spans v1/v2/v2a × seeds "
      "42/101/202 × all 10 tasks and **grows as the running sweep completes** "
      "(re-run `ml_pipeline/build_report_synth.py` to refresh).\n")
    w("- **Plots** are produced by `ml_pipeline/plot_synth.py`.\n")
    w("\n---\n")

    # ---- headline ----
    w("## 2. Headline — best synthetic cell per task, and what survives transfer\n\n")
    w("| task | best synth cell | synth test | same cell, experimental (v1) | gap |\n")
    w("|---|---|---|---|---|\n")
    for task in [t for t in TASK_ORDER if t in by_task]:
        rows = by_task[task]
        best = max(rows, key=lambda r: r["metric_test"])
        key = f"{task}/{best['model']}/{best['feature']}"
        exp_txt, gap_txt = "—", "—"
        if key in cells and cells[key]["metrics"].get("v1"):
            em = cells[key]["metrics"]["v1"]["mean"]
            if task == "severity":
                e = em.get("r2"); exp_txt = f"R²={fmt(e)}"
            else:
                e = em.get("accuracy")
                exp_txt = f"acc={fmt(e)}"
                # also note macro-F1 for the honest view
                mf = em.get("macro_f1")
                if mf is not None:
                    exp_txt += f" (macroF1={fmt(mf)})"
            if e is not None:
                gap_txt = f"−{best['metric_test'] - e:.3f}"
        w(f"| `{task}` | {best['model']}/{best['feature']} "
          f"| **{fmt(best['metric_test'])}** | {exp_txt} | {gap_txt} |\n")
    w("\n")
    w("Read this top-down: the synthetic ceilings are high for everything "
      "except localisation (`col_location`) and `severity`, yet the matching "
      "experimental numbers collapse — most starkly for `binary` and "
      "`mass_location`, which are near-perfect in-domain and near-chance (by "
      "macro-F1) on the real structure. **That collapse is the central result "
      "of the study, and it is a domain-shift effect, not a failure to "
      "learn.**\n\n")
    w("![per-task synthetic test scores](figures/synth/fig1_bespoke_by_task.png)\n\n")
    w("*Per-task synthetic test scores for every bespoke cell (red dotted = "
      "chance). Modal-feature cells dominate the learnable tasks.*\n")
    w("\n---\n")

    # ---- generalisation + cost ----
    w("## 3. In-domain generalisation and cost\n\n")
    w("![val vs test](figures/synth/fig2_val_vs_test.png)\n\n")
    w("*Synthetic validation vs synthetic test. Points hug the diagonal — "
      "there is essentially no val→test overfitting within the synthetic "
      "domain; the models that fit the val fold also fit the test fold. The "
      "generalisation problem is entirely cross-domain, not in-domain.*\n\n")
    w("![runtime vs test](figures/synth/fig3_runtime_vs_test.png)\n\n")
    w("*Training cost vs in-domain accuracy. The cheapest models (modal MLP / "
      "RF / XGB, ~1–4 s) are also the strongest in-domain; the expensive "
      "sequence models (transformer on raw time series, ~30 s) do not buy "
      "extra in-domain accuracy.*\n")
    w("\n---\n")

    # ---- per-task bespoke detail ----
    w("## 4. Per-task synthetic results (bespoke, v1, HPO-tuned)\n")
    for ti, task in enumerate([t for t in TASK_ORDER if t in by_task], 1):
        info = TASK_INFO[task]
        rows = sorted(by_task[task], key=lambda r: r["metric_test"], reverse=True)
        w(f"### 4.{ti} `{task}` — chance ≈ {info['chance']}\n\n")
        w(info["blurb"] + "\n\n")
        w(f"| model | feature | synth val | synth test ({info['metric']}) "
          f"| runtime (s) | key hyper-params |\n")
        w("|---|---|---|---|---|---|\n")
        for r in rows:
            w(f"| {r['model']} | {r['feature']} | {fmt(r.get('metric_val'))} "
              f"| **{fmt(r.get('metric_test'))}** | {r.get('runtime_s', 0):.0f} "
              f"| {hp_str(r.get('best_hyperparams', {}))} |\n")
        w("\n")
    w("\n---\n")

    # ---- the gap ----
    w("## 5. The sim-to-real gap, cell by cell\n\n")
    w("For every bespoke cell that also exists in the experimental study "
      "(30 cells, matched on task/model/feature, v1), the synthetic test "
      "score is plotted against the experimental zero-shot score in the same "
      "metric.\n\n")
    w("![sim-to-real gap](figures/synth/fig4_gap.png)\n\n")
    # gap table
    pairs = []
    for r in tm:
        key = f"{r['task']}/{r['model']}/{r['feature']}"
        if key not in cells or not cells[key]["metrics"].get("v1"):
            continue
        em = cells[key]["metrics"]["v1"]["mean"]
        if r["task"] == "severity":
            e = em.get("r2"); metric = "R²"
        else:
            e = em.get("accuracy"); metric = "acc"
        if e is None:
            continue
        pairs.append((r["task"], r["model"], r["feature"],
                      r["metric_test"], e, e - r["metric_test"], metric))
    pairs.sort(key=lambda p: p[5])  # most negative gap (biggest drop) first
    w("| cell | metric | synth test | experimental | Δ (exp − synth) |\n")
    w("|---|---|---|---|---|\n")
    for task, model, feat, s, e, d, metric in pairs:
        w(f"| `{task}/{model}/{feat}` | {metric} | {fmt(s)} | {fmt(e)} "
          f"| {d:+.3f} |\n")
    w("\n")
    w("Every classification cell loses 0.3–0.5 absolute accuracy crossing to "
      "the real data, and the binary/localisation cells that looked perfect "
      "in-domain are the ones that fall furthest. This is the quantitative "
      "statement of the sim-to-real problem the consolidated report then "
      "dissects per variant and per damage-severity tier.\n")
    w("\n---\n")

    # ---- vision ----
    w("## 6. Vision backbones — synthetic test (running sweep)\n\n")
    total = 3 * 3 * 10 * 3 * 3
    w(f"Source: `meta.synth_test` of every vision per-case JSON produced so "
      f"far — **{len(vision)} / {total}** cells complete. This section "
      f"refreshes when the script is re-run.\n\n")
    if vision:
        w("![vision synthetic test](figures/synth/fig5_vision_synth.png)\n\n")
        cov = defaultdict(int)
        for r in vision:
            cov[(r["variant"], r["seed"])] += 1
        w("**Coverage (cells done per variant×seed, of 90):**\n\n")
        w("| variant | seed42 | seed101 | seed202 |\n|---|---|---|---|\n")
        for variant in ("v1", "v2", "v2a"):
            w(f"| {variant} | " +
              " | ".join(str(cov.get((variant, s), 0))
                          for s in ("42", "101", "202")) + " |\n")
        w("\n")
        agg = defaultdict(list)
        for r in vision:
            if r["synth_test"] is not None:
                agg[(r["task"], r["backbone"], r["feature"])].append(r["synth_test"])
        if agg:
            w("**Synthetic test by cell (mean over variants+seeds done):**\n\n")
            w("| task | backbone | feature | synth test | n |\n|---|---|---|---|---|\n")
            for (task, bk, ft) in sorted(agg):
                vals = agg[(task, bk, ft)]
                w(f"| {task} | {bk} | {ft} | {sum(vals)/len(vals):.3f} "
                  f"| {len(vals)} |\n")
            w("\n")
    else:
        w("_No vision cells have finished yet; this section populates as the "
          "sweep runs._\n")
    w("\n---\n")

    # ---- reading guide ----
    w("## 7. How to read this against the consolidated report\n")
    w("- **High synth test + low experimental score = sim-to-real gap**, not "
      "a learning failure. §4–§5 show the models fit the synthetic task well; "
      "the consolidated report shows how little survives zero-shot transfer.\n")
    w("- **Accuracy vs macro-F1.** On the real set, accuracy is inflated by "
      "class imbalance (82.5 % damaged); the consolidated report uses macro-F1 "
      "as the metric of record. The headline table above shows both so the "
      "collapse is not hidden by the accuracy floor.\n")
    w("- **In-domain is solved; cross-domain is the research problem.** The "
      "val=test diagonal (§3) proves the models are not overfitting the "
      "synthetic fold — the entire generalisation gap lives at the "
      "synthetic→real boundary, which is what the variant study (v1/v2/v2a) "
      "and the DT-stratified analysis in the consolidated report attack.\n")

    OUT.write_text("".join(out))
    print(f"wrote {OUT}  (bespoke={len(tm)}, vision={len(vision)}, "
          f"{len(''.join(out)):,} chars)")


if __name__ == "__main__":
    build()
