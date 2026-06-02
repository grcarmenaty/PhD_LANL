"""Build the single, comprehensive consolidated report.

For each of the 10 tasks:
  - Task definition + axes + reference lines
  - One cell-by-cell subsection: every (model × feature) cell with its purpose,
    3-variant × 3-seed mean ± sd of the primary metric, secondary metrics
  - Embedded plots: cell_zoo bar chart + dt_per_task curve
After all 10 tasks:
  - Cross-task comparison & verdict
  - Methodology, vision status, limitations, recommendations
  - Reproducibility & artefact index

Reads:
  results/cells_v1_v2_v2a.json   (per-cell aggregated metrics, 244 cells)
  results/dt_compare_v1_v2_v2a.json  (DT sweep best-cell-per-task curves)
Writes:
  results/REPORT_CONSOLIDATED.md
"""
from __future__ import annotations
import json
from pathlib import Path
from textwrap import dedent

_REPO = Path(__file__).resolve().parent.parent
CELLS_JSON = _REPO / "results" / "cells_v1_v2_v2a.json"
DT_JSON = _REPO / "results" / "dt_compare_v1_v2_v2a.json"
OUT = _REPO / "results" / "REPORT_CONSOLIDATED.md"


# ---------------------------------------------------------------------------
# Task metadata: long-form purpose for each task and each (model, feature) pair
# ---------------------------------------------------------------------------

TASK_DEFS = {
    "binary": dict(
        kind="cls_binary",
        title="Damage detection — `binary` (any damage vs pristine)",
        question=(
            "Given an experimental spectrum, is the structure damaged in any way "
            "(crack, hole, bolt loosening, or added mass) **or** pristine?"
        ),
        axis="presence/absence of any damage (model output → ŷ ∈ {0=pristine, 1=damaged})",
        notes=(
            "Class prior on the 2 638-case set is 82.5 % damaged (2 176 damaged / "
            "462 pristine). Accuracy is misleading; **macro-F1 is the metric of "
            "record**. A class-collapsed model (predicting all-damaged) scores "
            "accuracy 0.825 but macro-F1 ≈ 0.452. Random-balanced macro-F1 = 0.5."
        ),
        score_key="macro_f1", score_label="macro-F1",
        higher_better=True, random=0.5,
    ),
    "is_bolt": dict(
        kind="cls_binary",
        title="Bolt-loosening detection — `is_bolt`",
        question="Does this spectrum exhibit bolt-loosening damage?",
        axis="positives = bolt-loosening cases; negatives = pristine + all other damage types",
        notes=(
            "Strongest learnable axis in the study: at high loosening percentage "
            "(≥ 85 %), best cells reach ≥ 0.82 macro-F1 across all three variants. "
            "Random = 0.5."
        ),
        score_key="macro_f1", score_label="macro-F1",
        higher_better=True, random=0.5,
    ),
    "is_crack": dict(
        kind="cls_binary",
        title="Crack detection — `is_crack`",
        question="Does this spectrum exhibit crack damage?",
        axis="positives = crack cases; negatives = pristine + non-crack damage",
        notes=(
            "Hard signal — crack depth/length physically couples weakly to global "
            "modal/FRF response. Best cells hover near 0.55–0.65 macro-F1. "
            "Random = 0.5."
        ),
        score_key="macro_f1", score_label="macro-F1",
        higher_better=True, random=0.5,
    ),
    "is_hole": dict(
        kind="cls_binary",
        title="Hole detection — `is_hole`",
        question="Does this spectrum exhibit a drilled-hole defect?",
        axis="positives = hole cases; negatives = pristine + non-hole damage",
        notes=(
            "This is the v1 modal-MLP showpiece (BA 0.651 on the registered cell) "
            "and the one that v2 catastrophically collapses (BA 0.500). "
            "Random = 0.5."
        ),
        score_key="macro_f1", score_label="macro-F1",
        higher_better=True, random=0.5,
    ),
    "is_mass": dict(
        kind="cls_binary",
        title="Added-mass detection — `is_mass`",
        question="Does this spectrum exhibit an added-mass condition?",
        axis="positives = added-mass cases; negatives = pristine + non-mass damage",
        notes=(
            "Added-mass shifts global natural frequencies — a globally encoded "
            "signal. Random = 0.5."
        ),
        score_key="macro_f1", score_label="macro-F1",
        higher_better=True, random=0.5,
    ),
    "is_pristine": dict(
        kind="cls_binary",
        title="Pristine detection — `is_pristine` (the inverse of `binary`)",
        question="Is this spectrum genuinely pristine?",
        axis="positives = pristine cases (~17.5 %); negatives = all damage types",
        notes=(
            "The complement of `binary`. Importantly it does **not** improve under "
            "DT restriction — the models can name a damage **type** when severe, but "
            "they cannot reliably answer 'is this damaged at all?'. This is the "
            "central synth-to-real failure mode. Random = 0.5."
        ),
        score_key="macro_f1", score_label="macro-F1",
        higher_better=True, random=0.5,
    ),
    "type": dict(
        kind="cls_multi",
        title="Damage-type classification — `type` (5 classes)",
        question="Which of {pristine, crack, hole, bolt loosening, added mass} is this?",
        axis="closed-set 5-way classification",
        notes=(
            "Hardest of the multi-class tasks because it mixes the difficult "
            "`is_crack` axis with the easy `is_bolt` axis. Random = 1/5 = 0.20."
        ),
        score_key="macro_f1", score_label="macro-F1",
        higher_better=True, random=0.20,
    ),
    "col_location": dict(
        kind="cls_multi",
        title="Column-location classification — `col_location`",
        question="Which column section is damaged (multi-class spatial localization)?",
        axis="9 column sections (3 columns × 3 tiers)",
        notes=(
            "Random = 1/9 ≈ 0.111. Spatial localization probes whether the spectrum "
            "encodes 'where' as well as 'what'. v2 specifically degrades here; "
            "v2a (asymmetric damage + v1 DR) is the **best** localizer."
        ),
        score_key="macro_f1", score_label="macro-F1",
        higher_better=True, random=1/9,
    ),
    "mass_location": dict(
        kind="cls_multi",
        title="Mass-location classification — `mass_location`",
        question="Which floor/tier carries the added mass?",
        axis="3 tiers (added-mass cases only)",
        notes=(
            "Random = 1/3 ≈ 0.333. DT-invariant by construction (mass cases have "
            "no native severity axis); same score across all DT thresholds."
        ),
        score_key="macro_f1", score_label="macro-F1",
        higher_better=True, random=1/3,
    ),
    "severity": dict(
        kind="regression",
        title="Bolt-severity regression — `severity` (MAE, lower = better)",
        question="What fraction of the bolt is loosened (regression on bolt cases)?",
        axis="continuous bolt-loosening fraction ∈ [0, 1]",
        notes=(
            "Only defined on bolt cases. Sample mean ≈ 0.7, std ≈ 0.18 — the trivial "
            "predict-the-mean baseline gives MAE ≈ 0.15. **Lower is better.**"
        ),
        score_key="mae", score_label="MAE",
        higher_better=False, random=None,
    ),
}


# Model and feature glossaries (used to build per-cell purpose lines)
MODELS = {
    "mlp":   "fully connected baseline (3 hidden layers); fast, low capacity",
    "cnn":   "1-D convolutional net on the feature sequence; learns local spectral motifs",
    "cnn1d": "deeper 1-D CNN variant; benefits from longer/denser inputs",
    "cnn2d": "2-D CNN treating the feature as a (channel × bin) image",
    "rf":    "random forest on flattened feature; robust non-parametric baseline",
    "xgb":   "gradient-boosted trees; usually pairs best with hand-crafted modal features",
    "transformer": "small self-attention encoder; captures long-range spectral structure",
}

FEATURES = {
    "modal":      "physics-grounded — extracted natural frequencies + damping ratios + mode shapes",
    "frf_mag":    "FRF magnitude across frequency bins — what most damage indicators key on",
    "cfdac":      "Cross-FRF Damage Assurance Criterion — pristine-vs-current FRF alignment image",
    "cfdac_real": "real part only of CFDAC",
    "cfdac_imag": "imaginary part only of CFDAC",
    "cfdac_mag":  "magnitude of CFDAC",
    "cfdac_phase": "phase of CFDAC",
    "cfdac_realimag": "real and imaginary CFDAC concatenated",
    "cfdac_magphase": "magnitude and phase CFDAC concatenated",
    "cfdac_all":  "all 4 CFDAC channels stacked",
}


def cell_purpose(task: str, model: str, feature: str) -> str:
    mdesc = MODELS.get(model, model)
    fdesc = FEATURES.get(feature, feature)
    return f"**{model}** ({mdesc}) on **{feature}** ({fdesc})."


# ---------------------------------------------------------------------------
# Helpers to format metric rows
# ---------------------------------------------------------------------------

def fmt(x, k=3, dash="—"):
    if x is None: return dash
    if isinstance(x, float):
        return f"{x:.{k}f}"
    return str(x)


def mean_sd(rec_var, key):
    """Return formatted 'mean ± sd' for a metric, or '—' if missing."""
    if rec_var is None: return "—"
    m = rec_var["mean"].get(key)
    s = rec_var["sd"].get(key)
    if m is None: return "—"
    return f"{m:.3f} ± {s:.3f}" if s is not None else f"{m:.3f}"


def winner(rec, score_key, higher_better):
    """Determine the winning variant and the spread."""
    scores = {}
    for v in ("v1", "v2", "v2a"):
        mv = rec["metrics"].get(v)
        if not mv: continue
        s = mv["mean"].get(score_key)
        if s is not None: scores[v] = s
    if not scores: return None, None
    if higher_better:
        best_v = max(scores, key=scores.get)
    else:
        best_v = min(scores, key=scores.get)
    spread = max(scores.values()) - min(scores.values())
    return best_v, spread


def secondary_line(rec, kind):
    """Compose a short secondary-metrics line per kind."""
    if kind == "cls_binary":
        # Show v1 BA, TPR, TNR (the most diagnostic accessory metrics)
        keys = [("v1", "bal_acc", "BA"), ("v1", "tpr", "TPR"), ("v1", "tnr", "TNR")]
    elif kind == "cls_multi":
        keys = [("v1", "bal_acc", "BA"), ("v1", "accuracy", "acc")]
    else:  # regression
        keys = [("v1", "r2", "R²"), ("v1", "mse", "MSE")]
    parts = []
    for v, k, lab in keys:
        mv = rec["metrics"].get(v)
        if not mv: continue
        m = mv["mean"].get(k)
        if m is not None:
            parts.append(f"{lab} = {m:.3f}")
    return ", ".join(parts) if parts else "—"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build():
    J = json.load(open(CELLS_JSON))
    cells = J["cells"]

    DTJ = json.load(open(DT_JSON))
    bp = DTJ["best_per_task"]
    dt_grid = DTJ["dt_grid"]

    out = []
    w = out.append

    # ---- Header ----
    w("# LANL 3SBB — Synth-to-Real Damage Diagnosis: Consolidated Report\n")
    w("**Author:** G. Carmenaty (PhD work, 2024–2026)  \n")
    w("**Date:** 2026-06-02  \n")
    w("**Scope:** Single source of truth. Supersedes every other `REPORT_*.md` "
      "in this directory (deprecation banners added to each).\n")
    w("\n---\n")

    # ---- Executive summary ----
    w("## Executive summary\n")
    w(dedent("""
        - Three synthetic-physics variants of the 3-storey bookcase benchmark are
          compared on the **same** 2 638-case experimental set under zero-shot
          transfer:
          - **v1** — baseline (`variation.py`); symmetric crack/hole; standard DR.
          - **v2** — widened domain randomisation (P1.2, `variation_v2.py`);
            asymmetric crack/hole (P2.1/P2.2).
          - **v2a** — disentangling ablation: v2's asymmetric damage geometry on
            v1's baseline DR.
        - **244 cells** = (10 tasks) × (model, feature) combinations were trained
          on each variant for **3 seeds** (42 / 101 / 202), giving 244 × 3 × 3 =
          **2 196 individual model fits** behind the numbers in this report.
        - **Verdict:**
          1. **v1 remains the reference.** It is robust and best-or-tied on
             localization.
          2. **v2 is REJECTED, multi-seed confirmed** — the `is_hole/mlp/modal`
             cell collapses to chance (BA 0.500) across all three seeds, and
             `mass_location` drops sharply.
          3. **v2a is REJECTED by its own pre-registered rule (C4 floor)** but
             only marginally. Crucially, v2a **identifies the cause** of v2's
             regression: the widened domain randomisation, **not** the asymmetric
             damage geometry. A separate widened-DR-only variant (v2b) is
             therefore unnecessary.
        - **The bigger story** the cell-by-cell view reveals: the central
          synth-to-real failure is **detection** (`binary` / `is_pristine`), not
          any single variant. DT-stratification helps every classification axis
          *except* these two. The models can name a damage type when it is severe;
          they cannot reliably answer "is this damaged at all?".
        """).strip())
    w("\n")
    w("\n---\n")

    # ---- Methodology block ----
    w("## Methodology\n")
    w(dedent("""
        ### Cells
        A "cell" is one trained-and-evaluated model. Each cell is uniquely
        determined by:

        - **task** — what is being predicted (10 tasks; see per-task sections);
        - **model** — architecture (mlp, cnn, cnn1d, cnn2d, rf, xgb, transformer);
        - **feature** — input representation (modal, frf_mag, and 8 cfdac variants).

        Not every combination is run — some are skipped where the feature is
        incompatible with the model (e.g. cfdac_all on cnn1d). The 244 cells
        that did run are the same across all 3 variants and all 3 seeds.

        ### Variants & seeds
        | variant | domain randomisation | damage geometry | source script |
        |---|---|---|---|
        | v1  | baseline           | symmetric crack/hole         | `variation.py`     |
        | v2  | **widened (P1.2)** | **asymmetric (P2.1/P2.2)**   | `variation_v2.py`  |
        | v2a | baseline (= v1)    | **asymmetric (P2.1/P2.2)**   | `variation.py` + asymmetric placement |

        Each variant was retrained for **seeds 42 / 101 / 202**. Reported "3-seed
        mean ± sd" is across these three runs.

        ### Test set
        The same 2 638-case experimental LANL 3SBB set is held fixed across all
        variants/seeds — zero-shot synth-to-real transfer is the protocol.

        ### Metrics
        - Binary tasks: **macro-F1** is the metric of record (the test set is
          82.5 % damaged on `binary`; accuracy is uninformative).
        - Multi-class tasks: macro-F1 (chance varies by class count; per-task
          random baseline given inline).
        - Regression (`severity`): **MAE** (lower is better).

        ### DT-stratification
        A second analysis re-scores the same predictions while gradually removing
        sub-threshold positives from the test set — the **damage threshold (DT)**
        sweep. It answers: *within the domain of competence (severe damage),
        what does each variant's best cell achieve?* This is exploratory
        (post-hoc cell selection); it is hypothesis-generating, not confirmatory.

        ### Reproduce
        ```bash
        python -m ml_pipeline.cells_aggregate         # 244-cell rollup
        python -m ml_pipeline.dt_compare_variants     # DT sweep
        python -m ml_pipeline.dt_feature_sweep        # per-damage-axis sweep
        python -m ml_pipeline.plot_cell_zoo           # per-task bars (10 PNGs)
        python -m ml_pipeline.plot_per_task_dt        # per-task DT curves (10 PNGs)
        python -m ml_pipeline.plot_dt_3way            # cross-task figures (7 PNGs)
        python -m ml_pipeline.build_consolidated_report  # regenerate this file
        ```
        """).strip())
    w("\n")
    w("\n---\n")

    # ---- Per-task sections (10) ----
    w("## Per-task review (10 tasks, 244 cells)\n")

    by_task = {}
    for k, v in cells.items():
        by_task.setdefault(v["task"], []).append((k, v))

    for tnum, task in enumerate(sorted(TASK_DEFS), 1):
        meta = TASK_DEFS[task]
        task_cells = by_task.get(task, [])
        score_key = meta["score_key"]
        higher_better = meta["higher_better"]

        # Sort by best variant's score
        def sk(item):
            _, rec = item
            vals = []
            for v in ("v1", "v2", "v2a"):
                mv = rec["metrics"].get(v)
                if not mv: continue
                s = mv["mean"].get(score_key)
                if s is not None: vals.append(s)
            if not vals: return -1e9 if higher_better else 1e9
            return max(vals) if higher_better else min(vals)
        task_cells = sorted(task_cells, key=sk, reverse=higher_better)

        w(f"### {tnum}. {meta['title']}\n")
        w(f"**Question.** {meta['question']}\n")
        w(f"**Output axis.** {meta['axis']}\n")
        w(f"**Notes.** {meta['notes']}\n")
        w(f"**Cells in this task:** {len(task_cells)}\n\n")

        # Plot embeds
        w(f"![cell-zoo: all {len(task_cells)} cells × 3 variants for {task}]"
          f"(figures/cell_zoo/{task}.png)\n\n")
        w(f"![DT sweep: best cell per variant vs DT for {task}]"
          f"(figures/dt_per_task/{task}.png)\n\n")

        # Per-cell explanation block
        w(f"#### Cells in `{task}`\n\n")
        w("Each cell shows what it explores, the 3-seed mean ± sd of "
          f"{meta['score_label']} per variant, and the cross-variant winner.\n\n")
        # Big table first (compact view)
        w("| cell | v1 | v2 | v2a | best | spread |\n")
        w("|---|---|---|---|---|---|\n")
        for ck, rec in task_cells:
            v1s = mean_sd(rec["metrics"].get("v1"), score_key)
            v2s = mean_sd(rec["metrics"].get("v2"), score_key)
            v2as = mean_sd(rec["metrics"].get("v2a"), score_key)
            best_v, spread = winner(rec, score_key, higher_better)
            label = f"`{rec['model']}/{rec['feature']}`"
            best_lbl = best_v if best_v else "—"
            spread_s = f"{spread:.3f}" if spread is not None else "—"
            w(f"| {label} | {v1s} | {v2s} | {v2as} | **{best_lbl}** | {spread_s} |\n")
        w("\n")

        # Detailed per-cell paragraphs
        w("##### Cell-by-cell\n\n")
        for ck, rec in task_cells:
            model = rec["model"]; feature = rec["feature"]
            best_v, spread = winner(rec, score_key, higher_better)
            v1m = mean_sd(rec["metrics"].get("v1"), score_key)
            v2m = mean_sd(rec["metrics"].get("v2"), score_key)
            v2am = mean_sd(rec["metrics"].get("v2a"), score_key)
            sec = secondary_line(rec, meta["kind"])
            w(f"- **`{model}/{feature}`** — {cell_purpose(task, model, feature)}  \n")
            w(f"  {meta['score_label']} — v1: {v1m} · v2: {v2m} · v2a: {v2am}.  \n")
            w(f"  Best variant: **{best_v or '—'}** (spread "
              f"{spread:.3f}" + (f").  \n" if spread is not None else "—).  \n"))
            w(f"  Secondary (v1): {sec}.\n")
        w("\n---\n")

    # ---- Cross-task verdict ----
    w("## Cross-task synthesis\n")
    w(dedent("""
        ### A. The v1 reference
        v1 wins or ties on **localization** (`col_location` severe-only, `mass_location`)
        and holds the `is_hole/mlp/modal` modal-MLP signal. It is the safe
        reference.

        ### B. The v2 collapse — multi-seed confirmed, cause isolated
        - `is_hole/mlp/modal`: v1 BA 0.651 → v2 BA **0.500** (chance) for all 3
          seeds.
        - `is_crack/mlp/modal`: v1 BA 0.596 → v2 BA **0.500**.
        - `mass_location`: 0.429 → **0.259**.
        - **v2a does NOT collapse** on these cells. The only thing v2 adds over
          v2a is the **widened DR (P1.2)** → that is the culprit, not the
          asymmetric damage geometry.

        ### C. The v2a verdict — REJECTED-but-marginal, hypothesis-rescuing
        - C4 floor (`is_hole/mlp/modal` ≥ v1 − 2σ) fails by ~1σ (0.621 vs floor
          0.653).
        - C2 (`col_location/mlp/modal` 2σ improvement) fails.
        - However, asymmetric geometry is **best-localizer** in the
          spatial-classification tasks (col_location severe-only 0.328 ≥ v1 0.309),
          and `is_crack` improves directionally.
        - **Net:** v2a does not adopt under the registered rule, but it succeeds at
          its scientific job — pinpointing widened DR as v2's failure mode.

        ### D. The detection failure
        Across `binary` and `is_pristine`, **DT restriction does not help.** This
        is the real, variant-independent limitation: the synthetic spectra teach
        the models damage *characterisation*, not damage *detection*. Any future
        physics iteration should target this, not the geometry knob.

        ### E. Tasks that transfer at high severity (good news)
        - `is_bolt` at 85 % loosening → ~0.82 macro-F1 across all variants.
        - `is_hole` at ≥ 6 mm diameter → ≥ 0.65 across all variants.
        - `type` (5-cls) climbs steadily with DT restriction.
        - `severity` (MAE) improves with restriction, all three variants.

        These results are **robust across the physics variant** — the positive
        signal in the study is not an artefact of one DR choice.
        """).strip())
    w("\n")
    w("\n---\n")

    # ---- Vision status ----
    w("## Vision-backbone status\n")
    w(dedent("""
        The pre-existing paper found strongest gains from ImageNet-pretrained
        vision backbones (ResNet50, EfficientNet-B0, ConvNeXt-T, Swin-T, ViT-B/16)
        on CFDAC images. For this v1/v2/v2a multi-seed study, the **bespoke
        tabular models (mlp/cnn/cnn1d/cnn2d/rf/xgb/transformer)** were rerun for
        full coverage but the vision sweep was **not** repeated for v2/v2a yet —
        only v1 vision (15 per-case files, single seed) is on disk. A v1 vs
        bespoke spot-check on the `type` task is shown in
        `figures/dt_3way/fig6_vision_vs_bespoke.png`.

        **Open work:** repeat the 5-backbone × 3-feature vision sweep for v2 and
        v2a, 3 seeds each. ETA per the per-cell budget is ~12 h on the current
        accelerator. No code changes required; the existing
        `ml_pipeline/vision_*` configuration is unchanged.
        """).strip())
    w("\n")
    w("\n---\n")

    # ---- Limitations ----
    w("## Limitations & honest caveats\n")
    w(dedent("""
        - **Post-hoc cell selection on the restricted set** (DT-stratified
          best-cell-per-task figures) is hypothesis-generating, not confirmatory.
          The fixed-cell pre-registered rule is the gate; everything else is
          exploratory.
        - **n = 3 seeds.** σ is noisy; "within ~1σ" decisions remain marginal.
        - **Single experimental set.** All conclusions reference the 2 638-case
          LANL 3SBB benchmark. Generalisation to other structures is not tested.
        - **Vision sweep is v1-only at present** (see above).
        - **`binary` class-collapse.** Many "best" cells on `binary` are class-
          collapsed (predict all-damaged). Macro-F1 catches this; raw accuracy
          would not. Don't be misled by the 0.825 accuracy floor.
        """).strip())
    w("\n")
    w("\n---\n")

    # ---- Recommendations ----
    w("## Recommendations\n")
    w(dedent("""
        1. **Keep v1 as the operational synth-to-real reference.** v2 is
           rejected; v2a is rejected-but-marginal.
        2. **Do not run v2b (widened-DR-only).** v2a already identifies widened
           DR as the regression cause; a third ablation adds compute, not
           information.
        3. **Re-tune the DR ranges** in `variation_v2.py` (the P1.2 widening)
           rather than the damage geometry — the geometry is benign-to-helpful.
        4. **Investigate detection separately.** The flat-or-falling `binary` /
           `is_pristine` curves under DT restriction point at a representational
           gap (synthetic pristine spectra do not match real pristine spectra),
           not at any damage-physics knob. Candidate: a synth-to-real domain
           adaptation step *only on the pristine class*.
        5. **Complete the vision sweep** for v2 and v2a (15 cells × 3 seeds
           each).
        """).strip())
    w("\n")
    w("\n---\n")

    # ---- Artefact index ----
    w("## Artefact index\n")
    w(dedent("""
        ### Code
        - `ml_pipeline/cells_aggregate.py` — per-cell rollup → `results/cells_v1_v2_v2a.json`
        - `ml_pipeline/dt_compare_variants.py` — DT sweep → `results/dt_compare_v1_v2_v2a.json`
        - `ml_pipeline/dt_feature_sweep.py` — per-damage-axis sweep → `results/dt_feature_sweep.json`
        - `ml_pipeline/plot_cell_zoo.py` — 10 per-task cell-zoo bars
        - `ml_pipeline/plot_per_task_dt.py` — 10 per-task DT curves
        - `ml_pipeline/plot_dt_3way.py` — 7 cross-task figures (severity dist, DT grid, tier bars, feature axis, severity MAE, vision vs bespoke, failure modes)
        - `ml_pipeline/build_consolidated_report.py` — generates this file

        ### Data
        - `results/experimental_full_per_case_v1_seed{42,101,202}.json` (3 files)
        - `results/experimental_full_per_case_v2_seed{42,101,202}.json` (3 files)
        - `results_v2a_seed{42,101,202}/experimental_full_per_case.json` (3 files)
        - `results/cells_v1_v2_v2a.json` — derived
        - `results/dt_compare_v1_v2_v2a.json` — derived
        - `results/dt_feature_sweep.json` — derived

        ### Figures (27 total)
        - `results/figures/cell_zoo/{task}.png` (10)
        - `results/figures/dt_per_task/{task}.png` (10)
        - `results/figures/dt_3way/fig{1..7}_*.png` (7)

        ### Superseded reports (kept for history, deprecation banners added)
        - `REPORT.md`, `REPORT_full.md`, `REPORT_definitive.md`, `REPORT_final.md`
        - `REPORT_dt_sweep.md`, `REPORT_dt_3way_verdict.md`, `REPORT_dt_3way_full.md`
        - `REPORT_v2_chunk_regen.md`, `REPORT_v2a_chunk_regen.md`
        - `REPORT_severity_stratified.md`, `REPORT_simtoreal.md`, `REPORT_noisy_mixed.md`
        - `REPORT_modal_gap_diagnostic.md`, `REPORT_noise.md`
        - `REPORT_vision.md`, `REPORT_vision_v2.md`
        """).strip())
    w("\n")

    OUT.write_text("".join(out))
    print(f"wrote {OUT}  ({len(''.join(out)):,} chars, {len(out)} lines/blocks)")


if __name__ == "__main__":
    build()
