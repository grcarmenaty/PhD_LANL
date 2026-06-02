"""Build REPORT_synth.md — the *synthetic-domain* (in-distribution)
training results, BEFORE any zero-shot test against the real
experimental data.

Why a separate report? The consolidated report
(`REPORT_CONSOLIDATED.md`) scores every model on the 2 638-case real
LANL set — the sim-to-real transfer. This report is the companion:
how well each model fits the *synthetic* task on a held-out synthetic
test fold. It is the learnability ceiling — the gap between the numbers
here and those in the consolidated report *is* the sim-to-real gap.

Two model families:
  * Bespoke models (mlp/cnn/cnn1d/cnn2d/rf/xgb/transformer) — synth
    val + synth test from `results/training_metrics.json` (HPO-tuned,
    v1 baseline, the 5 original tasks).
  * Vision backbones (convnext_tiny/resnet50/vit_b_16) — synth test
    pulled from the `meta.synth_test` field of every per-case JSON the
    vision sweep has produced so far, across variants and seeds. This
    section grows as the sweep completes; re-run this script to refresh.

Reads:
  results/training_metrics.json
  results_vision/<variant>_seed<seed>/per_case_vision/<task>_<bk>_<ft>.json
Writes:
  results/REPORT_synth.md
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
TM = _REPO / "results" / "training_metrics.json"
VIS_ROOT = _REPO / "results_vision"
OUT = _REPO / "results" / "REPORT_synth.md"

RANDOM = {  # chance baselines per task (for reading the numbers)
    "binary": "0.5 (balanced) / 0.825 acc by majority-class",
    "is_pristine": "0.5", "is_bolt": "0.5", "is_crack": "0.5",
    "is_hole": "0.5", "is_mass": "0.5",
    "type": "0.20 (5-class)", "col_location": "0.11 (9-class)",
    "mass_location": "0.33 (3-class)", "severity": "R²=0 = predict-mean",
}

MODEL_DESC = {
    "mlp": "MLP", "cnn": "1-D CNN", "cnn1d": "1-D CNN (deep)",
    "cnn2d": "2-D CNN", "rf": "random forest", "xgb": "gradient boosting",
    "transformer": "transformer",
    "convnext_tiny": "ConvNeXt-Tiny (timm, IN1k)",
    "resnet50": "ResNet50 (timm, IN1k)",
    "vit_b_16": "ViT-B/16 (timm, IN21k→IN1k)",
}


def load_bespoke():
    if not TM.exists():
        return []
    return json.load(open(TM))


def load_vision():
    """Return rows: {variant, seed, task, backbone, feature, kind, synth_test}."""
    rows = []
    if not VIS_ROOT.exists():
        return rows
    for d in sorted(VIS_ROOT.iterdir()):
        if not d.is_dir():
            continue
        name = d.name  # <variant>_seed<seed>
        if "_seed" not in name:
            continue
        variant, seed = name.split("_seed")
        pc = d / "per_case_vision"
        if not pc.exists():
            continue
        for jf in sorted(pc.glob("*.json")):
            try:
                meta = json.load(open(jf)).get("meta", {})
            except Exception:
                continue
            rows.append({
                "variant": variant, "seed": seed,
                "task": meta.get("task"), "backbone": meta.get("backbone"),
                "feature": meta.get("feature"), "kind": meta.get("kind"),
                "synth_test": meta.get("synth_test"),
            })
    return rows


def fmt(x):
    return "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def build():
    bespoke = load_bespoke()
    vision = load_vision()
    out = []
    w = out.append

    w("# LANL 3SBB — Synthetic-domain training results (pre-transfer)\n")
    w("**Companion to** [`REPORT_CONSOLIDATED.md`](REPORT_CONSOLIDATED.md). "
      "Date: 2026-06-02.\n")
    w("\n")
    w("This report shows **in-distribution** performance: every model is "
      "trained on synthetic data and scored on a **held-out synthetic test "
      "fold** — *before* any contact with the real experimental set. These "
      "are the numbers to compare against the cross-domain (experimental) "
      "numbers in the consolidated report; the difference between them is the "
      "sim-to-real gap.\n")
    w("\n")
    w("- Classification metric: accuracy on the synthetic test fold "
      "(70/15/15 stratified split).\n")
    w("- Regression (`severity`): R² on the synthetic test fold.\n")
    w("- Bespoke numbers are HPO-tuned on the **v1** baseline physics (the 5 "
      "original tasks). Vision numbers span **v1/v2/v2a × seeds 42/101/202** "
      "and all tasks attempted, and grow as the sweep completes.\n")
    w("\n---\n")

    # ---------------- Bespoke ----------------
    w("## 1. Bespoke models — synthetic val/test (v1, HPO-tuned)\n")
    w(f"Source: `results/training_metrics.json` ({len(bespoke)} cells).\n\n")
    by_task = defaultdict(list)
    for r in bespoke:
        by_task[r["task"]].append(r)
    for task in sorted(by_task):
        rows = sorted(by_task[task],
                      key=lambda r: r.get("metric_test", 0), reverse=True)
        mname = rows[0]["metric_name"]
        w(f"### `{task}` — chance ≈ {RANDOM.get(task, '?')}\n\n")
        w(f"| model | feature | synth val ({mname}) | synth test ({mname}) | runtime (s) |\n")
        w("|---|---|---|---|---|\n")
        for r in rows:
            w(f"| {r['model']} | {r['feature']} | {fmt(r.get('metric_val'))} "
              f"| **{fmt(r.get('metric_test'))}** | {r.get('runtime_s', 0):.0f} |\n")
        w("\n")
    w("\n---\n")

    # ---------------- Vision ----------------
    w("## 2. Vision backbones — synthetic test (running sweep)\n")
    total = 3 * 3 * 10 * 3 * 3
    w(f"Source: `meta.synth_test` of every per-case JSON produced so far — "
      f"**{len(vision)} / {total}** vision cells complete.\n\n")
    if not vision:
        w("_No vision cells finished yet; this section will populate as the "
          "sweep runs._\n")
    else:
        # Coverage by variant/seed
        cov = defaultdict(int)
        for r in vision:
            cov[(r["variant"], r["seed"])] += 1
        w("**Coverage (cells done per variant×seed, of 90):**\n\n")
        w("| variant | seed42 | seed101 | seed202 |\n|---|---|---|---|\n")
        for variant in ("v1", "v2", "v2a"):
            cells = " | ".join(str(cov.get((variant, s), 0))
                                for s in ("42", "101", "202"))
            w(f"| {variant} | {cells} |\n")
        w("\n")
        # Per-task synth test, averaged across seeds, per variant×backbone×feature
        agg = defaultdict(list)  # (task, variant, backbone, feature) -> [synth_test]
        for r in vision:
            if r["synth_test"] is None:
                continue
            agg[(r["task"], r["variant"], r["backbone"], r["feature"])].append(
                r["synth_test"])
        tasks = sorted({k[0] for k in agg})
        for task in tasks:
            w(f"### `{task}` — chance ≈ {RANDOM.get(task, '?')}\n\n")
            w("| backbone | feature | v1 | v2 | v2a |\n|---|---|---|---|---|\n")
            combos = sorted({(k[2], k[3]) for k in agg if k[0] == task})
            for bk, ft in combos:
                def cell(variant):
                    vals = agg.get((task, variant, bk, ft))
                    if not vals:
                        return "—"
                    m = sum(vals) / len(vals)
                    return f"{m:.3f}" + (f" (n={len(vals)})" if len(vals) < 3 else "")
                w(f"| {bk} | {ft} | {cell('v1')} | {cell('v2')} | {cell('v2a')} |\n")
            w("\n")
    w("\n---\n")

    # ---------------- Reading the gap ----------------
    w("## 3. How to read this against the consolidated (experimental) report\n")
    w("- **High synth test + low experimental score = sim-to-real gap**, not "
      "a failure to learn. Most cells here fit the synthetic task well; the "
      "consolidated report shows how little of that survives zero-shot "
      "transfer to the real structure.\n")
    w("- The synthetic test fold and the synthetic train fold come from the "
      "*same* generator, so these numbers are optimistic by construction — "
      "they are the ceiling, not the deployment estimate.\n")
    w("- For the cross-domain story, per-cell and per-variant, see "
      "[`REPORT_CONSOLIDATED.md`](REPORT_CONSOLIDATED.md).\n")

    OUT.write_text("".join(out))
    print(f"wrote {OUT}  (bespoke={len(bespoke)} cells, vision={len(vision)} cells)")


if __name__ == "__main__":
    build()
