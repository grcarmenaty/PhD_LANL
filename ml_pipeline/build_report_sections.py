"""Emit Markdown fragments that REPORT.md needs after every HPO run.

Three pieces:

  cross_model_tables.md   one cross-model comparison table per task
                          (binary / type / severity / col_location /
                          mass_location), built from
                          `results/hpo/*.json` + the
                          full-experimental eval.  Each table includes
                          every (model, feature) cell that has a
                          `<task>__<model>__<feature>.json` HPO log.

  variant_plots.md        markdown image references for every cell.
                          Used to embed the HPO + confusion / scatter
                          plots inline below the cross-model table.

  indicator_subsections.md   §7.6.k subsection per indicator (22 total),
                            with per-cell table, scatter plot,
                            HPO surface and one-paragraph narrative.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

INDICATOR_ORDER = [
    "SCI", "unsigned_SCI", "DRQ", "AIGAC", "FRFRMS", "FRFSF",
    "FRFSM_6dB", "ODS_diff", "r2_imag",
    "RVAC_mean", "RVAC_std", "RVAC_min", "RVAC_max",
    "GAC_mean", "GAC_std", "GAC_min", "GAC_max",
    "M2L_mean", "M2L_std", "M2L_min", "M2L_max", "M2L_abs_sum",
]
INDICATOR_DESC = {
    "SCI":          "Signed Structural Change Indicator — positive when CFDAC mass mostly sits below the diagonal, negative above.",
    "unsigned_SCI": "Magnitude version of SCI: `1 − |Pearson(CFDAC_ref, CFDAC_dmg)|`, bounded in `[0, 1]`.",
    "DRQ":          "Mean of the per-frequency Response Vector Assurance Criterion (RVAC); 1 = identical mode-shape, 0 = orthogonal.",
    "AIGAC":        "Mean of the per-frequency Generalised Assurance Criterion (GAC); cross-channel structural similarity.",
    "FRFRMS":       "RMS deviation between the log-magnitude FRF of the damaged sample and the pristine reference.",
    "FRFSF":        "FRF Shape Factor — peak / band-energy ratio difference vs. the pristine reference.",
    "FRFSM_6dB":    "Standard Mean with 6 dB band: total FRF deviation summed inside the 6 dB-down band around each peak.",
    "ODS_diff":     "`Σ |FRF − FRF_ref|`, an unbounded L1 distance between Operating Deflection Shapes.",
    "r2_imag":      "R² of `Im(FRF_dmg)` against `Im(FRF_ref)`.",
    "RVAC_mean":    "Mean of the RVAC vector (across frequency bins).",
    "RVAC_std":     "Standard deviation of the RVAC vector.",
    "RVAC_min":     "Minimum RVAC value across the frequency band — captures the worst-case shape mismatch.",
    "RVAC_max":     "Maximum RVAC value across the frequency band — captures the best-case agreement.",
    "GAC_mean":     "Mean of the GAC vector.",
    "GAC_std":      "Standard deviation of the GAC vector.",
    "GAC_min":      "Minimum GAC value across the frequency band.",
    "GAC_max":      "Maximum GAC value across the frequency band.",
    "M2L_mean":     "Mean of the Modal-to-Linearity ratio across the frequency band.",
    "M2L_std":      "Standard deviation of the M2L vector.",
    "M2L_min":      "Minimum M2L value.",
    "M2L_max":      "Maximum M2L value.",
    "M2L_abs_sum":  "`Σ |M2L_dmg − M2L_ref|`, an unbounded L1 sum.",
}


def _read_hpo_cells(hpo_dir: Path) -> Dict[Tuple[str, str, str], dict]:
    out: Dict[Tuple[str, str, str], dict] = {}
    for p in sorted(hpo_dir.glob("*.json")):
        blob = json.loads(p.read_text())
        key = (blob["task"], blob["model"], blob["feature"])
        out[key] = blob
    return out


def _read_exp_metrics(exp_path: Path) -> Dict[Tuple[str, str, str], dict]:
    if not exp_path.exists():
        return {}
    return {(r["task"], r["model"], r["feature"]): r
            for r in json.loads(exp_path.read_text())}


def _fmt(x, fmt="{:.3f}"):
    if x is None:
        return "—"
    return fmt.format(x)


def cross_model_tables(hpo_cells, exp_cells, out_dir: Path) -> None:
    by_task: Dict[str, list[dict]] = defaultdict(list)
    for (task, model, feat), blob in hpo_cells.items():
        exp = exp_cells.get((task, model, feat))
        by_task[task].append({
            "task": task, "model": model, "feature": feat,
            "val":  blob.get("best_metric_val"),
            "test": blob.get("best_metric_test"),
            "exp":  exp["value"] if exp else None,
        })

    pieces: list[str] = []
    for task in ("binary", "type", "severity", "col_location",
                    "mass_location"):
        rows = by_task.get(task, [])
        if not rows:
            continue
        # Sort by test desc; missing test → last.
        rows.sort(key=lambda r: (-(r["test"] if r["test"] is not None
                                           else -1e30)))
        metric_label = "R²" if task == "severity" else "acc"
        pieces.append(f"#### {task}\n")
        pieces.append(
            f"| model | feature | val {metric_label} | "
            f"test {metric_label} | exp {metric_label} |"
        )
        pieces.append("|---|---|---|---|---|")
        for r in rows:
            pieces.append(
                f"| {r['model']:<11} | `{r['feature']}` | "
                f"{_fmt(r['val'])} | {_fmt(r['test'])} | "
                f"{_fmt(r['exp'])} |"
            )
        pieces.append("")
    out = out_dir / "cross_model_tables.md"
    out.write_text("\n".join(pieces))
    print(f"wrote {out}  ({len(pieces)} lines, {len(by_task)} tasks)")


def variant_plots(hpo_cells, out_dir: Path) -> None:
    """Embed for every cell the HPO + confusion / scatter image."""
    by_task: Dict[str, list[Tuple[str, str]]] = defaultdict(list)
    for (task, model, feat), _blob in hpo_cells.items():
        by_task[task].append((model, feat))

    pieces: list[str] = []
    for task in ("binary", "type", "severity", "col_location",
                    "mass_location"):
        cells = by_task.get(task, [])
        if not cells:
            continue
        cells.sort()
        pieces.append(f"#### {task} — figures\n")
        for model, feat in cells:
            hpo_img = f"figures/hpo/{task}__{model}__{feat}.png"
            if task == "severity":
                second_img = f"figures/scatter/{task}_{model}_{feat}.png"
                second_label = "scatter"
            else:
                second_img = f"figures/confusion/{task}_{model}_{feat}.png"
                second_label = "confusion"
            pieces.append(f"![HPO — {task}/{model}/{feat}]({hpo_img})")
            pieces.append(f"![{second_label} — {task}/{model}/{feat}]({second_img})")
        pieces.append("")
    out = out_dir / "variant_plots.md"
    out.write_text("\n".join(pieces))
    print(f"wrote {out}")


def indicator_subsections(out_dir: Path) -> None:
    hpo_ind_dir = out_dir.parent / "results" / "hpo_indicators"
    if not hpo_ind_dir.exists():
        hpo_ind_dir = _REPO / "results" / "hpo_indicators"
    full_path = _REPO / "results" / "indicator_predictions_full.json"
    full_rows = (json.loads(full_path.read_text())
                     if full_path.exists() else [])
    # exp R² lookup: (indicator, model, feature) → row
    exp_lookup = {(r["indicator"], r["model"], r["feature"]): r
                     for r in full_rows}

    pieces: list[str] = []
    for k, ind in enumerate(INDICATOR_ORDER, start=7):
        # Collect all models that have a per-indicator HPO log for `modal`.
        rows = []
        for model in ("rf", "xgb", "mlp"):
            p = hpo_ind_dir / f"{ind}__{model}__modal.json"
            if not p.exists():
                continue
            blob = json.loads(p.read_text())
            erow = exp_lookup.get((ind, model, "modal"))
            rows.append({
                "model": model,
                "val":  blob.get("best_metric_val"),
                "test": blob.get("best_metric_test"),
                "hp":   blob.get("best_hyperparams"),
                "exp":  erow["exp_R2"] if erow else None,
                "mae":  erow["exp_MAE"] if erow else None,
            })
        # Pick best by exp R² (skip MLP if its exp R² is < -1e3 because
        # of the unscaled-output blowup).
        sane = [r for r in rows
                  if r["exp"] is not None and r["exp"] > -1e3]
        if sane:
            best = max(sane, key=lambda r: r["exp"])
        elif rows:
            best = max(rows, key=lambda r: r["test"] or -1e30)
        else:
            best = None
        # Transferability sentence.
        if best is None:
            transfer = ("No cross-domain metric recorded for this "
                            "indicator.")
        elif best["exp"] is None:
            transfer = (f"Best in-domain cell is {best['model']}, "
                            f"synth test {_fmt(best['test'])}.")
        elif best["exp"] >= 0.49:
            transfer = (f"**Transfers cleanly** (exp R² "
                            f"{_fmt(best['exp'], '{:+.3f}')} for "
                            f"{best['model']}/modal); usable as a "
                            "cross-domain damage proxy.")
        elif best["exp"] >= 0:
            transfer = (f"Marginal cross-domain transfer "
                            f"(exp R² {_fmt(best['exp'], '{:+.3f}')}); "
                            "no better than the experimental mean.")
        else:
            transfer = (f"**Does not transfer** — exp R² "
                            f"{_fmt(best['exp'], '{:+.3f}')} for "
                            f"{best['model']}/modal; synth-only signal.")

        pieces.append(f"### 7.6.{k} `{ind}`")
        pieces.append("")
        pieces.append(INDICATOR_DESC[ind])
        pieces.append("")
        pieces.append("| model | val R² | test R² | exp R² | exp MAE | best HPs |")
        pieces.append("|---|---|---|---|---|---|")
        for r in rows:
            pieces.append(
                f"| {r['model']:<3} | {_fmt(r['val'])} | "
                f"{_fmt(r['test'])} | {_fmt(r['exp'], '{:+.3f}')} | "
                f"{_fmt(r['mae'], '{:.4g}')} | `{r['hp']}` |"
            )
        pieces.append("")
        if best is not None:
            scatter = f"figures/indicators/scatter/{ind}_{best['model']}_modal.png"
            hpo_img = f"figures/indicators/hpo/{ind}__{best['model']}__modal.png"
            pieces.append(f"![{ind} scatter — {best['model']}/modal]({scatter})")
            pieces.append(f"![{ind} HPO — {best['model']}/modal]({hpo_img})")
            pieces.append("")
        pieces.append(transfer)
        pieces.append("")

    out = out_dir / "indicator_subsections.md"
    out.write_text("\n".join(pieces))
    print(f"wrote {out}  ({len(pieces)} lines, 22 indicators)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=_REPO / "results")
    p.add_argument("--out", type=Path, default=Path("/tmp"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    hpo_cells = _read_hpo_cells(args.results / "hpo")
    exp_cells = _read_exp_metrics(args.results / "balanced" / "experimental_full_evaluation.json")
    cross_model_tables(hpo_cells, exp_cells, args.out)
    variant_plots(hpo_cells, args.out)
    indicator_subsections(args.out)


if __name__ == "__main__":
    main()
