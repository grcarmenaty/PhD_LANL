"""Compare indicator regression against direct damage-parameter regression.

For each ``(model, feature)`` cell common to the indicator-prediction
HPO (``results/hpo_indicators/``) and the damage-parameter evaluation
(``results/experimental_full_evaluation.json``) this script builds a
side-by-side table:

  | model | feature | best indicator   | exp R² | severity exp R² |
                                  type exp acc | col_location exp acc |
                                  mass_location exp acc | binary exp acc |

The objective is to answer "does predicting an indicator transfer
better cross-domain than predicting the underlying damage parameter
directly?".  Inputs that are shared between the two pipelines
(currently `modal`) yield the most interpretable comparison; the
script also reports the indicator-only rows when no matching damage
cell exists.

Outputs:
    results/indicator_vs_damage.json   detailed records
    results/indicator_vs_damage_table.md   markdown table fragment
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _best_indicator(ind_rows: list[dict], model: str, feature: str
                      ) -> tuple[str | None, float | None, float | None]:
    """Return (best indicator name, exp R², exp MAE) for the
    `(model, feature)` cell, evaluating across the 22 indicators."""
    cell = [r for r in ind_rows
            if r["model"] == model and r["feature"] == feature]
    cell = [r for r in cell if r.get("exp_R2") is not None]
    if not cell:
        return None, None, None
    # Filter pathological MLP cases (exp R² < -1e6) so they don't dominate.
    sane = [r for r in cell if r["exp_R2"] > -1e3]
    pool = sane if sane else cell
    best = max(pool, key=lambda r: r["exp_R2"])
    return best["indicator"], float(best["exp_R2"]), float(best["exp_MAE"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=_REPO / "results")
    args = p.parse_args()

    ind_path = args.results / "indicator_predictions_full.json"
    dmg_path = args.results / "experimental_full_evaluation.json"
    if not ind_path.exists() or not dmg_path.exists():
        sys.exit(f"missing {ind_path} or {dmg_path}")

    ind_rows = json.loads(ind_path.read_text())
    dmg_rows = json.loads(dmg_path.read_text())

    # Index damage rows by (task, model, feature).
    dmg_index: dict[tuple[str, str, str], dict] = {}
    for r in dmg_rows:
        dmg_index[(r["task"], r["model"], r["feature"])] = r

    # Gather every (model, feature) cell that appears anywhere.
    cells: set[tuple[str, str]] = set()
    for r in ind_rows:
        cells.add((r["model"], r["feature"]))
    for r in dmg_rows:
        cells.add((r["model"], r["feature"]))

    tasks = ("binary", "type", "severity", "col_location", "mass_location")
    records = []
    for (model, feature) in sorted(cells):
        ind_name, ind_r2, ind_mae = _best_indicator(ind_rows, model, feature)
        row = {
            "model": model, "feature": feature,
            "best_indicator": ind_name,
            "best_indicator_exp_R2": ind_r2,
            "best_indicator_exp_MAE": ind_mae,
        }
        for t in tasks:
            v = dmg_index.get((t, model, feature))
            row[f"{t}_exp_value"] = float(v["value"]) if v else None
            row[f"{t}_exp_metric"] = v["metric"] if v else None
            row[f"{t}_exp_mae"]   = (float(v["mae"]) if v and v.get("mae") is not None
                                       else None)
        records.append(row)

    out_json = args.results / "indicator_vs_damage.json"
    out_json.write_text(json.dumps(records, indent=2))
    print(f"wrote {out_json}  ({len(records)} rows)")

    # Markdown table.  Only show rows where at least one direct task
    # exists, to keep the comparison meaningful.
    lines = []
    lines.append(
        "| model | feature | best indicator (exp R²) | severity exp R² | "
        "type exp acc | col_loc exp acc | mass_loc exp acc | binary exp acc |"
    )
    lines.append("|" + "---|" * 8)
    def _fmt(x, fmt):
        return "—" if x is None else fmt.format(x)
    for r in records:
        if not any(r.get(f"{t}_exp_value") is not None for t in tasks):
            continue
        ind_str = (f"`{r['best_indicator']}` ({_fmt(r['best_indicator_exp_R2'], '{:+.2f}')})"
                       if r["best_indicator"] else "—")
        lines.append(
            f"| {r['model']} | `{r['feature']}` | {ind_str} | "
            f"{_fmt(r['severity_exp_value'], '{:+.2f}')} | "
            f"{_fmt(r['type_exp_value'], '{:.2f}')} | "
            f"{_fmt(r['col_location_exp_value'], '{:.2f}')} | "
            f"{_fmt(r['mass_location_exp_value'], '{:.2f}')} | "
            f"{_fmt(r['binary_exp_value'], '{:.2f}')} |"
        )
    out_md = args.results / "indicator_vs_damage_table.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_md}  ({len(lines)} lines)")


if __name__ == "__main__":
    main()
