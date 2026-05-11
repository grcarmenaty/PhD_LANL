"""Aggregate training and experimental-evaluation metrics into a tidy report.

Produces two markdown tables (training / experimental), one per task, and
prints them to stdout.  Optionally writes ``results/report.md`` and a flat
CSV with every row for downstream spreadsheet use.
"""
from __future__ import annotations

import csv

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml_pipeline.tasks import TASK_DESCRIPTION   # noqa: E402


def _format_table(rows: List[Dict], task: str) -> str:
    rows = [r for r in rows if r.get("task") == task]
    if not rows:
        return f"### {task}\n_(no rows)_\n"
    head = (
        f"### {task} — {TASK_DESCRIPTION.get(task, '')}\n\n"
        "| model        | feature      | metric   | test    | val     | mae     |\n"
        "|--------------|--------------|----------|---------|---------|---------|\n"
    )
    out = [head]

    def _test_val(r: Dict) -> float:
        v = r.get("metric_test", r.get("value"))
        return -float(v) if v is not None else 0.0

    rows.sort(key=_test_val)
    for r in rows:
        metric = r.get("metric_name") or r.get("metric") or ""
        test   = r.get("metric_test", r.get("value"))
        val    = r.get("metric_val", None)
        mae    = (r.get("extra") or {}).get("mae_test", r.get("mae"))
        if mae is None and r.get("metric_name") == "R2":
            mae = r.get("extra", {}).get("mae_test")
        test_s = f"{test:.4f}" if isinstance(test, float) else "—"
        val_s  = f"{val:.4f}"  if isinstance(val,  float) else "—"
        mae_s  = f"{mae:.4f}"  if isinstance(mae,  float) else "—"
        out.append(
            f"| {r['model']:<12s} | {r['feature']:<12s} | {metric:<8s} | "
            f"{test_s:<7s} | {val_s:<7s} | {mae_s:<7s} |\n"
        )
    return "".join(out)


def build_report(results_dir: Path) -> str:
    train_path = results_dir / "training_metrics.json"
    exp_path   = results_dir / "experimental_evaluation.json"
    train_rows = json.loads(train_path.read_text()) if train_path.exists() else []
    exp_rows   = json.loads(exp_path.read_text())   if exp_path.exists()   else []

    tasks = ["binary", "type", "severity", "col_location", "mass_location"]
    out: list[str] = []
    out.append("# Training metrics (synthetic, hold-out test split)\n")
    for t in tasks:
        out.append(_format_table(train_rows, t))
    out.append("\n# Experimental-data evaluation (61 IQS cases)\n")
    out.append("\n*Note: composite damage scenarios in the experimental file "
               "are reduced to a single primary op (bolt > crack > hole > "
               "mass > pristine) for label assignment; OOD generalisation is "
               "expected to be partial.*\n\n")
    for t in tasks:
        out.append(_format_table(exp_rows, t))
    return "".join(out)


def _write_csv(rows: list[dict], path: Path, kind: str) -> None:
    if not rows:
        return
    fieldnames = ["task", "model", "feature", "n"]
    if kind == "train":
        fieldnames += ["metric_name", "metric_val", "metric_test",
                        "runtime_s", "n_train", "n_val", "n_test"]
    else:
        fieldnames += ["metric", "value", "mae"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            row = {k: r.get(k, "") for k in fieldnames}
            extras = r.get("extra") or {}
            if "mae" not in row and "mae_test" in extras:
                row["mae"] = extras["mae_test"]
            w.writerow(row)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=_REPO / "results")
    p.add_argument("--out",     type=Path, default=None)
    args = p.parse_args()
    md = build_report(args.results)
    sys.stdout.write(md)
    out = args.out or (args.results / "report.md")
    out.write_text(md)

    train_rows = json.loads((args.results / "training_metrics.json").read_text())
    _write_csv(train_rows, args.results / "training_metrics.csv", "train")
    exp_path = args.results / "experimental_evaluation.json"
    if exp_path.exists():
        _write_csv(json.loads(exp_path.read_text()),
                     args.results / "experimental_metrics.csv", "exp")


if __name__ == "__main__":
    main()
