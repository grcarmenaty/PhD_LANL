"""Run after HPO completes: rebuild metrics summary, render every plot,
evaluate trained models on the IQS experimental dataset.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def _run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(res.stdout)
    sys.stderr.write(res.stderr)
    if res.returncode != 0:
        print(f"  ! exit code {res.returncode}")


def collapse_hpo_into_training_metrics(results_dir: Path) -> None:
    """Convert every HPO best-trial row into the training_metrics.json schema
    so downstream plot/report scripts keep working unchanged.
    """
    rows: list[dict] = []
    for path in sorted((results_dir / "hpo").glob("*.json")):
        d = json.loads(path.read_text())
        if d.get("best_metric_val") is None:
            continue
        # Find the matching trial row to also extract MAE / runtime.
        best_h = d["best_hyperparams"]
        best_trial = None
        for t in d["trials"]:
            if t["hyperparams"] == best_h:
                best_trial = t; break
        if best_trial is None:
            continue
        rows.append({
            "task":          d["task"],
            "model":         d["model"],
            "feature":       d["feature"],
            "n_train":       0,
            "n_val":         0,
            "n_test":        0,
            "metric_name":   best_trial["metric_name"],
            "metric_val":    best_trial["metric_val"],
            "metric_test":   best_trial["metric_test"],
            "runtime_s":     best_trial["runtime_s"],
            "extra":         best_trial.get("extras", {}),
            "best_hyperparams": best_h,
        })
    out = results_dir / "training_metrics.json"
    out.write_text(json.dumps(rows, indent=2, default=str))
    print(f"Wrote {out}  ({len(rows)} rows)")


def main() -> None:
    results = _REPO / "results"
    print(">>> Collapsing HPO into training metrics …")
    collapse_hpo_into_training_metrics(results)

    print("\n>>> Running experimental evaluation …")
    _run([sys.executable, "ml_pipeline/evaluate.py"])

    print("\n>>> Rendering all plots …")
    _run([sys.executable, "ml_pipeline/plots_advanced.py"])
    _run([sys.executable, "ml_pipeline/plot_results.py"])

    print("\n>>> Building report …")
    _run([sys.executable, "ml_pipeline/report.py"])


if __name__ == "__main__":
    main()
