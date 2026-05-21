"""Re-score zero-shot predictions on a *Pristine + severe-damage* test set.

Deployment-relevant evaluation: restrict the experimental test set to cases
that are either Pristine or *clearly* damaged — Pristine ∪ {damage with
per-type-normalised severity >= tau} — and recompute accuracy, macro-F1 and
balanced accuracy.  This is the artefact behind REPORT_definitive.md
sections 4.5 / 5.5 (it differs from ``severity_stratified.json``, which is
damage-only and excludes Pristine).

Per-type severity normalisation uses the experimental severity ranges
(Bolt 11-85, Crack 5-8, Hole 4-6); every Mass case sits at a single
physical severity (1.2 kg), mapped to 0.458 to match the original
severity-stratified analysis.

Usage:
    python -m ml_pipeline.severity_inclusive_eval \\
        --per-case results/experimental_full_per_case_basescore.json \\
        --out results/severity_inclusive_eval.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
)

_REPO = Path(__file__).resolve().parent.parent

# Experimental per-type severity ranges (raw units); type codes:
# 0 Pristine, 1 Bolt, 2 Crack, 3 Hole, 4 Mass.
_SEV_RANGE = {1: (11.0, 85.0), 2: (5.0, 8.0), 3: (4.0, 6.0)}
_MASS_NORM = 0.458   # every Mass case is at 1.2 kg -> fixed normalised value
_THRESHOLDS = (0.0, 0.5, 0.7)
_TASKS = ("binary", "type")   # Pristine-inclusive stratification only meaningful here


def _normalised_severity(exp_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(exp_path, "r") as f:
        sev = f["severity"][:].astype(float)
        tc = f["type_code"][:].astype(int)
    snorm = np.zeros(len(sev), dtype=float)
    for t, (lo, hi) in _SEV_RANGE.items():
        m = tc == t
        snorm[m] = (sev[m] - lo) / (hi - lo)
    snorm[tc == 4] = _MASS_NORM
    return snorm, tc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-case", type=Path,
                   default=_REPO / "results"
                   / "experimental_full_per_case_basescore.json")
    p.add_argument("--exp", type=Path,
                   default=_REPO / "dataset" / "experimental_features.h5")
    p.add_argument("--out", type=Path,
                   default=_REPO / "results" / "severity_inclusive_eval.json")
    args = p.parse_args()

    snorm, tc = _normalised_severity(args.exp)
    per_case = json.loads(args.per_case.read_text())

    # Group per-case rows by cell; rows are in experimental-dataset order.
    cells: dict[tuple, list] = defaultdict(list)
    for r in per_case:
        if r["task"] in _TASKS:
            cells[(r["task"], r["model"], r["feature"])].append(r)

    out: list[dict] = []
    for (task, model, feature), rows in sorted(cells.items()):
        yt = np.array([r["y_true"] for r in rows])
        yp = np.array([r["y_pred"] for r in rows])
        if len(rows) != len(tc):
            print(f"  skip {task}/{model}/{feature}: "
                  f"{len(rows)} rows != {len(tc)} exp cases")
            continue
        for tau in _THRESHOLDS:
            keep = (tc == 0) | (snorm >= tau)   # Pristine always + damage>=tau
            yk, pk = yt[keep], yp[keep]
            out.append({
                "task": task, "model": model, "feature": feature,
                "tau": tau, "n_cases": int(keep.sum()),
                "accuracy": float(accuracy_score(yk, pk)),
                "macro_f1": float(f1_score(yk, pk, average="macro")),
                "balanced_acc": float(balanced_accuracy_score(yk, pk)),
            })
            print(f"  {task:7s}/{model:5s}/{feature:14s} tau>={tau}  "
                  f"n={int(keep.sum()):4d}  acc={out[-1]['accuracy']:.3f}  "
                  f"macroF1={out[-1]['macro_f1']:.3f}  "
                  f"balAcc={out[-1]['balanced_acc']:.3f}")

    args.out.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
