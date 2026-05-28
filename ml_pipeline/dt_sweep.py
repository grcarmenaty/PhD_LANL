"""Damage-Threshold / Improvement-Threshold swept evaluation.

Re-scores existing per-case predictions (``experimental_full_per_case*.json``)
as a function of a minimum stiffness-alteration threshold, reproducing the
DT/IT sensitivity methodology of Carmenaty & Pérez (the CFDAC transfer-
learning paper): a damaged test sample is included only if its stiffness
*reduction* exceeds the Damage Threshold. Sweeping the threshold yields the
"accuracy vs. minimum alteration" relationship that a single pooled metric
hides.

No retraining: this is pure post-processing on predictions already on disk.
Per-case rows for a given (task, model, feature) are written by
``evaluate_full_experimental`` in ``np.where(mask)[0]`` order, so they align
positionally with the experimental HDF5 arrays via the task mask
(verified: y_true reproduces the task target exactly).

Stiffness reduction per experimental sample is derived from the calibrated
severity→ratio functions in ``variation.py`` (the same physics the synth
generator uses):
  bolt  : 1 − bolt_jsr_ratio(percent)
  crack : 1 − crack_ratio(mm)
  hole  : 1 − hole_ratio(mm)
  pristine: 0 ;  mass: NaN (inertia change, not a stiffness alteration —
  excluded from the DT axis, reported separately).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml_pipeline.case_design import (                                   # noqa: E402
    TYPE_PRISTINE, TYPE_BOLT, TYPE_CRACK, TYPE_HOLE, TYPE_MASS)
from ml_pipeline.tasks import build_targets                             # noqa: E402
from ml_pipeline.variation import (                                       # noqa: E402
    crack_ratio, hole_ratio, bolt_jsr_ratio)

DEFAULT_DT_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60]


def stiffness_reduction(type_code: np.ndarray,
                         severity: np.ndarray) -> np.ndarray:
    """Per-sample fractional stiffness reduction (NaN for mass)."""
    sr = np.zeros(len(type_code), dtype=float)
    for i, (tc, sev) in enumerate(zip(type_code, severity)):
        if tc == TYPE_PRISTINE:
            sr[i] = 0.0
        elif tc == TYPE_BOLT:
            sr[i] = 1.0 - bolt_jsr_ratio(float(sev))
        elif tc == TYPE_CRACK:
            sr[i] = 1.0 - crack_ratio(float(sev))
        elif tc == TYPE_HOLE:
            sr[i] = 1.0 - hole_ratio(float(sev))
        else:  # mass — not a stiffness alteration
            sr[i] = np.nan
    return sr


def _load_exp(exp_path: Path):
    with h5py.File(exp_path, "r") as f:
        tc = f["type_code"][:].astype(int)
        sev = f["severity"][:].astype(float)
        sto = f["storey"][:].astype(int)
        end = f["end"][:].astype(int)
    return tc, sev, sto, end


def _index_per_case(per_case_paths: List[Path]
                    ) -> Dict[Tuple[str, str, str], List[List[dict]]]:
    """Group per-case rows by (task, model, feature), preserving order.

    Returns a dict mapping the cell to a *list of seed-runs*; each run is the
    ordered list of rows from one per-case file.
    """
    out: Dict[Tuple[str, str, str], List[List[dict]]] = defaultdict(list)
    for p in per_case_paths:
        rows = json.loads(p.read_text())
        by_cell: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
        for r in rows:
            by_cell[(r["task"], r["model"], r["feature"])].append(r)
        for cell, rs in by_cell.items():
            out[cell].append(rs)
    return out


def _sweep_cell(runs: List[List[dict]], idx: np.ndarray, y_task: np.ndarray,
                sr: np.ndarray, tc: np.ndarray, dt_grid: List[float]) -> dict:
    """DT-swept balanced-acc / macro-F1 averaged across seed runs.

    Positives below the DT (by stiffness reduction) are removed from the
    test set; negatives (pristine / other classes) are always kept.
    Mass positives (sr NaN) are dropped at any DT > 0.
    """
    sr_idx = sr[idx]
    curve = {f"{dt:.2f}": {"bal_acc": [], "macro_f1": [],
                            "n_pos": None, "n_neg": None,
                            "drop_to_neg_rate": []} for dt in dt_grid}
    for run in runs:
        yt = np.array([r["y_true"] for r in run]).astype(int)
        yp = np.array([r["y_pred"] for r in run]).astype(int)
        if len(yt) != len(idx) or not np.array_equal(yt, y_task):
            # alignment guard — skip a run that doesn't match the task target
            continue
        for dt in dt_grid:
            pos = yt == 1
            neg = yt == 0
            if dt <= 0:
                keep = np.ones(len(yt), bool)
            else:
                keep_pos = pos & (sr_idx >= dt)   # NaN >= dt is False → mass dropped
                keep = neg | keep_pos
            k = f"{dt:.2f}"
            n_pos = int((yt[keep] == 1).sum())
            n_neg = int((yt[keep] == 0).sum())
            curve[k]["n_pos"] = n_pos
            curve[k]["n_neg"] = n_neg
            # Balanced accuracy is only defined with both classes present.
            # Below the minimum-positives floor the cell has no damage left
            # to detect, so the metric is undefined (not "high").
            if n_pos >= 10 and n_neg >= 10:
                curve[k]["bal_acc"].append(
                    balanced_accuracy_score(yt[keep], yp[keep]))
                curve[k]["macro_f1"].append(
                    f1_score(yt[keep], yp[keep], average="macro",
                             labels=[0, 1], zero_division=0))
            # desirable failure: sub-threshold positives predicted negative
            dropped = pos & ~keep
            if dropped.any():
                curve[k]["drop_to_neg_rate"].append(
                    float((yp[dropped] == 0).mean()))
    # reduce
    summary = {}
    for k, v in curve.items():
        if v["n_pos"] is None:
            continue
        defined = bool(v["bal_acc"])
        summary[k] = {
            "dt": float(k),
            "bal_acc_mean": float(np.mean(v["bal_acc"])) if defined else None,
            "bal_acc_sd": float(np.std(v["bal_acc"])) if defined else None,
            "macro_f1_mean": float(np.mean(v["macro_f1"])) if defined else None,
            "n_pos": v["n_pos"], "n_neg": v["n_neg"],
            "dropped_pos_to_pristine_rate":
                float(np.mean(v["drop_to_neg_rate"]))
                if v["drop_to_neg_rate"] else None,
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", type=Path,
                    default=_REPO / "dataset" / "experimental_features.h5")
    ap.add_argument("--per-case", type=Path, nargs="+", required=True,
                    help="One or more per-case JSONs for the main arm "
                         "(multiple = seeds, averaged).")
    ap.add_argument("--baseline", type=Path, nargs="*", default=None,
                    help="Optional per-case JSON(s) for a comparison arm.")
    ap.add_argument("--cells", nargs="+",
                    default=["binary/mlp/cfdac_magphase",
                             "is_hole/mlp/modal",
                             "is_crack/mlp/modal",
                             "is_bolt/mlp/modal"],
                    help="task/model/feature triples to sweep.")
    ap.add_argument("--dt-grid", type=float, nargs="+", default=DEFAULT_DT_GRID)
    ap.add_argument("--out", type=Path,
                    default=_REPO / "results" / "dt_sweep.json")
    args = ap.parse_args()

    tc, sev, sto, end = _load_exp(args.exp)
    tasks = build_targets(tc, sto, end, sev)
    sr = stiffness_reduction(tc, sev)

    main_runs = _index_per_case(args.per_case)
    base_runs = _index_per_case(args.baseline) if args.baseline else {}

    report = {"dt_grid": args.dt_grid, "cells": {}}
    for cell in args.cells:
        task, model, feat = cell.split("/")
        if task not in tasks:
            print(f"[skip] unknown task {task}")
            continue
        mask, y_pool, kind = tasks[task]
        if kind != "cls":
            print(f"[skip] {cell}: DT sweep is for classification tasks")
            continue
        idx = np.where(mask)[0]
        key = (task, model, feat)
        if key not in main_runs:
            print(f"[skip] {cell}: not in per-case file(s)")
            continue
        main_sum = _sweep_cell(main_runs[key], idx, y_pool, sr, tc, args.dt_grid)
        base_sum = (_sweep_cell(base_runs[key], idx, y_pool, sr, tc,
                                args.dt_grid)
                    if key in base_runs else None)
        report["cells"][cell] = {"main": main_sum, "baseline": base_sum}

        print(f"\n=== {cell} — balanced-acc vs minimum stiffness reduction ===")
        hdr = f"  {'DT':>6} {'n_pos':>6} {'n_neg':>6} {'main_BA':>9}"
        if base_sum:
            hdr += f" {'base_BA':>9} {'Δ':>7}"
        hdr += f"  {'dropped→prist':>13}"
        print(hdr)
        for dt in args.dt_grid:
            k = f"{dt:.2f}"
            if k not in main_sum:
                continue
            m = main_sum[k]
            ba = m["bal_acc_mean"]
            ba_s = "  undef  " if ba is None else f"{ba:>9.3f}"
            line = f"  {dt:>6.2f} {m['n_pos']:>6} {m['n_neg']:>6} {ba_s}"
            if base_sum and k in base_sum:
                b = base_sum[k]["bal_acc_mean"]
                if ba is not None and b is not None:
                    line += f" {b:>9.3f} {ba - b:>+7.3f}"
                else:
                    line += f" {('undef' if b is None else f'{b:.3f}'):>9} {'—':>7}"
            dr = m["dropped_pos_to_pristine_rate"]
            line += f"  {('—' if dr is None else f'{dr:.2f}'):>13}"
            print(line)

    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
