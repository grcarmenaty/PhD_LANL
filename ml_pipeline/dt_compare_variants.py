"""Cross-variant DT-stratified comparison.

For each (task, cell) and each Damage-Threshold (DT) in the sweep grid,
compute balanced-accuracy + macro-F1 averaged across seeds, separately
for v1, v2, and v2a per-case predictions. Then for each task at each DT,
select the *best cell on the restricted test set* (not the pooled one) —
this is the methodology that lets us see whether v2 is competitive with
v1 when the model's domain-of-competence applies.

Output JSON layout:
{
  "dt_grid": [...],
  "by_cell": {
     "<task>/<model>/<feature>": {
        "v1": {"0.00": {bal_acc, macro_f1, n_pos, n_neg, ...}, ...},
        "v2": {...},
        "v2a": {...},
     }, ...
  },
  "best_per_task": {
     "<task>": {
        "v1": {"0.00": {"cell": "...", "bal_acc": x, "macro_f1": y}, ...},
        ...
     }, ...
  }
}

Multi-class tasks: macroF1 over all classes. Regression (severity): skipped
(MAE-vs-DT belongs in a separate script).
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml_pipeline.case_design import (
    TYPE_PRISTINE, TYPE_BOLT, TYPE_CRACK, TYPE_HOLE, TYPE_MASS)
from ml_pipeline.tasks import build_targets
from ml_pipeline.dt_sweep import stiffness_reduction, _load_exp, _index_per_case

DT_GRID = [0.0, 0.02, 0.04, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60]

VARIANT_FILES = {
    "v1": [_REPO / f"results/experimental_full_per_case_v1_seed{s}.json"
           for s in (42, 101, 202)],
    "v2": [_REPO / f"results/experimental_full_per_case_v2_seed{s}.json"
           for s in (42, 101, 202)],
    "v2a": [_REPO / f"results_v2a_seed{s}/experimental_full_per_case.json"
            for s in (42, 101, 202)],
}


def sweep_cell_multiclass(runs, idx, y_task, sr, dt_grid):
    """Sweep DT for a multi-class classification cell.

    A class-c sample is positive (kept above its DT) iff its stiffness
    reduction ≥ DT. Pristine and mass-like classes are always retained as
    they represent reference / no-stiffness-change cases.
    For multi-class tasks (col_location, mass_location, type), DT applies
    only to stiffness-altering damage classes; everything else is kept.
    Practically we filter on stiffness_reduction[idx] ≥ DT for samples
    with sr > 0 (damage), keep all sr == 0 / NaN.
    """
    sr_idx = sr[idx]
    out = {}
    for dt in dt_grid:
        per_seed_bal, per_seed_f1 = [], []
        n_kept = 0; classes_present = set()
        for run in runs:
            yt = np.array([r["y_true"] for r in run]).astype(int)
            yp = np.array([r["y_pred"] for r in run]).astype(int)
            if len(yt) != len(idx) or not np.array_equal(yt, y_task):
                continue
            if dt <= 0:
                keep = np.ones(len(yt), bool)
            else:
                # Keep: stiffness-reduction sample ≥ DT, OR sr==0 (pristine), OR NaN (mass)
                ks = np.nan_to_num(sr_idx, nan=-1.0)
                keep_above = ks >= dt          # damage above threshold
                keep_pristine = (sr_idx == 0.0)
                keep_mass = np.isnan(sr_idx)
                keep = keep_above | keep_pristine | keep_mass
            n_kept = int(keep.sum())
            classes_present = set(yt[keep].tolist())
            if len(classes_present) >= 2 and n_kept >= 20:
                per_seed_bal.append(balanced_accuracy_score(yt[keep], yp[keep]))
                per_seed_f1.append(f1_score(yt[keep], yp[keep], average="macro",
                                            zero_division=0))
        out[f"{dt:.2f}"] = {
            "dt": dt,
            "n_kept": n_kept,
            "n_classes_present": len(classes_present),
            "bal_acc_mean": float(np.mean(per_seed_bal)) if per_seed_bal else None,
            "bal_acc_sd": float(np.std(per_seed_bal)) if per_seed_bal else None,
            "macro_f1_mean": float(np.mean(per_seed_f1)) if per_seed_f1 else None,
            "macro_f1_sd": float(np.std(per_seed_f1)) if per_seed_f1 else None,
            "n_seeds": len(per_seed_bal),
        }
    return out


def sweep_cell_regression(runs, idx, y_task, sr, dt_grid):
    """Sweep DT for severity regression. MAE over kept samples (sr ≥ DT)."""
    sr_idx = sr[idx]
    out = {}
    for dt in dt_grid:
        per_seed_mae = []
        n_kept = 0
        for run in runs:
            yt = np.array([r["y_true"] for r in run]).astype(float)
            yp = np.array([r["y_pred"] for r in run]).astype(float)
            if len(yt) != len(idx):
                continue
            if dt <= 0:
                keep = np.ones(len(yt), bool)
            else:
                ks = np.nan_to_num(sr_idx, nan=-1.0)
                keep = ks >= dt
            n_kept = int(keep.sum())
            if n_kept >= 10:
                per_seed_mae.append(float(np.mean(np.abs(yt[keep] - yp[keep]))))
        out[f"{dt:.2f}"] = {
            "dt": dt, "n_kept": n_kept,
            "mae_mean": float(np.mean(per_seed_mae)) if per_seed_mae else None,
            "mae_sd": float(np.std(per_seed_mae)) if per_seed_mae else None,
            "n_seeds": len(per_seed_mae),
        }
    return out


def sweep_cell_binary(runs, idx, y_task, sr, dt_grid):
    """Sweep DT for a binary classification cell.

    Positives (y==1) require stiffness reduction ≥ DT; negatives (y==0) all kept.
    Mass positives (NaN sr) are dropped at any DT > 0 (mass is not a stiffness
    alteration, reported separately).
    """
    sr_idx = sr[idx]
    out = {}
    for dt in dt_grid:
        per_seed_bal, per_seed_f1, per_seed_drop = [], [], []
        n_pos = n_neg = 0
        for run in runs:
            yt = np.array([r["y_true"] for r in run]).astype(int)
            yp = np.array([r["y_pred"] for r in run]).astype(int)
            if len(yt) != len(idx) or not np.array_equal(yt, y_task):
                continue
            pos = yt == 1
            neg = yt == 0
            if dt <= 0:
                keep = np.ones(len(yt), bool)
            else:
                keep_pos = pos & (sr_idx >= dt)   # NaN >= dt is False → mass dropped
                keep = neg | keep_pos
            n_pos = int((yt[keep] == 1).sum())
            n_neg = int((yt[keep] == 0).sum())
            if n_pos >= 10 and n_neg >= 10:
                per_seed_bal.append(balanced_accuracy_score(yt[keep], yp[keep]))
                per_seed_f1.append(f1_score(yt[keep], yp[keep], average="macro",
                                            labels=[0, 1], zero_division=0))
            dropped = pos & ~keep
            if dropped.any():
                per_seed_drop.append(float((yp[dropped] == 0).mean()))
        out[f"{dt:.2f}"] = {
            "dt": dt,
            "n_pos": n_pos, "n_neg": n_neg,
            "bal_acc_mean": float(np.mean(per_seed_bal)) if per_seed_bal else None,
            "bal_acc_sd": float(np.std(per_seed_bal)) if per_seed_bal else None,
            "macro_f1_mean": float(np.mean(per_seed_f1)) if per_seed_f1 else None,
            "macro_f1_sd": float(np.std(per_seed_f1)) if per_seed_f1 else None,
            "dropped_to_neg_rate": float(np.mean(per_seed_drop)) if per_seed_drop else None,
            "n_seeds": len(per_seed_bal),
        }
    return out


def main():
    exp = _REPO / "dataset" / "experimental_features.h5"
    tc, sev, sto, end = _load_exp(exp)
    tasks_info = build_targets(tc, sto, end, sev)
    sr = stiffness_reduction(tc, sev)

    # Load all 3 variants' per_case files
    variants = {}
    for v, paths in VARIANT_FILES.items():
        existing = [p for p in paths if p.exists()]
        if len(existing) != len(paths):
            print(f"[warn] {v}: {len(existing)}/{len(paths)} files present")
        variants[v] = _index_per_case(existing)

    # Enumerate all (task, model, feature) cells from union of variants
    all_cells = set()
    for v in variants.values():
        all_cells.update(v.keys())
    print(f"cells discovered: {len(all_cells)}")

    # Sweep every cell × variant
    by_cell = {}
    skipped_reg = 0
    for cell in sorted(all_cells):
        task, model, feat = cell
        if task not in tasks_info:
            continue
        mask, y_pool, kind = tasks_info[task]
        idx = np.where(mask)[0]
        if kind == "reg":
            # Regression severity — sweep MAE on filtered positives only.
            sweep_fn = sweep_cell_regression
            n_classes = None
        else:
            n_classes = len(set(y_pool.tolist()))
            sweep_fn = sweep_cell_binary if n_classes == 2 else sweep_cell_multiclass
        cell_str = "/".join(cell)
        by_cell[cell_str] = {}
        for v_name, v_runs in variants.items():
            if cell not in v_runs:
                continue
            by_cell[cell_str][v_name] = sweep_fn(v_runs[cell], idx, y_pool, sr, DT_GRID)
    print(f"swept {len(by_cell)} cells; skipped {skipped_reg} regression cells")

    # Best-cell-per-task at each DT threshold, per variant.
    # Classification: select by macroF1.  Regression: select by min MAE.
    best_per_task = defaultdict(lambda: defaultdict(dict))
    tasks_seen = set(c.split("/")[0] for c in by_cell)
    for task in sorted(tasks_seen):
        task_cells = [c for c in by_cell if c.startswith(task + "/")]
        is_reg = (task == "severity")
        for v in ("v1", "v2", "v2a"):
            for dt in DT_GRID:
                k = f"{dt:.2f}"
                best = None
                for cell in task_cells:
                    if v not in by_cell[cell]: continue
                    if k not in by_cell[cell][v]: continue
                    rec = by_cell[cell][v][k]
                    if is_reg:
                        mae = rec.get("mae_mean")
                        if mae is None: continue
                        if best is None or mae < best["mae_mean"]:
                            best = {"cell": cell.split("/", 1)[1],
                                    "mae_mean": mae, "mae_sd": rec.get("mae_sd"),
                                    "n_kept": rec.get("n_kept"),
                                    "n_seeds": rec.get("n_seeds")}
                    else:
                        mf1 = rec.get("macro_f1_mean")
                        if mf1 is None: continue
                        if best is None or mf1 > best["macro_f1_mean"]:
                            best = {"cell": cell.split("/", 1)[1],
                                    "macro_f1_mean": mf1, "macro_f1_sd": rec.get("macro_f1_sd"),
                                    "bal_acc_mean": rec.get("bal_acc_mean"),
                                    "n_pos": rec.get("n_pos"), "n_neg": rec.get("n_neg"),
                                    "n_kept": rec.get("n_kept"),
                                    "n_seeds": rec.get("n_seeds")}
                if best:
                    best_per_task[task][v][k] = best

    out = {
        "dt_grid": DT_GRID,
        "by_cell": by_cell,
        "best_per_task": best_per_task,
    }
    out_path = _REPO / "results" / "dt_compare_v1_v2_v2a.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {out_path}")

    # Print verdict table: best-cell-per-task across DT, by variant
    print("\n" + "=" * 100)
    print("BEST CELL PER TASK — macroF1 by DT threshold (v1 / v2 / v2a)")
    print("=" * 100)
    for task in sorted(tasks_seen):
        is_reg = (task == "severity")
        metric_name = "MAE↓" if is_reg else "macroF1↑"
        print(f"\n— {task} ({metric_name}) —")
        hdr = f"  {'DT':>5} " + " ".join(f"{v:>20}" for v in ("v1", "v2", "v2a"))
        print(hdr)
        for dt in DT_GRID:
            k = f"{dt:.2f}"
            row = f"  {dt:>5.2f} "
            for v in ("v1", "v2", "v2a"):
                r = best_per_task[task].get(v, {}).get(k)
                if r is None:
                    row += f" {'(undef)':>20}"
                else:
                    val = r["mae_mean"] if is_reg else r["macro_f1_mean"]
                    row += f" {val:>6.3f} {('('+r['cell'][:12]+')'):>13}"
            print(row)


if __name__ == "__main__":
    main()
