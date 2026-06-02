"""Feature-dimension DT sweep (per damage-type physical axis) + multi-axis
tier filter for is_pristine.

Unlike dt_compare_variants.py (which uses fractional stiffness reduction —
limited to bolt for this experimental test set), this sweep uses each damage
type's natural axis:
  - bolt   : % loosening (raw `severity`)  — levels [11, 50, 85]
  - crack  : depth in mm                   — levels [5, 8]
  - hole   : diameter in mm                — levels [4, 6]
  - mass   : kg (single level 1.2, no sweep)
  - storey / end: location stratification (column section)

For multi-class / multi-type tasks (binary, type, col_location), a *tier*
filter is applied per-damage-type: tier 0 keeps everything, higher tiers
progressively retain only severer damage of each type.

For is_pristine (positives = pristine, negatives = damaged), the tier
filter is applied to NEGATIVES — at higher tiers only severe damaged
samples are in the negative set, so the test becomes "pristine vs
unambiguously damaged".

Output JSON: results/dt_feature_sweep.json
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import h5py, numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml_pipeline.case_design import (
    TYPE_PRISTINE, TYPE_BOLT, TYPE_CRACK, TYPE_HOLE, TYPE_MASS)
from ml_pipeline.tasks import build_targets
from ml_pipeline.dt_sweep import _load_exp, _index_per_case

# Per-type severity thresholds for the tier filter.
# Tier 0: keep all damage of that type.
# Tier 1: keep medium+ damage (≥ 1st non-zero severity).
# Tier 2: keep only the maximum severity for each type.
TIERS = {
    "all":   {TYPE_BOLT: 0,   TYPE_CRACK: 0, TYPE_HOLE: 0, TYPE_MASS: 0},
    "med+":  {TYPE_BOLT: 50,  TYPE_CRACK: 5, TYPE_HOLE: 4, TYPE_MASS: 0},
    "severe":{TYPE_BOLT: 85,  TYPE_CRACK: 8, TYPE_HOLE: 6, TYPE_MASS: 0},
}

# Per-type natural sweep grids (the severity values present in the test set).
TYPE_GRID = {
    TYPE_BOLT:  [0, 11, 50, 85],   # 0 = no filter
    TYPE_CRACK: [0, 5, 8],
    TYPE_HOLE:  [0, 4, 6],
    TYPE_MASS:  [0],                # single level
}
TYPE_NAME = {TYPE_BOLT: "bolt", TYPE_CRACK: "crack", TYPE_HOLE: "hole",
             TYPE_MASS: "mass", TYPE_PRISTINE: "pristine"}

VARIANT_FILES = {
    "v1": [_REPO / f"results/experimental_full_per_case_v1_seed{s}.json"
           for s in (42, 101, 202)],
    "v2": [_REPO / f"results/experimental_full_per_case_v2_seed{s}.json"
           for s in (42, 101, 202)],
    "v2a": [_REPO / f"results_v2a_seed{s}/experimental_full_per_case.json"
            for s in (42, 101, 202)],
}


def metrics_binary(yt, yp):
    n_pos = int((yt == 1).sum()); n_neg = int((yt == 0).sum())
    if n_pos < 10 or n_neg < 10:
        return None
    return {
        "bal_acc": balanced_accuracy_score(yt, yp),
        "macro_f1": f1_score(yt, yp, average="macro", labels=[0, 1], zero_division=0),
        "n_pos": n_pos, "n_neg": n_neg,
    }


def metrics_multiclass(yt, yp):
    classes = sorted(set(yt.tolist()))
    if len(classes) < 2 or len(yt) < 20:
        return None
    return {
        "bal_acc": balanced_accuracy_score(yt, yp),
        "macro_f1": f1_score(yt, yp, average="macro", zero_division=0),
        "n_kept": int(len(yt)), "n_classes_present": len(classes),
    }


def metrics_regression(yt, yp):
    if len(yt) < 10:
        return None
    return {"mae": float(np.mean(np.abs(yt - yp))), "n_kept": int(len(yt))}


def sweep_per_type_threshold(runs, idx, y_task, tc_idx, sev_idx,
                             damage_type, kind):
    """Sweep one damage type's severity threshold.

    Test set composition at each threshold:
      - All samples NOT of the swept damage type are kept unchanged.
      - Samples of the swept type are kept only if their severity ≥ threshold.

    For binary task is_<type>, this naturally filters positives. For
    multi-class tasks (col_location, type, binary), it filters all
    positives that match the swept type.
    """
    grid = TYPE_GRID[damage_type]
    type_mask = (tc_idx == damage_type)
    out = {}
    for thr in grid:
        per_seed = []
        for run in runs:
            yt = np.array([r["y_true"] for r in run])
            yp = np.array([r["y_pred"] for r in run])
            if len(yt) != len(idx) or (kind == "cls" and not np.array_equal(yt.astype(int), y_task)):
                continue
            if thr == 0:
                keep = np.ones(len(yt), bool)
            else:
                keep = (~type_mask) | (type_mask & (sev_idx >= thr))
            ytk, ypk = yt[keep], yp[keep]
            if kind == "reg":
                m = metrics_regression(ytk.astype(float), ypk.astype(float))
            elif len(set(y_task.tolist())) == 2:
                m = metrics_binary(ytk.astype(int), ypk.astype(int))
            else:
                m = metrics_multiclass(ytk.astype(int), ypk.astype(int))
            if m is not None:
                per_seed.append(m)
        if per_seed:
            agg = {k: float(np.mean([s[k] for s in per_seed]))
                   for k in per_seed[0] if isinstance(per_seed[0][k], (int, float))}
            agg["n_seeds"] = len(per_seed)
            out[f"{thr}"] = agg
    return out


def sweep_tier_filter(runs, idx, y_task, tc_idx, sev_idx, kind, *,
                      apply_to: str):
    """Multi-axis tier filter.

    apply_to='positives': keep only damaged samples (any type) with
      severity ≥ tier threshold for that type. Used for binary, type,
      col_location — tightens the positive set across all damage types.
    apply_to='negatives': for is_pristine — keep all pristine samples,
      keep only damaged negatives at tier threshold. The pristine class
      then competes against progressively severer damage.
    """
    out = {}
    for tier_name, tdict in TIERS.items():
        per_seed = []
        for run in runs:
            yt = np.array([r["y_true"] for r in run])
            yp = np.array([r["y_pred"] for r in run])
            if len(yt) != len(idx) or (kind == "cls" and not np.array_equal(yt.astype(int), y_task)):
                continue
            keep = np.ones(len(yt), bool)
            for damage_type, thr in tdict.items():
                tm = (tc_idx == damage_type)
                if apply_to == "positives":
                    # for samples of this damage type, require severity ≥ thr
                    drop = tm & (sev_idx < thr)
                else:  # negatives
                    drop = tm & (sev_idx < thr)
                keep &= ~drop
            ytk, ypk = yt[keep], yp[keep]
            if kind == "reg":
                m = metrics_regression(ytk.astype(float), ypk.astype(float))
            elif len(set(y_task.tolist())) == 2:
                m = metrics_binary(ytk.astype(int), ypk.astype(int))
            else:
                m = metrics_multiclass(ytk.astype(int), ypk.astype(int))
            if m is not None:
                per_seed.append(m)
        if per_seed:
            agg = {k: float(np.mean([s[k] for s in per_seed]))
                   for k in per_seed[0] if isinstance(per_seed[0][k], (int, float))}
            agg["n_seeds"] = len(per_seed)
            out[tier_name] = agg
    return out


def main():
    exp = _REPO / "dataset" / "experimental_features.h5"
    tc, sev, sto, end = _load_exp(exp)
    tasks_info = build_targets(tc, sto, end, sev)

    variants = {v: _index_per_case([p for p in paths if p.exists()])
                for v, paths in VARIANT_FILES.items()}

    # Tasks to evaluate.  is_<type> tasks → sweep that type's axis.
    # Multi-class / mixed-positive tasks → sweep tier (multi-axis).
    # is_pristine → tier applied to negatives.
    PER_TYPE_AXIS = {
        "is_bolt":  TYPE_BOLT,
        "is_crack": TYPE_CRACK,
        "is_hole":  TYPE_HOLE,
        "is_mass":  TYPE_MASS,
    }
    MULTI_AXIS_TASKS_POS = ["binary", "type", "col_location", "mass_location"]
    MULTI_AXIS_NEG = "is_pristine"

    all_cells = set()
    for v in variants.values():
        all_cells.update(v.keys())

    by_cell = {}
    for cell in sorted(all_cells):
        task, model, feat = cell
        if task not in tasks_info: continue
        mask, y_pool, kind = tasks_info[task]
        idx = np.where(mask)[0]
        tc_idx = tc[idx]; sev_idx = sev[idx]
        cell_str = "/".join(cell)
        rec = {"task": task, "model": model, "feature": feat,
               "kind": kind, "by_variant": {}}
        for v_name, v_runs in variants.items():
            if cell not in v_runs: continue
            v_out = {}
            if task in PER_TYPE_AXIS:
                damage_type = PER_TYPE_AXIS[task]
                v_out["per_type"] = {TYPE_NAME[damage_type]:
                    sweep_per_type_threshold(v_runs[cell], idx, y_pool,
                                              tc_idx, sev_idx,
                                              damage_type, kind)}
            elif task in MULTI_AXIS_TASKS_POS:
                v_out["tier_pos"] = sweep_tier_filter(
                    v_runs[cell], idx, y_pool, tc_idx, sev_idx, kind,
                    apply_to="positives")
            elif task == MULTI_AXIS_NEG:
                v_out["tier_neg"] = sweep_tier_filter(
                    v_runs[cell], idx, y_pool, tc_idx, sev_idx, kind,
                    apply_to="negatives")
            elif task == "severity":
                # regression — sweep each damage type
                v_out["per_type"] = {}
                for dt in (TYPE_BOLT, TYPE_CRACK, TYPE_HOLE):
                    v_out["per_type"][TYPE_NAME[dt]] = \
                        sweep_per_type_threshold(v_runs[cell], idx, y_pool,
                                                  tc_idx, sev_idx, dt, kind)
            rec["by_variant"][v_name] = v_out
        if rec["by_variant"]:
            by_cell[cell_str] = rec
    print(f"swept {len(by_cell)} cells")

    # Best cell per (task, variant, threshold-or-tier)
    # Tasks bucket by selection metric:
    #   classification → max macro_f1
    #   regression     → min mae
    is_reg_task = {"severity"}

    best = defaultdict(lambda: defaultdict(dict))
    for task, _ in {(c["task"], None) for c in by_cell.values()}:
        is_reg = task in is_reg_task
        task_cells = [k for k, v in by_cell.items() if v["task"] == task]
        # collect threshold keys present
        for v_name in ("v1", "v2", "v2a"):
            agg_keys = set()
            for c in task_cells:
                bv = by_cell[c]["by_variant"].get(v_name, {})
                for axis in ("per_type", "tier_pos", "tier_neg"):
                    sub = bv.get(axis)
                    if isinstance(sub, dict):
                        # per_type → dict of {damage_name: {thr: {...}}}
                        # tier_* → dict of {tier: {...}}
                        if axis == "per_type":
                            for dname, dgrid in sub.items():
                                for thr in dgrid:
                                    agg_keys.add((axis, dname, thr))
                        else:
                            for tier in sub:
                                agg_keys.add((axis, None, tier))
            for ax_key in agg_keys:
                axis, dname, thr = ax_key
                champ = None
                for c in task_cells:
                    bv = by_cell[c]["by_variant"].get(v_name, {})
                    sub = bv.get(axis)
                    if not isinstance(sub, dict): continue
                    rec = (sub.get(dname, {}).get(thr) if dname is not None
                           else sub.get(thr))
                    if rec is None: continue
                    score = rec.get("mae") if is_reg else rec.get("macro_f1")
                    if score is None: continue
                    if champ is None:
                        champ = (c, score, rec)
                    else:
                        better = score < champ[1] if is_reg else score > champ[1]
                        if better: champ = (c, score, rec)
                if champ:
                    bucket = best[task].setdefault(v_name, {}).setdefault(axis, {})
                    if axis == "per_type":
                        bucket.setdefault(dname, {})[thr] = {
                            "cell": "/".join(champ[0].split("/")[1:]),
                            **champ[2],
                        }
                    else:
                        bucket[thr] = {
                            "cell": "/".join(champ[0].split("/")[1:]),
                            **champ[2],
                        }

    out_path = _REPO / "results" / "dt_feature_sweep.json"
    out_path.write_text(json.dumps({"by_cell": by_cell, "best": best},
                                    indent=2, default=str))
    print(f"wrote {out_path}")

    # Pretty-print verdict
    print("\n" + "=" * 96)
    print("BEST CELL PER TASK — feature-dimensional sweep")
    print("=" * 96)
    for task in sorted(best):
        print(f"\n— {task} —")
        for v in ("v1", "v2", "v2a"):
            bv = best[task].get(v, {})
            if not bv: continue
            for axis, content in bv.items():
                if axis == "per_type":
                    for dname, grid in content.items():
                        keys = sorted(grid.keys(), key=lambda x: float(x))
                        for thr in keys:
                            r = grid[thr]
                            mf = r.get("macro_f1"); mae = r.get("mae")
                            n_pos = r.get("n_pos"); n_neg = r.get("n_neg")
                            n_kept = r.get("n_kept")
                            cell = r["cell"]
                            metric = (f"mF1={mf:.3f}" if mf is not None
                                       else f"MAE={mae:.3f}")
                            extra = (f"pos={n_pos} neg={n_neg}" if n_pos is not None
                                      else f"n={n_kept}")
                            print(f"  {v:>3}  {dname:>5} ≥ {thr:>3}  {metric}  "
                                  f"({extra})  cell={cell}")
                else:
                    tier_order = ("all", "med+", "severe")
                    for tier in tier_order:
                        r = content.get(tier)
                        if r is None: continue
                        mf = r.get("macro_f1"); mae = r.get("mae")
                        n_pos = r.get("n_pos"); n_neg = r.get("n_neg")
                        n_kept = r.get("n_kept")
                        cell = r["cell"]
                        metric = (f"mF1={mf:.3f}" if mf is not None
                                   else f"MAE={mae:.3f}")
                        extra = (f"pos={n_pos} neg={n_neg}" if n_pos is not None
                                  else f"n={n_kept}")
                        print(f"  {v:>3}  tier={tier:<6}  {metric}  "
                              f"({extra})  cell={cell}")


if __name__ == "__main__":
    main()
