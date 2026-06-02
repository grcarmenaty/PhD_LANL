"""Quick comparison: vision-backbone vs bespoke cells on the `type` task
under per-tier severity filtering — v1 only (vision per-case files exist
only for v1, single seed, May 20).

If vision substantially beats bespoke at any tier, that justifies running
the vision training for v2 and v2a. If not, the old "vision did not beat
bespoke on type" verdict survives the DT-stratified correction and we can
defer v2/v2a vision compute.
"""
from __future__ import annotations
import json, sys, glob
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml_pipeline.case_design import (
    TYPE_PRISTINE, TYPE_BOLT, TYPE_CRACK, TYPE_HOLE, TYPE_MASS)
from ml_pipeline.tasks import build_targets
from ml_pipeline.dt_sweep import _load_exp

TIERS = {
    "all":    {TYPE_BOLT: 0,  TYPE_CRACK: 0, TYPE_HOLE: 0, TYPE_MASS: 0},
    "med+":   {TYPE_BOLT: 50, TYPE_CRACK: 5, TYPE_HOLE: 4, TYPE_MASS: 0},
    "severe": {TYPE_BOLT: 85, TYPE_CRACK: 8, TYPE_HOLE: 6, TYPE_MASS: 0},
}


def load_bespoke_type_runs(paths: List[Path]):
    """Returns dict {(model, feature): [run0_y, run1_y, ...]}.

    Each run is (y_true, y_pred) arrays in mask order.
    """
    out = defaultdict(list)
    for p in paths:
        rows = json.loads(p.read_text())
        by_cell = defaultdict(list)
        for r in rows:
            if r.get("task") != "type":
                continue
            by_cell[(r["model"], r["feature"])].append(r)
        for cell, rs in by_cell.items():
            yt = np.array([r["y_true"] for r in rs], dtype=int)
            yp = np.array([r["y_pred"] for r in rs], dtype=int)
            out[cell].append((yt, yp))
    return out


def load_vision_type_runs(vision_dir: Path):
    """Returns dict {(backbone, feature): [(yt, yp)]} — single seed each."""
    out = defaultdict(list)
    for p in sorted(vision_dir.glob("type_*.json")):
        d = json.loads(p.read_text())
        meta = d.get("meta", {})
        rows = d.get("rows", [])
        if meta.get("task") != "type":
            continue
        yt = np.array([r["y_true"] for r in rows], dtype=int)
        yp = np.array([r["y_pred"] for r in rows], dtype=int)
        cell = (meta["backbone"], meta["feature"])
        out[cell].append((yt, yp))
    return out


def sweep_tier(runs, idx, y_task, tc_idx, sev_idx):
    """Returns {tier_name: {macro_f1_mean, bal_acc_mean, n_kept, n_seeds, sd}}."""
    out = {}
    for tier_name, tdict in TIERS.items():
        per_mf1, per_bal, n_kept = [], [], 0
        for (yt, yp) in runs:
            if len(yt) != len(idx) or not np.array_equal(yt, y_task):
                continue
            keep = np.ones(len(yt), bool)
            for damage_type, thr in tdict.items():
                tm = (tc_idx == damage_type)
                keep &= ~(tm & (sev_idx < thr))
            ytk, ypk = yt[keep], yp[keep]
            classes = sorted(set(ytk.tolist()))
            if len(classes) < 2 or len(ytk) < 20:
                continue
            per_mf1.append(f1_score(ytk, ypk, average="macro", zero_division=0))
            per_bal.append(balanced_accuracy_score(ytk, ypk))
            n_kept = int(len(ytk))
        if per_mf1:
            out[tier_name] = {
                "macro_f1_mean": float(np.mean(per_mf1)),
                "macro_f1_sd":   float(np.std(per_mf1)),
                "bal_acc_mean":  float(np.mean(per_bal)),
                "n_kept": n_kept, "n_seeds": len(per_mf1),
            }
    return out


def main():
    exp = _REPO / "dataset" / "experimental_features.h5"
    tc, sev, sto, end = _load_exp(exp)
    tasks_info = build_targets(tc, sto, end, sev)
    mask, y_pool, kind = tasks_info["type"]
    idx = np.where(mask)[0]
    tc_idx = tc[idx]; sev_idx = sev[idx]

    # v1 bespoke (3 seeds)
    bespoke_paths = [_REPO / f"results/experimental_full_per_case_v1_seed{s}.json"
                      for s in (42, 101, 202)]
    bespoke = load_bespoke_type_runs([p for p in bespoke_paths if p.exists()])
    vision  = load_vision_type_runs(_REPO / "results/per_case_vision")
    print(f"bespoke cells: {len(bespoke)}  vision cells: {len(vision)}")

    results = {}  # {("bespoke"|"vision", model, feature): {tier: {...}}}
    for cell, runs in bespoke.items():
        results[("bespoke",) + cell] = sweep_tier(runs, idx, y_pool, tc_idx, sev_idx)
    for cell, runs in vision.items():
        results[("vision",) + cell] = sweep_tier(runs, idx, y_pool, tc_idx, sev_idx)

    # Rank by macro_f1 at each tier
    print("\n" + "=" * 92)
    print("TYPE task — best cells per tier (v1 only)")
    print("=" * 92)
    for tier in ("all", "med+", "severe"):
        ranked = sorted(
            ((k, v.get(tier)) for k, v in results.items() if v.get(tier) is not None),
            key=lambda x: -x[1]["macro_f1_mean"]
        )
        print(f"\n— tier = {tier} —  (top 8 by macroF1)")
        hdr = f"  {'rank':>4} {'family':>8} {'model':>16} {'feature':>16}  {'mF1':>7} {'±sd':>7}  {'balacc':>7} {'n_seeds':>7}"
        print(hdr)
        for i, (k, r) in enumerate(ranked[:8], 1):
            family, model, feat = k
            print(f"  {i:>4} {family:>8} {model:>16} {feat:>16}  "
                  f"{r['macro_f1_mean']:>7.3f} {r['macro_f1_sd']:>7.3f}  "
                  f"{r['bal_acc_mean']:>7.3f} {r['n_seeds']:>7}")

    # Best vision vs best bespoke at each tier (head-to-head)
    print("\n" + "=" * 92)
    print("Head-to-head: best VISION vs best BESPOKE per tier")
    print("=" * 92)
    for tier in ("all", "med+", "severe"):
        bv = max(
            ((k, v) for k, v in results.items() if k[0] == "vision" and v.get(tier)),
            key=lambda x: x[1][tier]["macro_f1_mean"],
            default=(None, None),
        )
        bb = max(
            ((k, v) for k, v in results.items() if k[0] == "bespoke" and v.get(tier)),
            key=lambda x: x[1][tier]["macro_f1_mean"],
            default=(None, None),
        )
        if bv[0] is None or bb[0] is None:
            print(f"  tier={tier}: missing data"); continue
        bvm = bv[1][tier]["macro_f1_mean"]
        bbm = bb[1][tier]["macro_f1_mean"]
        bb_sd = bb[1][tier]["macro_f1_sd"]
        delta = bvm - bbm
        print(f"  tier={tier:<7}  vision: {bv[0][1]}/{bv[0][2]:<16}  mF1={bvm:.3f}  "
              f"|  bespoke: {bb[0][1]}/{bb[0][2]:<16}  mF1={bbm:.3f} ± {bb_sd:.3f}  "
              f"|  Δ(vision-bespoke) = {delta:+.3f}")

    out_path = _REPO / "results" / "dt_vision_check_v1_type.json"
    # JSON-serialise (tuple keys to strings)
    serial = {f"{k[0]}/{k[1]}/{k[2]}": v for k, v in results.items()}
    out_path.write_text(json.dumps(serial, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
