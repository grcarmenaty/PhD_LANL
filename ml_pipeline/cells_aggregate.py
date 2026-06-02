"""Per-cell aggregation across all 9 per-case files (3 variants × 3 seeds).

For each (task, model, feature) cell, computes 3-seed mean ± sd of:
  - classification: macroF1, balanced_acc, accuracy, n_pos, n_neg, class_prior
  - multi-class: macroF1, balanced_acc, accuracy, per-class F1 (compact)
  - regression: MAE, MSE, R2

Output: results/cells_v1_v2_v2a.json — keyed by "task/model/feature"

This file is the data source for the long consolidated report.
"""
from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score

_REPO = Path(__file__).resolve().parent.parent

VARIANT_FILES = {
    "v1":  [(s, _REPO / f"results/experimental_full_per_case_v1_seed{s}.json") for s in (42, 101, 202)],
    "v2":  [(s, _REPO / f"results/experimental_full_per_case_v2_seed{s}.json") for s in (42, 101, 202)],
    "v2a": [(s, _REPO / f"results_v2a_seed{s}/experimental_full_per_case.json") for s in (42, 101, 202)],
}


def classify_kind(y_true: np.ndarray) -> str:
    u = sorted(set(y_true.tolist()))
    if all(isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer()) for v in u):
        u_int = sorted({int(v) for v in u})
        if len(u_int) == 2: return "cls_binary"
        if len(u_int) <= 12: return "cls_multi"
    return "regression"


def metrics_binary(yt, yp):
    n_pos = int((yt == 1).sum()); n_neg = int((yt == 0).sum())
    return {
        "n_pos": n_pos, "n_neg": n_neg,
        "class_prior": float(max(n_pos, n_neg) / (n_pos + n_neg)),
        "accuracy":   float(accuracy_score(yt, yp)),
        "bal_acc":    float(balanced_accuracy_score(yt, yp)),
        "macro_f1":   float(f1_score(yt, yp, average="macro", labels=[0, 1], zero_division=0)),
        "tpr":        float((yp[yt == 1] == 1).mean()) if n_pos else 0.0,
        "tnr":        float((yp[yt == 0] == 0).mean()) if n_neg else 0.0,
    }


def metrics_multi(yt, yp, n_classes):
    classes = list(range(n_classes))
    from collections import Counter
    ct = Counter(yt.tolist())
    prior = max(ct.values()) / len(yt) if len(yt) else 0
    return {
        "n_samples": int(len(yt)),
        "n_classes": int(n_classes),
        "class_prior": float(prior),
        "accuracy":   float(accuracy_score(yt, yp)),
        "bal_acc":    float(balanced_accuracy_score(yt, yp)),
        "macro_f1":   float(f1_score(yt, yp, average="macro", labels=classes, zero_division=0)),
        "per_class_f1": [
            float(f1_score(yt == c, yp == c, zero_division=0))
            for c in classes
        ],
    }


def metrics_regression(yt, yp):
    yt = yt.astype(float); yp = yp.astype(float)
    mae = float(np.mean(np.abs(yt - yp)))
    mse = float(np.mean((yt - yp) ** 2))
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {
        "n_samples": int(len(yt)),
        "mae": mae, "mse": mse, "r2": r2,
        "y_true_mean": float(np.mean(yt)),
        "y_true_std":  float(np.std(yt)),
    }


def main():
    # Discover cells & kind
    print("scanning per-case files…")
    cells: dict[tuple[str, str, str], dict] = {}
    runs: dict[tuple[str, str, str], dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]] = defaultdict(lambda: defaultdict(dict))
    for variant, files in VARIANT_FILES.items():
        for seed, p in files:
            rows = json.load(open(p))
            by_cell: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
            for r in rows:
                by_cell[(r["task"], r["model"], r["feature"])].append(r)
            for ck, rs in by_cell.items():
                yt = np.array([r["y_true"] for r in rs])
                yp = np.array([r["y_pred"] for r in rs])
                # Decide kind once per cell (from first seen run)
                if ck not in cells:
                    cells[ck] = {"task": ck[0], "model": ck[1], "feature": ck[2],
                                  "kind": classify_kind(yt)}
                runs[ck][variant][seed] = (yt, yp)

    # Compute per (cell, variant, seed) metrics
    print(f"computing metrics for {len(cells)} cells…")
    out = {"cells": {}}
    for ck in sorted(cells):
        cell_str = "/".join(ck)
        meta = cells[ck]
        kind = meta["kind"]
        rec = {**meta, "metrics": {}}
        for variant in ("v1", "v2", "v2a"):
            if variant not in runs[ck]:
                rec["metrics"][variant] = None
                continue
            per_seed = {}
            for seed, (yt, yp) in sorted(runs[ck][variant].items()):
                if kind == "cls_binary":
                    m = metrics_binary(yt.astype(int), yp.astype(int))
                elif kind == "cls_multi":
                    n_classes = int(max(yt.max(), yp.max())) + 1
                    m = metrics_multi(yt.astype(int), yp.astype(int), n_classes)
                else:
                    m = metrics_regression(yt, yp)
                per_seed[str(seed)] = m
            # aggregate mean ± sd of scalar metrics
            keys = [k for k, v in next(iter(per_seed.values())).items() if isinstance(v, (int, float))]
            mean = {k: float(np.mean([per_seed[s][k] for s in per_seed])) for k in keys}
            sd   = {k: float(np.std ([per_seed[s][k] for s in per_seed])) for k in keys}
            rec["metrics"][variant] = {
                "seeds": per_seed,
                "mean":  mean,
                "sd":    sd,
                "n_seeds": len(per_seed),
            }
        # cross-variant headline
        score_key = "mae" if kind == "regression" else "macro_f1"
        best = None
        for v in ("v1", "v2", "v2a"):
            mv = rec["metrics"].get(v)
            if not mv: continue
            s = mv["mean"].get(score_key)
            if s is None: continue
            if best is None or (kind == "regression" and s < best[1]) or (kind != "regression" and s > best[1]):
                best = (v, s)
        spread = None
        scores = [rec["metrics"][v]["mean"].get(score_key) for v in ("v1", "v2", "v2a")
                  if rec["metrics"].get(v)]
        scores = [s for s in scores if s is not None]
        if scores:
            spread = float(max(scores) - min(scores))
        rec["cross_variant"] = {
            "selection_metric": score_key,
            "best_variant": best[0] if best else None,
            "best_score":   best[1] if best else None,
            "spread":       spread,
        }
        out["cells"][cell_str] = rec

    out_path = _REPO / "results" / "cells_v1_v2_v2a.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}  ({len(out['cells'])} cells)")


if __name__ == "__main__":
    main()
