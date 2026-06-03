"""Summarise the hi-res (1601 CFDAC) sweep with honest metrics and compare
against the 128 baseline (results/cells_v1_v2_v2a.json, v1 reference seed-mean).

For each completed hires cell (results_hires/per_case/*.json) computes the
experimental zero-shot metrics — balanced accuracy + macro-F1 (NOT raw
accuracy) for classifiers, MAE/R2 for regression — flags class-collapse,
and pulls the in-domain synth metric from results_hires/synth_test.json.

Writes results_hires/hires_summary.json and prints a console table.
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                             accuracy_score)

_REPO = Path(__file__).resolve().parent.parent
PC = _REPO / "results_hires" / "per_case"
SYNTH = _REPO / "results_hires" / "synth_test.json"
BASELINE = _REPO / "results" / "cells_v1_v2_v2a.json"


def cls_metrics(yt, yp, n_out):
    yt = yt.astype(int); yp = yp.astype(int)
    classes = list(range(n_out))
    ct = Counter(yt.tolist())
    prior = max(ct.values()) / len(yt) if len(yt) else 0.0
    n_pred_classes = len(set(yp.tolist()))
    bal = float(balanced_accuracy_score(yt, yp))
    mf1 = float(f1_score(yt, yp, labels=classes, average="macro", zero_division=0))
    acc = float(accuracy_score(yt, yp))
    # collapse: predicts (almost) one class, or balanced acc at/below chance
    chance = 1.0 / n_out
    collapse = (n_pred_classes <= 1) or (bal <= chance + 0.02)
    return {"n": int(len(yt)), "n_classes": int(n_out),
            "n_pred_classes": int(n_pred_classes),
            "class_prior": float(prior), "chance_bal_acc": float(chance),
            "accuracy": acc, "balanced_acc": bal, "macro_f1": mf1,
            "collapse": bool(collapse)}


def reg_metrics(yt, yp):
    yt = yt.astype(float); yp = yp.astype(float)
    mae = float(np.mean(np.abs(yt - yp)))
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"n": int(len(yt)), "mae": mae, "r2": r2,
            "y_true_mean": float(np.mean(yt)), "y_true_std": float(np.std(yt))}


def main():
    synth = json.loads(SYNTH.read_text()) if SYNTH.exists() else {}
    base = json.loads(BASELINE.read_text())["cells"] if BASELINE.exists() else {}

    def baseline_for(task, model, feature):
        """matching-cell (same model/feature) and task-best v1 baseline."""
        match = base.get(f"{task}/{model}/{feature}")
        # task-best by v1 macro_f1 (or min MAE for severity)
        cand = [(k, v) for k, v in base.items() if v["task"] == task
                and v["metrics"].get("v1")]
        best = None
        for k, v in cand:
            m = v["metrics"]["v1"]["mean"]
            if task == "severity":
                s = m.get("mae")
                if s is not None and (best is None or s < best[2]):
                    best = (k, v, s)
            else:
                s = m.get("macro_f1")
                if s is not None and (best is None or s > best[2]):
                    best = (k, v, s)
        def pack(v):
            if not v: return None
            m = v["metrics"]["v1"]["mean"]
            if v["kind"] == "regression":
                return {"cell": f"{v['model']}/{v['feature']}",
                        "mae": m.get("mae"), "r2": m.get("r2")}
            return {"cell": f"{v['model']}/{v['feature']}",
                    "macro_f1": m.get("macro_f1"),
                    "balanced_acc": m.get("bal_acc"),
                    "accuracy": m.get("accuracy"),
                    "class_prior": m.get("class_prior")}
        return (pack(match) if match else None,
                (pack(best[1]) | {"cell": f"{best[1]['model']}/{best[1]['feature']}"}
                 if best else None))

    out = {}
    for p in sorted(PC.glob("*_hires1601.json")):
        d = json.loads(p.read_text())
        meta, rows = d["meta"], d["rows"]
        task, model, feature = meta["task"], meta["model"], meta["feature"]
        kind = meta["kind"]
        yt = np.array([r["y_true"] for r in rows])
        yp = np.array([r["y_pred"] for r in rows])
        if kind == "cls":
            exp = cls_metrics(yt, yp, int(meta["n_out"]))
        else:
            exp = reg_metrics(yt, yp)
        sk = f"{task}_{model}_{feature}_hires1601"
        srec = synth.get(sk, {})
        match_b, best_b = baseline_for(task, model, feature)
        out[sk] = {
            "task": task, "model": model, "feature": feature, "kind": kind,
            "n_out": int(meta["n_out"]),
            "synth": {"val_macro_f1_or_r2": srec.get("synth_val"),
                      "test_metric": srec.get("synth_test_metric"),
                      "test_macro_f1": srec.get("synth_test_macro_f1"),
                      "runtime_s": srec.get("runtime_s")},
            "exp": exp,
            "baseline_128_same_cell_v1": match_b,
            "baseline_128_task_best_v1": best_b,
        }

    (_REPO / "results_hires" / "hires_summary.json").write_text(
        json.dumps(out, indent=2))

    # Console table
    print(f"\n{'cell':<46} {'kind':<4} {'synthF1/R2':>10} "
          f"{'expMF1/R2':>10} {'expBalAcc':>9} {'expAcc':>7} "
          f"{'prior':>6} {'collapse':>8}  base128(taskbest v1 MF1/MAE)")
    print("-" * 140)
    for sk, r in sorted(out.items(), key=lambda kv: kv[1]["task"]):
        e = r["exp"]; s = r["synth"]
        bb = r["baseline_128_task_best_v1"] or {}
        if r["kind"] == "cls":
            base_s = bb.get("macro_f1")
            print(f"{sk:<46} {'cls':<4} "
                  f"{(s['test_macro_f1'] or 0):>10.3f} "
                  f"{e['macro_f1']:>10.3f} {e['balanced_acc']:>9.3f} "
                  f"{e['accuracy']:>7.3f} {e['class_prior']:>6.3f} "
                  f"{str(e['collapse']):>8}  "
                  f"{bb.get('cell','-')} MF1={base_s if base_s is None else round(base_s,3)}")
        else:
            base_s = bb.get("mae")
            print(f"{sk:<46} {'reg':<4} "
                  f"{(s['test_metric'] or 0):>10.3f} "
                  f"{e['r2']:>10.3f} {'-':>9} {'-':>7} {'-':>6} {'-':>8}  "
                  f"{bb.get('cell','-')} MAE={base_s if base_s is None else round(base_s,3)}")
    print(f"\nwrote results_hires/hires_summary.json  ({len(out)} cells)")


if __name__ == "__main__":
    main()
