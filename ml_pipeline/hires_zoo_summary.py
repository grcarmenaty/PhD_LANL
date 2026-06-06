"""Distill the hi-res model-zoo results (per-case JSONs from the colab-hires-*
branches) into small committable artefacts:

  results_hires/zoo_summary.json   per-cell exp metrics (bal-acc, macro-F1,
                                   collapse, synth) — NO per-case rows
  results_hires/zoo_dt_sweep.json  damage-severity (DT) sweep for the detection
                                   tasks, best cell per task vs severity
  results/figures/hires/zoo_best_per_task.png
  results/figures/hires/zoo_dt_is_bolt.png

Raw per-case predictions stay on the colab-hires-* branches (too large for main).
Point --root at a dir holding the extracted branch trees (e.g. several
results_hires_zoo/ copies). Severity for the DT join comes from the local
experimental_features_hires.h5.
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score, r2_score

_REPO = Path(__file__).resolve().parent.parent
DET_TASKS = ["binary", "is_bolt", "is_crack", "is_hole", "is_mass", "is_pristine"]


def load_sev():
    import h5py
    p = _REPO / "dataset" / "experimental_features_hires.h5"
    if not p.exists():
        return {}
    with h5py.File(p, "r") as f:
        return dict(zip([str(s) for s in f["names"][:]], f["severity"][:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/tmp/anal", help="dir tree containing per_case/*.json")
    a = ap.parse_args()
    sev = load_sev()
    cells = {}
    dt_rows = defaultdict(list)              # task -> list of (cell, dt_curve)
    thr = [0, 20, 40, 60, 80]
    for p in glob.glob(f"{a.root}/**/per_case/*_hires1601.json", recursive=True):
        d = json.load(open(p)); m = d["meta"]; r = d["rows"]
        if not r:
            continue
        yt = np.array([x["y_true"] for x in r]); yp = np.array([x["y_pred"] for x in r])
        key = f"{m['task']}/{m['model']}/{m['feature']}"
        rec = {"task": m["task"], "model": m["model"], "feature": m["feature"],
               "kind": m["kind"], "n_out": int(m.get("n_out", 2)),
               "synth": m.get("synth_test_macro_f1") if m.get("synth_test_macro_f1") is not None
                        else m.get("synth_test_metric")}
        if m["kind"] == "cls":
            n = rec["n_out"]
            rec["exp_bal_acc"] = float(balanced_accuracy_score(yt, yp))
            rec["exp_macro_f1"] = float(f1_score(yt, yp, labels=list(range(n)), average="macro", zero_division=0))
            rec["exp_acc"] = float(accuracy_score(yt, yp))
            rec["chance"] = 1.0 / n
            rec["collapse"] = bool(len(set(yp.tolist())) <= 1 or rec["exp_bal_acc"] <= 1.0 / n + 0.02)
            # DT sweep for detection tasks (binary positives stratified by severity)
            if m["task"] in DET_TASKS and sev:
                sv = np.array([sev.get(x["case"], 0.0) for x in r])
                curve = []
                for t in thr:
                    keep = (yt == 0) | ((yt == 1) & (sv >= t))
                    curve.append(float(balanced_accuracy_score(yt[keep], yp[keep]))
                                 if (yt[keep] == 1).sum() >= 5 else None)
                dt_rows[m["task"]].append((key, curve))
        else:
            rec["exp_r2"] = float(r2_score(yt.astype(float), yp.astype(float)))
            rec["collapse"] = bool(rec["exp_r2"] < 0)
        cells[key] = rec

    (_REPO / "results_hires" / "zoo_summary.json").write_text(json.dumps(cells, indent=1))

    # best per task + DT (best detection cell at DT=0)
    bytask = defaultdict(list)
    for k, v in cells.items():
        bytask[v["task"]].append(v)
    best = {}
    for t, cc in bytask.items():
        if cc[0]["kind"] == "reg":
            b = max(cc, key=lambda c: c["exp_r2"]); best[t] = {"cell": f"{b['model']}/{b['feature']}", "exp_r2": b["exp_r2"]}
        else:
            b = max(cc, key=lambda c: c["exp_bal_acc"])
            best[t] = {"cell": f"{b['model']}/{b['feature']}", "exp_bal_acc": b["exp_bal_acc"],
                       "exp_macro_f1": b["exp_macro_f1"], "chance": b["chance"]}
    dt_out = {"dt_grid": thr, "best_per_task": best,
              "dt_sweeps": {t: dict(rows) for t, rows in dt_rows.items()}}
    (_REPO / "results_hires" / "zoo_dt_sweep.json").write_text(json.dumps(dt_out, indent=1))

    # ---- plot 1: best exp bal-acc per task vs chance ----
    OUT = _REPO / "results" / "figures" / "hires"; OUT.mkdir(parents=True, exist_ok=True)
    cls_tasks = [t for t in bytask if bytask[t][0]["kind"] == "cls"]
    cls_tasks.sort()
    bals = [best[t]["exp_bal_acc"] for t in cls_tasks]; ch = [best[t]["chance"] for t in cls_tasks]
    x = np.arange(len(cls_tasks))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x, bals, 0.6, color="#1f77b4", edgecolor="black", lw=0.4, label="best cell exp balanced-acc")
    ax.plot(x, ch, "k_", ms=22, mew=2, label="chance (1/n_classes)")
    for i, t in enumerate(cls_tasks):
        ax.annotate(best[t]["cell"], (x[i], bals[i]), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=6, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(cls_tasks, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("experimental balanced accuracy"); ax.set_ylim(0, 0.9)
    ax.set_title("Hi-res model zoo — best-transferring cell per task (zero-shot on experiment)", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT / "zoo_best_per_task.png", dpi=130); plt.close(fig)

    # ---- plot 2: is_bolt DT curves (top cells) ----
    if "is_bolt" in dt_rows:
        fig, ax = plt.subplots(figsize=(9, 5.2))
        top = sorted(dt_rows["is_bolt"], key=lambda kv: -(kv[1][0] or 0))[:4]
        for key, curve in top:
            xs = [thr[i] for i, v in enumerate(curve) if v is not None]
            ys = [v for v in curve if v is not None]
            ax.plot(xs, ys, "o-", lw=2, label=key.split("/", 1)[1])
        ax.axhline(0.5, ls=":", color="black", alpha=0.5)
        ax.set_xlabel("min bolt-loosening severity kept in positives (%)")
        ax.set_ylabel("experimental balanced accuracy")
        ax.set_title("is_bolt — transfer improves with damage severity (top cells)", fontweight="bold")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(OUT / "zoo_dt_is_bolt.png", dpi=130); plt.close(fig)

    n_noncol = sum(1 for v in cells.values() if v["kind"] == "cls" and not v["collapse"])
    n_cls = sum(1 for v in cells.values() if v["kind"] == "cls")
    print(f"summarised {len(cells)} cells ({n_noncol}/{n_cls} cls non-collapsed on exp)")
    print("wrote results_hires/zoo_summary.json, results_hires/zoo_dt_sweep.json, 2 figures")


if __name__ == "__main__":
    main()
