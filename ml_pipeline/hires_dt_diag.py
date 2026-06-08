"""DT-swept DIAGNOSTICS: how AUC, macro-F1, sensitivity/specificity and the
confusion matrix evolve as we keep only the more-severe positives.

The plain DT sweep (hires_dt_1601.py) tracks best-cell balanced-accuracy vs
severity percentile. This goes further: for the best cell of each detection
task it recomputes the FULL diagnostic suite at each percentile threshold
(using the stored class probabilities), so we can see AUC and the confusion
matrix — not just one scalar — improve with damage.

Reads 1601 per-case JSONs (with `proba`) from --root + severity from the
experimental h5. Writes results_hires/dt_diag.json and:
  figures/hires/dt_auc.png           AUC + macro-F1 vs severity percentile, per task
  figures/hires/dt_confusion_evo.png confusion-matrix strip (task × percentile)

Run: python ml_pipeline/hires_dt_diag.py --root /tmp/allres
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (balanced_accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix)

_REPO = Path(__file__).resolve().parent.parent
DET = ["binary", "is_bolt", "is_crack", "is_hole", "is_mass"]
PCTS = [0, 25, 50, 75, 90]


def best_detection_cells():
    """task -> (model, feature) chosen by experimental balanced-acc at 1601."""
    S = json.loads((_REPO/"results_hires"/"zoo_summary.json").read_text())
    best = {}
    for k, v in S.items():
        if v.get("res") != 1601 or v["task"] not in DET or v["kind"] != "cls":
            continue
        s = v.get("exp_bal_acc", 0)
        if v["task"] not in best or s > best[v["task"]][0]:
            best[v["task"]] = (s, v["model"], v["feature"])
    return {t: (m, f) for t, (s, m, f) in best.items()}


def load_rows(root, task, model, feat):
    hits = glob.glob(f"{root}/**/per_case/{task}_{model}_{feat}_hires1601.json", recursive=True)
    if not hits:
        return None
    return json.load(open(hits[0]))["rows"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); a = ap.parse_args()
    from ml_pipeline import figdata
    root = a.root or figdata.percase_root()
    names, _tc, svs = figdata.load_exp_labels()
    sev = dict(zip(names, svs))
    cells = best_detection_cells()

    out = {"percentiles": PCTS, "per_task": {}}
    for task in DET:
        if task not in cells:
            continue
        mo, fe = cells[task]
        rows = load_rows(root, task, mo, fe)
        if not rows or "proba" not in rows[0] or rows[0]["proba"] is None:
            continue
        yt = np.array([r["y_true"] for r in rows])
        yp = np.array([r["y_pred"] for r in rows])
        sc = np.array([r["proba"][1] for r in rows])           # P(positive)
        sv = np.array([sev.get(r["case"], 0.0) for r in rows])
        pos_sev = sv[yt == 1]
        thr = [np.percentile(pos_sev, q) for q in PCTS] if len(pos_sev) else [0]*len(PCTS)
        rec = {"cell": f"{mo}/{fe}", "sev_thresholds": [float(x) for x in thr],
               "auc": [], "macro_f1": [], "bal_acc": [], "sens": [], "spec": [], "n_pos": [], "cm": []}
        for th in thr:
            keep = (yt == 0) | ((yt == 1) & (sv >= th))
            ytk, ypk, sck = yt[keep], yp[keep], sc[keep]
            npos = int((ytk == 1).sum())
            rec["n_pos"].append(npos)
            if npos < 5 or len(set(ytk.tolist())) < 2:
                for kk in ("auc", "macro_f1", "bal_acc", "sens", "spec"): rec[kk].append(None)
                rec["cm"].append(None); continue
            cm = confusion_matrix(ytk, ypk, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            rec["auc"].append(float(roc_auc_score(ytk, sck)))
            rec["macro_f1"].append(float(f1_score(ytk, ypk, labels=[0, 1], average="macro", zero_division=0)))
            rec["bal_acc"].append(float(balanced_accuracy_score(ytk, ypk)))
            rec["sens"].append(float(tp/(tp+fn)) if tp+fn else None)
            rec["spec"].append(float(tn/(tn+fp)) if tn+fp else None)
            cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
            rec["cm"].append([[round(float(x), 3) for x in r] for r in cmn])
        out["per_task"][task] = rec
    (_REPO/"results_hires"/"dt_diag.json").write_text(json.dumps(out, indent=1))

    OUT = _REPO/"results"/"figures"/"hires"
    # ---- (a) AUC + macro-F1 vs percentile ----
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.2))
    cmap = plt.get_cmap("tab10")
    for i, task in enumerate(DET):
        r = out["per_task"].get(task)
        if not r: continue
        xs = [PCTS[j] for j in range(len(PCTS)) if r["auc"][j] is not None]
        ya = [v for v in r["auc"] if v is not None]
        yf = [r["macro_f1"][j] for j in range(len(PCTS)) if r["auc"][j] is not None]
        if xs:
            ax[0].plot(xs, ya, "o-", lw=2, color=cmap(i), label=f"{task} ({r['cell']})")
            ax[1].plot(xs, yf, "o-", lw=2, color=cmap(i), label=task)
    for a_, ttl, yl in [(ax[0], "(a) ROC-AUC vs severity kept", "experimental AUC"),
                        (ax[1], "(b) macro-F1 vs severity kept", "experimental macro-F1")]:
        a_.axhline(0.5, ls=":", color="black", alpha=.5)
        a_.set_xlabel("keep positives with severity ≥ this percentile (more-damaged →)")
        a_.set_ylabel(yl); a_.set_title(ttl, fontweight="bold", fontsize=10); a_.grid(alpha=.3)
    ax[0].legend(fontsize=8, loc="lower right")
    plt.tight_layout(); plt.savefig(OUT/"dt_auc.png", dpi=130); plt.close(fig)

    # ---- (b) confusion-matrix evolution strip: tasks × percentiles ----
    tasks = [t for t in DET if t in out["per_task"]]
    cols = len(PCTS)
    fig, axs = plt.subplots(len(tasks), cols, figsize=(2.5*cols, 2.5*len(tasks)))
    axs = np.atleast_2d(axs)
    for ri, task in enumerate(tasks):
        r = out["per_task"][task]
        for ci, p in enumerate(PCTS):
            a = axs[ri, ci]; cm = r["cm"][ci]
            if cm is None:
                a.axis("off"); continue
            a.imshow(cm, cmap="Blues", vmin=0, vmax=1)
            for ii in range(2):
                for jj in range(2):
                    a.text(jj, ii, f"{cm[ii][jj]:.2f}", ha="center", va="center", fontsize=8,
                           color="white" if cm[ii][jj] > .5 else "black")
            a.set_xticks([0, 1]); a.set_yticks([0, 1])
            a.set_xticklabels(["neg", "pos"], fontsize=6); a.set_yticklabels(["neg", "pos"], fontsize=6)
            au = r["auc"][ci]
            if ri == 0:
                a.set_title(f"≥p{p}", fontsize=9, fontweight="bold")
            if ci == 0:
                a.set_ylabel(f"{task}\n{r['cell']}", fontsize=7)
            a.text(0.5, 1.18 if ri else 1.32, f"AUC {au:.2f}" if au else "", transform=a.transAxes,
                   ha="center", fontsize=6.5, color="#444")
    fig.suptitle("Confusion matrix (row-normalised) evolving along the DT severity sweep — "
                 "columns keep progressively more-severe positives (all negatives retained)",
                 fontweight="bold", fontsize=11, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97]); plt.savefig(OUT/"dt_confusion_evo.png", dpi=130); plt.close(fig)

    print("=== DT-swept diagnostics (best cell per detection task) ===")
    for task in tasks:
        r = out["per_task"][task]
        print(f"{task:9s} {r['cell']:28s} AUC " +
              " ".join((f"{v:.2f}" if v is not None else " -- ") for v in r["auc"]) +
              "  sens " + " ".join((f"{v:.2f}" if v is not None else " -- ") for v in r["sens"]))
    print("wrote results_hires/dt_diag.json + dt_auc.png + dt_confusion_evo.png")


if __name__ == "__main__":
    main()
