"""DT sweep on a COMMON PHYSICAL axis: storey-stiffness loss (%).

The severity DT sweep (hires_dt_1601 / hires_dt_diag) stratifies each task on
its own native severity unit (bolt %, crack mm, hole mm, mass kg), so the tasks
are not directly comparable. Here every stiffness-reducing damage is mapped to
the actual fraction of storey stiffness it removes, using the SAME calibrated
functions the simulator used to generate the data
(ml_pipeline.variation.{bolt_jsr_ratio, crack_ratio, hole_ratio}):

    stiffness_loss = 1 - stiffness_ratio(severity)        (bolt / crack / hole)
    stiffness_loss = 0                                    (mass — adds inertia,
                                                           not compliance)

This puts bolt, crack and hole on one physical x-axis and reveals *why* they
transfer so differently: experimentally bolt removes 15–61% of storey stiffness
while crack/hole remove only 2–6% — the latter barely perturb the structure, so
there is no high-stiffness-loss regime to climb into.

Reads 1601 per-case JSONs (with `proba`) from --root + type_code/severity from
the experimental h5. Writes results_hires/dt_stiffness.json and
  figures/hires/dt_stiffness.png      balanced-acc + AUC vs min stiffness loss kept
  figures/hires/dt_stiffness_map.png  severity→stiffness-loss map + exp distribution

Run: python ml_pipeline/hires_dt_stiffness.py --root /tmp/allres
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
from ml_pipeline.variation import bolt_jsr_ratio, crack_ratio, hole_ratio

# tasks whose positives reduce stiffness (mass excluded — it adds mass)
STIFF_TASKS = ["binary", "is_bolt", "is_crack", "is_hole"]
# absolute stiffness-loss thresholds (%) — a physical, cross-task-comparable grid
THR = [0.0, 2.0, 5.0, 10.0, 20.0, 40.0]
TYPE_NAMES = ["pristine", "bolt", "crack", "hole", "mass"]
TCOL = {"bolt": "#1f77b4", "crack": "#d62728", "hole": "#2ca02c", "mass": "#9467bd", "binary": "#000000"}


def stiffness_loss(tc: int, sev: float) -> float:
    """Fraction of storey stiffness removed (0..1). 0 for mass/pristine."""
    if tc == 1: return 1.0 - bolt_jsr_ratio(sev)
    if tc == 2: return 1.0 - crack_ratio(sev)
    if tc == 3: return 1.0 - hole_ratio(sev)
    return 0.0


def best_cells():
    S = json.loads((_REPO/"results_hires"/"zoo_summary.json").read_text())
    best = {}
    for k, v in S.items():
        if v.get("res") != 1601 or v["task"] not in STIFF_TASKS or v["kind"] != "cls":
            continue
        s = v.get("exp_bal_acc", 0)
        if v["task"] not in best or s > best[v["task"]][0]:
            best[v["task"]] = (s, v["model"], v["feature"])
    return {t: (m, f) for t, (s, m, f) in best.items()}


def load_rows(root, task, model, feat):
    hits = glob.glob(f"{root}/**/per_case/{task}_{model}_{feat}_hires1601.json", recursive=True)
    return json.load(open(hits[0]))["rows"] if hits else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); a = ap.parse_args()
    from ml_pipeline import figdata
    root = a.root or figdata.percase_root()
    names, tcs, svs = figdata.load_exp_labels()
    tcs = tcs.astype(int); svs = svs.astype(float)
    tc_by = dict(zip(names, tcs)); sev_by = dict(zip(names, svs))
    sl_by = {n: 100.0 * stiffness_loss(tc_by[n], sev_by[n]) for n in names}   # percent
    cells = best_cells()

    out = {"thresholds_pct": THR, "per_task": {}}
    for task in STIFF_TASKS:
        if task not in cells:
            continue
        mo, fe = cells[task]
        rows = load_rows(root, task, mo, fe)
        if not rows:
            continue
        yt = np.array([r["y_true"] for r in rows])
        yp = np.array([r["y_pred"] for r in rows])
        has_prob = "proba" in rows[0] and rows[0]["proba"] is not None
        sc = np.array([r["proba"][1] for r in rows]) if has_prob else None
        sl = np.array([sl_by.get(r["case"], 0.0) for r in rows])
        rec = {"cell": f"{mo}/{fe}", "bal_acc": [], "auc": [], "macro_f1": [], "sens": [], "n_pos": []}
        for th in THR:
            keep = (yt == 0) | ((yt == 1) & (sl >= th))
            ytk, ypk = yt[keep], yp[keep]
            npos = int((ytk == 1).sum()); rec["n_pos"].append(npos)
            if npos < 5 or len(set(ytk.tolist())) < 2:
                for kk in ("bal_acc", "auc", "macro_f1", "sens"): rec[kk].append(None)
                continue
            rec["bal_acc"].append(float(balanced_accuracy_score(ytk, ypk)))
            rec["macro_f1"].append(float(f1_score(ytk, ypk, labels=[0, 1], average="macro", zero_division=0)))
            tp = int(((ytk == 1) & (ypk == 1)).sum()); rec["sens"].append(float(tp / npos))
            rec["auc"].append(float(roc_auc_score(ytk, sc[keep])) if sc is not None else None)
        out["per_task"][task] = rec
    (_REPO/"results_hires"/"dt_stiffness.json").write_text(json.dumps(out, indent=1))

    OUT = _REPO/"results"/"figures"/"hires"
    # ---- (1) balanced-acc + AUC vs min stiffness loss ----
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.3))
    for task in STIFF_TASKS:
        r = out["per_task"].get(task)
        if not r: continue
        col = TCOL.get(task.replace("is_", ""), "#000") if task != "binary" else "#000000"
        xb = [THR[i] for i in range(len(THR)) if r["bal_acc"][i] is not None]
        yb = [v for v in r["bal_acc"] if v is not None]
        np_lbl = r["n_pos"]
        if xb:
            ax[0].plot(xb, yb, "o-", lw=2, color=col, label=f"{task} ({r['cell']})")
            # annotate where the curve truncates (positives run out)
            ax[0].annotate(f"n+={np_lbl[len(xb)-1]}", (xb[-1], yb[-1]), fontsize=7,
                           color=col, xytext=(3, 3), textcoords="offset points")
        xa = [THR[i] for i in range(len(THR)) if r["auc"][i] is not None]
        ya = [v for v in r["auc"] if v is not None]
        if xa:
            ax[1].plot(xa, ya, "o-", lw=2, color=col, label=task)
    for a_, ttl, yl in [(ax[0], "(a) balanced accuracy", "experimental balanced-acc"),
                        (ax[1], "(b) ROC-AUC", "experimental AUC")]:
        a_.axhline(0.5, ls=":", color="black", alpha=.5, label="chance")
        a_.axvspan(0, 6.4, color="grey", alpha=.08)
        a_.set_xlabel("keep positives with ≥ this storey-stiffness loss (%)")
        a_.set_ylabel(yl); a_.set_title(ttl, fontweight="bold", fontsize=10); a_.grid(alpha=.3)
    ax[0].legend(fontsize=8, loc="lower right")
    ax[0].text(6.6, 0.52, "← crack/hole live\nentirely in here\n(≤6.4% loss)", fontsize=7.5, color="#555")
    fig.suptitle("DT sweep on a PHYSICAL axis — transfer vs storey-stiffness loss (best cell per task)\n"
                 "bolt removes 15–61% stiffness and keeps improving; crack/hole remove ≤6% and never enter the high-loss regime",
                 fontweight="bold", fontsize=11, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.92]); plt.savefig(OUT/"dt_stiffness.png", dpi=130); plt.close(fig)

    # ---- (2) severity->stiffness map + experimental distribution ----
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.0))
    grids = {"bolt": (np.linspace(5, 95, 100), bolt_jsr_ratio, "% loosening"),
             "crack": (np.linspace(1, 8, 100), crack_ratio, "mm depth"),
             "hole": (np.linspace(1, 6, 100), hole_ratio, "mm diameter")}
    for name, (xs, fn, unit) in grids.items():
        ys = [100*(1-fn(x)) for x in xs]
        ax[0].plot(xs, ys, lw=2, color=TCOL[name], label=f"{name} ({unit})")
    ax[0].set_xlabel("native severity (per-type units)"); ax[0].set_ylabel("storey-stiffness loss (%)")
    ax[0].set_title("(a) calibrated severity → stiffness-loss map\n(the simulator's own damage model)",
                    fontweight="bold", fontsize=10); ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)
    # experimental distribution per type
    data = []
    for c in range(1, 5):
        m = tcs == c
        data.append(100*np.array([stiffness_loss(c, s) for s in svs[m]]))
    bp = ax[1].boxplot(data, labels=TYPE_NAMES[1:], patch_artist=True, showfliers=True, widths=.6)
    for patch, c in zip(bp["boxes"], range(1, 5)):
        patch.set_facecolor(TCOL[TYPE_NAMES[c]]); patch.set_alpha(.7)
    ax[1].set_ylabel("storey-stiffness loss (%)")
    ax[1].set_title("(b) experimental stiffness loss by damage type\n"
                    "bolt dominates; mass removes none", fontweight="bold", fontsize=10)
    ax[1].grid(axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig(OUT/"dt_stiffness_map.png", dpi=130); plt.close(fig)

    print("=== DT sweep vs storey-stiffness loss (best cell per task) ===")
    print("task".ljust(10) + "".join(f"≥{t:g}%".rjust(9) for t in THR))
    for task in STIFF_TASKS:
        r = out["per_task"].get(task)
        if not r: continue
        ba = r["bal_acc"]; nP = r["n_pos"]
        print(task.ljust(10) + "".join((f"{ba[i]:.3f}" if i < len(ba) and ba[i] is not None else "  -- ").rjust(9)
                                        for i in range(len(THR))))
        print("   n_pos:".ljust(10) + "".join(f"{n}".rjust(9) for n in nP))
    print("wrote results_hires/dt_stiffness.json + dt_stiffness.png + dt_stiffness_map.png")


if __name__ == "__main__":
    main()
