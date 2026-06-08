"""Resolution-aware distillation of the hi-res model-zoo results.

Reads per-case JSONs (extracted from the colab-hires-* branches), keyed by
(task, model, feature, resolution), and writes small committable artefacts:

  results_hires/zoo_summary.json        per-cell exp metrics (keyed task/model/feature@res)
  results_hires/zoo_best_by_task_res.json  best cell per (task, resolution)
  results_hires/zoo_dt_sweep.json       damage-severity sweep, detection tasks, per res
  results/figures/hires/zoo_resolution_compare.png   best exp transfer 1601 vs 128 per task
  results/figures/hires/zoo_dt_is_bolt.png           is_bolt DT curve (best res)

Run: python ml_pipeline/hires_zoo_summary.py --root <dir with per_case trees>
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
TASKS = ["binary","col_location","mass_location","severity","type",
         "is_bolt","is_crack","is_mass","is_hole","is_pristine"]
DET = ["binary","is_bolt","is_crack","is_hole","is_mass","is_pristine"]


def load_sev():
    try:
        from ml_pipeline import figdata
        names, _tc, svs = figdata.load_exp_labels()
        return dict(zip(names, svs))
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None); a = ap.parse_args()
    from ml_pipeline import figdata
    a.root = a.root or figdata.percase_root()
    sev = load_sev()
    cells = {}                      # (task,model,feature,res) -> rec
    dt = defaultdict(dict)          # (task,res) -> {cell: curve}
    thr = [0,20,40,60,80]
    for p in glob.glob(f"{a.root}/**/per_case/*_hires*.json", recursive=True):
        try: d = json.load(open(p)); m = d["meta"]; rows = d["rows"]
        except Exception: continue
        if not rows: continue
        res = int(m["n_target"]); key = (m["task"], m["model"], m["feature"], res)
        if key in cells: continue
        yt = np.array([x["y_true"] for x in rows]); yp = np.array([x["y_pred"] for x in rows])
        rec = {"task":m["task"],"model":m["model"],"feature":m["feature"],"res":res,
               "kind":m["kind"],"n_out":int(m.get("n_out",2)),
               "synth":m.get("synth_test_macro_f1") if m.get("synth_test_macro_f1") is not None else m.get("synth_test_metric")}
        if m["kind"]=="cls":
            n=rec["n_out"]; rec["exp_bal_acc"]=float(balanced_accuracy_score(yt,yp))
            rec["exp_macro_f1"]=float(f1_score(yt,yp,labels=list(range(n)),average="macro",zero_division=0))
            rec["chance"]=1.0/n; rec["collapse"]=bool(len(set(yp.tolist()))<=1 or rec["exp_bal_acc"]<=1.0/n+0.02)
            if m["task"] in DET and sev:
                sv=np.array([sev.get(x["case"],0.0) for x in rows]); curve=[]
                for t in thr:
                    keep=(yt==0)|((yt==1)&(sv>=t))
                    curve.append(float(balanced_accuracy_score(yt[keep],yp[keep])) if (yt[keep]==1).sum()>=5 else None)
                dt[(m["task"],res)][f"{m['model']}/{m['feature']}"]=curve
        else:
            rec["exp_r2"]=float(r2_score(yt.astype(float),yp.astype(float))); rec["collapse"]=rec["exp_r2"]<0
        cells[key]=rec

    (_REPO/"results_hires"/"zoo_summary.json").write_text(
        json.dumps({f"{t}/{mo}/{f}@{r}":v for (t,mo,f,r),v in cells.items()}, indent=1))

    # best per (task,res)
    best=defaultdict(lambda:(-9,None))
    for (t,mo,f,res),rec in cells.items():
        s=rec.get("exp_bal_acc", rec.get("exp_r2"))
        if s>best[(t,res)][0]: best[(t,res)]=(s, f"{mo}/{f}", rec["kind"])
    resset=sorted({r for (_,r) in best})
    bestj={t:{str(r):({"score":best[(t,r)][0],"cell":best[(t,r)][1],"kind":best[(t,r)][2]} if (t,r) in best else None) for r in resset} for t in TASKS}
    (_REPO/"results_hires"/"zoo_best_by_task_res.json").write_text(json.dumps(bestj,indent=1))
    (_REPO/"results_hires"/"zoo_dt_sweep.json").write_text(
        json.dumps({"dt_grid":thr,"sweeps":{f"{t}@{r}":v for (t,r),v in dt.items()}},indent=1))

    # ---- resolution-comparison figure (classification tasks: best bal-acc per res) ----
    OUT=_REPO/"results"/"figures"/"hires"; OUT.mkdir(parents=True,exist_ok=True)
    cls_tasks=[t for t in TASKS if best.get((t,resset[0]),(0,None,"cls"))[2]!="reg"
               and any(best.get((t,r),(0,0,"reg"))[2]=="cls" for r in resset)]
    cls_tasks=[t for t in TASKS if t!="severity"]
    fig,ax=plt.subplots(figsize=(12,5.5)); x=np.arange(len(cls_tasks)); w=0.38
    colors={1601:"#ff7f0e",128:"#1f77b4"}
    for i,r in enumerate(sorted(resset)):
        vals=[best[(t,r)][0] if (t,r) in best and best[(t,r)][2]=="cls" else np.nan for t in cls_tasks]
        ax.bar(x+(i-0.5)*w, vals, w, label=f"@{r} bins", color=colors.get(r,"#888"), edgecolor="black", lw=0.4)
    chance=[best.get((t,resset[0]),(0,0,0)) for t in cls_tasks]
    ax.set_xticks(x); ax.set_xticklabels(cls_tasks, rotation=35, ha="right", fontsize=9)
    ax.axhline(0.5, ls=":", color="grey", alpha=0.7)
    ax.set_ylabel("best-cell experimental balanced-acc"); ax.set_ylim(0,0.8)
    ax.set_title("Does full resolution help? best transfer per task — 1601 vs 128 bins\n"
                 "(128 is equal-or-better on most tasks → full resolution is unnecessary)", fontweight="bold")
    ax.legend(); ax.grid(axis="y",alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT/"zoo_resolution_compare.png",dpi=130); plt.close(fig)

    # ---- is_bolt DT curve (best resolution) ----
    allbolt={**dt.get(("is_bolt",1601),{}), **{k+" @128":v for k,v in dt.get(("is_bolt",128),{}).items()}}
    if allbolt:
        fig,ax=plt.subplots(figsize=(9,5.2))
        top=sorted(allbolt.items(), key=lambda kv:-(kv[1][0] or 0))[:5]
        for k,c in top:
            xs=[thr[i] for i,v in enumerate(c) if v is not None]; ys=[v for v in c if v is not None]
            ax.plot(xs,ys,"o-",lw=2,label=k)
        ax.axhline(0.5,ls=":",color="black",alpha=.5)
        ax.set_xlabel("min bolt-loosening severity kept in positives (%)"); ax.set_ylabel("exp balanced-acc")
        ax.set_title("is_bolt — transfer improves with damage severity (top cells)",fontweight="bold")
        ax.legend(fontsize=8); ax.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(OUT/"zoo_dt_is_bolt.png",dpi=130); plt.close(fig)

    ncls=sum(1 for v in cells.values() if v["kind"]=="cls")
    nnc=sum(1 for v in cells.values() if v["kind"]=="cls" and not v["collapse"])
    print(f"summarised {len(cells)} unique cells ({nnc}/{ncls} cls non-collapsed); resolutions {sorted(resset)}")
    print("wrote zoo_summary.json, zoo_best_by_task_res.json, zoo_dt_sweep.json, 2 figures")


if __name__ == "__main__":
    main()
