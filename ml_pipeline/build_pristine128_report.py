"""Build the detailed report for the pristine-anchored 128-bin study.

Trains-once-elsewhere; this script only *reads* committed per-case predictions
and renders the report + every figure, so it reproduces from the repo alone:

  inputs (committed):
    results_hires_zoo/pristine128/per_case_pristine128.tar.gz   (30 cells)
    results_hires/per_case_hires128.tar.gz                      (570 calibrated cells)
  outputs:
    results/figures/pristine128/*.png
    results/REPORT_PRISTINE128.md

The story is a paired comparison: the *same* (model, feature) cell trained on a
model whose damage was sized by first-principles, pristine-anchored physics
(`ml_pipeline.pristine_physics`) vs the damage-calibrated model whose magnitudes
were fitted to the damaged experimental FRFs.
"""
from __future__ import annotations
import json, os, re, tarfile, tempfile, glob
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (balanced_accuracy_score, f1_score, accuracy_score,
                             r2_score, roc_curve, auc, confusion_matrix)

_REPO = Path(__file__).resolve().parent.parent
FIG = _REPO/"results"/"figures"/"pristine128"; FIG.mkdir(parents=True, exist_ok=True)
PRISTINE_TGZ = _REPO/"results_hires_zoo"/"pristine128"/"per_case_pristine128.tar.gz"
CALIB_TGZ    = _REPO/"results_hires"/"per_case_hires128.tar.gz"

TASKS = ["binary","is_pristine","is_bolt","is_crack","is_hole","is_mass",
         "type","col_location","mass_location","severity"]
DET   = ["binary","is_pristine","is_bolt","is_crack","is_hole","is_mass"]
DESC  = {"binary":"Pristine vs Damage","is_pristine":"Is pristine","is_bolt":"Is bolt",
  "is_crack":"Is crack","is_hole":"Is hole","is_mass":"Is mass","type":"Damage type (5)",
  "col_location":"Column location (6)","mass_location":"Mass plate (4)","severity":"Severity (reg)"}
TYPE_LBL = ["Pristine","Bolt","Crack","Hole","Mass"]
COL_LBL  = ["S1·BD","S1·AD","S2·BD","S2·AD","S3·BD","S3·AD"]
MASS_LBL = ["Base","Fl1","Fl2","Fl3"]
PC, CC = "#2c7fb8", "#9aa7b5"   # pristine / calibrated colours


# ── load per-case predictions ────────────────────────────────────────────────
def _clean(case):
    s=str(case)
    m=re.match(r"^b['\"](.*)['\"]$", s)
    return m.group(1) if m else s

def load(tgz):
    tmp=tempfile.mkdtemp();
    with tarfile.open(tgz) as t: t.extractall(tmp)
    out={}
    for p in glob.glob(os.path.join(tmp,"**","*_hires128.json"),recursive=True):
        d=json.load(open(p)); m=d["meta"]; r=d["rows"]
        yt=np.array([x["y_true"] for x in r]); yp=np.array([x["y_pred"] for x in r])
        pr=[x.get("proba") for x in r]          # list of per-row prob lists (keep as list!)
        cs=[_clean(x.get("case")) for x in r]
        out[(m["task"],m["model"],m["feature"])]=dict(meta=m,yt=yt,yp=yp,proba=pr,case=cs,kind=m["kind"])
    return out

def score(c):
    yt,yp=c["yt"],c["yp"]
    if c["kind"]=="cls":
        n=c["meta"]["n_out"]; bal=balanced_accuracy_score(yt,yp)
        return dict(metric=bal,bal=bal,mf1=f1_score(yt,yp,labels=list(range(n)),average="macro",zero_division=0),
                    acc=accuracy_score(yt,yp),chance=1/n,n_out=n,
                    collapse=(len(set(yp.tolist()))<=1) or (bal<=1/n+0.02))
    yt=yt.astype(float); yp=yp.astype(float)
    return dict(metric=(r2_score(yt,yp) if np.var(yt)>0 else 0.0),
                mae=float(np.mean(np.abs(yt-yp))),chance=0.0,collapse=False)

def best_cell(D,task):
    cs=[(k,v) for k,v in D.items() if k[0]==task]
    cs.sort(key=lambda kv:(score(kv[1])["collapse"], -score(kv[1])["metric"]))
    return cs[0]

def pos_proba(c):
    """positive-class probability for a 2-class detection cell."""
    out=np.full(len(c["yt"]),np.nan)
    for i,p in enumerate(c["proba"]):
        if isinstance(p,(list,tuple,np.ndarray)) and len(p)>=2: out[i]=float(p[1])
    return out


# ── figures ───────────────────────────────────────────────────────────────────
def fig_overview(P,C):
    cls=[t for t in TASKS if t!="severity"]
    pv=[score(best_cell(P,t)[1])["metric"] for t in cls]
    cv=[score(C[best_cell(P,t)[0]])["metric"] for t in cls]
    ch=[score(best_cell(P,t)[1])["chance"] for t in cls]
    x=np.arange(len(cls)); w=0.38
    fig,ax=plt.subplots(figsize=(11,4.3))
    ax.bar(x-w/2,cv,w,color=CC,label="calibrated (damage fitted to test set)")
    ax.bar(x+w/2,pv,w,color=PC,label="pristine-anchored (first-principles damage)")
    for xi,c in zip(x,ch): ax.plot([xi-.5,xi+.5],[c,c],color="crimson",ls="--",lw=1.1)
    ax.plot([],[],color="crimson",ls="--",lw=1.1,label="chance")
    ax.set_xticks(x); ax.set_xticklabels([DESC[t] for t in cls],rotation=30,ha="right",fontsize=8)
    ax.set_ylabel("experimental balanced accuracy\n(zero-shot)"); ax.set_ylim(0,.8)
    ax.legend(fontsize=8); ax.set_title("Best of top-3 cells per task — pristine-anchored vs calibrated")
    fig.tight_layout(); fig.savefig(FIG/"overview_best.png",dpi=140); plt.close(fig)

def fig_delta(P,C):
    cls=[t for t in TASKS if t!="severity"]
    d=[(t, score(C[best_cell(P,t)[0]])["metric"]-score(best_cell(P,t)[1])["metric"]) for t in cls]
    d.sort(key=lambda z:z[1])
    fig,ax=plt.subplots(figsize=(7,4))
    ys=np.arange(len(d)); cols=["#2ca25f" if z[1]<=0 else "#de2d26" for z in d]
    ax.barh(ys,[z[1] for z in d],color=cols)
    ax.set_yticks(ys); ax.set_yticklabels([DESC[z[0]] for z in d],fontsize=8)
    ax.axvline(0,color="k",lw=.8); ax.set_xlabel("calibrated − pristine  (balanced-acc lost by going pristine-only)")
    ax.set_title("Cost of refusing damaged data, per task"); fig.tight_layout()
    fig.savefig(FIG/"delta_per_task.png",dpi=140); plt.close(fig)

def fig_submodels():
    import ml_pipeline.pristine_physics as PP
    import ml_pipeline.variation as V
    fig,axs=plt.subplots(1,3,figsize=(12,3.6))
    # bolt
    pct=np.linspace(5,95,50)
    axs[0].plot(pct,[PP.bolt_jsr_ratio(p) for p in pct],color=PC,lw=2,label="pristine: 1−p/100")
    axs[0].plot(pct,[V.bolt_jsr_ratio(p) for p in pct],color=CC,lw=2,ls="--",label="calibrated (fit to damaged FRFs)")
    axs[0].scatter([11,20,50,85],[.85,.70,.55,.39],color=CC,zorder=3,s=18)
    axs[0].set_title("Bolt — per-end JSR ratio"); axs[0].set_xlabel("loosening %")
    # crack
    mm=np.linspace(1,8,50)
    axs[1].plot(mm,[PP.crack_ratio(m) for m in mm],color=PC,lw=2,label="pristine: (lx−a)/lx")
    axs[1].plot(mm,[V.crack_ratio(m) for m in mm],color=CC,lw=2,ls="--",label="calibrated")
    axs[1].scatter([5,8],[.96,.94],color=CC,zorder=3,s=18)
    axs[1].set_title("Crack — storey-stiffness ratio"); axs[1].set_xlabel("crack length (mm)")
    # hole
    mm=np.linspace(1,6,50)
    axs[2].plot(mm,[PP.hole_ratio(m) for m in mm],color=PC,lw=2,label="pristine: 1−πφ⁴/64 / I")
    axs[2].plot(mm,[V.hole_ratio(m) for m in mm],color=CC,lw=2,ls="--",label="calibrated")
    axs[2].scatter([4,6],[.98,.97],color=CC,zorder=3,s=18)
    axs[2].set_title("Hole — storey-stiffness ratio"); axs[2].set_xlabel("hole diameter (mm)")
    for a in axs: a.set_ylabel("k_damaged / k_pristine"); a.legend(fontsize=7); a.grid(alpha=.3)
    fig.suptitle("Damage submodels: pristine-anchored physics (solid) vs damaged-data-fitted tables (dashed + anchors)")
    fig.tight_layout(); fig.savefig(FIG/"damage_submodels.png",dpi=140); plt.close(fig)

def fig_confusion(P,C,task,labels):
    kp=best_cell(P,task); kc=C[kp[0]]
    fig,axs=plt.subplots(1,2,figsize=(2.4+1.0*len(labels),0.9+0.5*len(labels)))
    for ax,(c,title) in zip(axs,[(kp[1],f"pristine-anchored\n{kp[0][1]}/{kp[0][2]}"),
                                 (kc,f"calibrated\n{kp[0][1]}/{kp[0][2]}")]):
        n=len(labels); cm=confusion_matrix(c["yt"],c["yp"],labels=list(range(n)))
        cmn=cm/np.clip(cm.sum(1,keepdims=True),1,None)
        im=ax.imshow(cmn,cmap="Blues",vmin=0,vmax=1)
        ax.set_xticks(range(n)); ax.set_xticklabels(labels,rotation=45,ha="right",fontsize=7)
        ax.set_yticks(range(n)); ax.set_yticklabels(labels,fontsize=7)
        for i in range(n):
            for j in range(n):
                ax.text(j,i,f"{cmn[i,j]:.2f}",ha="center",va="center",fontsize=6,
                        color="white" if cmn[i,j]>.5 else "black")
        ax.set_title(title,fontsize=8); ax.set_xlabel("predicted",fontsize=7)
    axs[0].set_ylabel("true",fontsize=7)
    fig.suptitle(f"{DESC[task]} — experimental confusion (row-normalised)",fontsize=10)
    fig.tight_layout(); fig.savefig(FIG/f"confusion_{task}.png",dpi=140); plt.close(fig)

def fig_roc(P,C):
    fig,axs=plt.subplots(2,3,figsize=(12,7))
    for ax,task in zip(axs.ravel(),DET):
        kp=best_cell(P,task); kc=C[kp[0]]
        for c,col,lab in [(kp[1],PC,"pristine"),(kc,CC,"calibrated")]:
            s=pos_proba(c)
            if np.isnan(s).all(): continue
            fpr,tpr,_=roc_curve(c["yt"],s); a=auc(fpr,tpr)
            ax.plot(fpr,tpr,color=col,lw=2,label=f"{lab} (AUC {a:.2f})")
        ax.plot([0,1],[0,1],color="crimson",ls="--",lw=.8)
        ax.set_title(f"{DESC[task]}  ({kp[0][1]}/{kp[0][2]})",fontsize=8)
        ax.set_xlabel("FPR",fontsize=7); ax.set_ylabel("TPR",fontsize=7); ax.legend(fontsize=7)
    fig.suptitle("Detection ROC — best cell per task, experimental zero-shot",fontsize=11)
    fig.tight_layout(); fig.savefig(FIG/"roc_detection.png",dpi=140); plt.close(fig)

def fig_cellzoo(P,C,task):
    cells=sorted([k for k in P if k[0]==task],key=lambda k:-score(P[k])["metric"])
    reg=(P[cells[0]]["kind"]=="reg")
    pv=[score(P[k])["metric"] for k in cells]; cv=[score(C[k])["metric"] for k in cells]
    ch=score(P[cells[0]])["chance"]
    x=np.arange(len(cells)); w=.38
    fig,ax=plt.subplots(figsize=(6.5,3.4))
    ax.bar(x-w/2,cv,w,color=CC,label="calibrated"); ax.bar(x+w/2,pv,w,color=PC,label="pristine")
    if not reg: ax.axhline(ch,color="crimson",ls="--",lw=1,label="chance")
    ax.set_xticks(x); ax.set_xticklabels([f"{k[1]}\n{k[2]}" for k in cells],fontsize=7)
    ax.set_ylabel("exp R²" if reg else "exp balanced-acc")
    ax.set_title(f"{DESC[task]} — top-3 cells"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG/f"cellzoo_{task}.png",dpi=140); plt.close(fig)

def fig_severity(P,C):
    kp=best_cell(P,"severity"); kc=C[kp[0]]
    fig,axs=plt.subplots(1,2,figsize=(9,4.2))
    for ax,(c,title) in zip(axs,[(kp[1],"pristine-anchored"),(kc,"calibrated")]):
        yt=c["yt"].astype(float); yp=c["yp"].astype(float)
        ax.scatter(yt,yp,s=6,alpha=.3,color=PC if title.startswith("pristine") else CC)
        lim=[min(yt.min(),yp.min()),max(yt.max(),yp.max())]
        ax.plot(lim,lim,color="crimson",ls="--",lw=.8)
        ax.set_xlabel("true severity (norm.)"); ax.set_ylabel("predicted")
        ax.set_title(f"{title}  R²={r2_score(yt,yp) if np.var(yt)>0 else 0:.3f}",fontsize=9)
    fig.suptitle(f"Severity regression ({kp[0][1]}/{kp[0][2]}) — experimental")
    fig.tight_layout(); fig.savefig(FIG/"severity_scatter.png",dpi=140); plt.close(fig)

def _sev_of(cases):
    from ml_pipeline.evaluate import primary_op
    return np.array([primary_op(n)["severity"] for n in cases],dtype=float), \
           np.array([primary_op(n)["type_code"] for n in cases],dtype=int)

# ── Damage-threshold (DT) severity sweep ──────────────────────────────────────
# Central thesis: detection should improve with damage size, because a bigger
# perturbation outruns the synth→real domain gap. We stratify positives by their
# damage-severity percentile, keep only the more-severe ones (all negatives kept)
# and recompute balanced-acc / ROC-AUC / sensitivity — for BOTH the pristine-
# anchored and the calibrated cell, so we can see whether the pristine-only
# physics preserves the size→detectability ordering.
DT_TASKS = ["binary","is_bolt","is_crack","is_hole","is_mass"]   # is_pristine pos sev=0
PCTS = [0,25,50,75,90]
SEV_UNIT = {"is_bolt":"% loosening","is_crack":"mm","is_hole":"mm","is_mass":"kg","binary":"pooled %ile"}

def _sweep_one(c, pcts=PCTS):
    sev,_=_sev_of(c["case"]); yt=c["yt"]; yp=c["yp"]; s=pos_proba(c)
    posv=sev[yt==1]; thr=np.percentile(posv, pcts) if posv.size else np.zeros(len(pcts))
    bal=[]; au=[]; sen=[]; npos=[]
    for th in thr:
        keep=(yt==0)|((yt==1)&(sev>=th)); ytk=yt[keep]; ypk=yp[keep]; sk=s[keep]
        n=int((ytk==1).sum()); npos.append(n)
        if n>=5:
            bal.append(balanced_accuracy_score(ytk,ypk))
            sen.append(float(((ypk==1)&(ytk==1)).sum()/max(1,(ytk==1).sum())))
            au.append(auc(*roc_curve(ytk,sk)[:2]) if not np.isnan(sk).all() else np.nan)
        else: bal.append(np.nan); sen.append(np.nan); au.append(np.nan)
    return dict(thr=thr,bal=bal,auc=au,sens=sen,npos=npos)

def dt_data(P,C):
    out={}
    for t in DT_TASKS:
        kp=best_cell(P,t); kc=C[kp[0]]
        out[t]=dict(cell=f"{kp[0][1]}/{kp[0][2]}",p=_sweep_one(kp[1]),c=_sweep_one(kc))
    return out

def fig_dt_combined(D):
    fig,ax=plt.subplots(figsize=(8.5,4.6))
    cmap=plt.cm.tab10(np.linspace(0,1,len(DT_TASKS)))
    for col,t in zip(cmap,DT_TASKS):
        ax.plot(PCTS,D[t]["p"]["bal"],color=col,lw=2,marker="o",ms=4,label=f"{DESC[t]} · pristine")
        ax.plot(PCTS,D[t]["c"]["bal"],color=col,lw=1.6,ls="--",marker="s",ms=3,alpha=.8)
    ax.plot([],[],color="k",lw=2,label="— pristine"); ax.plot([],[],color="k",ls="--",lw=1.6,label="-- calibrated")
    ax.set_xlabel("keep positives with damage severity ≥ this percentile  (more-damaged →)")
    ax.set_ylabel("experimental balanced-acc"); ax.set_xticks(PCTS); ax.grid(alpha=.3)
    ax.set_title("DT sweep — detection vs damage size  (pristine solid, calibrated dashed)")
    ax.legend(fontsize=7,ncol=2,loc="lower right"); fig.tight_layout()
    fig.savefig(FIG/"dt_combined.png",dpi=140); plt.close(fig)

def fig_dt_pertask(D):
    fig,axs=plt.subplots(2,3,figsize=(12,7))
    for ax,t in zip(axs.ravel(),DT_TASKS):
        ax.plot(PCTS,D[t]["c"]["bal"],color=CC,lw=2,ls="--",marker="s",ms=4,label="calibrated")
        ax.plot(PCTS,D[t]["p"]["bal"],color=PC,lw=2,marker="o",ms=4,label="pristine")
        ax.axhline(0.5,color="crimson",ls=":",lw=.9)
        sev=D[t]["p"]["thr"]
        ax.set_title(f"{DESC[t]}  ({D[t]['cell']})",fontsize=8)
        ax.set_xlabel(f"severity ≥ percentile  [{SEV_UNIT[t]}: {sev[0]:.0f}→{sev[-1]:.0f}]",fontsize=7)
        ax.set_ylabel("balanced-acc",fontsize=7); ax.set_xticks(PCTS); ax.set_ylim(.45,.95)
        ax.grid(alpha=.3); ax.legend(fontsize=7)
        for x,n in zip(PCTS,D[t]["p"]["npos"]): ax.annotate(f"n={n}",(x,.47),fontsize=6,ha="center",color="gray")
    axs.ravel()[-1].axis("off")
    fig.suptitle("DT sweep per detection task — balanced-acc vs damage severity (n = positives kept)",fontsize=11)
    fig.tight_layout(); fig.savefig(FIG/"dt_pertask.png",dpi=140); plt.close(fig)

def fig_dt_auc(D):
    fig,axs=plt.subplots(1,2,figsize=(12,4.4))
    cmap=plt.cm.tab10(np.linspace(0,1,len(DT_TASKS)))
    for metric,ax,ylab in [("auc",axs[0],"ROC-AUC"),("sens",axs[1],"sensitivity (recall on positives)")]:
        for col,t in zip(cmap,DT_TASKS):
            ax.plot(PCTS,D[t]["p"][metric],color=col,lw=2,marker="o",ms=4,label=DESC[t])
            ax.plot(PCTS,D[t]["c"][metric],color=col,lw=1.6,ls="--",marker="s",ms=3,alpha=.8)
        ax.set_xlabel("keep positives with severity ≥ percentile"); ax.set_ylabel(ylab)
        ax.set_xticks(PCTS); ax.grid(alpha=.3)
    axs[0].legend(fontsize=7,ncol=2); axs[0].plot([],[],color="k",lw=2,label="pristine")
    fig.suptitle("DT sweep — ranking (AUC) and sensitivity vs damage size  (pristine solid, calibrated dashed)",fontsize=11)
    fig.tight_layout(); fig.savefig(FIG/"dt_auc.png",dpi=140); plt.close(fig)

def md_dt_table(D):
    L=["| Task | Cell | metric | p0 (all) | ≥p50 | ≥p75 | ≥p90 |",
       "|---|---|---|--:|--:|--:|--:|"]
    idx={0:1,1:2,2:3}  # p50,p75,p90 are indices 2,3,4
    for t in DT_TASKS:
        for who,lab in [("p","pristine"),("c","calibrated")]:
            b=D[t][who]["bal"]
            L.append(f"| {DESC[t]} | {D[t]['cell']} | {lab} bal-acc | "
                     f"{b[0]:.3f} | {b[2]:.3f} | {b[3]:.3f} | {b[4]:.3f} |")
    return "\n".join(L)


# ── markdown ──────────────────────────────────────────────────────────────────
def md_table_headline(P,C):
    L=["| Task | Best pristine cell | Metric | Pristine | Chance | Calibrated\\* | Δ |",
       "|---|---|---|--:|--:|--:|--:|"]
    for t in TASKS:
        kp=best_cell(P,t); sp=score(kp[1]); sc=score(C[kp[0]])
        mn="R²" if kp[1]["kind"]=="reg" else "bal-acc"
        flag=" ⚠" if sp["collapse"] else ""
        L.append(f"| {DESC[t]} | `{kp[0][1]}/{kp[0][2]}` | {mn} | **{sp['metric']:.3f}**{flag} | "
                 f"{sp['chance']:.3f} | {sc['metric']:.3f} | {sp['metric']-sc['metric']:+.3f} |")
    return "\n".join(L)

def md_full_table(P,C):
    L=["| Task | Model | Feature | Kind | Pristine | macro-F1 | Calibrated\\* | Chance | Collapse |",
       "|---|---|---|---|--:|--:|--:|--:|:--:|"]
    for t in TASKS:
        for k in sorted([k for k in P if k[0]==t],key=lambda k:-score(P[k])["metric"]):
            sp=score(P[k]); sc=score(C[k])
            mf1=f"{sp.get('mf1',float('nan')):.3f}" if sp.get("mf1") is not None and k[1] else "—"
            mf1=f"{sp['mf1']:.3f}" if "mf1" in sp else "—"
            L.append(f"| {t} | {k[1]} | {k[2]} | {P[k]['kind']} | {sp['metric']:.3f} | {mf1} | "
                     f"{sc['metric']:.3f} | {sp['chance']:.3f} | {'yes' if sp['collapse'] else ''} |")
    return "\n".join(L)

def main():
    P=load(PRISTINE_TGZ); C=load(CALIB_TGZ)
    NROW=max(len(v["yt"]) for v in P.values())
    fig_overview(P,C); fig_delta(P,C); fig_submodels(); fig_roc(P,C)
    fig_confusion(P,C,"type",TYPE_LBL); fig_confusion(P,C,"col_location",COL_LBL)
    fig_confusion(P,C,"mass_location",MASS_LBL); fig_severity(P,C)
    D=dt_data(P,C); fig_dt_combined(D); fig_dt_pertask(D); fig_dt_auc(D)
    for t in TASKS: fig_cellzoo(P,C,t)

    cls=[t for t in TASKS if t!="severity"]
    pm=np.mean([score(best_cell(P,t)[1])["metric"] for t in cls])
    cm=np.mean([score(C[best_cell(P,t)[0]])["metric"] for t in cls])
    R=_REPO/"results"/"REPORT_PRISTINE128.md"
    G="figures/pristine128"
    cz="\n".join(f"### {DESC[t]} (`{t}`)\n\n![{t} cell zoo]({G}/cellzoo_{t}.png)\n" for t in TASKS)
    txt=f"""# LANL 3SBB — Pristine-anchored damage diagnosis: detailed report (128 bins)

*Top-3 `(model,feature)` cells per task · {NROW:,} experimental measurements (zero-shot) · 30 cells*

## Contents
1. Overview & headline
2. The damage submodels (what "pristine-anchored" changes)
3. Cost of refusing damaged data
4. Detection diagnostics — ROC
5. Confusion matrices (type / location)
6. Severity regression
7. Damage-threshold (DT) sweep
8. Per-task catalogue (every cell)
9. Full 30-cell table · Reproduce

---

## 1 · Overview & headline

Every model is trained **only on synthetic FRFs** from the reduced-order 3SBB model **adjusted as well as possible from the pristine case alone**: the calibrated pristine baseline plus *first-principles, pristine-anchored* damage submodels (`ml_pipeline/pristine_physics.py`). **No damaged measurement informs the training data.** Models are evaluated **zero-shot** on the real experimental cases. For each task we trained the **top-3 cells** (ranked by experimental transfer in the damage-calibrated 128-bin study) and report the best. The *calibrated* column is the **same cell** from the main study, whose synthetic damage magnitudes **were** fitted to the damaged FRFs — so the gap is the share of performance bought by peeking at the damage.

![overview]({G}/overview_best.png)

{md_table_headline(P,C)}

\\*same cell, damage-calibrated study. ⚠ = collapsed / at chance.

**Mean best-cell experimental balanced accuracy over the 9 classification tasks: pristine `{pm:.3f}` vs calibrated `{cm:.3f}` (−{cm-pm:.3f}).**

## 2 · The damage submodels — what "pristine-anchored" changes

The only thing that differs from the calibrated generator is *how big* each damage is. The calibrated model reads magnitudes from tables whose anchors were fitted to the damaged FRFs; the pristine model derives them from geometry + mechanics, anchored only at the undamaged state.

![damage submodels]({G}/damage_submodels.png)

Bolt loosening is the biggest divergence: the pristine `1−p/100` preload law removes *more* stiffness at high severity than the fitted `0.39 @ 85%`. Crack/hole follow geometric section-loss. Mass is identical (a known kg). These curves are the entire difference between the two columns everywhere else in this report.

## 3 · Cost of refusing damaged data

![delta per task]({G}/delta_per_task.png)

Detection of bolt/hole/mass survives the honesty test; `is_hole` and `mass_location` lose essentially nothing — their physics generalises unaided. Fine typing (`type`) and crack detection (`is_crack`) fall most, because the crack stiffness effect is small and its fitted magnitude was doing real work.

## 4 · Detection diagnostics — ROC

![ROC]({G}/roc_detection.png)

## 5 · Confusion matrices (experimental, row-normalised)

![type confusion]({G}/confusion_type.png)

![col_location confusion]({G}/confusion_col_location.png)

![mass_location confusion]({G}/confusion_mass_location.png)

## 6 · Severity regression

![severity scatter]({G}/severity_scatter.png)

## 7 · Damage-threshold (DT) sweep — the central test

The thesis of synth-to-real SHM is that **detection improves with damage size**, because a larger perturbation outruns the synthetic→experimental domain gap. We stratify each detection task's positives by their damage-severity percentile, keep only the more-severe ones (all negatives retained), and recompute the metrics — for **both** the pristine-anchored cell and the calibrated cell. The question this study asks: *does the pristine-only physics preserve that size→detectability ordering, or did the fitted magnitudes own it?*

![DT combined]({G}/dt_combined.png)

*Balanced-accuracy vs the severity percentile kept (pristine solid, calibrated dashed). Both rise together — the ordering is a property of the physics, not the fitting.*

![DT per task]({G}/dt_pertask.png)

*Per task, with the positive count `n` kept at each threshold and the actual severity span annotated. `is_bolt` is the clean win (loosening spans 5–85%); `is_hole`/`is_mass` are flat because their experimental severity barely varies — there is no "more-severe" subset to climb into, not a model failure.*

![DT AUC and sensitivity]({G}/dt_auc.png)

*Ranking (ROC-AUC) and sensitivity (recall on positives) tell the same story across damage size.*

{md_dt_table(D)}

## 8 · Per-task catalogue (every cell)

{cz}

## 9 · Full 30-cell table (experimental zero-shot)

{md_full_table(P,C)}

## Reproduce

```bash
python ml_pipeline/build_pristine128_report.py
```

Reads the committed per-case predictions (`results_hires_zoo/pristine128/per_case_pristine128.tar.gz` for the pristine model, `results_hires/per_case_hires128.tar.gz` for the calibrated baseline) and rewrites this report + every figure under `results/figures/pristine128/`. Training notebook: `notebooks/hires_pristine128_top3_gpu.ipynb` (L4 GPU); raw predictions also on branch `colab-hires-pristine128`.
"""
    R.write_text(txt)
    print("wrote", R, "and", len(list(FIG.glob('*.png'))), "figures")

if __name__ == "__main__":
    main()
