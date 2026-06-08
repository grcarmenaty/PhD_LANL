"""Damage-threshold (DT) sweep on the FULL-resolution (1601) results.

For each detection task, stratify the POSITIVES by their damage severity and
recompute experimental balanced-accuracy as we keep only the more-severe
positives (all negatives retained). Thresholds are per-task severity
PERCENTILES, so each task is swept on its own damage axis (bolt %, hole mm,
mass kg, crack depth). Tests the thesis: models trained on synth transfer to
real data *better at higher damage*.

Reads 1601 per-case JSONs (from --root, extracted from the colab-hires-* branches)
+ severity from dataset/experimental_features_hires.h5.
Writes results_hires/dt_1601.json and results/figures/hires/dt_1601_<task>.png + a combined figure.
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score

_REPO = Path(__file__).resolve().parent.parent
DET = ["binary", "is_bolt", "is_crack", "is_hole", "is_mass"]   # is_pristine positives have sev 0
PCTS = [0, 25, 50, 75, 90]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="/tmp/allres"); a = ap.parse_args()
    import h5py
    with h5py.File(_REPO/"dataset"/"experimental_features_hires.h5","r") as f:
        sev = dict(zip([str(s) for s in f["names"][:]], f["severity"][:]))

    # collect 1601 detection cells, dedup by (task,model,feature)
    cells = {}
    for p in glob.glob(f"{a.root}/**/per_case/*_hires1601.json", recursive=True):
        try: d=json.load(open(p)); m=d["meta"]; rows=d["rows"]
        except Exception: continue
        if m["task"] not in DET or m["kind"]!="cls" or not rows: continue
        k=(m["task"],m["model"],m["feature"])
        if k in cells: continue
        cells[k]=rows

    out={"percentiles":PCTS,"per_task":{}}
    for task in DET:
        tcells={k:v for k,v in cells.items() if k[0]==task}
        if not tcells: continue
        # per-task positive-severity percentile thresholds (from any cell — same exp set)
        any_rows=next(iter(tcells.values()))
        pos_sev=np.array([sev.get(r["case"],0.0) for r in any_rows if r["y_true"]==1])
        thr=[np.percentile(pos_sev,q) for q in PCTS] if len(pos_sev) else [0]*len(PCTS)
        rows_by_cell={}
        for (t,mo,f),rows in tcells.items():
            yt=np.array([r["y_true"] for r in rows]); yp=np.array([r["y_pred"] for r in rows])
            sv=np.array([sev.get(r["case"],0.0) for r in rows])
            curve=[]
            for th in thr:
                keep=(yt==0)|((yt==1)&(sv>=th))
                curve.append(float(balanced_accuracy_score(yt[keep],yp[keep])) if (yt[keep]==1).sum()>=5 else None)
            rows_by_cell[f"{mo}/{f}"]=curve
        # best cell at each DT level
        best=[]
        for i in range(len(PCTS)):
            vals=[(c,v[i]) for c,v in rows_by_cell.items() if v[i] is not None]
            best.append(max(vals,key=lambda x:x[1]) if vals else (None,None))
        out["per_task"][task]={"sev_thresholds":[float(x) for x in thr],
                                "best_per_pct":[{"pct":PCTS[i],"cell":best[i][0],"bal_acc":best[i][1]} for i in range(len(PCTS))],
                                "all_cells":rows_by_cell}
    (_REPO/"results_hires"/"dt_1601.json").write_text(json.dumps(out,indent=1))

    # ---- combined figure: best-cell balanced-acc vs severity percentile, per task ----
    OUT=_REPO/"results"/"figures"/"hires"; OUT.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(9.5,5.6))
    for task in DET:
        if task not in out["per_task"]: continue
        bp=out["per_task"][task]["best_per_pct"]
        xs=[b["pct"] for b in bp if b["bal_acc"] is not None]
        ys=[b["bal_acc"] for b in bp if b["bal_acc"] is not None]
        if xs: ax.plot(xs,ys,"o-",lw=2,label=task)
    ax.axhline(0.5,ls=":",color="black",alpha=.5,label="chance")
    ax.set_xlabel("keep positives with severity ≥ this percentile (more-damaged →)")
    ax.set_ylabel("best-cell experimental balanced-acc")
    ax.set_title("DT sweep @1601 — transfer vs damage severity (best cell per task)",fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=.3); ax.set_ylim(0.45,1.0)
    plt.tight_layout(); plt.savefig(OUT/"dt_1601_combined.png",dpi=130); plt.close(fig)

    print("=== DT sweep @1601 — best-cell balanced-acc vs severity percentile ===")
    print("task".ljust(13)+"".join(f"p{p}".rjust(9) for p in PCTS))
    for task in DET:
        if task not in out["per_task"]: continue
        bp=out["per_task"][task]["best_per_pct"]
        print(task.ljust(13)+"".join((f"{b['bal_acc']:.3f}" if b['bal_acc'] is not None else "  -- ").rjust(9) for b in bp))
        print("   best@p90:", bp[-1]["cell"])
    print("\nwrote results_hires/dt_1601.json + figures/hires/dt_1601_combined.png")


if __name__ == "__main__":
    main()
