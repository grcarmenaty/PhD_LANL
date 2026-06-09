"""Exploratory + diagnostic analysis for the hi-res (1601) sim-to-real study.

Two halves:
  (A) Exploratory data analysis of BOTH domains (synthetic train set + IQS
      experimental set): class balance, severity distributions, per-class FRF
      signatures, and a quantitative covariate-shift diagnosis (domain-classifier
      AUC + PCA of the spectral feature).
  (B) Per-task diagnostics for the best cell per task: confusion matrices,
      ROC curves with AUC (detection tasks), and the severity-regression
      scatter / residuals.

Writes figures to results/figures/hires/ and a machine-readable stats blob
results_hires/analysis.json consumed by build_hires_report.py.

Run: python ml_pipeline/hires_analysis.py --root /tmp/allres
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score,
                             balanced_accuracy_score, f1_score, r2_score,
                             mean_absolute_error)
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict

_REPO = Path(__file__).resolve().parent.parent
FIG = _REPO/"results"/"figures"/"hires"; FIG.mkdir(parents=True, exist_ok=True)
SYN = _REPO/"dataset"/"features_hires.h5"
EXP = _REPO/"dataset"/"experimental_features_hires.h5"
TYPE_NAMES = ["pristine","bolt","crack","hole","mass"]
DET = ["binary","is_pristine","is_bolt","is_crack","is_hole","is_mass"]
CFAM = {"pristine":"#7f7f7f","bolt":"#1f77b4","crack":"#d62728","hole":"#2ca02c","mass":"#9467bd"}


# ----------------------------------------------------------------------------- helpers
def _decimate(a, res, axis=1):
    """Bin-average `a` along `axis` to `res` bins (the same freq decimation the
    training engines use). No-op if res >= current length."""
    n = a.shape[axis]
    if res is None or res >= n:
        return a
    edges = np.linspace(0, n, res + 1).astype(int); starts = edges[:-1]
    s = np.add.reduceat(a, starts, axis=axis)
    sizes = np.maximum(np.diff(edges), 1)
    shape = [1] * a.ndim; shape[axis] = res
    return (s / sizes.reshape(shape)).astype(a.dtype)


def logmag_chanmean(h5, key="frf_mag", sub=None, res=1601):
    """log10 channel-averaged FRF magnitude -> (N, res). Optionally subsample N
    and decimate the frequency axis to `res` bins (for the 128-bin study)."""
    import h5py
    with h5py.File(h5, "r") as f:
        N = f[key].shape[0]
        idx = np.arange(N) if sub is None or sub >= N else np.sort(
            np.random.RandomState(0).choice(N, sub, replace=False))
        mag = f[key][:][idx] if sub is None else f[key][np.sort(idx)]
        tc = f["type_code"][:][idx]
        sev = f["severity"][:][idx]
        fr = f["freqs"][:]
    lm = np.log10(np.abs(mag) + 1e-9).mean(axis=2)   # (n,1601)
    lm = _decimate(lm, res, axis=1); fr = _decimate(fr[None, :], res, axis=1)[0]
    return lm, tc.astype(int), sev.astype(float), fr


def load_meta_labels(h5):
    import h5py
    with h5py.File(h5, "r") as f:
        return (f["type_code"][:].astype(int), f["severity"][:].astype(float),
                f["storey"][:].astype(int), f["end"][:].astype(int))


def best_cells(summary, res=1601):
    """task -> (model, feature, rec) chosen by exp_bal_acc (cls) / exp_r2 (reg)."""
    by = {}
    for k, v in summary.items():
        if v.get("res") != res:
            continue
        t = v["task"]
        s = v.get("exp_bal_acc") if v["kind"] == "cls" else v.get("exp_r2")
        if s is None:
            continue
        if t not in by or s > by[t][0]:
            by[t] = (s, v)
    return {t: by[t][1] for t in by}


def percase(root, task, model, feat, res=1601):
    p = f"{root}/**/per_case/{task}_{model}_{feat}_hires{res}.json"
    hits = glob.glob(p, recursive=True)
    if not hits:
        return None
    d = json.load(open(hits[0]))
    rows = d["rows"]
    yt = np.array([r["y_true"] for r in rows])
    yp = np.array([r["y_pred"] for r in rows])
    pr = np.array([r["proba"] for r in rows]) if "proba" in rows[0] and rows[0]["proba"] is not None else None
    cases = [r["case"] for r in rows]
    return yt, yp, pr, cases, d["meta"]


# ----------------------------------------------------------------------------- (A) EDA
def eda(stats, res=1601):
    from ml_pipeline import figdata
    B = figdata.load_eda_arrays(res)
    if B is not None:                               # committed bundle (no HDF5 needed)
        syn_tc = B["syn_tc"].astype(int); syn_sev = B["syn_sev"].astype(float)
        exp_tc = B["exp_tc"].astype(int); exp_sev = B["exp_sev"].astype(float)
    else:                                           # fall back to the hi-res HDF5
        syn_tc, syn_sev, _, _ = load_meta_labels(SYN)
        exp_tc, exp_sev, _, _ = load_meta_labels(EXP)
    stats["counts"] = {"synth": {TYPE_NAMES[i]: int((syn_tc == i).sum()) for i in range(5)},
                       "exp":   {TYPE_NAMES[i]: int((exp_tc == i).sum()) for i in range(5)}}

    # --- Fig 1: class balance + severity distributions -----------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    x = np.arange(5); w = 0.38
    sc = [(syn_tc == i).mean() for i in range(5)]
    ec = [(exp_tc == i).mean() for i in range(5)]
    ax[0].bar(x - w/2, sc, w, label=f"synth (N={len(syn_tc)})", color="#ff7f0e", edgecolor="k", lw=.4)
    ax[0].bar(x + w/2, ec, w, label=f"exp (N={len(exp_tc)})", color="#1f77b4", edgecolor="k", lw=.4)
    ax[0].set_xticks(x); ax[0].set_xticklabels(TYPE_NAMES, rotation=20); ax[0].set_ylabel("class proportion")
    ax[0].set_title("(a) Damage-type balance\nsynth balanced · exp bolt-heavy", fontweight="bold", fontsize=10)
    ax[0].legend(fontsize=8); ax[0].grid(axis="y", alpha=.3)

    # severity per type (exp) — boxplot of damaged classes only
    data = [exp_sev[exp_tc == i] for i in range(1, 5)]
    bp = ax[1].boxplot(data, labels=TYPE_NAMES[1:], patch_artist=True, showfliers=False)
    for patch, i in zip(bp["boxes"], range(1, 5)):
        patch.set_facecolor(CFAM[TYPE_NAMES[i]]); patch.set_alpha(.7)
    ax[1].set_ylabel("severity (native units)")
    ax[1].set_title("(b) Experimental severity by type\n(bolt 0–85%, hole narrow)", fontweight="bold", fontsize=10)
    ax[1].grid(axis="y", alpha=.3)

    # severity histogram overall, synth vs exp (normalised 0..1 within damaged)
    def norm01(s, tc):
        out = np.zeros_like(s, float)
        for i in range(1, 5):
            m = tc == i; r = s[m]
            if r.max() > r.min():
                out[m] = (r - r.min()) / (r.max() - r.min())
        return out[tc > 0]
    ax[2].hist(norm01(syn_sev, syn_tc), bins=25, alpha=.55, density=True, label="synth", color="#ff7f0e")
    ax[2].hist(norm01(exp_sev, exp_tc), bins=25, alpha=.55, density=True, label="exp", color="#1f77b4")
    ax[2].set_xlabel("per-type-normalised severity"); ax[2].set_ylabel("density")
    ax[2].set_title("(c) Severity coverage\n(synth uniform · exp clustered)", fontweight="bold", fontsize=10)
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(FIG/"eda_class_severity.png", dpi=130); plt.close(fig)

    # --- per-class FRF signatures + domain gap -------------------------------
    if B is not None:
        syn_lm = B["syn_lm"]; syn_tc2 = B["syn_lm_tc"].astype(int); fr = B["freqs"]
        exp_lm = B["exp_lm"]; exp_tc2 = B["exp_tc"].astype(int)
    else:
        syn_lm, syn_tc2, _, fr = logmag_chanmean(SYN, sub=4000, res=res)
        exp_lm, exp_tc2, _, _ = logmag_chanmean(EXP, res=res)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    for i in range(5):
        ax[0].plot(fr, syn_lm[syn_tc2 == i].mean(0), color=CFAM[TYPE_NAMES[i]], lw=1.4, label=TYPE_NAMES[i])
    ax[0].set_title("(a) Synthetic mean log|FRF| by class", fontweight="bold", fontsize=10)
    ax[0].set_xlabel("frequency (Hz)"); ax[0].set_ylabel("log10 |H(f)| (channel mean)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3); ax[0].set_xlim(0, 100)
    # domain gap: pristine synth vs exp + damaged synth vs exp
    ax[1].plot(fr, syn_lm[syn_tc2 == 0].mean(0), color="#ff7f0e", lw=1.6, label="synth pristine")
    ax[1].plot(fr, exp_lm[exp_tc2 == 0].mean(0), color="#ff7f0e", lw=1.6, ls="--", label="exp pristine")
    ax[1].plot(fr, syn_lm[syn_tc2 == 1].mean(0), color="#1f77b4", lw=1.6, label="synth bolt")
    ax[1].plot(fr, exp_lm[exp_tc2 == 1].mean(0), color="#1f77b4", lw=1.6, ls="--", label="exp bolt")
    ax[1].set_title("(b) Domain gap: synth (solid) vs exp (dashed)", fontweight="bold", fontsize=10)
    ax[1].set_xlabel("frequency (Hz)"); ax[1].grid(alpha=.3); ax[1].set_xlim(0, 100); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIG/"eda_frf_signatures.png", dpi=130); plt.close(fig)

    # --- covariate shift: PCA + domain classifier ----------------------------
    X = np.vstack([syn_lm, exp_lm]); dom = np.r_[np.zeros(len(syn_lm)), np.ones(len(exp_lm))]
    Xs = StandardScaler().fit_transform(X)
    pc = PCA(n_components=2, random_state=0).fit(Xs); Z = pc.transform(Xs)
    # domain classifier AUC (5-fold) = covariate-shift strength (0.5 = identical)
    proba = cross_val_predict(LogisticRegression(max_iter=2000, C=1.0), Xs, dom,
                              cv=5, method="predict_proba")[:, 1]
    dom_auc = float(roc_auc_score(dom, proba))
    stats["domain_classifier_auc"] = dom_auc
    stats["pca_explained_var"] = [float(v) for v in pc.explained_variance_ratio_]

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
    ax[0].scatter(Z[dom == 0, 0], Z[dom == 0, 1], s=6, alpha=.35, color="#ff7f0e", label="synth")
    ax[0].scatter(Z[dom == 1, 0], Z[dom == 1, 1], s=6, alpha=.35, color="#1f77b4", label="exp")
    ax[0].set_title(f"(a) PCA of log|FRF| by DOMAIN\ndomain-classifier AUC={dom_auc:.3f} (1.0=fully separable)",
                    fontweight="bold", fontsize=10)
    ax[0].set_xlabel(f"PC1 ({100*pc.explained_variance_ratio_[0]:.0f}%)")
    ax[0].set_ylabel(f"PC2 ({100*pc.explained_variance_ratio_[1]:.0f}%)"); ax[0].legend(); ax[0].grid(alpha=.3)
    tcall = np.r_[syn_tc2, exp_tc2]
    for i in range(5):
        m = tcall == i
        ax[1].scatter(Z[m, 0], Z[m, 1], s=6, alpha=.35, color=CFAM[TYPE_NAMES[i]], label=TYPE_NAMES[i])
    ax[1].set_title("(b) Same projection by CLASS\n(classes overlap → spectral overlap)", fontweight="bold", fontsize=10)
    ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    # clip to the bulk (robust to a few spectral outliers)
    xb = np.percentile(Z[:, 0], [0.5, 99.5]); yb = np.percentile(Z[:, 1], [0.5, 99.5])
    pad = lambda lo, hi: (lo - 0.08*(hi-lo), hi + 0.08*(hi-lo))
    for a_ in ax:
        a_.set_xlim(*pad(*xb)); a_.set_ylim(*pad(*yb))
    plt.tight_layout(); plt.savefig(FIG/"eda_domain_shift.png", dpi=130); plt.close(fig)
    print(f"EDA: domain-classifier AUC={dom_auc:.3f}; PCA var={pc.explained_variance_ratio_[:2]}")


# ----------------------------------------------------------------------------- (B) diagnostics
def diagnostics(root, summary, stats, res=1601):
    bc = best_cells(summary, res)
    stats["best_cells"] = {}

    # ---- confusion-matrix grid for classification tasks ----
    cls_tasks = ["binary", "type", "mass_location", "col_location",
                 "is_pristine", "is_bolt", "is_crack", "is_hole", "is_mass"]
    cls_tasks = [t for t in cls_tasks if t in bc]
    n = len(cls_tasks); cols = 3; rows = int(np.ceil(n/cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4.3*cols, 3.8*rows)); axs = np.array(axs).ravel()
    for ax, t in zip(axs, cls_tasks):
        rec = bc[t]; pc = percase(root, t, rec["model"], rec["feature"], res)
        if pc is None:
            ax.axis("off"); continue
        yt, yp, pr, _, _ = pc
        nlab = rec["n_out"]
        cm = confusion_matrix(yt, yp, labels=list(range(nlab)))
        cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(nlab):
            for j in range(nlab):
                ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if cmn[i, j] > .5 else "black")
        labs = TYPE_NAMES if t == "type" else (["neg","pos"] if nlab == 2 else list(range(nlab)))
        ax.set_xticks(range(nlab)); ax.set_yticks(range(nlab))
        ax.set_xticklabels(labs, fontsize=7, rotation=30); ax.set_yticklabels(labs, fontsize=7)
        ax.set_xlabel("predicted", fontsize=8); ax.set_ylabel("true", fontsize=8)
        ba = balanced_accuracy_score(yt, yp)
        ax.set_title(f"{t}\n{rec['model']}/{rec['feature']} · bal-acc {ba:.2f}", fontweight="bold", fontsize=9)
        stats["best_cells"].setdefault(t, {})["cm_rownorm"] = [[round(float(x), 3) for x in row] for row in cmn]
    for ax in axs[len(cls_tasks):]:
        ax.axis("off")
    fig.suptitle("Experimental confusion matrices (row-normalised) — best cell per task",
                 fontweight="bold", fontsize=12, y=1.002)
    plt.tight_layout(); plt.savefig(FIG/"diag_confusion.png", dpi=130); plt.close(fig)

    # ---- ROC curves for detection tasks (proba available) ----
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    for t in DET:
        if t not in bc:
            continue
        rec = bc[t]; pc = percase(root, t, rec["model"], rec["feature"], res)
        if pc is None:
            continue
        yt, yp, pr, _, _ = pc
        if pr is None or pr.ndim != 2 or pr.shape[1] < 2 or len(set(yt.tolist())) < 2:
            continue
        score = pr[:, 1]
        fprr, tprr, _ = roc_curve(yt, score); auc = roc_auc_score(yt, score)
        ax.plot(fprr, tprr, lw=2, label=f"{t} ({rec['model']}/{rec['feature']}) AUC={auc:.3f}")
        stats["best_cells"].setdefault(t, {})["auc"] = float(auc)
    ax.plot([0, 1], [0, 1], ls=":", color="black", alpha=.6, label="chance")
    ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate")
    ax.set_title("Experimental ROC — detection tasks (best cell each)\nAUC is threshold-free, immune to the 82.5% prior",
                 fontweight="bold", fontsize=10)
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(FIG/"diag_roc.png", dpi=130); plt.close(fig)

    # ---- severity regression scatter + residuals ----
    if "severity" in bc:
        rec = bc["severity"]; pc = percase(root, "severity", rec["model"], rec["feature"])
        if pc is not None:
            yt, yp, _, _, _ = pc; yt = yt.astype(float); yp = yp.astype(float)
            r2 = r2_score(yt, yp); r = float(np.corrcoef(yt, yp)[0, 1]); mae = mean_absolute_error(yt, yp)
            stats["best_cells"]["severity"] = {"r2": float(r2), "pearson_r": r, "mae": float(mae),
                                               "cell": f"{rec['model']}/{rec['feature']}"}
            fig, ax = plt.subplots(1, 2, figsize=(12.5, 5))
            ax[0].scatter(yt, yp, s=10, alpha=.3, color="#1f77b4")
            lim = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
            ax[0].plot(lim, lim, "k--", lw=1, label="ideal")
            # least-squares trend
            b1, b0 = np.polyfit(yt, yp, 1)
            ax[0].plot(np.array(lim), b1*np.array(lim)+b0, "r-", lw=1.5, label=f"fit (slope {b1:.2f})")
            ax[0].set_xlabel("true severity (normalised)"); ax[0].set_ylabel("predicted")
            ax[0].set_title(f"(a) Severity: pred vs true\nR²={r2:+.3f} · r={r:.3f} · MAE={mae:.3f}", fontweight="bold", fontsize=10)
            ax[0].legend(); ax[0].grid(alpha=.3)
            res = yp - yt
            ax[1].scatter(yt, res, s=10, alpha=.3, color="#d62728"); ax[1].axhline(0, color="k", ls="--", lw=1)
            ax[1].set_xlabel("true severity"); ax[1].set_ylabel("residual (pred − true)")
            ax[1].set_title("(b) Residuals — systematic regression to the mean", fontweight="bold", fontsize=10)
            ax[1].grid(alpha=.3)
            plt.tight_layout(); plt.savefig(FIG/"diag_severity.png", dpi=130); plt.close(fig)

    # record bal-acc/macroF1 for best cells
    for t, rec in bc.items():
        e = stats["best_cells"].setdefault(t, {})
        e["cell"] = f"{rec['model']}/{rec['feature']}"
        if rec["kind"] == "cls":
            e["bal_acc"] = rec.get("exp_bal_acc"); e["macro_f1"] = rec.get("exp_macro_f1")
            e["chance"] = rec.get("chance"); e["in_domain"] = rec.get("synth")
    print("diagnostics: AUCs", {k: round(v.get("auc", 0), 3) for k, v in stats["best_cells"].items() if "auc" in v})


def main():
    global FIG
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=None)
    ap.add_argument("--res", type=int, default=1601); a = ap.parse_args()
    res = a.res
    from ml_pipeline import figdata
    FIG = figdata.figdir(res)
    root = a.root or figdata.percase_root()
    summary = json.loads((_REPO/"results_hires"/"zoo_summary.json").read_text())
    stats = {}
    eda(stats, res)
    diagnostics(root, summary, stats, res)
    (_REPO/"results_hires"/f"analysis{figdata.sfx(res)}.json").write_text(json.dumps(stats, indent=1))
    print(f"wrote results_hires/analysis{figdata.sfx(res)}.json + EDA/diagnostic figures (res={res})")


if __name__ == "__main__":
    main()
