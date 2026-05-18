"""Diagnostic plots for the FINAL pipeline state (P0+P1.1+P1.4).

Reads per-case predictions from results/per_case_final/<task>.json
(produced by ml_pipeline.eval_final) and emits:

  results/figures/final/confusion_<task>.png        per classification task
  results/figures/final/roc_binary.png              binary ROC + AUC
  results/figures/final/roc_type_ovr.png            type one-vs-rest ROC curves
  results/figures/final/severity_scatter.png        severity scatter w/ residuals
  results/figures/final/per_class_f1_<task>.png     per-class F1 bar
  results/figures/final/headline_metrics_bar.png    final-only metrics bar

Plus zero-shot baselines (no fine-tune, raw artefact predictions) when
useful, for the binary task that the P1.4 sweep didn't cover.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, mean_absolute_error, r2_score, roc_auc_score, roc_curve,
    auc,
)

_REPO = Path(__file__).resolve().parent.parent

# Class labels per task
TYPE_NAMES = ["Pristine", "Bolt", "Crack", "Hole", "Mass"]
COL_LOC_NAMES = ["S1BD", "S1AD", "S2BD", "S2AD", "S3BD", "S3AD"]
MASS_LOC_NAMES = ["Base", "F1", "F2", "F3"]

OUT_DIR = _REPO / "results" / "figures" / "final"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _load(task):
    p = _REPO / "results" / "per_case_final" / f"{task}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


# ── confusion matrices ──────────────────────────────────────────────────────
def plot_confusion(task, class_names):
    d = _load(task)
    if d is None:
        print(f"  no per-case file for {task}; skipping confusion")
        return
    meta = d["meta"]
    if meta["kind"] != "cls":
        return
    rows = d["rows"]
    y = np.array([r["y_true"] for r in rows])
    yhat = np.array([r["y_pred"] for r in rows])
    cm = confusion_matrix(y, yhat, labels=list(range(meta["n_classes"])))
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    acc = accuracy_score(y, yhat)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (M, title, fmt) in zip(axes, [
        (cm,      f"counts (n={len(rows)})", "{}"),
        (cm_norm, "row-normalised (recall)", "{:.2f}")
    ]):
        im = ax.imshow(M, cmap="Blues", vmin=0,
                          vmax=max(1, np.max(cm)) if "count" in title else 1)
        ax.set_xticks(np.arange(meta["n_classes"]))
        ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(np.arange(meta["n_classes"]))
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(title)
        for i in range(meta["n_classes"]):
            for j in range(meta["n_classes"]):
                txt = fmt.format(M[i, j]) if "count" in title else fmt.format(M[i, j])
                colr = "white" if M[i, j] > (0.5 * np.max(M) if M is cm else 0.5) else "black"
                ax.text(j, i, txt, ha="center", va="center",
                          fontsize=8, color=colr)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    fig.suptitle(f"{task} confusion — {meta['model']}/{meta['feature']} "
                  f"(P1.4 'all' k=50%, accuracy={acc:.3f})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT_DIR / f"confusion_{task}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")
    return cm, cm_norm


# ── per-class F1 ────────────────────────────────────────────────────────────
def plot_per_class_f1(task, class_names):
    d = _load(task)
    if d is None or d["meta"]["kind"] != "cls":
        return
    meta = d["meta"]
    rows = d["rows"]
    y = np.array([r["y_true"] for r in rows])
    yhat = np.array([r["y_pred"] for r in rows])
    f1 = f1_score(y, yhat, labels=list(range(meta["n_classes"])),
                     average=None, zero_division=0)
    support = np.bincount(y, minlength=meta["n_classes"])

    fig, ax = plt.subplots(figsize=(max(5, meta["n_classes"] * 1.4), 4))
    x = np.arange(meta["n_classes"])
    bars = ax.bar(x, f1, color="#4a90d9", edgecolor="black", linewidth=0.5)
    for xi, f1i, n in zip(x, f1, support):
        ax.text(xi, f1i + 0.02, f"{f1i:.2f}\n(n={n})",
                  ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylabel("per-class F1")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"{task} per-class F1 — {meta['model']}/{meta['feature']} "
                  f"(P1.4 'all' k=50%)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    out = OUT_DIR / f"per_class_f1_{task}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


# ── ROC for binary (pristine vs damage) ─────────────────────────────────────
def plot_binary_roc():
    """Binary wasn't in the transfer-learning sweep; pull the zero-shot
    per-case predictions from results/baseline / results/p1_1 instead.
    Compare baseline vs P1.1 for the binary task."""
    fig, ax = plt.subplots(figsize=(6, 5.5))

    for label, snap_dir, color in [
        ("baseline (zero-shot)", "baseline", "#888888"),
        ("P1.1 (zero-shot, retrained)", "p1_1",  "#4a90d9"),
    ]:
        pc_path = _REPO / "results" / snap_dir / "experimental_full_per_case.json"
        if not pc_path.exists():
            continue
        rows = json.loads(pc_path.read_text())
        bin_rows = [r for r in rows if r["task"] == "binary"]
        if not bin_rows:
            continue
        # Pick the best cell from the eval JSON
        eval_rows = json.loads(
            (_REPO / "results" / snap_dir / "experimental_full_evaluation.json").read_text())
        best = max((r for r in eval_rows if r["task"] == "binary"),
                       key=lambda r: r["value"])
        cell_rows = [r for r in bin_rows
                          if r["model"] == best["model"]
                          and r["feature"] == best["feature"]]
        if not cell_rows:
            continue
        y = np.array([r["y_true"] for r in cell_rows])
        yhat = np.array([r["y_pred"] for r in cell_rows])
        try:
            # treat yhat as score (0/1 from argmax; AUC on argmax labels)
            fpr, tpr, _ = roc_curve(y, yhat)
            roc_auc = auc(fpr, tpr)
        except Exception:
            continue
        ax.plot(fpr, tpr, color=color, linewidth=2,
                  label=f"{label}: {best['model']}/{best['feature']} "
                          f"(AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("binary ROC (zero-shot, full 2638-case experimental)")
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    out = OUT_DIR / "roc_binary.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


# ── ROC for type (one-vs-rest) ──────────────────────────────────────────────
def plot_type_roc():
    d = _load("type")
    if d is None:
        return
    meta = d["meta"]
    rows = d["rows"]
    y = np.array([r["y_true"] for r in rows])
    proba = np.array([r["proba"] for r in rows])

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    colors = ["#888888", "#d9534f", "#e6a23c", "#7eb854", "#4a90d9"]
    aucs = []
    for k, (name, color) in enumerate(zip(TYPE_NAMES, colors)):
        y_bin = (y == k).astype(int)
        if y_bin.sum() == 0:
            continue
        try:
            fpr, tpr, _ = roc_curve(y_bin, proba[:, k])
            roc_auc = auc(fpr, tpr)
        except Exception:
            continue
        ax.plot(fpr, tpr, color=color, linewidth=2,
                  label=f"{name} (AUC={roc_auc:.3f}, n={int(y_bin.sum())})")
        aucs.append(roc_auc)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    mean_auc = float(np.mean(aucs)) if aucs else float("nan")
    ax.set_title(f"type ROC (one-vs-rest) — {meta['model']}/{meta['feature']}\n"
                  f"P1.4 'all' k=50%, macro-AUC={mean_auc:.3f}")
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = OUT_DIR / "roc_type_ovr.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


# ── Severity scatter + residuals ────────────────────────────────────────────
def plot_severity_scatter_final():
    d = _load("severity")
    if d is None:
        return
    meta = d["meta"]
    rows = d["rows"]
    y = np.array([r["y_true"] for r in rows])
    yhat = np.array([r["y_pred"] for r in rows])
    r2 = r2_score(y, yhat); mae = mean_absolute_error(y, yhat)

    # Per-damage-type breakdown (true-severity bin into 4 quartiles to
    # roughly correspond to Bolt/Crack/Hole/Mass since each type has its
    # own [0,1] severity normalisation).
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].scatter(y, yhat, s=14, alpha=0.55, color="#d9534f",
                       edgecolor="black", linewidth=0.3)
    axes[0].plot([0, 1], [0, 1], "--", color="black", linewidth=1)
    axes[0].set_xlim(-0.05, 1.05); axes[0].set_ylim(-0.05, 1.05)
    axes[0].set_xlabel("true severity (normalised per damage type)")
    axes[0].set_ylabel("predicted severity")
    axes[0].set_title(f"P1.4 best severity: {meta['model']}/{meta['feature']}\n"
                          f"R²={r2:.3f}, MAE={mae:.3f}, n={len(rows)}")
    axes[0].grid(linestyle=":", alpha=0.4)

    # Residuals
    resid = yhat - y
    axes[1].hist(resid, bins=30, color="#d9534f", edgecolor="black",
                    alpha=0.8)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].axvline(np.median(resid), color="green", linestyle="-",
                          linewidth=1.5, label=f"median={np.median(resid):+.2f}")
    axes[1].set_xlabel("residual (pred − true)")
    axes[1].set_ylabel("count")
    axes[1].set_title("Severity residual histogram")
    axes[1].legend(); axes[1].grid(linestyle=":", alpha=0.4)

    fig.tight_layout()
    out = OUT_DIR / "severity_scatter.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


# ── Final-state metrics bar ─────────────────────────────────────────────────
def plot_final_metrics_bar():
    summary = json.loads(
        (_REPO / "results" / "per_case_final" / "summary.json").read_text())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    tasks_present = [m["task"] for m in summary]
    vals = [m["metric_value"] for m in summary]
    cells = [f"{m['model']}/{m['feature']}" for m in summary]
    metric_lbl = [m["metric_name"] for m in summary]
    x = np.arange(len(tasks_present))
    bars = ax.bar(x, vals, color="#d9534f", edgecolor="black", linewidth=0.4)
    for xi, vi, ci, ml in zip(x, vals, cells, metric_lbl):
        ax.text(xi, vi + 0.02, f"{vi:.3f}", ha="center", va="bottom",
                  fontsize=10)
        ax.text(xi, -0.06, f"{ml}\n{ci}", ha="center", va="top",
                  fontsize=7, color="gray")
    ax.set_xticks(x); ax.set_xticklabels(tasks_present, fontsize=10)
    ax.set_ylim(-0.15, 1.15)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("metric on held-out 50 % balanced experimental")
    ax.set_title("Final pipeline (P0+P1.1+P1.4 'all' k=50%) — per-task best cell")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    out = OUT_DIR / "headline_metrics_bar.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    plot_confusion("type", TYPE_NAMES)
    plot_confusion("col_location", COL_LOC_NAMES)
    plot_confusion("mass_location", MASS_LOC_NAMES)
    plot_per_class_f1("type", TYPE_NAMES)
    plot_per_class_f1("col_location", COL_LOC_NAMES)
    plot_per_class_f1("mass_location", MASS_LOC_NAMES)
    plot_binary_roc()
    plot_type_roc()
    plot_severity_scatter_final()
    plot_final_metrics_bar()
    print(f"\nfigures in {OUT_DIR}")


if __name__ == "__main__":
    main()
