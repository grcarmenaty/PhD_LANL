"""Experimental-data twins for the test-time figures.

For every classifier / regressor under ``<results>/models/``, predict on
the full ``experimental_features_balanced.h5`` set and write the same
four visual types that ``plots_advanced.py`` produces from the synth
test split:

  confusion/<task>_<model>_<feature>.png      (classification)
  perclass_f1/<task>.png                       (classification)
  scatter/severity_<model>_<feature>.png       (regression - severity only)
  roc/binary_{roc,pr}.png                      (binary only - overlay)

Titles end with ``(exp balanced)`` so they cannot be confused with the
synth twins under ``<results>/figures/``.

Reuses all helpers from ``plots_advanced.py``; the only behavioural
difference is that the "test split" is the entire 680-row experimental
set instead of a held-out fold of the synth file.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.metrics import (
    auc, confusion_matrix, f1_score, precision_recall_curve, roc_curve,
)
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml_pipeline.plots_advanced import (  # noqa: E402
    SCALERS,
    _LazyFeatures,
    _class_labels,
    _fit_scalers_per_task,
    _load_model_predict,
    _parse_tag,
    _safe_savefig,
)
from ml_pipeline.tasks import (  # noqa: E402
    TASK_N_CLASSES, build_targets,
)
from ml_pipeline.train import load_labels  # noqa: E402


def _exp_splits(features_path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """For the experimental file, use the ENTIRE pool of each task as
    the test set - no train/val/test split (exp is held out by
    construction)."""
    L = load_labels(features_path)
    tasks = build_targets(L["type_code"], L["storey"], L["end"], L["severity"])
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for tn, (mask, y_pool, kind) in tasks.items():
        ipool = np.where(mask)[0]
        out[tn] = {
            "idx_te": ipool,
            "y_te":   y_pool,   # build_targets returns y already masked
            "kind":   kind,
        }
    return out


def plot_confusions_exp(exp_path: Path, results_dir: Path,
                          out_dir: Path) -> None:
    L = load_labels(exp_path)
    tasks = build_targets(L["type_code"], L["storey"], L["end"], L["severity"])
    splits = _exp_splits(exp_path)
    feats = _LazyFeatures(exp_path)

    for art in sorted((results_dir / "models").iterdir()):
        tag = art.stem
        task_name, model_name, feature = _parse_tag(tag, tasks)
        if task_name is None:
            continue
        sp = splits[task_name]
        if sp["kind"] != "cls":
            continue
        try:
            X = feats[feature][sp["idx_te"]]
        except KeyError:
            continue
        n_out = TASK_N_CLASSES[task_name]
        try:
            scaler = SCALERS.get(task_name, {}).get(feature)
            y_pred, _ = _load_model_predict(art, X, "cls", n_out,
                                                  scaler=scaler)
        except Exception:
            continue
        y_te = sp["y_te"]
        labels = _class_labels(task_name)
        cm = confusion_matrix(y_te, y_pred, labels=list(range(len(labels))))
        denom = np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        cm_norm = cm.astype(float) / denom
        fig, ax = plt.subplots(figsize=(0.7 + 0.6 * len(labels),
                                            0.7 + 0.6 * len(labels)))
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                          color="white" if cm_norm[i, j] > 0.5 else "black",
                          fontsize=9)
        ax.set_title(f"{task_name} - {model_name} / {feature}  (exp balanced)")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        _safe_savefig(fig, out_dir / "confusion" / f"{tag}.png")


def plot_perclass_f1_exp(exp_path: Path, results_dir: Path,
                           out_dir: Path) -> None:
    L = load_labels(exp_path)
    tasks = build_targets(L["type_code"], L["storey"], L["end"], L["severity"])
    splits = _exp_splits(exp_path)
    feats = _LazyFeatures(exp_path)

    by_task: Dict[str, list] = {}
    for art in sorted((results_dir / "models").iterdir()):
        tag = art.stem
        task_name, model_name, feature = _parse_tag(tag, tasks)
        if task_name is None:
            continue
        sp = splits[task_name]
        if sp["kind"] != "cls":
            continue
        try:
            X = feats[feature][sp["idx_te"]]
        except KeyError:
            continue
        n_out = TASK_N_CLASSES[task_name]
        try:
            scaler = SCALERS.get(task_name, {}).get(feature)
            y_pred, _ = _load_model_predict(art, X, "cls", n_out,
                                                  scaler=scaler)
        except Exception:
            continue
        f1 = f1_score(sp["y_te"], y_pred, labels=list(range(n_out)),
                          average=None, zero_division=0)
        by_task.setdefault(task_name, []).append(
            (f"{model_name}/{feature}", f1))

    # Per-task summary JSON for downstream report commentary.
    summary: Dict[str, dict] = {}
    for task_name, rows in by_task.items():
        labels = _class_labels(task_name)
        rows_data = [
            {"cell": name, "f1_per_class": [float(x) for x in f1]}
            for (name, f1) in rows
        ]
        # Pick the best cell by mean F1 and report its worst class.
        best_cell = max(rows, key=lambda r: float(np.mean(r[1])))
        worst_idx = int(np.argmin(best_cell[1]))
        best_idx = int(np.argmax(best_cell[1]))
        summary[task_name] = {
            "labels": labels,
            "best_cell": best_cell[0],
            "best_cell_mean_f1": float(np.mean(best_cell[1])),
            "best_class": labels[best_idx],
            "best_class_f1": float(best_cell[1][best_idx]),
            "worst_class": labels[worst_idx],
            "worst_class_f1": float(best_cell[1][worst_idx]),
            "rows": rows_data,
        }
    import json
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "perclass_summary.json").write_text(json.dumps(summary, indent=2))

    for task_name, rows in by_task.items():
        labels = _class_labels(task_name)
        rows = sorted(rows, key=lambda r: -np.mean(r[1]))
        names = [r[0] for r in rows]
        mat = np.stack([r[1] for r in rows])
        fig, ax = plt.subplots(figsize=(0.7 + 0.6 * len(labels),
                                            0.4 + 0.3 * len(names)))
        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                          color="white" if mat[i, j] < 0.5 else "black",
                          fontsize=7)
        ax.set_title(f"per-class F1 - {task_name}  (exp balanced)")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        _safe_savefig(fig, out_dir / "perclass_f1" / f"{task_name}.png")


def plot_roc_pr_binary_exp(exp_path: Path, results_dir: Path,
                              out_dir: Path) -> None:
    L = load_labels(exp_path)
    tasks = build_targets(L["type_code"], L["storey"], L["end"], L["severity"])
    if "binary" not in tasks:
        return
    splits = _exp_splits(exp_path)
    sp = splits["binary"]
    if sp["kind"] != "cls":
        return
    idx_te = sp["idx_te"]; y_te = sp["y_te"]
    feats = _LazyFeatures(exp_path)

    fig_roc, ax_roc = plt.subplots(figsize=(7, 7))
    fig_pr,  ax_pr  = plt.subplots(figsize=(7, 7))
    any_plotted = False
    for art in sorted((results_dir / "models").iterdir()):
        tag = art.stem
        if not tag.startswith("binary_"):
            continue
        task_name, model_name, feat = _parse_tag(tag, tasks)
        if task_name != "binary":
            continue
        try:
            X = feats[feat][idx_te]
        except KeyError:
            continue
        try:
            scaler = SCALERS.get("binary", {}).get(feat)
            _, proba = _load_model_predict(art, X, "cls", 2, scaler=scaler)
        except Exception:
            continue
        if proba is None or proba.shape[1] < 2:
            continue
        scores = proba[:, 1]
        fpr, tpr, _ = roc_curve(y_te, scores)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, lw=1.0,
                       label=f"{model_name}/{feat}  AUC={roc_auc:.3f}")
        p, r, _ = precision_recall_curve(y_te, scores)
        ax_pr.plot(r, p, lw=1.0, label=f"{model_name}/{feat}")
        any_plotted = True

    if not any_plotted:
        plt.close(fig_roc); plt.close(fig_pr)
        return
    ax_roc.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax_roc.set_xlabel("false positive rate"); ax_roc.set_ylabel("true positive rate")
    ax_roc.set_title("ROC - binary (exp balanced)")
    ax_roc.legend(loc="lower right", fontsize=7)
    ax_roc.grid(linestyle=":", alpha=0.5)
    ax_pr.set_xlabel("recall"); ax_pr.set_ylabel("precision")
    ax_pr.set_title("Precision-Recall - binary (exp balanced)")
    ax_pr.legend(loc="lower left", fontsize=7)
    ax_pr.grid(linestyle=":", alpha=0.5)
    _safe_savefig(fig_roc, out_dir / "roc" / "binary_roc.png")
    _safe_savefig(fig_pr,  out_dir / "roc" / "binary_pr.png")


def plot_severity_scatter_exp(exp_path: Path, results_dir: Path,
                                out_dir: Path) -> None:
    L = load_labels(exp_path)
    tasks = build_targets(L["type_code"], L["storey"], L["end"], L["severity"])
    if "severity" not in tasks:
        return
    splits = _exp_splits(exp_path)
    sp = splits["severity"]
    if sp["kind"] != "reg":
        return
    idx_te = sp["idx_te"]; y_te = sp["y_te"]
    feats = _LazyFeatures(exp_path)

    for art in sorted((results_dir / "models").iterdir()):
        tag = art.stem
        if not tag.startswith("severity_"):
            continue
        task_name, model_name, feat = _parse_tag(tag, tasks)
        if task_name != "severity":
            continue
        try:
            X = feats[feat][idx_te]
        except KeyError:
            continue
        try:
            scaler = SCALERS.get("severity", {}).get(feat)
            y_pred, _ = _load_model_predict(art, X, "reg", 1, scaler=scaler)
        except Exception:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].scatter(y_te, y_pred, s=8, alpha=0.5)
        axes[0].plot([0, 1], [0, 1], "k--", lw=0.7)
        axes[0].set_xlabel("true severity (normalised)")
        axes[0].set_ylabel("predicted")
        axes[0].set_title(f"{model_name}/{feat}  (exp balanced)")
        axes[0].grid(linestyle=":", alpha=0.5)
        res = y_pred - y_te
        axes[1].hist(res, bins=40, alpha=0.7)
        axes[1].set_xlabel("residual (pred - true)")
        axes[1].set_ylabel("count")
        axes[1].set_title("residuals  (exp balanced)")
        axes[1].grid(linestyle=":", alpha=0.5)
        _safe_savefig(fig, out_dir / "scatter" / f"{tag}.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp", type=Path,
                      default=_REPO / "dataset"
                              / "experimental_features_balanced.h5",
                      help="Experimental features HDF5 file.")
    p.add_argument("--syn-features", type=Path,
                      default=_REPO / "dataset" / "features.h5",
                      help="Synth features file the models were trained "
                              "on (used to re-fit StandardScaler for torch "
                              "MLP cells).")
    p.add_argument("--results", type=Path,
                      default=_REPO / "results",
                      help="Results directory containing models/.")
    p.add_argument("--out", type=Path,
                      default=_REPO / "results" / "figures_exp",
                      help="Output directory for experimental figures.")
    args = p.parse_args()

    if not (args.results / "models").exists():
        raise SystemExit(f"missing {args.results / 'models'}")
    if not args.exp.exists():
        raise SystemExit(f"missing {args.exp}")

    if args.syn_features.exists():
        print(f"fitting scalers from {args.syn_features}...", flush=True)
        scalers = _fit_scalers_per_task(args.syn_features)
        SCALERS.clear()
        SCALERS.update(scalers)
    else:
        print(f"warn: {args.syn_features} not found - torch MLP cells "
                  "may produce incorrect predictions without the right "
                  "scaler.", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    print("confusion matrices...", flush=True)
    plot_confusions_exp(args.exp, args.results, args.out)
    print("per-class F1...", flush=True)
    plot_perclass_f1_exp(args.exp, args.results, args.out)
    print("binary ROC/PR...", flush=True)
    plot_roc_pr_binary_exp(args.exp, args.results, args.out)
    print("severity scatter...", flush=True)
    plot_severity_scatter_exp(args.exp, args.results, args.out)

    n = sum(1 for _ in args.out.rglob("*.png"))
    print(f"wrote {n} PNGs to {args.out}", flush=True)


if __name__ == "__main__":
    main()
