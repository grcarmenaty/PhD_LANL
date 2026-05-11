"""Diagnostic / paper-quality plots for the ML pipeline.

Categories produced (under ``results/figures/``):

  signals/          example time series, FRF magnitudes and CFDAC matrices
                    per damage class (1 sample / class)
  dataset/          class-count and severity-distribution histograms
  confusion/        confusion matrix for every classifier (per task,
                    model, feature)
  perclass_f1/      per-class F1 bar chart (every classifier)
  roc/              ROC + PR curves for the binary task
  scatter/          severity prediction scatter + residual histogram
  feat_importance/  RF / XGB feature importances per task per feature
  embedding/        PCA + t-SNE 2-D embeddings of every flat feature
  hpo/              response-surface heatmaps from ``results/hpo/``

The script is idempotent — call it after every training / HPO run.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_recall_curve, roc_curve, auc,
)
from sklearn.preprocessing import StandardScaler

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml_pipeline.case_design import TYPE_NAMES   # noqa: E402
from ml_pipeline.features import INDICATOR_NAMES  # noqa: E402
from ml_pipeline.tasks import (   # noqa: E402
    build_targets, TASK_DESCRIPTION, TASK_N_CLASSES,
)
from ml_pipeline.train import (   # noqa: E402
    FEATURES_FLAT, FEATURES_SEQ, FEATURES_MAT,
    load_labels, load_feature, make_split, SEED,
)
from ml_pipeline.models import MLP, Conv1DStack, SmallTransformer, Conv2DStack  # noqa: E402

# Per-(task, flat-feature) StandardScaler table.  Populated in main().
SCALERS: Dict[str, Dict[str, "StandardScaler"]] = {}


# ── helpers ─────────────────────────────────────────────────────────────────
def _safe_savefig(fig, path: Path, dpi: int = 110) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _class_labels(task: str) -> List[str]:
    if task == "binary":
        return ["Pristine", "Damage"]
    if task == "type":
        return [TYPE_NAMES[i] for i in range(5)]
    if task == "col_location":
        return [f"S{(i // 2) + 1}{['BD', 'AD'][i % 2]}" for i in range(6)]
    if task == "mass_location":
        return ["Base", "F1", "F2", "F3"]
    if task == "severity":
        return ["severity"]
    return [str(i) for i in range(TASK_N_CLASSES.get(task) or 4)]


# ── dataset summary plots ───────────────────────────────────────────────────
def plot_dataset_summary(features_path: Path, out_dir: Path) -> None:
    L = load_labels(features_path)
    tc = L["type_code"]
    sev = L["severity"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    counts = np.bincount(tc.astype(int))
    axes[0].bar([TYPE_NAMES[i] for i in range(len(counts))], counts,
                  color=["#888", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    axes[0].set_title("Class counts")
    axes[0].set_ylabel("samples")
    axes[0].grid(axis="y", linestyle=":", alpha=0.5)

    for idx, tc_val in enumerate([1, 2, 3, 4]):
        mask = tc == tc_val
        axes[1].hist(sev[mask], bins=40, alpha=0.5,
                       label=TYPE_NAMES[tc_val])
    axes[1].set_title("Severity distributions")
    axes[1].set_xlabel("severity (units depend on type)")
    axes[1].set_ylabel("count")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(axis="y", linestyle=":", alpha=0.5)
    fig.suptitle("Synthetic dataset overview")
    _safe_savefig(fig, out_dir / "dataset" / "class_severity.png")


# ── example signal / FRF / CFDAC per class ──────────────────────────────────
def plot_examples_per_class(features_path: Path, out_dir: Path) -> None:
    L = load_labels(features_path)
    tc = L["type_code"]
    rng = np.random.default_rng(0)
    picks: Dict[int, int] = {}
    for k in range(5):
        idx = np.where(tc == k)[0]
        if len(idx):
            picks[k] = int(rng.choice(idx))

    with h5py.File(features_path, "r") as f:
        freqs = f["freqs"][:]
        # time axis: assume features were extracted at FS = 256 Hz, N_T = 1024.
        time_axis = np.arange(1024) / 256.0
        fig_ts, axts = plt.subplots(5, 1, figsize=(9, 11), sharex=True)
        fig_fr, axfr = plt.subplots(5, 1, figsize=(9, 11), sharex=True)
        fig_cf, axcf = plt.subplots(1, 5, figsize=(20, 4))
        for ax_row, (k, i) in enumerate(picks.items()):
            ts = f["timeseries"][i]
            for ch in range(ts.shape[1]):
                axts[ax_row].plot(time_axis, ts[:, ch], lw=0.5,
                                     alpha=0.7)
            axts[ax_row].set_title(f"{TYPE_NAMES[k]} — sample id {i}")
            axts[ax_row].grid(linestyle=":", alpha=0.5)

            mag = f["frf_mag"][i]
            for ch in range(mag.shape[1]):
                axfr[ax_row].semilogy(freqs, mag[:, ch], lw=0.7, alpha=0.7)
            axfr[ax_row].set_title(f"|H(f)| — {TYPE_NAMES[k]}  (id {i})")
            axfr[ax_row].grid(linestyle=":", alpha=0.5)

            if "cfdac_real" in f:
                cre = f["cfdac_real"][i]; cim = f["cfdac_imag"][i]
                mag = np.sqrt(cre ** 2 + cim ** 2)
                im = axcf[ax_row].imshow(mag, origin="lower",
                                              cmap="viridis", vmin=0, vmax=1)
                axcf[ax_row].set_title(f"|CFDAC| — {TYPE_NAMES[k]}")
                axcf[ax_row].set_xticks([]); axcf[ax_row].set_yticks([])
        axts[-1].set_xlabel("time [s]")
        axfr[-1].set_xlabel("frequency [Hz]")
        fig_ts.suptitle("Acceleration time series (1 sample / class, all 9 sensors)")
        fig_fr.suptitle("FRF magnitude (1 sample / class, all 9 sensors)")
        fig_cf.suptitle("|CFDAC| vs the pristine reference (1 sample / class)")
        _safe_savefig(fig_ts, out_dir / "signals" / "timeseries.png")
        _safe_savefig(fig_fr, out_dir / "signals" / "frf_mag.png")
        _safe_savefig(fig_cf, out_dir / "signals" / "cfdac.png")


# ── confusion matrices, per-class F1, ROC/PR ────────────────────────────────
def _load_model_predict(art_path: Path, X: np.ndarray, kind: str,
                          n_out: int, scaler=None
                          ) -> Tuple[np.ndarray, np.ndarray | None]:
    """Return (predicted_labels_or_values, probabilities_if_cls).

    ``scaler``: optional StandardScaler for flat features.  Sklearn
    (.pkl) blobs already embed their own scaler; Torch (.pt) blobs do
    not, so the caller must pass the same scaler that was fit on the
    HPO train fold for flat features.
    """
    if art_path.suffix == ".pkl":
        with open(art_path, "rb") as f:
            blob = pickle.load(f)
        mdl = blob["model"]
        sk_scaler = blob.get("scaler") or scaler
        Xf = X.reshape(len(X), -1)
        if sk_scaler is not None:
            Xf = sk_scaler.transform(Xf)
        if kind == "cls" and hasattr(mdl, "predict_proba"):
            proba = mdl.predict_proba(Xf)
        else:
            proba = None
        return mdl.predict(Xf), proba
    # Torch.
    blob = torch.load(art_path, map_location="cpu", weights_only=False)
    in_shape = blob["in_shape"]; name = blob["model_name"]
    hp = blob.get("hyperparams") or {}
    seq = X.ndim == 3; mat = X.ndim == 4

    # For flat features (modal / indicators) MLP models were trained on
    # StandardScaler-transformed inputs — replicate that transform here.
    Xa = X
    if name == "mlp" and scaler is not None:
        Xa = scaler.transform(X.reshape(len(X), -1))
    t = torch.as_tensor(np.asarray(Xa)).float()
    if seq and t.ndim == 3:
        t = t.permute(0, 2, 1)

    if name == "mlp":
        in_dim = t.shape[-1] if t.ndim == 2 else int(np.prod(in_shape))
        if t.ndim == 3:
            t = t.flatten(1)
        hidden = tuple(hp.get("hidden", (256, 128, 64)))
        mdl = MLP(in_dim=in_dim, n_out=n_out, hidden=hidden,
                     regression=(kind == "reg"))
    elif name == "cnn":
        ch = in_shape[1] if len(in_shape) == 2 else in_shape[-1]
        mdl = Conv1DStack(n_channels=ch, n_out=n_out,
                              widths=tuple(hp.get("widths", (32, 64, 128))),
                              kernel_size=int(hp.get("kernel_size", 7)),
                              regression=(kind == "reg"))
    elif name == "transformer":
        ch = in_shape[1] if len(in_shape) == 2 else in_shape[-1]
        mdl = SmallTransformer(n_channels=ch, n_out=n_out,
                                   d_model=int(hp.get("d_model", 48)),
                                   n_layers=int(hp.get("n_layers", 2)),
                                   regression=(kind == "reg"))
    elif name == "cnn2d":
        mdl = Conv2DStack(n_channels=in_shape[0], n_out=n_out,
                              widths=tuple(hp.get("widths", (16, 32, 64))),
                              kernel_size=int(hp.get("kernel_size", 5)),
                              regression=(kind == "reg"))
    else:
        raise ValueError(name)
    mdl.load_state_dict(blob["state_dict"]); mdl.eval()
    with torch.no_grad():
        out = mdl(t)
    if kind == "cls":
        proba = torch.softmax(out, dim=1).numpy()
        return out.argmax(1).numpy(), proba
    return out.squeeze(1).numpy(), None


def _fit_scalers_per_task(features_path: Path) -> Dict[str, Dict[str, "StandardScaler"]]:
    """Re-fit one StandardScaler per (task, flat-feature) on the same
    train fold that HPO used.  Required because the Torch artefacts
    don't embed their scaler.
    """
    L = load_labels(features_path)
    tasks = build_targets(L["type_code"], L["storey"], L["end"], L["severity"])
    out: Dict[str, Dict[str, "StandardScaler"]] = {}
    flat_data = {name: load_feature(features_path, name) for name in FEATURES_FLAT}
    for tn, (mask, y_pool, kind) in tasks.items():
        out[tn] = {}
        ipool = np.where(mask)[0]
        i_tr, _, _ = make_split(y_pool, kind)
        idx_tr = ipool[i_tr]
        for feat in FEATURES_FLAT:
            X_tr = flat_data[feat][idx_tr].reshape(len(idx_tr), -1)
            out[tn][feat] = StandardScaler().fit(X_tr)
    return out


def plot_confusions(features_path: Path, results_dir: Path,
                      out_dir: Path) -> None:
    L = load_labels(features_path)
    tasks = build_targets(L["type_code"], L["storey"], L["end"], L["severity"])

    # We re-create the same splits used during training.
    splits: Dict[str, Dict[str, np.ndarray]] = {}
    for tn, (mask, y_pool, kind) in tasks.items():
        ipool = np.where(mask)[0]
        i_tr, i_va, i_te = make_split(y_pool, kind)
        splits[tn] = {
            "idx_te": ipool[i_te],
            "y_te":   y_pool[i_te],
            "kind":   kind,
        }

    feats: Dict[str, np.ndarray] = {}
    for name in (*FEATURES_FLAT, *FEATURES_SEQ, *FEATURES_MAT):
        feats[name] = load_feature(features_path, name)

    models_dir = results_dir / "models"
    for art in sorted(models_dir.iterdir()):
        tag = art.stem
        # tag = "<task>_<model>_<feature>" — task may contain underscore.
        for k in (3, 2):
            cand = "_".join(tag.split("_")[:-k])
            if cand in tasks:
                task_name = cand; rest = tag.split("_")[-k:]; break
        else:
            continue
        model_name = rest[0]; feature = "_".join(rest[1:])
        sp = splits[task_name]
        if sp["kind"] != "cls":
            continue
        X_te = feats[feature][sp["idx_te"]]
        n_out = TASK_N_CLASSES[task_name]
        try:
            scaler = SCALERS.get(task_name, {}).get(feature)
            y_pred, _ = _load_model_predict(art, X_te, "cls", n_out,
                                                  scaler=scaler)
        except Exception as e:
            print(f"  skip {tag}: {e}")
            continue
        y_te = sp["y_te"]
        labels = _class_labels(task_name)
        cm = confusion_matrix(y_te, y_pred, labels=list(range(len(labels))))
        cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
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
        ax.set_title(f"{task_name} — {model_name} / {feature}")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        _safe_savefig(fig, out_dir / "confusion" / f"{tag}.png")


# ── per-class F1 ────────────────────────────────────────────────────────────
def plot_perclass_f1(features_path: Path, results_dir: Path,
                       out_dir: Path) -> None:
    L = load_labels(features_path)
    tasks = build_targets(L["type_code"], L["storey"], L["end"], L["severity"])
    splits: Dict[str, Dict[str, np.ndarray]] = {}
    for tn, (mask, y_pool, kind) in tasks.items():
        ipool = np.where(mask)[0]
        _, _, i_te = make_split(y_pool, kind)
        splits[tn] = {"idx_te": ipool[i_te], "y_te": y_pool[i_te], "kind": kind}

    feats: Dict[str, np.ndarray] = {}
    for name in (*FEATURES_FLAT, *FEATURES_SEQ, *FEATURES_MAT):
        feats[name] = load_feature(features_path, name)

    # Aggregate per task: matrix (model x class) of F1.
    by_task: Dict[str, list[Tuple[str, str, np.ndarray]]] = {}
    for art in sorted((results_dir / "models").iterdir()):
        tag = art.stem
        for k in (3, 2):
            cand = "_".join(tag.split("_")[:-k])
            if cand in tasks: task_name = cand; rest = tag.split("_")[-k:]; break
        else:
            continue
        model_name = rest[0]; feature = "_".join(rest[1:])
        sp = splits[task_name]
        if sp["kind"] != "cls": continue
        X_te = feats[feature][sp["idx_te"]]
        n_out = TASK_N_CLASSES[task_name]
        try:
            scaler = SCALERS.get(task_name, {}).get(feature)
            y_pred, _ = _load_model_predict(art, X_te, "cls", n_out,
                                                  scaler=scaler)
        except Exception as e:
            continue
        f1 = f1_score(sp["y_te"], y_pred,
                       labels=list(range(n_out)),
                       average=None, zero_division=0)
        by_task.setdefault(task_name, []).append((f"{model_name}/{feature}", f1))

    for task_name, rows in by_task.items():
        labels = _class_labels(task_name)
        rows = sorted(rows, key=lambda r: -np.mean(r[1]))
        names = [r[0] for r in rows]
        mat = np.stack([r[1] for r in rows])
        fig, ax = plt.subplots(figsize=(0.7 + 0.6 * len(labels),
                                            0.4 + 0.3 * len(names)))
        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                          color="white" if mat[i, j] < 0.5 else "black", fontsize=7)
        ax.set_title(f"per-class F1 — {task_name}")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        _safe_savefig(fig, out_dir / "perclass_f1" / f"{task_name}.png")


# ── ROC + PR for binary task ────────────────────────────────────────────────
def plot_roc_pr_binary(features_path: Path, results_dir: Path,
                         out_dir: Path) -> None:
    L = load_labels(features_path)
    tasks = build_targets(L["type_code"], L["storey"], L["end"], L["severity"])
    mask, y_all, kind = tasks["binary"]
    _, _, i_te = make_split(y_all, kind)
    ipool = np.where(mask)[0]; idx_te = ipool[i_te]; y_te = y_all[i_te]

    feats: Dict[str, np.ndarray] = {}
    for name in (*FEATURES_FLAT, *FEATURES_SEQ, *FEATURES_MAT):
        feats[name] = load_feature(features_path, name)

    fig_roc, ax_roc = plt.subplots(figsize=(7, 7))
    fig_pr,  ax_pr  = plt.subplots(figsize=(7, 7))

    for art in sorted((results_dir / "models").iterdir()):
        tag = art.stem
        if not tag.startswith("binary_"):
            continue
        rest = tag[len("binary_"):]
        # rest = "<model>_<feature>"
        for f in (*FEATURES_FLAT, *FEATURES_SEQ, *FEATURES_MAT):
            if rest.endswith("_" + f):
                feat = f
                model_name = rest[:-(len(f) + 1)]
                break
        else:
            continue
        X = feats[feat][idx_te]
        try:
            scaler = SCALERS.get("binary", {}).get(feat)
            _, proba = _load_model_predict(art, X, "cls", 2, scaler=scaler)
        except Exception as e:
            continue
        if proba is None or proba.shape[1] < 2:
            continue
        scores = proba[:, 1]
        fpr, tpr, _ = roc_curve(y_te, scores)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, lw=1.0, label=f"{model_name}/{feat}  AUC={roc_auc:.3f}")
        p, r, _ = precision_recall_curve(y_te, scores)
        ax_pr.plot(r, p, lw=1.0, label=f"{model_name}/{feat}")
    ax_roc.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax_roc.set_xlabel("false positive rate"); ax_roc.set_ylabel("true positive rate")
    ax_roc.set_title("ROC — binary (Pristine vs Damage)")
    ax_roc.legend(loc="lower right", fontsize=7)
    ax_roc.grid(linestyle=":", alpha=0.5)
    ax_pr.set_xlabel("recall"); ax_pr.set_ylabel("precision")
    ax_pr.set_title("Precision–Recall — binary")
    ax_pr.legend(loc="lower left", fontsize=7)
    ax_pr.grid(linestyle=":", alpha=0.5)
    _safe_savefig(fig_roc, out_dir / "roc" / "binary_roc.png")
    _safe_savefig(fig_pr,  out_dir / "roc" / "binary_pr.png")


# ── severity scatter + residual ─────────────────────────────────────────────
def plot_severity_scatter(features_path: Path, results_dir: Path,
                            out_dir: Path) -> None:
    L = load_labels(features_path)
    tasks = build_targets(L["type_code"], L["storey"], L["end"], L["severity"])
    mask, y_all, kind = tasks["severity"]
    ipool = np.where(mask)[0]
    i_tr, i_va, i_te = make_split(y_all, kind)
    idx_te = ipool[i_te]; y_te = y_all[i_te]

    feats: Dict[str, np.ndarray] = {}
    for name in (*FEATURES_FLAT, *FEATURES_SEQ, *FEATURES_MAT):
        feats[name] = load_feature(features_path, name)

    for art in sorted((results_dir / "models").iterdir()):
        tag = art.stem
        if not tag.startswith("severity_"):
            continue
        rest = tag[len("severity_"):]
        for f in (*FEATURES_FLAT, *FEATURES_SEQ, *FEATURES_MAT):
            if rest.endswith("_" + f):
                feat = f; model_name = rest[:-(len(f) + 1)]; break
        else:
            continue
        X = feats[feat][idx_te]
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
        axes[0].set_title(f"{model_name}/{feat}")
        axes[0].grid(linestyle=":", alpha=0.5)
        res = y_pred - y_te
        axes[1].hist(res, bins=40, alpha=0.7)
        axes[1].set_xlabel("residual (pred − true)")
        axes[1].set_ylabel("count")
        axes[1].set_title("residuals")
        axes[1].grid(linestyle=":", alpha=0.5)
        _safe_savefig(fig, out_dir / "scatter" / f"{tag}.png")


# ── feature importances ─────────────────────────────────────────────────────
def plot_feature_importance(features_path: Path, results_dir: Path,
                              out_dir: Path) -> None:
    modal_names = [f"ch{c}_{lbl}"
                       for c in range(9)
                       for lbl in (
                           "peak1_f", "peak1_a", "peak2_f", "peak2_a",
                           "peak3_f", "peak3_a", "mean_logA", "std_logA", "bandE",
                       )]
    name_lookup = {"modal": modal_names, "indicators": INDICATOR_NAMES}
    for art in sorted((results_dir / "models").iterdir()):
        tag = art.stem
        if art.suffix != ".pkl":
            continue
        for f in FEATURES_FLAT:
            if tag.endswith("_" + f):
                feat = f
                rest = tag[:-(len(f) + 1)]
                break
        else:
            continue
        model_name = rest.split("_")[-1]
        if model_name not in {"rf", "xgb"}:
            continue
        task_name = "_".join(rest.split("_")[:-1])
        with open(art, "rb") as fh:
            blob = pickle.load(fh)
        mdl = blob["model"]
        if not hasattr(mdl, "feature_importances_"):
            continue
        imp = np.asarray(mdl.feature_importances_)
        names = name_lookup[feat]
        if len(names) != len(imp):
            names = [f"f{i}" for i in range(len(imp))]
        order = np.argsort(imp)[::-1][:20]
        fig, ax = plt.subplots(figsize=(7, max(3, len(order) * 0.25)))
        ax.barh([names[i] for i in order][::-1], imp[order][::-1])
        ax.set_xlabel("importance")
        ax.set_title(f"{task_name} — {model_name}/{feat} (top 20)")
        ax.grid(axis="x", linestyle=":", alpha=0.5)
        _safe_savefig(fig, out_dir / "feat_importance" / f"{tag}.png")


# ── PCA / t-SNE embeddings ──────────────────────────────────────────────────
def plot_embeddings(features_path: Path, out_dir: Path,
                      max_points: int = 3000) -> None:
    L = load_labels(features_path)
    tc = L["type_code"]
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(tc), size=min(max_points, len(tc)), replace=False)
    y = tc[idx]

    feats: Dict[str, np.ndarray] = {}
    for name in ("modal", "indicators"):
        feats[name] = load_feature(features_path, name)[idx]

    for feat_name, X in feats.items():
        X_flat = X.reshape(len(X), -1)
        X_s = StandardScaler().fit_transform(X_flat)

        # PCA
        pca = PCA(n_components=2).fit(X_s)
        Z = pca.transform(X_s)
        fig, ax = plt.subplots(figsize=(7, 6))
        for k in range(5):
            m = y == k
            ax.scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.6, label=TYPE_NAMES[k])
        ax.set_title(f"PCA — {feat_name}  (var explained {pca.explained_variance_ratio_.sum():.2f})")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f} %)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f} %)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(linestyle=":", alpha=0.5)
        _safe_savefig(fig, out_dir / "embedding" / f"pca_{feat_name}.png")

        # t-SNE
        try:
            tsne = TSNE(n_components=2, perplexity=30,
                          init="pca", random_state=SEED)
            Z = tsne.fit_transform(X_s)
            fig, ax = plt.subplots(figsize=(7, 6))
            for k in range(5):
                m = y == k
                ax.scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.6, label=TYPE_NAMES[k])
            ax.set_title(f"t-SNE — {feat_name}")
            ax.legend(loc="best", fontsize=8)
            ax.grid(linestyle=":", alpha=0.5)
            _safe_savefig(fig, out_dir / "embedding" / f"tsne_{feat_name}.png")
        except Exception as e:
            print(f"  t-SNE failed for {feat_name}: {e}")


# ── HPO response surfaces ───────────────────────────────────────────────────
def plot_hpo_surfaces(results_dir: Path, out_dir: Path) -> None:
    hpo_dir = results_dir / "hpo"
    if not hpo_dir.exists():
        return
    for path in sorted(hpo_dir.glob("*.json")):
        data = json.loads(path.read_text())
        trials = data["trials"]
        if not trials:
            continue
        # Identify the two varying hyperparameters.
        h_keys = list(trials[0]["hyperparams"].keys())
        if len(h_keys) < 2:
            continue
        k0, k1 = h_keys[0], h_keys[1]
        v0 = sorted({str(t["hyperparams"][k0]) for t in trials})
        v1 = sorted({str(t["hyperparams"][k1]) for t in trials})
        if len(v0) <= 1 or len(v1) <= 1:
            continue
        mat = np.full((len(v0), len(v1)), np.nan)
        for t in trials:
            i = v0.index(str(t["hyperparams"][k0]))
            j = v1.index(str(t["hyperparams"][k1]))
            mat[i, j] = t["metric_val"]
        fig, ax = plt.subplots(figsize=(4 + 0.5 * len(v1),
                                            3 + 0.4 * len(v0)))
        im = ax.imshow(mat, cmap="viridis", origin="lower",
                          aspect="auto")
        ax.set_xticks(range(len(v1))); ax.set_xticklabels(v1, rotation=30, ha="right")
        ax.set_yticks(range(len(v0))); ax.set_yticklabels(v0)
        ax.set_xlabel(k1); ax.set_ylabel(k0)
        ax.set_title(f"{data['task']} / {data['model']} / {data['feature']}")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.3f}", ha="center",
                              va="center",
                              color="white" if mat[i, j] < np.nanmean(mat) else "black",
                              fontsize=8)
        fig.colorbar(im, fraction=0.046, pad=0.04)
        out_path = out_dir / "hpo" / f"{path.stem}.png"
        _safe_savefig(fig, out_path)


# ── orchestrator ─────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, default=_REPO / "dataset" / "features.h5")
    p.add_argument("--results",  type=Path, default=_REPO / "results")
    p.add_argument("--skip", type=str, default="",
                      help="Comma-separated stage names to skip.")
    args = p.parse_args()
    out = args.results / "figures"; out.mkdir(parents=True, exist_ok=True)
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    # Module-global table of scalers, one per (task, flat-feature).
    # Plot functions read SCALERS at call time.
    global SCALERS
    print("fitting per-task StandardScalers …")
    SCALERS = _fit_scalers_per_task(args.features)

    stages = [
        ("dataset",         lambda: plot_dataset_summary(args.features, out)),
        ("signals",         lambda: plot_examples_per_class(args.features, out)),
        ("confusion",       lambda: plot_confusions(args.features, args.results, out)),
        ("perclass_f1",     lambda: plot_perclass_f1(args.features, args.results, out)),
        ("roc",             lambda: plot_roc_pr_binary(args.features, args.results, out)),
        ("scatter",         lambda: plot_severity_scatter(args.features, args.results, out)),
        ("feat_importance", lambda: plot_feature_importance(args.features, args.results, out)),
        ("embedding",       lambda: plot_embeddings(args.features, out)),
        ("hpo",             lambda: plot_hpo_surfaces(args.results, out)),
    ]
    for name, fn in stages:
        if name in skip:
            print(f"skip {name}")
            continue
        print(f"plotting: {name}")
        try:
            fn()
        except Exception as e:
            print(f"  FAILED {name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
