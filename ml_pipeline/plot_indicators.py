"""Plots for the § 7.6 indicator regression task.

Generates, under ``results/figures/indicators/``:

* ``hpo/<indicator>__<model>__modal.png`` — HPO response surface
  (val R² heatmap) for every (indicator, model) cell.
* ``scatter/<indicator>__<model>__modal.png`` — true-vs-predicted
  scatter on the *full 2 638-case experimental damage subset* (using
  the saved synth-trained model).
* ``r2_summary.png`` — bar chart of best exp R² per (indicator, model)
  with bars sorted by indicator R² and coloured by model.

Run with::

  python ml_pipeline/plot_indicators.py
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml_pipeline.case_design import TYPE_PRISTINE                     # noqa: E402
from ml_pipeline.features import INDICATOR_NAMES                      # noqa: E402
from ml_pipeline.models import MLP                                      # noqa: E402
from ml_pipeline.train import (                                          # noqa: E402
    load_labels, load_feature, make_split, SEED,
)
from sklearn.preprocessing import StandardScaler                       # noqa: E402


def _safe_savefig(fig, path: Path, dpi: int = 110) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _hpo_heatmap(blob: dict, out_path: Path) -> None:
    trials = blob.get("trials", [])
    if not trials:
        return
    h_keys = list(trials[0]["hyperparams"].keys())
    if len(h_keys) < 2:
        return
    k0, k1 = h_keys[0], h_keys[1]
    v0 = sorted({str(t["hyperparams"][k0]) for t in trials})
    v1 = sorted({str(t["hyperparams"][k1]) for t in trials})
    mat = np.full((len(v0), len(v1)), np.nan)
    for t in trials:
        i = v0.index(str(t["hyperparams"][k0]))
        j = v1.index(str(t["hyperparams"][k1]))
        mat[i, j] = t["metric_val"]
    fig, ax = plt.subplots(figsize=(2.5 + 0.5 * len(v1),
                                        2 + 0.4 * len(v0)))
    im = ax.imshow(mat, cmap="viridis", origin="lower", aspect="auto")
    ax.set_xticks(range(len(v1))); ax.set_xticklabels(v1, rotation=30, ha="right")
    ax.set_yticks(range(len(v0))); ax.set_yticklabels(v0)
    ax.set_xlabel(k1); ax.set_ylabel(k0)
    ax.set_title(f"{blob['indicator']} / {blob['model']} / {blob['feature']}  "
                    f"val R²")
    mid = np.nanmean(mat) if np.isfinite(mat).any() else 0.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                          color="white" if mat[i, j] < mid else "black",
                          fontsize=8)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    _safe_savefig(fig, out_path)


def _load_indicator_model(art: Path, X_syn_train: np.ndarray,
                            in_dim: int):
    """Load a (model, modal-scaler) pair from disk."""
    scaler = StandardScaler().fit(X_syn_train.reshape(len(X_syn_train), -1))
    if art.suffix == ".pkl":
        with open(art, "rb") as fh:
            blob = pickle.load(fh)
        return ("sklearn", blob["model"], blob.get("scaler") or scaler)
    blob = torch.load(art, map_location="cpu", weights_only=False)
    hp = blob.get("hyperparams") or {}
    mdl = MLP(in_dim=in_dim, n_out=1,
                  hidden=tuple(hp.get("hidden", (256, 128, 64))),
                  regression=True)
    mdl.load_state_dict(blob["state_dict"]); mdl.eval()
    return ("mlp", mdl, scaler)


def _predict(kind: str, mdl, scaler, X: np.ndarray) -> np.ndarray:
    Xf = X.reshape(len(X), -1)
    Xs = scaler.transform(Xf)
    if kind == "sklearn":
        return mdl.predict(Xs)
    with torch.no_grad():
        return mdl(torch.as_tensor(Xs).float()).squeeze(1).numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--synth", type=Path,
                      default=_REPO / "dataset" / "features.h5")
    p.add_argument("--exp", type=Path,
                      default=_REPO / "dataset" / "experimental_features.h5")
    p.add_argument("--results", type=Path, default=_REPO / "results")
    args = p.parse_args()

    out_root = args.results / "figures" / "indicators"
    out_root.mkdir(parents=True, exist_ok=True)

    # ── HPO heatmaps ────────────────────────────────────────────────────
    hpo_dir = args.results / "hpo_indicators"
    for path in sorted(hpo_dir.glob("*.json")):
        blob = json.loads(path.read_text())
        _hpo_heatmap(blob,
                       out_root / "hpo" / f"{path.stem}.png")
    print(f"wrote HPO heatmaps for {len(list(hpo_dir.glob('*.json')))} cells")

    # ── exp-set scatter plots ────────────────────────────────────────────
    L = load_labels(args.synth)
    type_code = L["type_code"]
    dmg_idx = np.where(type_code != TYPE_PRISTINE)[0]
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(dmg_idx))
    n = len(dmg_idx); n_test = int(0.15 * n); n_val = int(0.15 * n)
    idx_train_syn = dmg_idx[perm[n_test + n_val:]]
    X_syn_modal = load_feature(args.synth, "modal")
    X_syn_train_modal = X_syn_modal[idx_train_syn]

    with h5py.File(args.exp, "r") as f:
        e_type = f["type_code"][:]
        e_modal = f["modal"][:]
        e_ind   = f["indicators"][:]
    edmg = np.where(e_type != TYPE_PRISTINE)[0]
    X_exp_modal = e_modal[edmg]
    Y_exp_ind   = e_ind[edmg]

    in_dim = X_syn_modal.shape[1] if X_syn_modal.ndim == 2 else int(np.prod(X_syn_modal.shape[1:]))

    models_dir = args.results / "models_indicators"
    for art in sorted(models_dir.iterdir()):
        # tag = "<indicator>_<model>_<feature>".  Indicator name may have underscores.
        tag = art.stem
        if not tag.endswith("_modal"):
            continue
        rest = tag[: -len("_modal")]
        for m in ("rf", "xgb", "mlp"):
            if rest.endswith("_" + m):
                ind_name = rest[: -len(m) - 1]; model_name = m
                break
        else:
            continue
        if ind_name not in INDICATOR_NAMES:
            continue
        ind_idx = INDICATOR_NAMES.index(ind_name)
        try:
            kind, mdl, scaler = _load_indicator_model(art, X_syn_train_modal, in_dim)
            y_hat = _predict(kind, mdl, scaler, X_exp_modal)
        except Exception as e:
            print(f"  skip {tag}: {e}")
            continue
        y_true = Y_exp_ind[:, ind_idx]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].scatter(y_true, y_hat, s=8, alpha=0.5)
        lo, hi = float(np.nanmin([y_true, y_hat])), float(np.nanmax([y_true, y_hat]))
        axes[0].plot([lo, hi], [lo, hi], "k--", lw=0.7)
        axes[0].set_xlabel(f"true {ind_name}")
        axes[0].set_ylabel(f"predicted")
        axes[0].set_title(f"{ind_name} / {model_name}/modal — exp scatter (n={len(y_true)})")
        axes[0].grid(linestyle=":", alpha=0.5)
        res = y_hat - y_true
        axes[1].hist(res[np.isfinite(res)], bins=40, alpha=0.7)
        axes[1].set_xlabel("residual (pred − true)")
        axes[1].set_ylabel("count")
        axes[1].set_title("residuals")
        axes[1].grid(linestyle=":", alpha=0.5)
        _safe_savefig(fig, out_root / "scatter" / f"{tag}.png")
    print(f"wrote scatter plots")

    # ── R² summary bar chart ────────────────────────────────────────────
    pred_path = args.results / "indicator_predictions_full.json"
    if pred_path.exists():
        rows = json.loads(pred_path.read_text())
        per_ind = {}
        for r in rows:
            per_ind.setdefault(r["indicator"], {})[r["model"]] = r["exp_R2"]
        # Order indicators by best non-MLP exp R² so the readers see the
        # transferable ones first.
        order = sorted(per_ind.keys(),
                          key=lambda n: -max(per_ind[n].get("rf", -1e9),
                                                per_ind[n].get("xgb", -1e9)))
        models = ["rf", "xgb", "mlp"]
        fig, ax = plt.subplots(figsize=(11, 6))
        x = np.arange(len(order))
        w = 0.27
        for i, m in enumerate(models):
            vals = np.array([per_ind[n].get(m, np.nan) for n in order])
            ax.bar(x + (i - 1) * w,
                    np.where(np.isfinite(vals) & (vals > -1.5), vals, -1.5),
                    width=w, label=m)
        ax.set_xticks(x); ax.set_xticklabels(order, rotation=45, ha="right",
                                                  fontsize=8)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylim(-1.5, 1.0)
        ax.set_ylabel("exp R² (clipped at −1.5)")
        ax.set_title("Indicator regression — full-exp R² per (indicator, model)")
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        _safe_savefig(fig, out_root / "r2_summary.png")
        print("wrote r2_summary.png")


if __name__ == "__main__":
    main()
