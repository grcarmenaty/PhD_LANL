"""HPO for every model family on every CFDAC variant.

This is the closure script that completes the matrix:

  modal       → RF / XGB / MLP                        (already in `hpo.py`)
  frf_mag     → 1-D CNN / Transformer                  (already in `hpo.py`)
  timeseries  → 1-D CNN / Transformer                  (already in `hpo.py`)
  cfdac (legacy)  → 2-D CNN                            (already in `hpo.py`)
  cfdac_{real,imag,mag,phase,realimag,magphase,all}
      → 2-D CNN / 3-D CNN                              (`hpo_cfdac_variants.py`)
      → RF / XGB / MLP / 1-D CNN / Transformer         (THIS SCRIPT)

The 3-D CNN already covers the volumetric inductive bias, so the
`cfdac3d_*` variants are *not* re-run here — they would flatten to the
same vector as their 2-D analogue.

Resume-friendly: every cell whose JSON already exists is silently
skipped.  Re-running the script after a kill or after `evaluate.py`
adds new artefacts is therefore free.

Outputs (additive to the existing directories):
    results/hpo/<task>__<model>__<variant>.json
    results/models/<task>_<model>_<variant>.pkl    (rf / xgb)
    results/models/<task>_<model>_<variant>.pt     (mlp / cnn / transformer)
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, mean_absolute_error, r2_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier, XGBRegressor

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml_pipeline.models import (                                       # noqa: E402
    MLP, Conv1DStack, SmallTransformer,
)
from ml_pipeline.tasks import build_targets                              # noqa: E402
from ml_pipeline.train import (                                           # noqa: E402
    load_labels, load_feature, make_split, SEED, _CFDAC_VARIANTS,
)

torch.set_num_threads(4)


GRIDS = {
    "rf":  {"n_estimators": [200], "max_depth": [None]},
    # RF regression on 16 384-dim flat CFDAC is much slower than RF
    # classification (no built-in node-impurity shortcut); n_estimators=100
    # is enough for severity to land under ~90 s.
    "rf_reg":  {"n_estimators": [100], "max_depth": [None]},
    "xgb": {"n_estimators": [200], "max_depth": [6]},
    "mlp": {"hidden": [(256, 128, 64)], "lr": [1e-3]},
    "cnn": {"widths": [(32, 64, 128)], "kernel_size": [7]},
    "transformer": {"d_model": [48], "n_layers": [2]},
}
# Single-point grids per cell — 16 384-dim flattened CFDAC + 6 model
# families × 7 variants × 5 tasks already adds up to ~ 175 trials; a
# 4-point grid would push wall-clock past 6 h.

VARIANTS = [
    "cfdac_real", "cfdac_imag", "cfdac_mag", "cfdac_phase",     # 1-ch
    "cfdac_realimag", "cfdac_magphase",                         # 2-ch
    "cfdac_all",                                                # 4-ch
]

MODELS = ("rf", "xgb", "mlp", "cnn", "transformer")


def _flatten(X: np.ndarray) -> np.ndarray:
    return X.reshape(len(X), -1).astype(np.float32, copy=False)


def _to_seq(X: np.ndarray) -> np.ndarray:
    """Reshape ``(n, C, H, W)`` CFDAC matrix into a ``(n, C*W, H)`` tensor
    suitable for ``Conv1DStack`` / ``SmallTransformer``.  Rows (H) become
    the sequence length so that a stride-1 conv along time picks up
    column-of-frequency-bin correlations; the C * W channel stack lets
    every column carry its own kernel.
    """
    if X.ndim == 3:
        X = X[:, None, :, :]
    n, C, H, W = X.shape
    return X.transpose(0, 1, 3, 2).reshape(n, C * W, H).astype(np.float32, copy=False)


def _train_sklearn(model_name: str, kind: str, params: dict,
                    X_tr, y_tr, X_va, y_va, X_te, y_te) -> Tuple[dict, object]:
    t0 = time.time()
    if model_name == "rf":
        cls = RandomForestClassifier if kind == "cls" else RandomForestRegressor
        mdl = cls(n_jobs=-1, random_state=SEED, **params)
    else:
        cls = XGBClassifier if kind == "cls" else XGBRegressor
        mdl = cls(n_jobs=-1, random_state=SEED,
                     learning_rate=0.1,
                     tree_method="hist", max_bin=128,
                     **params)
    mdl.fit(X_tr, y_tr)
    pred_va = mdl.predict(X_va); pred_te = mdl.predict(X_te)
    if kind == "cls":
        val  = float(accuracy_score(y_va, pred_va))
        test = float(accuracy_score(y_te, pred_te))
        extras = {}
        mname = "accuracy"
    else:
        val  = float(r2_score(y_va, pred_va))
        test = float(r2_score(y_te, pred_te))
        extras = {"mae_test": float(mean_absolute_error(y_te, pred_te))}
        mname = "R2"
    return ({
        "hyperparams": params, "metric_name": mname,
        "metric_val": val, "metric_test": test, "extras": extras,
        "runtime_s": time.time() - t0,
    }, mdl)


def _train_torch(model_name: str, kind: str, n_out: int, params: dict,
                  X_tr, y_tr, X_va, y_va, X_te, y_te,
                  epochs: int = 4) -> Tuple[dict, nn.Module]:
    t0 = time.time()
    Xtr = torch.as_tensor(X_tr).float()
    Xva = torch.as_tensor(X_va).float()
    Xte = torch.as_tensor(X_te).float()
    if kind == "cls":
        ytr = torch.as_tensor(y_tr).long()
        yva = torch.as_tensor(y_va).long()
        yte = torch.as_tensor(y_te).long()
        loss_fn = nn.CrossEntropyLoss()
    else:
        ytr = torch.as_tensor(y_tr).float().unsqueeze(1)
        yva = torch.as_tensor(y_va).float().unsqueeze(1)
        yte = torch.as_tensor(y_te).float().unsqueeze(1)
        loss_fn = nn.MSELoss()

    if model_name == "mlp":
        mdl = MLP(in_dim=Xtr.shape[1], n_out=n_out,
                      hidden=tuple(params["hidden"]), dropout=0.2,
                      regression=(kind == "reg"))
        opt = torch.optim.AdamW(mdl.parameters(),
                                    lr=float(params["lr"]),
                                    weight_decay=1e-4)
    elif model_name == "cnn":
        ch = Xtr.shape[1]
        mdl = Conv1DStack(n_channels=ch, n_out=n_out,
                              widths=tuple(params["widths"]),
                              kernel_size=int(params["kernel_size"]),
                              regression=(kind == "reg"))
        opt = torch.optim.AdamW(mdl.parameters(), lr=1e-3,
                                    weight_decay=1e-4)
    elif model_name == "transformer":
        ch = Xtr.shape[1]
        mdl = SmallTransformer(n_channels=ch, n_out=n_out,
                                  d_model=int(params["d_model"]),
                                  n_layers=int(params["n_layers"]),
                                  downsample=16,
                                  regression=(kind == "reg"))
        opt = torch.optim.AdamW(mdl.parameters(), lr=5e-4,
                                    weight_decay=1e-4)
    else:
        raise ValueError(model_name)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)
    best, best_state = -np.inf, None
    for ep in range(epochs):
        mdl.train()
        for xb, yb in dl:
            opt.zero_grad(); loss_fn(mdl(xb), yb).backward(); opt.step()
        sched.step()
        mdl.eval()
        with torch.no_grad():
            out = mdl(Xva).cpu()
            metric = (accuracy_score(y_va, out.argmax(1).numpy())
                          if kind == "cls"
                          else r2_score(y_va, out.squeeze(1).numpy()))
        if metric > best:
            best = float(metric)
            best_state = {k: v.detach().clone() for k, v in mdl.state_dict().items()}
    if best_state is not None:
        mdl.load_state_dict(best_state)
    mdl.eval()
    with torch.no_grad():
        out_te = mdl(Xte).cpu()
        if kind == "cls":
            test = float(accuracy_score(y_te, out_te.argmax(1).numpy()))
            extras = {}; mname = "accuracy"
        else:
            test = float(r2_score(y_te, out_te.squeeze(1).numpy()))
            extras = {"mae_test": float(mean_absolute_error(
                y_te, out_te.squeeze(1).numpy()))}
            mname = "R2"
    return ({
        "hyperparams": {k: (list(v) if isinstance(v, tuple) else v)
                         for k, v in params.items()},
        "metric_name": mname, "metric_val": best, "metric_test": test,
        "extras": extras, "runtime_s": time.time() - t0,
    }, mdl)


def _existing_cell_done(path: Path) -> bool:
    """Skip-existing guard: if the JSON file already has a non-null
    `best_metric_val`, the cell is considered completed by an earlier
    run.  Returns True if the cell should be skipped."""
    if not path.exists():
        return False
    try:
        blob = json.loads(path.read_text())
        return blob.get("best_metric_val") is not None
    except Exception:
        return False


def run(features_path: Path, out_dir: Path, epochs: int = 4) -> None:
    hpo_dir    = out_dir / "hpo"
    models_dir = out_dir / "models"
    hpo_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(features_path)
    tasks = build_targets(labels["type_code"], labels["storey"],
                              labels["end"], labels["severity"])

    plan: List[Tuple[str, str, str]] = []
    for tname in tasks:
        for variant in VARIANTS:
            for m in MODELS:
                cell_path = hpo_dir / f"{tname}__{m}__{variant}.json"
                if _existing_cell_done(cell_path):
                    continue
                plan.append((tname, m, variant))
    print(f"plan: {len(plan)} cells (existing cells skipped)")

    # Group by variant, respecting the order declared in VARIANTS
    # (smallest → largest) so the 4-channel `cfdac_all` is last.
    variant_rank = {v: i for i, v in enumerate(VARIANTS)}
    plan.sort(key=lambda r: (variant_rank[r[2]], r[0], r[1]))
    current_feat, X_full = None, None
    done = 0
    for tname, model_name, feat in plan:
        mask, y_pool, kind = tasks[tname]
        ipool = np.where(mask)[0]
        i_tr, i_va, i_te = make_split(y_pool, kind)
        idx_tr = ipool[i_tr]; idx_va = ipool[i_va]; idx_te = ipool[i_te]
        y_tr = y_pool[i_tr]; y_va = y_pool[i_va]; y_te = y_pool[i_te]
        if feat != current_feat:
            del X_full; gc.collect()
            print(f">>> loading {feat} ...", flush=True)
            X_full = load_feature(features_path, feat)
            current_feat = feat
            print(f"    shape = {X_full.shape}, "
                    f"mem = {X_full.nbytes / 1e9:.2f} GB", flush=True)

        # Reshape view depends on model family.
        if model_name in ("rf", "xgb", "mlp"):
            X_tr = _flatten(X_full[idx_tr])
            X_va = _flatten(X_full[idx_va])
            X_te = _flatten(X_full[idx_te])
            flat_in_dim = X_tr.shape[1]
            scaler = StandardScaler().fit(X_tr)
            X_tr_s = scaler.transform(X_tr); del X_tr
            X_va_s = scaler.transform(X_va); del X_va
            X_te_s = scaler.transform(X_te); del X_te
            gc.collect()
        else:  # cnn / transformer
            X_tr_s = _to_seq(X_full[idx_tr])
            X_va_s = _to_seq(X_full[idx_va])
            X_te_s = _to_seq(X_full[idx_te])
            scaler = None
            flat_in_dim = X_tr_s.shape[1] * X_tr_s.shape[2]
        in_shape = list(X_tr_s.shape[1:])

        n_out = (int(y_pool.max()) + 1) if kind == "cls" else 1
        # RF regression uses the slimmer grid.
        grid_key = "rf_reg" if (model_name == "rf" and kind == "reg") else model_name
        grid  = GRIDS[grid_key]
        keys  = list(grid.keys()); vals = [grid[k] for k in keys]
        trials: List[Dict[str, Any]] = []
        best_trial, best_obj = None, None
        for combo in itertools.product(*vals):
            params = dict(zip(keys, combo))
            try:
                if model_name in ("rf", "xgb"):
                    row, mdl = _train_sklearn(model_name, kind, params,
                                                   X_tr_s, y_tr,
                                                   X_va_s, y_va,
                                                   X_te_s, y_te)
                else:
                    row, mdl = _train_torch(model_name, kind, n_out, params,
                                                 X_tr_s, y_tr,
                                                 X_va_s, y_va,
                                                 X_te_s, y_te,
                                                 epochs=epochs)
            except Exception as e:
                print(f"  FAIL {tname}/{model_name}/{feat} {params}: {e}",
                          flush=True)
                continue
            trials.append(row); done += 1
            print(f"  [{done}] {tname}/{model_name}/{feat} "
                    f"{params}  val={row['metric_val']:.4f}  "
                    f"test={row['metric_test']:.4f}  ({row['runtime_s']:.1f}s)",
                    flush=True)
            if best_trial is None or row["metric_val"] > best_trial["metric_val"]:
                best_trial = row; best_obj = mdl

        if best_trial is None:
            continue
        out_json = hpo_dir / f"{tname}__{model_name}__{feat}.json"
        out_json.write_text(json.dumps({
            "task": tname, "model": model_name, "feature": feat,
            "best_hyperparams": best_trial["hyperparams"],
            "best_metric_val":  best_trial["metric_val"],
            "best_metric_test": best_trial["metric_test"],
            "trials": trials,
        }, indent=2, default=str))
        tag = f"{tname}_{model_name}_{feat}"
        if model_name in ("rf", "xgb"):
            with open(models_dir / f"{tag}.pkl", "wb") as fh:
                pickle.dump({"model": best_obj, "scaler": scaler}, fh)
        else:
            torch.save({
                "state_dict": best_obj.state_dict(),
                "model_name": model_name,
                "n_out": n_out,
                "in_shape": in_shape,
                "hyperparams": best_trial["hyperparams"],
            }, models_dir / f"{tag}.pt")
    print("\ndone.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path,
                      default=_REPO / "dataset" / "features.h5")
    p.add_argument("--out", type=Path, default=_REPO / "results")
    p.add_argument("--epochs", type=int, default=4)
    args = p.parse_args()
    run(args.features, args.out, args.epochs)


if __name__ == "__main__":
    main()
