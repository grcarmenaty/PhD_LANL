"""Train 22 separate per-indicator regressors.

For each pymodal indicator (SCI, unsigned_SCI, DRQ, AIGAC, FRFRMS,
FRFSF, FRFSM_6dB, ODS_diff, r2_imag, RVAC_{mean,std,min,max},
GAC_{mean,std,min,max}, M2L_{mean,std,min,max,abs_sum}) train one
regressor per ``(model, feature)`` cell.

Trained on **damage samples only** (Pristine excluded) — same 70 / 15 / 15
split as the main HPO.  Scored on:
  * synthetic test fold
  * full 2638-case experimental dataset

Output:
  results/hpo_indicators/<indicator>__<model>__<feature>.json   per-trial log
  results/models_indicators/<indicator>_<model>_<feature>.pkl/.pt    artefact

Per the user request "models to predict each indicator independently
of damage typology or location" — these regressors take a *feature
representation* of the damaged trial and output the indicator value.
22 indicators × {RF, XGB, MLP} × {modal, frf_mag, timeseries, cfdac_realimag}
= 264 cells (subject to model–feature compatibility).
"""
from __future__ import annotations

import argparse
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml_pipeline.case_design import TYPE_PRISTINE                     # noqa: E402
from ml_pipeline.features import INDICATOR_NAMES                      # noqa: E402
from ml_pipeline.models import MLP, Conv2DStack                        # noqa: E402
from ml_pipeline.train import (                                          # noqa: E402
    load_labels, load_feature, make_split, SEED,
)

torch.set_num_threads(4)

# (model, feature) cells.  Limited to the configurations that win the
# severity task in the main HPO so each indicator gets a strong
# regressor without exploding the trial count.
CELLS: List[Tuple[str, str]] = [
    ("rf",    "modal"),
    ("xgb",   "modal"),
    ("mlp",   "modal"),
]

GRIDS = {
    "rf":     {"n_estimators": [100, 300], "max_depth": [12, None]},
    "xgb":    {"n_estimators": [100, 300], "max_depth": [4, 8]},
    "mlp":    {"hidden": [(128, 64), (256, 128, 64)], "lr": [1e-3, 3e-3]},
    "cnn2d":  {"widths": [(8, 16, 32), (16, 32, 64)], "kernel_size": [3, 5]},
}


def _train_sklearn(model_name: str, params, X_tr, y_tr, X_va, y_va,
                    X_te, y_te) -> Tuple[Dict[str, Any], object]:
    t0 = time.time()
    if model_name == "rf":
        mdl = RandomForestRegressor(n_jobs=-1, random_state=SEED, **params)
    else:
        mdl = XGBRegressor(n_jobs=-1, random_state=SEED,
                              learning_rate=0.1, **params)
    mdl.fit(X_tr, y_tr)
    yhat_va = mdl.predict(X_va); yhat_te = mdl.predict(X_te)
    return ({
        "hyperparams": params,
        "metric_name": "R2",
        "metric_val":  float(r2_score(y_va, yhat_va)),
        "metric_test": float(r2_score(y_te, yhat_te)),
        "extras":      {"mae_test": float(mean_absolute_error(y_te, yhat_te))},
        "runtime_s":   time.time() - t0,
    }, mdl)


def _train_torch(model_name: str, params, X_tr, y_tr, X_va, y_va,
                  X_te, y_te, epochs: int = 4) -> Tuple[Dict[str, Any], nn.Module]:
    t0 = time.time()
    Xtr = torch.as_tensor(X_tr).float()
    Xva = torch.as_tensor(X_va).float()
    Xte = torch.as_tensor(X_te).float()
    ytr = torch.as_tensor(y_tr).float().unsqueeze(1)
    yva = torch.as_tensor(y_va).float().unsqueeze(1)
    yte = torch.as_tensor(y_te).float().unsqueeze(1)
    if model_name == "mlp":
        in_dim = Xtr.shape[1] if Xtr.ndim == 2 else int(np.prod(Xtr.shape[1:]))
        if Xtr.ndim != 2:
            Xtr = Xtr.flatten(1); Xva = Xva.flatten(1); Xte = Xte.flatten(1)
        model = MLP(in_dim=in_dim, n_out=1,
                     hidden=tuple(params["hidden"]), dropout=0.2,
                     regression=True, bounded_output=False)
        opt = torch.optim.AdamW(model.parameters(),
                                 lr=float(params["lr"]),
                                 weight_decay=1e-4)
    elif model_name == "cnn2d":
        model = Conv2DStack(n_channels=Xtr.shape[1], n_out=1,
                              widths=tuple(params["widths"]),
                              kernel_size=int(params["kernel_size"]),
                              regression=True, bounded_output=False)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    else:
        raise ValueError(model_name)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)
    loss_fn = nn.MSELoss()
    best, best_state = -np.inf, None
    for _ep in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            r2 = float(r2_score(y_va, model(Xva).squeeze(1).cpu().numpy()))
        if r2 > best:
            best = r2
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        yhat_te = model(Xte).squeeze(1).cpu().numpy()
    return ({
        "hyperparams": {k: (list(v) if isinstance(v, tuple) else v)
                         for k, v in params.items()},
        "metric_name": "R2",
        "metric_val":  best,
        "metric_test": float(r2_score(y_te, yhat_te)),
        "extras":      {"mae_test": float(mean_absolute_error(y_te, yhat_te))},
        "runtime_s":   time.time() - t0,
    }, model)


def run(features_path: Path, out_dir: Path, epochs: int = 4) -> None:
    hpo_dir    = out_dir / "hpo_indicators"
    models_dir = out_dir / "models_indicators"
    hpo_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(features_path)
    type_code = labels["type_code"]
    damage_mask = type_code != TYPE_PRISTINE
    damage_idx_global = np.where(damage_mask)[0]
    print(f"using {len(damage_idx_global)} damage samples out of "
            f"{len(type_code)}")

    # 70/15/15 random split among damage samples (regression).
    n_dmg = len(damage_idx_global)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n_dmg)
    n_test = int(0.15 * n_dmg); n_val = int(0.15 * n_dmg)
    idx_test  = damage_idx_global[perm[:n_test]]
    idx_val   = damage_idx_global[perm[n_test:n_test + n_val]]
    idx_train = damage_idx_global[perm[n_test + n_val:]]

    with h5py.File(features_path, "r") as f:
        ind_all = f["indicators"][:]                     # (n, 22)

    feats_loaded: Dict[str, np.ndarray] = {}
    for _, fname in CELLS:
        if fname not in feats_loaded:
            feats_loaded[fname] = load_feature(features_path, fname)

    grand_total = len(INDICATOR_NAMES) * sum(
        int(np.prod([len(v) for v in GRIDS[m].values()])) for m, _ in CELLS)
    print(f"plan: 22 indicators × {len(CELLS)} cells, "
            f"~{grand_total} trials")

    done = 0
    for ind_idx, ind_name in enumerate(INDICATOR_NAMES):
        y_train = ind_all[idx_train, ind_idx].astype(np.float32)
        y_val   = ind_all[idx_val,   ind_idx].astype(np.float32)
        y_test  = ind_all[idx_test,  ind_idx].astype(np.float32)
        print(f"\n=== indicator {ind_idx:>2d} / 22  '{ind_name}'  "
                f"std(train)={y_train.std():.4f} ===")
        for model_name, feat_name in CELLS:
            tag = f"{ind_name}__{model_name}__{feat_name}"
            if (hpo_dir / f"{tag}.json").exists():
                print(f"  skip {tag} (cached)")
                continue
            X = feats_loaded[feat_name]
            X_tr = X[idx_train]; X_va = X[idx_val]; X_te = X[idx_test]
            scaler = None
            if feat_name in {"modal", "indicators"}:
                scaler = StandardScaler().fit(X_tr.reshape(len(X_tr), -1))
                X_tr = scaler.transform(X_tr.reshape(len(X_tr), -1))
                X_va = scaler.transform(X_va.reshape(len(X_va), -1))
                X_te = scaler.transform(X_te.reshape(len(X_te), -1))
            grid = GRIDS[model_name]
            keys = list(grid.keys()); vals = [grid[k] for k in keys]
            trials, best, best_obj = [], None, None
            for combo in itertools.product(*vals):
                params = dict(zip(keys, combo))
                try:
                    if model_name in ("rf", "xgb"):
                        row, mdl = _train_sklearn(model_name, params,
                                                       X_tr, y_train,
                                                       X_va, y_val,
                                                       X_te, y_test)
                    else:
                        row, mdl = _train_torch(model_name, params,
                                                     X_tr, y_train,
                                                     X_va, y_val,
                                                     X_te, y_test,
                                                     epochs=epochs)
                except Exception as e:
                    print(f"  FAIL {ind_name}/{model_name}/{feat_name} "
                            f"{params}: {e}")
                    continue
                trials.append(row); done += 1
                if best is None or row["metric_val"] > best["metric_val"]:
                    best, best_obj = row, mdl
            if best is None: continue
            print(f"  {model_name}/{feat_name:<14s}  "
                    f"val R²={best['metric_val']:.3f}  "
                    f"test R²={best['metric_test']:.3f}  "
                    f"MAE={best['extras']['mae_test']:.3f}  "
                    f"best={best['hyperparams']}")
            tag = f"{ind_name}__{model_name}__{feat_name}"
            (hpo_dir / f"{tag}.json").write_text(json.dumps({
                "indicator": ind_name, "model": model_name,
                "feature": feat_name,
                "best_hyperparams": best["hyperparams"],
                "best_metric_val": best["metric_val"],
                "best_metric_test": best["metric_test"],
                "best_mae_test": best["extras"]["mae_test"],
                "trials": trials,
            }, indent=2, default=str))
            if model_name in ("rf", "xgb"):
                with open(models_dir / f"{tag.replace('__', '_')}.pkl", "wb") as f:
                    pickle.dump({"model": best_obj, "scaler": scaler}, f)
            else:
                torch.save({
                    "state_dict": best_obj.state_dict(),
                    "model_name": model_name, "n_out": 1,
                    "in_shape":   list(X_tr.shape[1:]),
                    "hyperparams": best["hyperparams"],
                    "indicator":  ind_name,
                    "feature":    feat_name,
                }, models_dir / f"{tag.replace('__', '_')}.pt")

    print(f"\ndone — {done} trials.")


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
