"""HPO for the tabular models (RF / XGB / MLP) on every CFDAC variant.

Existing ``hpo.py`` only trains the 2-D CNN on the legacy `cfdac`.
``hpo_cfdac_variants.py`` adds Conv2DStack / Conv3DStack on the new
variant catalogue.  This script closes the matrix: every CFDAC variant
gets the same treatment as `modal` / `frf_mag` / `timeseries`, i.e.
Random Forest, XGBoost and an MLP fit on a flattened view of the
variant array.

Outputs (appended to the existing directories):
    results/hpo/<task>__<model>__<variant>.json   one per cell
    results/models/<task>_<model>_<variant>.pkl   for rf / xgb
    results/models/<task>_<model>_<variant>.pt    for mlp
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

from ml_pipeline.models import MLP                                       # noqa: E402
from ml_pipeline.tasks import build_targets                              # noqa: E402
from ml_pipeline.train import (                                           # noqa: E402
    load_labels, load_feature, make_split, SEED, _CFDAC_VARIANTS,
)

torch.set_num_threads(4)


GRIDS = {
    "rf":  {"n_estimators": [100, 300], "max_depth": [12, None]},
    "xgb": {"n_estimators": [100, 300], "max_depth": [4, 8]},
    "mlp": {"hidden": [(256, 128, 64), (512, 256, 128)],
            "lr":     [1e-3, 3e-3]},
}

# Order from smallest to largest so the 4-channel `cfdac_all`
# (the most RAM-hungry one) is processed last when other arrays
# have already been freed.  Skip both `cfdac` (legacy alias of
# `cfdac_realimag`) and every `cfdac3d_*` (would flatten to the
# same vector as its 2-D analogue and add nothing for tabular
# models).
VARIANTS = [
    "cfdac_real", "cfdac_imag", "cfdac_mag", "cfdac_phase",     # 1-ch
    "cfdac_realimag", "cfdac_magphase",                         # 2-ch
    "cfdac_all",                                                # 4-ch
]

MODELS_TABULAR = ("rf", "xgb", "mlp")


def _flatten(X: np.ndarray) -> np.ndarray:
    return X.reshape(len(X), -1).astype(np.float32, copy=False)


def _train_sklearn(model_name: str, kind: str, n_out: int,
                    params: dict,
                    X_tr, y_tr, X_va, y_va, X_te, y_te) -> Tuple[dict, object]:
    t0 = time.time()
    if model_name == "rf":
        cls = RandomForestClassifier if kind == "cls" else RandomForestRegressor
        mdl = cls(n_jobs=-1, random_state=SEED, **params)
    else:
        cls = XGBClassifier if kind == "cls" else XGBRegressor
        mdl = cls(n_jobs=-1, random_state=SEED,
                     learning_rate=0.1, **params)
    mdl.fit(X_tr, y_tr)
    if kind == "cls":
        pred_va = mdl.predict(X_va); pred_te = mdl.predict(X_te)
        val  = float(accuracy_score(y_va, pred_va))
        test = float(accuracy_score(y_te, pred_te))
        extras = {}
        mname = "accuracy"
    else:
        pred_va = mdl.predict(X_va); pred_te = mdl.predict(X_te)
        val  = float(r2_score(y_va, pred_va))
        test = float(r2_score(y_te, pred_te))
        extras = {"mae_test": float(mean_absolute_error(y_te, pred_te))}
        mname = "R2"
    return ({
        "hyperparams": params, "metric_name": mname,
        "metric_val": val, "metric_test": test, "extras": extras,
        "runtime_s": time.time() - t0,
    }, mdl)


def _train_mlp(kind: str, n_out: int, params: dict,
                X_tr_s, y_tr, X_va_s, y_va, X_te_s, y_te,
                epochs: int = 4) -> Tuple[dict, nn.Module]:
    """Train MLP on already-standardised inputs."""
    t0 = time.time()
    Xtr = torch.as_tensor(X_tr_s).float()
    Xva = torch.as_tensor(X_va_s).float()
    Xte = torch.as_tensor(X_te_s).float()
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
    mdl = MLP(in_dim=Xtr.shape[1], n_out=n_out,
                  hidden=tuple(params["hidden"]), dropout=0.2,
                  regression=(kind == "reg"))
    opt = torch.optim.AdamW(mdl.parameters(),
                                lr=float(params["lr"]),
                                weight_decay=1e-4)
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
            extras = {}
            mname = "accuracy"
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
            for m in MODELS_TABULAR:
                plan.append((tname, m, variant))
    grand_total = sum(int(np.prod([len(v) for v in GRIDS[m].values()]))
                        for _, m, _ in plan)
    print(f"plan: {len(plan)} cells, {grand_total} trials")

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
        X_tr = _flatten(X_full[idx_tr])
        X_va = _flatten(X_full[idx_va])
        X_te = _flatten(X_full[idx_te])
        # Tabular models on flat CFDAC consume *standardised*
        # inputs.  Transform in place to keep peak RAM bounded by
        # 3 × (n × dim × 4 bytes) instead of 6 ×.
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr); del X_tr
        X_va_s = scaler.transform(X_va); del X_va
        X_te_s = scaler.transform(X_te); del X_te
        gc.collect()

        n_out = (int(y_pool.max()) + 1) if kind == "cls" else 1
        grid  = GRIDS[model_name]
        keys  = list(grid.keys()); vals = [grid[k] for k in keys]
        trials: List[Dict[str, Any]] = []
        best_trial, best_obj = None, None
        for combo in itertools.product(*vals):
            params = dict(zip(keys, combo))
            try:
                if model_name in ("rf", "xgb"):
                    row, mdl = _train_sklearn(model_name, kind, n_out,
                                                   params,
                                                   X_tr_s, y_tr,
                                                   X_va_s, y_va,
                                                   X_te_s, y_te)
                else:
                    row, mdl = _train_mlp(kind, n_out, params,
                                              X_tr_s, y_tr,
                                              X_va_s, y_va,
                                              X_te_s, y_te,
                                              epochs=epochs)
            except Exception as e:
                print(f"  FAIL {tname}/{model_name}/{feat} {params}: {e}",
                          flush=True)
                continue
            trials.append(row); done += 1
            print(f"  [{done}/{grand_total}] {tname}/{model_name}/{feat} "
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
                "in_shape": list(X_tr.shape[1:]),
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
