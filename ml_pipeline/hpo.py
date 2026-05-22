"""Grid-search hyperparameter optimisation with response-surface logging.

For every ``(task, model, feature)`` cell we sweep the 2-D
hyperparameter grid declared in ``HPO_GRIDS`` and write every trial's
metric to ``results/hpo/<task>__<model>__<feature>.json``.  The best
trial's model is saved to ``results/models/<task>_<model>_<feature>.pkl``
or ``.pt`` (matching the train.py convention).

The grids are small enough to be exhaustive on CPU within a few
minutes per cell.  Sequence-input deep models (CNN, Transformer,
2-D CNN) use shorter HPO grids than the tabular ones.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier, XGBRegressor

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml_pipeline.models import MLP, Conv1DStack, SmallTransformer, Conv2DStack  # noqa: E402
from ml_pipeline.tasks import build_targets, TASK_DESCRIPTION  # noqa: E402
from ml_pipeline.train import (   # noqa: E402
    FEATURES_FLAT, FEATURES_SEQ, FEATURES_MAT,
    SK_MODELS, TORCH_FLAT, TORCH_SEQ, TORCH_MAT,
    load_labels, load_feature, make_split, SEED,
)

torch.set_num_threads(4)

# ── Hyperparameter grids ─────────────────────────────────────────────────────
HPO_GRIDS: Dict[str, Dict[str, list]] = {
    "rf": {
        "n_estimators": [100, 200, 300],
        "max_depth":    [6, 12, None],
    },
    "xgb": {
        "n_estimators": [100, 300, 600],
        "max_depth":    [4, 6, 8],
    },
    "mlp": {
        "hidden":  [(128, 64), (256, 128, 64), (512, 256, 128)],
        "lr":      [5e-4, 1e-3, 3e-3],
    },
    "cnn": {
        "widths":      [(16, 32, 64), (32, 64, 128)],
        "kernel_size": [5, 7],
    },
    "transformer": {
        "d_model":   [32, 48, 64],
        "n_layers":  [1, 2],
    },
    "cnn2d": {
        "widths":      [(8, 16, 32), (16, 32, 64)],
        "kernel_size": [3, 5],
    },
}


@dataclass
class TrialResult:
    task: str
    model: str
    feature: str
    hyperparams: Dict[str, Any]
    metric_name: str
    metric_val:  float
    metric_test: float
    extras:      Dict[str, float]
    runtime_s:   float


# ── Worker: train one trial ─────────────────────────────────────────────────
def _train_sklearn(model_name: str, kind: str, params: Dict[str, Any],
                    X_tr, y_tr, X_va, y_va, X_te, y_te) -> TrialResult:
    t0 = time.time()
    if kind == "cls":
        if model_name == "rf":
            mdl = RandomForestClassifier(n_jobs=-1, random_state=SEED,
                                            class_weight="balanced", **params)
        else:
            mdl = XGBClassifier(n_jobs=-1, random_state=SEED,
                                  use_label_encoder=False,
                                  eval_metric="mlogloss",
                                  learning_rate=0.1, **params)
    else:
        if model_name == "rf":
            mdl = RandomForestRegressor(n_jobs=-1, random_state=SEED, **params)
        else:
            mdl = XGBRegressor(n_jobs=-1, random_state=SEED,
                                 learning_rate=0.1, **params)
    mdl.fit(X_tr, y_tr)
    yhat_va = mdl.predict(X_va); yhat_te = mdl.predict(X_te)
    if kind == "cls":
        mv = float(accuracy_score(y_va, yhat_va))
        mt = float(accuracy_score(y_te, yhat_te))
        metric = "accuracy"; extras = {}
    else:
        mv = float(r2_score(y_va, yhat_va))
        mt = float(r2_score(y_te, yhat_te))
        metric = "R2"; extras = {"mae_test": float(mean_absolute_error(y_te, yhat_te))}
    return TrialResult(task="", model=model_name, feature="",
                          hyperparams=params, metric_name=metric,
                          metric_val=mv, metric_test=mt, extras=extras,
                          runtime_s=time.time() - t0), mdl


def _to_tensor(x: np.ndarray) -> torch.Tensor:
    t = torch.as_tensor(x)
    if t.ndim == 3:
        t = t.permute(0, 2, 1)
    return t.float()


def _train_torch(model_name: str, kind: str, n_out: int,
                  params: Dict[str, Any],
                  X_tr, y_tr, X_va, y_va, X_te, y_te,
                  epochs: int) -> Tuple[TrialResult, nn.Module]:
    t0 = time.time()
    # Deterministic per-trial seeding: torch was previously unseeded, so
    # every cnn/transformer/cnn2d cell was a single unreproducible draw.
    # Re-seeding here makes weight init + shuffling reproducible and turns
    # any A/B (e.g. plain vs augmented features) into a controlled
    # comparison where the only varying factor is the data.
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    # X_tr may be either a numpy/tensor array (eager) or a torch Dataset
    # (lazy, streaming from HDF5 row-by-row).  Detect and route.
    from torch.utils.data import Dataset as _TorchDataset
    streaming = isinstance(X_tr, _TorchDataset)
    if streaming:
        # Probe one sample for shape; the dataset already encodes y.
        x0, _ = X_tr[0]
        # x0 is already in tensor layout produced by LazyCFDACDataset
        Xtr_shape = (len(X_tr),) + tuple(x0.shape)
        ds_tr = X_tr
    else:
        Xtr = _to_tensor(X_tr)
        Xtr_shape = tuple(Xtr.shape)
        if kind == "cls":
            ytr = torch.as_tensor(y_tr).long()
        else:
            ytr = torch.as_tensor(y_tr).float().unsqueeze(1)
        ds_tr = TensorDataset(Xtr, ytr)
    Xva = _to_tensor(X_va); Xte = _to_tensor(X_te)
    if kind == "cls":
        yva = torch.as_tensor(y_va).long(); yte = torch.as_tensor(y_te).long()
        loss_fn = nn.CrossEntropyLoss()
    else:
        yva = torch.as_tensor(y_va).float().unsqueeze(1)
        yte = torch.as_tensor(y_te).float().unsqueeze(1)
        loss_fn = nn.MSELoss()

    seq = len(Xtr_shape) == 3; mat = len(Xtr_shape) == 4
    batch = 64
    if seq and Xtr_shape[2] >= 256:
        batch = 128
    if model_name == "mlp":
        in_dim = Xtr_shape[1] * Xtr_shape[2] if seq else Xtr_shape[1]
        model = MLP(in_dim=in_dim, n_out=n_out,
                     hidden=tuple(params["hidden"]), dropout=0.2,
                     regression=(kind == "reg"))
        lr = float(params["lr"])
    elif model_name == "cnn":
        model = Conv1DStack(n_channels=Xtr_shape[1], n_out=n_out,
                              widths=tuple(params["widths"]),
                              kernel_size=int(params["kernel_size"]),
                              regression=(kind == "reg"))
        lr = 1e-3
    elif model_name == "transformer":
        model = SmallTransformer(n_channels=Xtr_shape[1], n_out=n_out,
                                   d_model=int(params["d_model"]),
                                   n_layers=int(params["n_layers"]),
                                   regression=(kind == "reg"))
        lr = 1e-3
    elif model_name == "cnn2d":
        model = Conv2DStack(n_channels=Xtr_shape[1], n_out=n_out,
                              widths=tuple(params["widths"]),
                              kernel_size=int(params["kernel_size"]),
                              regression=(kind == "reg"))
        lr = 1e-3
    else:
        raise ValueError(model_name)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n_workers = 2 if streaming else 0
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True,
                       num_workers=n_workers,
                       persistent_workers=bool(n_workers),
                       generator=torch.Generator().manual_seed(SEED))

    best_metric = -np.inf
    best_state = None
    for ep in range(epochs):
        model.train()
        for xb, yb in dl_tr:
            opt.zero_grad()
            out = model(xb)
            loss_fn(out, yb).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            out_va = model(Xva).cpu()
            if kind == "cls":
                pred = out_va.argmax(1).numpy()
                metric = accuracy_score(y_va, pred)
            else:
                metric = r2_score(y_va, out_va.squeeze(1).numpy())
        if metric > best_metric:
            best_metric = metric
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_te = model(Xte).cpu()
        if kind == "cls":
            yhat_te = out_te.argmax(1).numpy()
            mt = float(accuracy_score(y_te, yhat_te))
            metric_name = "accuracy"; extras = {}
        else:
            yhat_te = out_te.squeeze(1).numpy()
            mt = float(r2_score(y_te, yhat_te))
            metric_name = "R2"
            extras = {"mae_test": float(mean_absolute_error(y_te, yhat_te))}
    return TrialResult(task="", model=model_name, feature="",
                          hyperparams=params, metric_name=metric_name,
                          metric_val=float(best_metric), metric_test=mt,
                          extras=extras, runtime_s=time.time() - t0), model


# ── Orchestrator ────────────────────────────────────────────────────────────
def run_hpo(features_path: Path, out_dir: Path, epochs: int = 4) -> None:
    hpo_dir    = out_dir / "hpo"
    models_dir = out_dir / "models"
    hpo_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(features_path)
    tasks  = build_targets(labels["type_code"], labels["storey"],
                            labels["end"], labels["severity"])

    # Per-cell lazy load (pymodal-style).  No global preload of any feature.
    # FEATURES_FLAT/SEQ: read only the rows this cell needs (tr ∪ va ∪ te).
    # FEATURES_MAT (cfdac, 7.5 GB): stream via LazyCFDACDataset for train
    # and bulk-read just val/test via batch_read().
    # Each cell is self-contained: a VM reboot mid-cell loses only that
    # cell, and finished cells are durable via best.json.
    from ml_pipeline.lazy_datasets import LazyCFDACDataset

    grand_total = 0
    grand_done  = 0
    plan: List[Tuple[str, str, str]] = []   # (task, model, feature)
    for task in tasks:
        for feat_name in (*FEATURES_FLAT, *FEATURES_SEQ, *FEATURES_MAT):
            if feat_name in FEATURES_FLAT:
                model_list = (*SK_MODELS, *TORCH_FLAT)
            elif feat_name in FEATURES_SEQ:
                model_list = TORCH_SEQ
            else:
                model_list = TORCH_MAT
            for m in model_list:
                grand_total += int(np.prod([len(v) for v in HPO_GRIDS[m].values()]))
                plan.append((task, m, feat_name))
    # Sort by feature *cost* so cheap cells finish first and we get early
    # durable wins.  cfdac is the slowest (gzip per-row chunks, 7.5 GB);
    # modal is tiny; frf_mag and timeseries are mid-sized SEQ features.
    _FEATURE_COST = {"modal": 0, "frf_mag": 1, "timeseries": 2, "cfdac": 3}
    plan.sort(key=lambda r: (_FEATURE_COST.get(r[2], 99), r[0], r[1]))
    print(f"HPO plan: {len(plan)} cells, {grand_total} total trials")

    for task_name, model_name, feat_name in plan:
        out_json = hpo_dir / f"{task_name}__{model_name}__{feat_name}.json"
        if out_json.exists():
            print(f"  skip {task_name}/{model_name}/{feat_name} (cached)")
            continue
        mask, y_all, kind = tasks[task_name]
        idx_pool = np.where(mask)[0]
        y_pool   = y_all
        idx_tr_local, idx_va_local, idx_te_local = make_split(y_pool, kind)
        idx_tr = idx_pool[idx_tr_local]; idx_va = idx_pool[idx_va_local]
        idx_te = idx_pool[idx_te_local]
        y_tr = y_pool[idx_tr_local]; y_va = y_pool[idx_va_local]
        y_te = y_pool[idx_te_local]

        scaler = None
        t_load = time.time()
        if feat_name in FEATURES_MAT:
            # Stream cfdac during training; bulk-read only val/test.
            ds_tr = LazyCFDACDataset(features_path, feat_name,
                                      rows=idx_tr, labels=y_tr, kind=kind,
                                      reshape="conv2d")
            X_tr = ds_tr  # marker — _train_torch detects Dataset
            X_va = ds_tr.batch_read(idx_va)
            X_te = ds_tr.batch_read(idx_te)
            print(f"  load(cfdac, va+te only)={time.time()-t_load:.1f}s "
                  f"tr=streaming({len(idx_tr)} rows) "
                  f"va={X_va.shape} te={X_te.shape}", flush=True)
        else:
            # Per-cell subset read for FEATURES_FLAT and FEATURES_SEQ.
            rows_all = np.concatenate([idx_tr, idx_va, idx_te])
            X_all = load_feature(features_path, feat_name, rows=rows_all)
            n_tr, n_va = len(idx_tr), len(idx_va)
            X_tr = X_all[:n_tr]
            X_va = X_all[n_tr:n_tr + n_va]
            X_te = X_all[n_tr + n_va:]
            del X_all
            if feat_name in FEATURES_FLAT:
                scaler = StandardScaler().fit(X_tr.reshape(len(X_tr), -1))
                X_tr = scaler.transform(X_tr.reshape(len(X_tr), -1))
                X_va = scaler.transform(X_va.reshape(len(X_va), -1))
                X_te = scaler.transform(X_te.reshape(len(X_te), -1))
            print(f"  load({feat_name})={time.time()-t_load:.1f}s "
                  f"tr={X_tr.shape} va={X_va.shape} te={X_te.shape}",
                  flush=True)

        n_out = (int(y_pool.max()) + 1) if kind == "cls" else 1
        grid  = HPO_GRIDS[model_name]
        keys  = list(grid.keys()); vals = [grid[k] for k in keys]
        trial_rows: list[Dict[str, Any]] = []
        best_trial: TrialResult | None = None
        best_obj   = None
        for combo in itertools.product(*vals):
            params = dict(zip(keys, combo))
            tag = f"{task_name}/{model_name}/{feat_name}  {params}"
            try:
                if model_name in SK_MODELS:
                    res, obj = _train_sklearn(model_name, kind, params,
                                                  X_tr, y_tr, X_va, y_va, X_te, y_te)
                else:
                    res, obj = _train_torch(model_name, kind, n_out, params,
                                                X_tr, y_tr, X_va, y_va, X_te, y_te,
                                                epochs=epochs)
            except Exception as e:
                print(f"  FAIL {tag}: {e}")
                continue
            res.task = task_name; res.feature = feat_name
            trial_rows.append({
                "hyperparams": {k: (list(v) if isinstance(v, tuple) else v)
                                  for k, v in params.items()},
                "metric_name": res.metric_name,
                "metric_val":  res.metric_val,
                "metric_test": res.metric_test,
                "extras":      res.extras,
                "runtime_s":   res.runtime_s,
            })
            grand_done += 1
            print(f"  [{grand_done}/{grand_total}] {tag}  "
                    f"val={res.metric_val:.4f}  test={res.metric_test:.4f}  "
                    f"({res.runtime_s:.1f}s)")
            if best_trial is None or res.metric_val > best_trial.metric_val:
                best_trial = res
                best_obj   = obj

        # Save trials.
        out_json = hpo_dir / f"{task_name}__{model_name}__{feat_name}.json"
        out_json.write_text(json.dumps({
            "task":  task_name, "model": model_name, "feature": feat_name,
            "best_hyperparams": best_trial.hyperparams if best_trial else None,
            "best_metric_val":  best_trial.metric_val  if best_trial else None,
            "best_metric_test": best_trial.metric_test if best_trial else None,
            "trials": trial_rows,
        }, indent=2, default=str))
        # Save best model.
        if best_trial is not None and best_obj is not None:
            tag = f"{task_name}_{model_name}_{feat_name}"
            if model_name in SK_MODELS:
                with open(models_dir / f"{tag}.pkl", "wb") as f:
                    pickle.dump({"model": best_obj, "scaler": scaler}, f)
            else:
                if hasattr(X_tr, "shape"):
                    in_shape = list(X_tr.shape[1:])
                else:
                    # X_tr is a LazyCFDACDataset; probe a sample.
                    in_shape = list(X_tr[0][0].shape)
                torch.save({
                    "state_dict": best_obj.state_dict(),
                    "model_name": model_name,
                    "n_out": n_out,
                    "in_shape": in_shape,
                    "hyperparams": best_trial.hyperparams,
                    # P1.1: load_feature() applies per-sample normalisation
                    # by default; flag the artefact so eval feeds the
                    # matching input distribution.
                    "input_normalized": True,
                }, models_dir / f"{tag}.pt")

    print("\nHPO complete.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path,
                          default=_REPO / "dataset" / "features.h5")
    parser.add_argument("--out", type=Path, default=_REPO / "results")
    parser.add_argument("--epochs", type=int, default=4,
                          help="Per-trial epochs for Torch models.")
    parser.add_argument("--seed", type=int, default=SEED,
                          help="Override the global seed (split + torch + "
                               "sklearn) for multi-seed variance runs.")
    args = parser.parse_args()
    # Multi-seed support: rebind the seed everywhere it is read — this
    # module's global (sklearn random_state + _train_torch.manual_seed) and
    # train.SEED (make_split's random_state).
    import ml_pipeline.train as _train
    globals()["SEED"] = args.seed
    _train.SEED = args.seed
    run_hpo(args.features, args.out, epochs=args.epochs)


if __name__ == "__main__":
    main()
