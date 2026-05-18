"""Train every (model, feature, task) combination on the synthetic dataset.

For each (model, feature) supported by that architecture we train all 5
tasks, evaluate on a stratified hold-out split, and append a row to
``results/training_metrics.json``.  Trained model artefacts go to
``results/models/<task>_<model>_<feature>.{pkl,pt}``.

Model x feature compatibility
-----------------------------
  RandomForest, XGBoost : `modal`, `indicators`                  (flat features)
  MLP                   : `modal`, `indicators`                  (flat features)
  Conv1D                : `frf_mag`, `timeseries`                (1-D sequences)
  Transformer           : `frf_mag`, `timeseries`                (1-D sequences)

5 tasks  x  models (5)  x  features = many combos; total runtime ~minutes
on CPU because the models are tiny and the dataset is small.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, r2_score,
)
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier, XGBRegressor

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml_pipeline.models import MLP, Conv1DStack, SmallTransformer, Conv2DStack  # noqa: E402
from ml_pipeline.tasks import (   # noqa: E402
    build_targets, TASK_DESCRIPTION, TASK_N_CLASSES,
)


FEATURES_FLAT  = ("modal",)
# P0.4: `timeseries` on experimental data is synthesised from FRF via
# H(f)*F(f) -> IFFT (see evaluate.py:synthesize_timeseries), so it
# carries no independent information beyond `frf_mag` on the exp side.
# Training a separate timeseries cell is therefore double-counting on
# any cross-domain test.  Keep FEATURES_SEQ_ALL = legacy enumeration
# for back-compat with existing artefacts; FEATURES_SEQ (the active
# training list) drops `timeseries`.
FEATURES_SEQ_ALL = ("frf_mag", "timeseries")
FEATURES_SEQ     = ("frf_mag",)
FEATURES_MAT     = ("cfdac",)        # 2-D matricial features
SK_MODELS      = ("rf", "xgb")
TORCH_FLAT     = ("mlp",)
TORCH_SEQ      = ("cnn", "transformer")
TORCH_MAT      = ("cnn2d",)

DEVICE = torch.device("cpu")
SEED   = 20260511

# Cap PyTorch BLAS to 4 threads — beyond that we get scheduler contention
# on this CPU and per-epoch time goes up significantly.
torch.set_num_threads(4)


@dataclass
class Result:
    task: str
    model: str
    feature: str
    n_train: int
    n_val: int
    n_test: int
    metric_name: str
    metric_val: float
    metric_test: float
    runtime_s: float
    extra: dict


# ── data loader ─────────────────────────────────────────────────────────────
def load_labels(features_path: Path) -> dict[str, np.ndarray]:
    """Read labels.  Supports two on-disk schemas: the original
    synth ``features.h5`` keeps labels under a ``labels/`` group;
    the experimental files (and the balanced subsets produced by
    ``rebalance_datasets.py``) store them at the top level."""
    with h5py.File(features_path, "r") as f:
        prefix = "labels/" if "labels" in f else ""
        sid = f[f"{prefix}sample_id"][:]
        sid = sid.astype(np.int64) if sid.dtype.kind != "O" else sid
        return {
            "type_code": f[f"{prefix}type_code"][:].astype(np.int64),
            "storey":    f[f"{prefix}storey"][:].astype(np.int64),
            "end":       f[f"{prefix}end"][:].astype(np.int64),
            "severity":  f[f"{prefix}severity"][:].astype(np.float32),
            "sample_id": sid,
        }


_CFDAC_VARIANTS = {
    # 2-D inputs (n, C, H, W) for Conv2d
    "cfdac_real":     (("cfdac_real",),                          "stack2d"),
    "cfdac_imag":     (("cfdac_imag",),                          "stack2d"),
    "cfdac_mag":      (("cfdac_mag",),                           "stack2d"),
    "cfdac_phase":    (("cfdac_phase",),                         "stack2d"),
    "cfdac_realimag": (("cfdac_real", "cfdac_imag"),             "stack2d"),
    "cfdac_magphase": (("cfdac_mag", "cfdac_phase"),             "stack2d"),
    "cfdac_all":      (("cfdac_real", "cfdac_imag",
                          "cfdac_mag",  "cfdac_phase"),            "stack2d"),
    # 3-D inputs (n, 1, D, H, W) for Conv3d
    "cfdac3d_realimag": (("cfdac_real", "cfdac_imag"),                  "stack3d"),
    "cfdac3d_magphase": (("cfdac_mag",  "cfdac_phase"),                 "stack3d"),
    "cfdac3d_all":      (("cfdac_real", "cfdac_imag",
                            "cfdac_mag",  "cfdac_phase"),                "stack3d"),
    # legacy alias kept for backward-compat with the early HPO log file
    "cfdac":          (("cfdac_real", "cfdac_imag"),             "stack2d"),
}


def _per_sample_normalize(name: str, X: np.ndarray) -> np.ndarray:
    """P1.1: per-sample, deterministic input normalisation.

    Synth and exp pipelines call this through the *same* function so
    that training and cross-domain inference see identical input
    statistics regardless of the absolute amplitude in each domain.
    Modal and indicators are left alone -- they go through a
    StandardScaler fitted per-fold by the caller.
    """
    if name == "frf_mag":
        # (n, n_freq, 9) -> log + per-sample z-score over (freq, channel)
        Xlog = np.log10(np.clip(X.astype(np.float32), 1e-8, None))
        mu  = Xlog.mean(axis=(-2, -1), keepdims=True)
        sig = Xlog.std (axis=(-2, -1), keepdims=True) + 1e-6
        return ((Xlog - mu) / sig).astype(np.float32)
    if name in ("frf_real", "frf_imag"):
        m = np.maximum(np.abs(X).max(axis=(-2, -1), keepdims=True), 1e-12)
        return (X / m).astype(np.float32)
    if name in ("cfdac_real", "cfdac_imag"):
        mu = X.mean(axis=(-2, -1), keepdims=True)
        return (X - mu).astype(np.float32)
    if name == "cfdac_mag":
        Xs = 2.0 * X.astype(np.float32) - 1.0
        mu = Xs.mean(axis=(-2, -1), keepdims=True)
        return (Xs - mu).astype(np.float32)
    if name == "cfdac_phase":
        return (X.astype(np.float32) / np.float32(np.pi))
    if name == "timeseries":
        mu  = X.mean(axis=-2, keepdims=True)
        sig = X.std (axis=-2, keepdims=True) + 1e-6
        return ((X - mu) / sig).astype(np.float32)
    return X


def load_feature(features_path: Path, name: str,
                  rows: np.ndarray | None = None,
                  normalize: bool = True) -> np.ndarray:
    """Load a feature array.

    For CFDAC variants, the on-disk arrays are individual H × W planes;
    this function stacks them into the right tensor shape:

      * ``stack2d`` →  (n, C, H, W)  for Conv2d
      * ``stack3d`` →  (n, 1, D, H, W)  for Conv3d

    With ``normalize=True`` (default) each *part* is run through
    ``_per_sample_normalize`` before stacking, and primitive features
    (frf_mag, timeseries, frf_real/imag) are normalised on the way
    out.  Modal and indicators are passed through as-is for a
    fold-fitted StandardScaler downstream.
    """
    if name in _CFDAC_VARIANTS:
        parts, mode = _CFDAC_VARIANTS[name]
        layers = []
        with h5py.File(features_path, "r") as f:
            for p in parts:
                arr = f[p][:] if rows is None else f[p][rows]
                if normalize:
                    arr = _per_sample_normalize(p, arr)
                layers.append(arr)
        if mode == "stack2d":
            return np.stack(layers, axis=1)
        # stack3d → add a singleton channel dim then stack along depth
        return np.stack(layers, axis=1)[:, np.newaxis, ...]
    with h5py.File(features_path, "r") as f:
        if rows is None:
            data = f[name][:]
        else:
            order = np.argsort(rows)
            tmp = f[name][rows[order]]
            data = np.empty_like(tmp)
            data[order] = tmp
    if normalize:
        data = _per_sample_normalize(name, data)
    return data


# ── splits ──────────────────────────────────────────────────────────────────
def make_split(y: np.ndarray, kind: str,
                test_frac: float = 0.15,
                val_frac: float = 0.15,
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(idx_train, idx_val, idx_test)`` arrays into ``y``.

    Classification uses stratified splits; regression uses random shuffles.
    """
    n = len(y)
    idx_all = np.arange(n)
    if kind == "cls":
        idx_trv, idx_test = train_test_split(
            idx_all, test_size=test_frac, stratify=y, random_state=SEED)
        idx_train, idx_val = train_test_split(
            idx_trv, test_size=val_frac / (1.0 - test_frac),
            stratify=y[idx_trv], random_state=SEED)
    else:
        idx_trv, idx_test = train_test_split(
            idx_all, test_size=test_frac, random_state=SEED)
        idx_train, idx_val = train_test_split(
            idx_trv, test_size=val_frac / (1.0 - test_frac),
            random_state=SEED)
    return idx_train, idx_val, idx_test


# ── sklearn training ────────────────────────────────────────────────────────
def train_sklearn(model_name: str, kind: str,
                   X_tr, y_tr, X_va, y_va, X_te, y_te) -> Dict:
    t0 = time.time()
    if kind == "cls":
        if model_name == "rf":
            mdl = RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                            random_state=SEED, class_weight="balanced")
        else:
            mdl = XGBClassifier(n_estimators=300, max_depth=6,
                                  learning_rate=0.1, n_jobs=-1,
                                  random_state=SEED, use_label_encoder=False,
                                  eval_metric="mlogloss")
        mdl.fit(X_tr, y_tr)
        yhat_va = mdl.predict(X_va)
        yhat_te = mdl.predict(X_te)
        return {
            "model": mdl,
            "metric_name": "accuracy",
            "metric_val":  float(accuracy_score(y_va, yhat_va)),
            "metric_test": float(accuracy_score(y_te, yhat_te)),
            "yhat_test":   yhat_te,
            "runtime_s":   time.time() - t0,
        }
    else:
        if model_name == "rf":
            mdl = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED)
        else:
            mdl = XGBRegressor(n_estimators=300, max_depth=6,
                                 learning_rate=0.1, n_jobs=-1, random_state=SEED)
        mdl.fit(X_tr, y_tr)
        yhat_va = mdl.predict(X_va)
        yhat_te = mdl.predict(X_te)
        return {
            "model": mdl,
            "metric_name": "R2",
            "metric_val":  float(r2_score(y_va, yhat_va)),
            "metric_test": float(r2_score(y_te, yhat_te)),
            "yhat_test":   yhat_te,
            "mae_test":    float(mean_absolute_error(y_te, yhat_te)),
            "runtime_s":   time.time() - t0,
        }


# ── torch training ──────────────────────────────────────────────────────────
def _to_tensor(x: np.ndarray, seq: bool) -> torch.Tensor:
    t = torch.as_tensor(x)
    if seq and t.ndim == 3:
        # (N, L, C) -> (N, C, L)
        t = t.permute(0, 2, 1)
    # 4-D inputs (N, C, H, W) — CFDAC — pass through unchanged.
    return t.float()


def train_torch(model_name: str, kind: str, n_out: int,
                 X_tr, y_tr, X_va, y_va, X_te, y_te,
                 epochs: int = 12, batch: int = 64,
                 lr: float = 1e-3,
                 init_from: Path | None = None,
                 feature: str | None = None) -> Dict:
    t0 = time.time()
    seq = X_tr.ndim == 3
    is_mat = X_tr.ndim == 4
    # Larger batches for sequence inputs to amortise per-iter Python overhead.
    if seq and X_tr.shape[1] >= 256:
        batch = 128
    if is_mat:
        batch = 64
    Xtr = _to_tensor(X_tr, seq); Xva = _to_tensor(X_va, seq); Xte = _to_tensor(X_te, seq)

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
        if seq:
            in_dim = Xtr.shape[1] * Xtr.shape[2]
        else:
            in_dim = Xtr.shape[1]
        model = MLP(in_dim=in_dim, n_out=n_out,
                     hidden=(256, 128, 64), dropout=0.2,
                     regression=(kind == "reg"))
    elif model_name == "cnn":
        model = Conv1DStack(n_channels=Xtr.shape[1], n_out=n_out,
                              regression=(kind == "reg"))
    elif model_name == "transformer":
        model = SmallTransformer(n_channels=Xtr.shape[1], n_out=n_out,
                                   regression=(kind == "reg"))
    elif model_name == "cnn2d":
        model = Conv2DStack(n_channels=Xtr.shape[1], n_out=n_out,
                              regression=(kind == "reg"))
    else:
        raise ValueError(model_name)

    # P2.3: optionally warm-start the backbone from an SSL-pretrained
    # checkpoint produced by pretrain_ssl.py.  We only load the
    # `backbone_state_dict` parameters whose names also exist on this
    # task model; unmatched keys (the SSL projection head) are dropped.
    if init_from is not None and feature is not None:
        ssl_path = Path(init_from) / f"{model_name}_{feature}_ssl.pt"
        if ssl_path.exists():
            ssl_blob = torch.load(ssl_path, map_location="cpu",
                                       weights_only=False)
            ssl_state = ssl_blob.get("backbone_state_dict", {})
            own_state = model.state_dict()
            matched = {k: v for k, v in ssl_state.items()
                          if k in own_state and v.shape == own_state[k].shape}
            own_state.update(matched)
            model.load_state_dict(own_state)
            print(f"    SSL warm-start: {len(matched)}/{len(ssl_state)} "
                      f"keys from {ssl_path.name}", flush=True)
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ds_tr = TensorDataset(Xtr, ytr)
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True, num_workers=0)

    best_metric = -np.inf if kind == "cls" else -np.inf  # higher = better
    best_state = None
    for ep in range(epochs):
        model.train()
        for xb, yb in dl_tr:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
        sched.step()
        # validation
        model.eval()
        with torch.no_grad():
            out_va = model(Xva.to(DEVICE)).cpu()
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
        out_te = model(Xte.to(DEVICE)).cpu()
        if kind == "cls":
            yhat = out_te.argmax(1).numpy()
            test_metric = accuracy_score(y_te, yhat)
            metric_name = "accuracy"
            extras = {}
        else:
            yhat = out_te.squeeze(1).numpy()
            test_metric = r2_score(y_te, yhat)
            metric_name = "R2"
            extras = {"mae_test": float(mean_absolute_error(y_te, yhat))}
    return {
        "model": model,
        "metric_name": metric_name,
        "metric_val":  float(best_metric),
        "metric_test": float(test_metric),
        "yhat_test":   yhat,
        "runtime_s":   time.time() - t0,
        **extras,
    }


# ── orchestrator ────────────────────────────────────────────────────────────
def run(features_path: Path, out_dir: Path,
        epochs: int = 12,
        init_from: Path | None = None) -> List[Result]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    labels = load_labels(features_path)
    tasks  = build_targets(labels["type_code"], labels["storey"],
                            labels["end"], labels["severity"])

    # Pre-load each feature representation once (still small enough at 339 MB total).
    feats: Dict[str, np.ndarray] = {}
    for name in (*FEATURES_FLAT, *FEATURES_SEQ, *FEATURES_MAT):
        print(f"loading feature: {name}")
        feats[name] = load_feature(features_path, name)

    results: List[Result] = []
    for task_name, (mask, y_all, kind) in tasks.items():
        idx_pool = np.where(mask)[0]
        y_pool   = y_all                # already restricted to mask
        if kind == "cls":
            n_classes = TASK_N_CLASSES[task_name]
            n_out = n_classes
        else:
            n_out = 1
        print(f"\n=== TASK {task_name}: {TASK_DESCRIPTION[task_name]}  "
              f"(n={len(idx_pool)}, kind={kind}) ===")

        idx_tr_local, idx_va_local, idx_te_local = make_split(y_pool, kind)
        # Map back to global indices via idx_pool.
        idx_tr = idx_pool[idx_tr_local]
        idx_va = idx_pool[idx_va_local]
        idx_te = idx_pool[idx_te_local]
        y_tr = y_pool[idx_tr_local]
        y_va = y_pool[idx_va_local]
        y_te = y_pool[idx_te_local]

        for feat_name in (*FEATURES_FLAT, *FEATURES_SEQ, *FEATURES_MAT):
            X = feats[feat_name]
            X_tr = X[idx_tr]; X_va = X[idx_va]; X_te = X[idx_te]

            # Flat features get standard-scaled for the linear / NN methods.
            if feat_name in FEATURES_FLAT:
                scaler = StandardScaler().fit(X_tr.reshape(len(X_tr), -1))
                X_tr_s = scaler.transform(X_tr.reshape(len(X_tr), -1))
                X_va_s = scaler.transform(X_va.reshape(len(X_va), -1))
                X_te_s = scaler.transform(X_te.reshape(len(X_te), -1))
            else:
                # Sequences are not standard-scaled.
                scaler = None

            # Compose model list per feature type.
            if feat_name in FEATURES_FLAT:
                model_list = (*SK_MODELS, "mlp")
                flat_X = (X_tr_s, X_va_s, X_te_s)
            elif feat_name in FEATURES_SEQ:
                model_list = TORCH_SEQ
                flat_X = (X_tr, X_va, X_te)
            else:  # FEATURES_MAT
                model_list = TORCH_MAT
                flat_X = (X_tr, X_va, X_te)

            for model_name in model_list:
                tag = f"{task_name}_{model_name}_{feat_name}"
                print(f"  → {tag}", end="  ", flush=True)
                if model_name in SK_MODELS:
                    out = train_sklearn(model_name, kind,
                                          flat_X[0], y_tr, flat_X[1], y_va,
                                          flat_X[2], y_te)
                    art_path = out_dir / "models" / f"{tag}.pkl"
                    with open(art_path, "wb") as f:
                        pickle.dump({"model": out["model"], "scaler": scaler}, f)
                else:
                    out = train_torch(model_name, kind, n_out,
                                       flat_X[0], y_tr, flat_X[1], y_va,
                                       flat_X[2], y_te, epochs=epochs,
                                       init_from=init_from,
                                       feature=feat_name)
                    art_path = out_dir / "models" / f"{tag}.pt"
                    torch.save({"state_dict": out["model"].state_dict(),
                                 "model_name": model_name,
                                 "n_out": n_out,
                                 "in_shape": list(X_tr.shape[1:]),
                                 # P1.1: artefacts produced after the
                                 # per-sample-normalisation roll-in
                                 # carry this flag so eval routines
                                 # know to feed normalised inputs.
                                 "input_normalized": True}, art_path)
                print(f"  val={out['metric_val']:.4f}  test={out['metric_test']:.4f}  "
                      f"({out['runtime_s']:.1f}s)")
                extra = {k: out[k] for k in out
                          if k in {"mae_test"}}
                results.append(Result(
                    task=task_name, model=model_name, feature=feat_name,
                    n_train=len(idx_tr), n_val=len(idx_va), n_test=len(idx_te),
                    metric_name=out["metric_name"],
                    metric_val=out["metric_val"], metric_test=out["metric_test"],
                    runtime_s=out["runtime_s"], extra=extra,
                ))

    (out_dir / "training_metrics.json").write_text(
        json.dumps([asdict(r) for r in results], indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path,
                          default=_REPO / "dataset" / "features.h5")
    parser.add_argument("--out", type=Path, default=_REPO / "results")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--init-from", type=Path, default=None,
                          help="P2.3: directory of SSL-pretrained "
                                  "backbones from pretrain_ssl.py to warm-"
                                  "start training (expects files named "
                                  "<model>_<feature>_ssl.pt).")
    args = parser.parse_args()
    results = run(args.features, args.out, epochs=args.epochs,
                      init_from=args.init_from)

    # Summary table.
    print("\n========= SUMMARY =========")
    print(f"{'task':<14s} {'model':<12s} {'feature':<12s} "
          f"{'val':>8s} {'test':>8s} {'time':>6s}")
    for r in results:
        print(f"{r.task:<14s} {r.model:<12s} {r.feature:<12s} "
              f"{r.metric_val:>8.4f} {r.metric_test:>8.4f} {r.runtime_s:>6.1f}")


if __name__ == "__main__":
    main()
