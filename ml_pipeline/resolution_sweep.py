"""Train every model on 5 feature-resolution levels (1.00 → 0.50).

For each ratio in [1.0, 0.875, 0.75, 0.625, 0.50] we resample each
feature axis to that resolution before training a fresh copy of every
``(model, feature)`` cell that has a matching synth-trained artefact.

Resampling rules (Fourier-resample = decimation + low-pass; see
``scipy.signal.resample``):

  modal       : ratio applied to the 81-d feature vector
                (81 → 71, 61, 51, 41).
  frf_mag     : ratio applied to the frequency axis (381 → 333, 286,
                238, 191) keeping all 9 channels.
  timeseries  : ratio applied to the time axis (1024 → 896, 768, 640,
                512).
  cfdac_*     : ratio applied to both spatial axes (128 → 112, 96, 80,
                64).

Outputs:
    results/resolution_sweep.json   one row per
                                      (task, model, feature, ratio)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import torch
from scipy.signal import resample
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

from ml_pipeline.models import (                                          # noqa: E402
    MLP, Conv1DStack, SmallTransformer, Conv2DStack, Conv3DStack,
)
from ml_pipeline.tasks import build_targets                               # noqa: E402
from ml_pipeline.train import (                                            # noqa: E402
    load_labels, load_feature, make_split, SEED, _CFDAC_VARIANTS,
)
from ml_pipeline.lazy_datasets import LazyCFDACDataset, _CFDAC_PARTS    # noqa: E402

RATIOS = (1.000, 0.875, 0.750, 0.625, 0.500)

# Multi-channel CFDAC variants don't fit in the 16 GB sandbox at full
# resolution: tr/va/te raw cache + the per-ratio downsampled copy peak
# around 16-32 GB.  Skip them here (same set excluded from HPO step 3).
SKIP_MULTICH_VARIANTS = {
    "cfdac",  # legacy alias for cfdac_realimag (2-ch) in lazy_datasets._CFDAC_PARTS
    "cfdac_realimag", "cfdac_magphase", "cfdac_all",
    "cfdac3d_realimag", "cfdac3d_magphase", "cfdac3d_all",
}
torch.set_num_threads(4)


def _resample_axis(x: np.ndarray, axis: int, n_new: int) -> np.ndarray:
    """Fourier-resample along ``axis`` to ``n_new`` points."""
    if x.shape[axis] == n_new:
        return x
    return resample(x, n_new, axis=axis).astype(np.float32, copy=False)


def _apply_ratio(name: str, X: np.ndarray, ratio: float) -> np.ndarray:
    if ratio >= 0.999:
        return X
    if name == "modal":
        n_new = max(8, int(round(X.shape[-1] * ratio)))
        return _resample_axis(X, axis=-1, n_new=n_new)
    if name in ("frf_mag", "frf_imag", "frf_real"):
        # (n, N_f, 9): resample N_f.
        n_new = max(8, int(round(X.shape[1] * ratio)))
        return _resample_axis(X, axis=1, n_new=n_new)
    if name == "timeseries":
        # (n, N_t, 9): resample N_t.
        n_new = max(8, int(round(X.shape[1] * ratio)))
        return _resample_axis(X, axis=1, n_new=n_new)
    if name in _CFDAC_VARIANTS:
        # (n, C, H, W) or (n, 1, D, H, W): resample H and W jointly.
        H = X.shape[-2]; W = X.shape[-1]
        h_new = max(8, int(round(H * ratio)))
        w_new = max(8, int(round(W * ratio)))
        X = _resample_axis(X, axis=-2, n_new=h_new)
        X = _resample_axis(X, axis=-1, n_new=w_new)
        return X
    # indicators (22-d) — already too small to subsample meaningfully
    n_new = max(4, int(round(X.shape[-1] * ratio)))
    return _resample_axis(X, axis=-1, n_new=n_new)


# ── training adapters (mirror hpo_cfdac_allmodels) ──────────────────
def _train_sklearn(model_name: str, kind: str, params: dict,
                    X_tr, y_tr, X_va, y_va, X_te, y_te) -> dict:
    t0 = time.time()
    cls = (RandomForestClassifier if (model_name == "rf" and kind == "cls")
              else RandomForestRegressor if model_name == "rf"
              else XGBClassifier if kind == "cls"
              else XGBRegressor)
    if model_name == "rf":
        mdl = cls(n_jobs=1, random_state=SEED, **params)
    else:
        mdl = cls(n_jobs=1, random_state=SEED, learning_rate=0.1,
                     tree_method="hist", max_bin=128, **params)
    mdl.fit(X_tr, y_tr)
    pred_te = mdl.predict(X_te)
    if kind == "cls":
        return {"metric": "accuracy",
                "value": float(accuracy_score(y_te, pred_te)),
                "runtime_s": time.time() - t0}
    return {"metric": "R2",
            "value": float(r2_score(y_te, pred_te)),
            "mae": float(mean_absolute_error(y_te, pred_te)),
            "runtime_s": time.time() - t0}


def _train_torch(model_name: str, kind: str, n_out: int, params: dict,
                  X_tr, y_tr, X_va, y_va, X_te, y_te,
                  epochs: int = 4) -> dict:
    t0 = time.time()
    Xtr = torch.as_tensor(X_tr).float()
    Xva = torch.as_tensor(X_va).float()
    Xte = torch.as_tensor(X_te).float()
    if kind == "cls":
        ytr = torch.as_tensor(y_tr).long(); loss_fn = nn.CrossEntropyLoss()
    else:
        ytr = torch.as_tensor(y_tr).float().unsqueeze(1)
        loss_fn = nn.MSELoss()
    if model_name == "mlp":
        in_dim = Xtr.shape[1] if Xtr.ndim == 2 else int(np.prod(Xtr.shape[1:]))
        if Xtr.ndim != 2:
            Xtr = Xtr.flatten(1); Xva = Xva.flatten(1); Xte = Xte.flatten(1)
        mdl = MLP(in_dim=in_dim, n_out=n_out,
                      hidden=tuple(params.get("hidden", (256, 128, 64))),
                      regression=(kind == "reg"))
    elif model_name == "cnn":
        ch = Xtr.shape[1]
        mdl = Conv1DStack(n_channels=ch, n_out=n_out,
                              widths=tuple(params.get("widths", (32, 64, 128))),
                              kernel_size=int(params.get("kernel_size", 7)),
                              regression=(kind == "reg"))
    elif model_name == "transformer":
        ch = Xtr.shape[1]
        mdl = SmallTransformer(n_channels=ch, n_out=n_out,
                                    d_model=int(params.get("d_model", 48)),
                                    n_layers=int(params.get("n_layers", 2)),
                                    regression=(kind == "reg"))
    elif model_name == "cnn2d":
        mdl = Conv2DStack(n_channels=Xtr.shape[1], n_out=n_out,
                              widths=tuple(params.get("widths", (16, 32, 64))),
                              kernel_size=int(params.get("kernel_size", 5)),
                              regression=(kind == "reg"))
    elif model_name == "cnn3d":
        depth = Xtr.shape[2]
        mdl = Conv3DStack(depth=depth, n_out=n_out,
                              widths=tuple(params.get("widths", (8, 16, 32))),
                              kernel_size=int(params.get("kernel_size", 3)),
                              regression=(kind == "reg"))
    else:
        raise ValueError(model_name)
    opt = torch.optim.AdamW(mdl.parameters(), lr=1e-3, weight_decay=1e-4)
    dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)
    for _ in range(epochs):
        mdl.train()
        for xb, yb in dl:
            opt.zero_grad(); loss_fn(mdl(xb), yb).backward(); opt.step()
    mdl.eval()
    with torch.no_grad():
        out = mdl(Xte).cpu().numpy()
    if kind == "cls":
        return {"metric": "accuracy",
                "value": float(accuracy_score(y_te, out.argmax(1))),
                "runtime_s": time.time() - t0}
    pred = out.squeeze(1)
    return {"metric": "R2",
            "value": float(r2_score(y_te, pred)),
            "mae": float(mean_absolute_error(y_te, pred)),
            "runtime_s": time.time() - t0}


def _parse_tag(tag: str, tasks: Dict) -> Tuple[str | None, str | None, str | None]:
    cat = (*_CFDAC_VARIANTS.keys(), "modal", "indicators", "frf_mag", "timeseries")
    for task in tasks:
        prefix = task + "_"
        if not tag.startswith(prefix):
            continue
        rest = tag[len(prefix):]
        for feat in cat:
            suffix = "_" + feat
            if rest.endswith(suffix):
                return task, rest[:-len(suffix)], feat
    return None, None, None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path,
                      default=_REPO / "dataset" / "features.h5")
    p.add_argument("--out", type=Path, default=_REPO / "results")
    p.add_argument("--epochs", type=int, default=4)
    args = p.parse_args()
    out_path = args.out / "resolution_sweep.json"

    labels = load_labels(args.features)
    tasks = build_targets(labels["type_code"], labels["storey"],
                              labels["end"], labels["severity"])

    # Inventory: every cell present in results/hpo/ (so the sweep
    # mirrors the same model menu).
    cells: List[Tuple[str, str, str]] = []
    seen = set()
    for hpo_path in sorted((args.out / "hpo").glob("*.json")):
        blob = json.loads(hpo_path.read_text())
        key = (blob["task"], blob["model"], blob["feature"])
        if blob["feature"] in SKIP_MULTICH_VARIANTS:
            continue
        if key not in seen:
            seen.add(key); cells.append(key)
    print(f"plan: {len(cells)} cells × {len(RATIOS)} ratios = "
              f"{len(cells)*len(RATIOS)} retrains", flush=True)

    # Resume guard: a results/resolution_sweep.json may already exist
    # from a previous partial run.
    existing = []
    if out_path.exists():
        existing = json.loads(out_path.read_text())
    done = {(r["task"], r["model"], r["feature"], r["ratio"]): True
                for r in existing}

    rows: List[Dict[str, Any]] = list(existing)
    current_feat = None
    X_full = None       # eager array for small (non-CFDAC) features
    ds = None           # LazyCFDACDataset for CFDAC variants
    cached_task = None  # for CFDAC: cache materialised tr/va/te slices per task
    X_tr_raw = X_va_raw = X_te_raw = None
    for task, model, feat in cells:
        is_cfdac = feat in _CFDAC_PARTS
        if feat != current_feat:
            del X_full; X_full = None
            ds = None
            X_tr_raw = X_va_raw = X_te_raw = None
            cached_task = None
            import gc; gc.collect()
            try:
                if is_cfdac:
                    ds = LazyCFDACDataset(args.features, feat)
                    print(f">>> lazy {feat}  H={ds.h} W={ds.w}", flush=True)
                else:
                    X_full = load_feature(args.features, feat)
                    print(f">>> {feat}  shape={X_full.shape}", flush=True)
            except Exception as e:
                print(f"  skip feat {feat}: {e}", flush=True)
                current_feat = feat
                continue
            current_feat = feat
        if not is_cfdac and X_full is None:
            continue
        mask, y_pool, kind = tasks[task]
        ipool = np.where(mask)[0]
        i_tr, i_va, i_te = make_split(y_pool, kind)
        idx_tr = ipool[i_tr]; idx_va = ipool[i_va]; idx_te = ipool[i_te]
        y_tr = y_pool[i_tr]; y_va = y_pool[i_va]; y_te = y_pool[i_te]
        # For CFDAC: read tr/va/te once per (variant, task); reuse across ratios + models.
        if is_cfdac and cached_task != task:
            X_tr_raw = ds.batch_read(idx_tr)
            X_va_raw = ds.batch_read(idx_va)
            X_te_raw = ds.batch_read(idx_te)
            cached_task = task
        n_out = (int(y_pool.max()) + 1) if kind == "cls" else 1
        for ratio in RATIOS:
            if (task, model, feat, ratio) in done:
                continue
            t0 = time.time()
            try:
                if is_cfdac:
                    X_tr = _apply_ratio(feat, X_tr_raw, ratio)
                    X_va = _apply_ratio(feat, X_va_raw, ratio)
                    X_te = _apply_ratio(feat, X_te_raw, ratio)
                else:
                    Xr = _apply_ratio(feat, X_full, ratio)
                    X_tr = Xr[idx_tr]; X_va = Xr[idx_va]; X_te = Xr[idx_te]
                if model in ("rf", "xgb"):
                    Xtr_s = X_tr.reshape(len(X_tr), -1).astype(np.float32)
                    Xte_s = X_te.reshape(len(X_te), -1).astype(np.float32)
                    if feat == "modal":
                        scaler = StandardScaler().fit(Xtr_s)
                        Xtr_s = scaler.transform(Xtr_s)
                        Xte_s = scaler.transform(Xte_s)
                    params = {"n_estimators": 100,
                                "max_depth": (None if model == "rf" else 6)}
                    row = _train_sklearn(model, kind, params,
                                              Xtr_s, y_tr,
                                              Xtr_s, y_tr,
                                              Xte_s, y_te)
                else:
                    row = _train_torch(model, kind, n_out, {},
                                            X_tr, y_tr, X_va, y_va,
                                            X_te, y_te,
                                            epochs=args.epochs)
            except Exception as e:
                print(f"  FAIL {task}/{model}/{feat} r={ratio}: {e}",
                          flush=True)
                continue
            row.update({"task": task, "model": model, "feature": feat,
                          "ratio": ratio})
            rows.append(row)
            # Persist after every cell so a kill is safe.
            out_path.write_text(json.dumps(rows, indent=2))
            print(f"  [{len(rows)}] {task}/{model}/{feat} r={ratio:.3f}  "
                      f"{row['metric']}={row['value']:+.3f}  "
                      f"({row['runtime_s']:.1f}s)",
                      flush=True)
            # Free per-ratio tensors before next iter so peak stays bounded
            # for the multi-fit sequence (esp. RF on flattened CFDAC).
            try:
                del X_tr, X_va, X_te
            except NameError:
                pass
            try:
                del Xtr_s, Xte_s
            except NameError:
                pass
            import gc; gc.collect()
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out_path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
