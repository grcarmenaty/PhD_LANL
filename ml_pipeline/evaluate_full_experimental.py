"""Run every trained model on the full 2638-case experimental dataset.

Reads features from ``dataset/experimental_features.h5`` (built by
``build_experimental_features.py``).  Iterates over every artefact in
``results/models/`` and ``results/models_indicators/`` and writes:

  results/experimental_full_evaluation.json     per-(task, model, feature) summary
  results/experimental_full_per_case.json       per-case predictions
  results/indicator_predictions_full.json       per-(indicator, model, feature) summary

This *replaces* the old 61-case evaluation that used ``median_frfs.h5``.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, mean_absolute_error, r2_score,
)
from sklearn.preprocessing import StandardScaler

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml_pipeline.case_design import TYPE_PRISTINE                     # noqa: E402
from ml_pipeline.features import INDICATOR_NAMES                      # noqa: E402
from ml_pipeline.models import (                                        # noqa: E402
    MLP, Conv1DStack, SmallTransformer, Conv2DStack, Conv3DStack,
)
from ml_pipeline.tasks import build_targets                           # noqa: E402
from ml_pipeline.train import (                                          # noqa: E402
    FEATURES_FLAT, FEATURES_SEQ, FEATURES_MAT,
    load_labels, load_feature, make_split, _CFDAC_VARIANTS,
    _per_sample_normalize,
)


def _exp_load_feature(exp_path: Path, name: str,
                       rows: np.ndarray | None = None,
                       normalize: bool = False) -> np.ndarray:
    """Read a feature out of experimental_features.h5 with the same
    CFDAC stacking convention as load_feature on the synth file.

    With ``normalize=True`` each part is run through the shared
    ``_per_sample_normalize`` (P1.1) so cross-domain inputs match the
    synth-train statistics. Used by the vision pipeline; default
    ``False`` preserves the bespoke-eval behaviour."""
    if name in _CFDAC_VARIANTS:
        parts, mode = _CFDAC_VARIANTS[name]
        layers = []
        with h5py.File(exp_path, "r") as f:
            for p in parts:
                arr = f[p][:] if rows is None else f[p][rows]
                if normalize:
                    arr = _per_sample_normalize(p, arr)
                layers.append(arr)
        if mode == "stack2d":
            return np.stack(layers, axis=1)
        return np.stack(layers, axis=1)[:, np.newaxis, ...]
    with h5py.File(exp_path, "r") as f:
        data = f[name][:] if rows is None else f[name][rows]
    if normalize:
        data = _per_sample_normalize(name, data)
    return data


def _build_scalers(syn_path: Path) -> Dict[str, Dict[str, StandardScaler]]:
    """Re-fit per-(task, flat-feature) StandardScaler on the synth
    train fold so MLP / sklearn cells receive the same input
    distribution they trained on."""
    L = load_labels(syn_path)
    tasks = build_targets(L["type_code"], L["storey"], L["end"], L["severity"])
    out: Dict[str, Dict[str, StandardScaler]] = {}
    flat = {n: load_feature(syn_path, n) for n in FEATURES_FLAT}
    for tn, (mask, y_pool, kind) in tasks.items():
        out[tn] = {}
        ipool = np.where(mask)[0]
        i_tr, _, _ = make_split(y_pool, kind)
        idx_tr = ipool[i_tr]
        for f in FEATURES_FLAT:
            out[tn][f] = StandardScaler().fit(
                flat[f][idx_tr].reshape(len(idx_tr), -1))
    return out


def _predict(art: Path, X: np.ndarray, kind: str, n_out: int,
              scaler: StandardScaler | None = None) -> np.ndarray:
    if art.suffix == ".pkl":
        with open(art, "rb") as f:
            blob = pickle.load(f)
        mdl = blob["model"]
        sk_scaler = blob.get("scaler") or scaler
        Xf = X.reshape(len(X), -1)
        if sk_scaler is not None:
            Xf = sk_scaler.transform(Xf)
        return mdl.predict(Xf)
    blob = torch.load(art, map_location="cpu", weights_only=False)
    in_shape = blob["in_shape"]; name = blob["model_name"]
    hp = blob.get("hyperparams") or {}
    Xa = X
    if name == "mlp" and scaler is not None:
        Xa = scaler.transform(X.reshape(len(X), -1))
    t = torch.as_tensor(np.asarray(Xa)).float()
    seq = (t.ndim == 3); mat = (t.ndim == 4); vol = (t.ndim == 5)
    if seq and t.shape[1] != in_shape[-1]:
        t = t.permute(0, 2, 1)
    if name == "mlp":
        in_dim = t.shape[-1] if t.ndim == 2 else int(np.prod(in_shape))
        if t.ndim != 2: t = t.flatten(1)
        mdl = MLP(in_dim=in_dim, n_out=n_out,
                     hidden=tuple(hp.get("hidden", (256, 128, 64))),
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
    elif name == "cnn3d":
        depth = in_shape[1]
        mdl = Conv3DStack(depth=depth, n_out=n_out,
                              widths=tuple(hp.get("widths", (8, 16, 32))),
                              kernel_size=int(hp.get("kernel_size", 3)),
                              regression=(kind == "reg"))
    else:
        raise ValueError(name)
    mdl.load_state_dict(blob["state_dict"]); mdl.eval()
    with torch.no_grad():
        out = mdl(t)
    if kind == "cls":
        return out.argmax(1).numpy()
    return out.squeeze(1).numpy()


def _parse_tag(tag: str, tasks: Dict[str, tuple]) -> tuple[str, str, str] | None:
    """Parse '<task>_<model>_<feature>' allowing task names with underscores."""
    for task in tasks:
        prefix = task + "_"
        if tag.startswith(prefix):
            rest = tag[len(prefix):]
            # try increasingly long feature names from the catalogue
            for feat in (*_CFDAC_VARIANTS.keys(), "modal", "indicators",
                            "frf_mag", "timeseries"):
                suffix = "_" + feat
                if rest.endswith(suffix):
                    return task, rest[:-len(suffix)], feat
    return None


def evaluate_classification_regression(syn_path: Path, exp_path: Path,
                                          out_dir: Path) -> None:
    print("loading synth labels + scalers …")
    syn_labels = load_labels(syn_path)
    syn_tasks  = build_targets(syn_labels["type_code"], syn_labels["storey"],
                                  syn_labels["end"], syn_labels["severity"])
    scalers = _build_scalers(syn_path)

    with h5py.File(exp_path, "r") as f:
        e_type = f["type_code"][:].astype(np.int64)
        e_storey = f["storey"][:].astype(np.int64)
        e_end = f["end"][:].astype(np.int64)
        e_sev = f["severity"][:].astype(np.float32)
        names = [str(s) for s in f["names"][:]]
    exp_tasks = build_targets(e_type, e_storey, e_end, e_sev)
    n = len(e_type)
    print(f"experimental: {n} cases")

    rows: List[Dict] = []
    per_case: List[Dict] = []
    models_dir = out_dir / "models"
    if not models_dir.exists():
        raise FileNotFoundError(models_dir)
    artefacts = sorted(models_dir.iterdir())
    print(f"artefacts: {len(artefacts)}")

    # Cache loaded features (only what we actually need, lazily).
    feat_cache: Dict[str, np.ndarray] = {}

    for art in artefacts:
        tag = art.stem
        parsed = _parse_tag(tag, syn_tasks)
        if parsed is None:
            print(f"  skip {tag}: cannot parse")
            continue
        task_name, model_name, feature = parsed
        mask, y_all, kind = exp_tasks[task_name]
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        if feature not in feat_cache:
            print(f"  loading exp feature: {feature}")
            feat_cache[feature] = _exp_load_feature(exp_path, feature)
        X = feat_cache[feature][idx]
        y = y_all
        scaler = scalers.get(task_name, {}).get(feature)
        try:
            n_out = (int(y.max()) + 1) if kind == "cls" else 1
            pred = _predict(art, X, kind, n_out, scaler=scaler)
        except Exception as e:
            print(f"  FAIL {tag}: {e}")
            continue
        if kind == "cls":
            metric = float(accuracy_score(y, pred))
            mae = None; mname = "accuracy"
        else:
            metric = float(r2_score(y, pred))
            mae = float(mean_absolute_error(y, pred))
            mname = "R2"
        print(f"  {tag:<55s} {mname}={metric:.3f}  n={len(X)}"
                + (f"  MAE={mae:.3f}" if mae is not None else ""))
        rows.append({
            "task": task_name, "model": model_name, "feature": feature,
            "metric": mname, "value": metric, "mae": mae,
            "n_cases": int(len(X)),
        })
        for nm, yt, yp in zip(np.array(names)[idx].tolist(), y.tolist(),
                                  pred.tolist()):
            per_case.append({
                "task": task_name, "model": model_name, "feature": feature,
                "case": nm, "y_true": float(yt), "y_pred": float(yp),
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "experimental_full_evaluation.json").write_text(
        json.dumps(rows, indent=2))
    (out_dir / "experimental_full_per_case.json").write_text(
        json.dumps(per_case, indent=2))
    print(f"\nWrote experimental_full_evaluation.json ({len(rows)} rows) "
            f"and experimental_full_per_case.json ({len(per_case)} rows)")


def evaluate_indicator_predictors(syn_path: Path, exp_path: Path,
                                      out_dir: Path) -> None:
    models_dir = out_dir / "models_indicators"
    if not models_dir.exists():
        print("(no indicator models — skipping)")
        return
    print("evaluating indicator predictors on full experimental set …")

    # Use damage cases only (Pristine has no meaningful "damage indicator").
    with h5py.File(exp_path, "r") as f:
        type_code = f["type_code"][:]
        ind_true  = f["indicators"][:]
    dmg_idx = np.where(type_code != TYPE_PRISTINE)[0]
    print(f"  {len(dmg_idx)} damage cases")

    syn_labels = load_labels(syn_path)
    syn_tasks = build_targets(syn_labels["type_code"], syn_labels["storey"],
                                  syn_labels["end"], syn_labels["severity"])
    scalers = _build_scalers(syn_path)

    feat_cache: Dict[str, np.ndarray] = {}
    rows: List[Dict] = []
    for art in sorted(models_dir.iterdir()):
        # tag = "<indicator>_<model>_<feature>" — but indicator names can
        # contain underscores; parse from the right by feature name.
        tag = art.stem
        for feat in (*_CFDAC_VARIANTS.keys(), "modal", "indicators",
                        "frf_mag", "timeseries"):
            if tag.endswith("_" + feat):
                rest = tag[: -len(feat) - 1]
                break
        else:
            continue
        # Now rest = "<indicator>_<model>".  Model is one of rf/xgb/mlp/cnn2d/cnn3d.
        for m in ("cnn2d", "cnn3d", "mlp", "xgb", "rf", "transformer", "cnn"):
            if rest.endswith("_" + m):
                ind_name = rest[: -len(m) - 1]; model_name = m
                break
        else:
            continue
        if ind_name not in INDICATOR_NAMES:
            continue
        ind_idx = INDICATOR_NAMES.index(ind_name)
        if feat not in feat_cache:
            print(f"  loading exp feature: {feat}")
            feat_cache[feat] = _exp_load_feature(exp_path, feat)
        X = feat_cache[feat][dmg_idx]
        y = ind_true[dmg_idx, ind_idx]
        # scaler from severity task fold (same train fold as indicator
        # predictors — they were trained on damage samples).
        scaler = None
        if feat in ("modal", "indicators"):
            scaler = scalers.get("severity", {}).get(feat)
        try:
            pred = _predict(art, X, "reg", 1, scaler=scaler)
        except Exception as e:
            print(f"  FAIL {tag}: {e}")
            continue
        rows.append({
            "indicator": ind_name, "model": model_name, "feature": feat,
            "exp_R2": float(r2_score(y, pred)),
            "exp_MAE": float(mean_absolute_error(y, pred)),
            "n_cases": int(len(X)),
        })
        print(f"    {ind_name:<14s}/{model_name:<5s}/{feat:<14s}  "
                f"R²={rows[-1]['exp_R2']:.3f}  MAE={rows[-1]['exp_MAE']:.3f}")
    (out_dir / "indicator_predictions_full.json").write_text(
        json.dumps(rows, indent=2))
    print(f"\nWrote indicator_predictions_full.json ({len(rows)} rows)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--syn", type=Path,
                      default=_REPO / "dataset" / "features.h5")
    p.add_argument("--exp", type=Path,
                      default=_REPO / "dataset" / "experimental_features.h5")
    p.add_argument("--out", type=Path, default=_REPO / "results")
    p.add_argument("--skip-cls", action="store_true")
    p.add_argument("--skip-ind", action="store_true")
    args = p.parse_args()
    if not args.skip_cls:
        evaluate_classification_regression(args.syn, args.exp, args.out)
    if not args.skip_ind:
        evaluate_indicator_predictors(args.syn, args.exp, args.out)


if __name__ == "__main__":
    main()
