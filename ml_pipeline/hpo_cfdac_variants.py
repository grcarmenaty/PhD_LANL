"""HPO for the new CFDAC variants — appends results to results/hpo/.

Variants run per task:
    cfdac_real, cfdac_imag, cfdac_mag, cfdac_phase           (1-channel 2D)
    cfdac_realimag (existing as cfdac), cfdac_magphase, cfdac_all   (multi-ch 2D)
    cfdac3d_realimag, cfdac3d_magphase, cfdac3d_all                (Conv3d 3D)

Per-task model:
    cnn2d  for the 2-D variants (Conv2DStack on (B, C, H, W))
    cnn3d  for the 3-D variants (Conv3DStack on (B, 1, D, H, W))

Existing ``results/hpo/binary__cnn2d__cfdac.json`` etc. are *not*
touched; the new files are named ``<task>__<model>__<variant>.json``.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml_pipeline.models import Conv2DStack, Conv3DStack       # noqa: E402
from ml_pipeline.lazy_datasets import LazyCFDACDataset         # noqa: E402
from ml_pipeline.tasks import build_targets                    # noqa: E402
from ml_pipeline.train import (                                  # noqa: E402
    load_labels, load_feature, make_split, SEED,
)

torch.set_num_threads(4)


def _hp_key(hp: dict) -> str:
    """Canonical, JSON-stable key for a hyperparameter dict (handles
    tuples vs. lists after a partial JSON round-trip)."""
    norm = {k: (list(v) if isinstance(v, (tuple, list)) else v)
            for k, v in hp.items()}
    return json.dumps(norm, sort_keys=True, default=str)


def _atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)

# (model_name, feature_name)  → grid
GRIDS = {
    "cnn2d": {"widths": [(8, 16, 32), (16, 32, 64)],
                 "kernel_size": [3, 5]},
    "cnn3d": {"widths": [(8, 16, 32), (16, 32, 64)],
                 "kernel_size": [3, 5]},
}

CELLS = [
    ("cnn2d", f) for f in (
        "cfdac_real", "cfdac_imag", "cfdac_mag", "cfdac_phase",
        "cfdac_magphase",
    )
]


def _train_torch(model_name: str, feat_name: str, kind: str, n_out: int,
                  params, X_tr, y_tr, X_va, y_va, X_te, y_te,
                  epochs: int = 4) -> tuple[dict, nn.Module]:
    t0 = time.time()
    torch.manual_seed(SEED)            # deterministic weight init + shuffling
    np.random.seed(SEED)
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

    if model_name == "cnn2d":
        model = Conv2DStack(n_channels=Xtr.shape[1], n_out=n_out,
                              widths=tuple(params["widths"]),
                              kernel_size=int(params["kernel_size"]),
                              regression=(kind == "reg"))
    else:  # cnn3d
        # Xtr shape: (n, 1, D, H, W).  depth = D.
        depth = Xtr.shape[2]
        model = Conv3DStack(depth=depth, n_out=n_out,
                              widths=tuple(params["widths"]),
                              kernel_size=int(params["kernel_size"]),
                              regression=(kind == "reg"))

    batch = 64
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch, shuffle=True)
    best, best_state = -np.inf, None
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            out = model(Xva).cpu()
            metric = (accuracy_score(y_va, out.argmax(1).numpy())
                        if kind == "cls"
                        else r2_score(y_va, out.squeeze(1).numpy()))
        if metric > best:
            best = float(metric)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_te = model(Xte).cpu()
        if kind == "cls":
            mt = float(accuracy_score(y_te, out_te.argmax(1).numpy()))
            extras = {}
        else:
            mt = float(r2_score(y_te, out_te.squeeze(1).numpy()))
            extras = {"mae_test": float(mean_absolute_error(
                y_te, out_te.squeeze(1).numpy()))}
    return ({
        "hyperparams": {k: (list(v) if isinstance(v, tuple) else v)
                         for k, v in params.items()},
        "metric_name": "accuracy" if kind == "cls" else "R2",
        "metric_val":  best,
        "metric_test": mt,
        "extras":      extras,
        "runtime_s":   time.time() - t0,
    }, model)


def _train_torch_streaming(model_name: str, feat_name: str, kind: str,
                            n_out: int, params, tr_ds, va_ds, te_ds,
                            epochs: int = 4,
                            batch_size: int = 64) -> tuple[dict, nn.Module]:
    """Streaming variant — train a Conv2DStack / Conv3DStack over a
    DataLoader fed by LazyCFDACDataset. The full (n, C, H, W) tensor
    is never materialised."""
    t0 = time.time()
    torch.manual_seed(SEED)            # deterministic weight init + shuffling
    np.random.seed(SEED)
    x0, _ = tr_ds[0]
    if model_name == "cnn2d":
        model = Conv2DStack(n_channels=x0.shape[0], n_out=n_out,
                              widths=tuple(params["widths"]),
                              kernel_size=int(params["kernel_size"]),
                              regression=(kind == "reg"))
    else:  # cnn3d, x0 shape (1, D, H, W)
        depth = x0.shape[1]
        model = Conv3DStack(depth=depth, n_out=n_out,
                              widths=tuple(params["widths"]),
                              kernel_size=int(params["kernel_size"]),
                              regression=(kind == "reg"))

    loss_fn = nn.CrossEntropyLoss() if kind == "cls" else nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    te_dl = DataLoader(te_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    def _eval(dl):
        outs, ys = [], []
        model.eval()
        with torch.no_grad():
            for xb, yb in dl:
                outs.append(model(xb).cpu())
                ys.append(yb)
        return torch.cat(outs, 0), torch.cat(ys, 0)

    best, best_state = -np.inf, None
    for _ in range(epochs):
        model.train()
        for xb, yb in tr_dl:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        sched.step()
        out_va, y_va_t = _eval(va_dl)
        if kind == "cls":
            metric = accuracy_score(y_va_t.numpy(), out_va.argmax(1).numpy())
        else:
            metric = r2_score(y_va_t.numpy(), out_va.squeeze(1).numpy())
        if metric > best:
            best = float(metric)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    out_te, y_te_t = _eval(te_dl)
    if kind == "cls":
        mt = float(accuracy_score(y_te_t.numpy(), out_te.argmax(1).numpy()))
        extras = {}
    else:
        mt = float(r2_score(y_te_t.numpy(), out_te.squeeze(1).numpy()))
        extras = {"mae_test": float(mean_absolute_error(
            y_te_t.numpy(), out_te.squeeze(1).numpy()))}
    return ({
        "hyperparams": {k: (list(v) if isinstance(v, tuple) else v)
                          for k, v in params.items()},
        "metric_name": "accuracy" if kind == "cls" else "R2",
        "metric_val":  best,
        "metric_test": mt,
        "extras":      extras,
        "runtime_s":   time.time() - t0,
    }, model)


def run(features_path: Path, out_dir: Path, epochs: int = 4) -> None:
    hpo_dir    = out_dir / "hpo"
    models_dir = out_dir / "models"
    hpo_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(features_path)
    tasks  = build_targets(labels["type_code"], labels["storey"],
                            labels["end"], labels["severity"])

    # Load each variant lazily inside the loop (each one is up to 2.6 GB
    # so pre-loading all of them blows past the RAM budget).
    plan = []
    for tname in tasks:
        for model_name, feat in CELLS:
            plan.append((tname, model_name, feat))
    grand_total = sum(int(np.prod([len(v) for v in GRIDS[m].values()]))
                        for _, m, _ in plan)
    print(f"plan: {len(plan)} cells, {grand_total} trials")

    done = 0
    # Group plan by feature so each variant is loaded from disk only once
    # per (variant, task) pair via LazyCFDACDataset.batch_read().
    plan.sort(key=lambda r: (r[2], r[0], r[1]))
    current_feat, ds = None, None
    for tname, model_name, feat in plan:
        out_json = hpo_dir / f"{tname}__{model_name}__{feat}.json"
        if out_json.exists():
            print(f"  skip {tname}/{model_name}/{feat} (cached)")
            continue
        mask, y_pool, kind = tasks[tname]
        ipool = np.where(mask)[0]
        i_tr, i_va, i_te = make_split(y_pool, kind)
        idx_tr = ipool[i_tr]; idx_va = ipool[i_va]; idx_te = ipool[i_te]
        y_tr = y_pool[i_tr]; y_va = y_pool[i_va]; y_te = y_pool[i_te]
        if feat != current_feat:
            current_feat = feat
            import gc; gc.collect()
            print(f">>> {feat}", flush=True)

        # Stream every cell via LazyCFDACDataset + DataLoader so the
        # 60k-sample CFDAC variants never have to fit fully in RAM.
        reshape = "conv3d" if model_name == "cnn3d" else "conv2d"
        tr_ds = LazyCFDACDataset(features_path, feat,
                                      rows=idx_tr, labels=y_tr, kind=kind,
                                      reshape=reshape)
        va_ds = LazyCFDACDataset(features_path, feat,
                                      rows=idx_va, labels=y_va, kind=kind,
                                      reshape=reshape)
        te_ds = LazyCFDACDataset(features_path, feat,
                                      rows=idx_te, labels=y_te, kind=kind,
                                      reshape=reshape)

        n_out = (int(y_pool.max()) + 1) if kind == "cls" else 1
        grid  = GRIDS[model_name]
        keys  = list(grid.keys()); vals = [grid[k] for k in keys]
        out_partial = hpo_dir / f"{tname}__{model_name}__{feat}.partial.json"
        trials, done_keys = [], set()
        best_trial, best_obj = None, None
        if out_partial.exists():
            try:
                prev = json.loads(out_partial.read_text())
                for r in prev.get("trials", []):
                    trials.append(r)
                    done_keys.add(_hp_key(r["hyperparams"]))
                    done += 1
                    if best_trial is None or r["metric_val"] > best_trial["metric_val"]:
                        best_trial = r
                print(f"  resume {tname}/{model_name}/{feat}: "
                      f"{len(trials)} trials from partial")
            except Exception as e:
                print(f"  WARN partial read failed ({out_partial.name}): {e}")
        for combo in itertools.product(*vals):
            params = dict(zip(keys, combo))
            if _hp_key(params) in done_keys:
                continue
            try:
                row, mdl = _train_torch_streaming(
                    model_name, feat, kind, n_out, params,
                    tr_ds, va_ds, te_ds, epochs=epochs)
            except Exception as e:
                print(f"  FAIL {tname}/{model_name}/{feat} {params}: {e}")
                continue
            trials.append(row)
            done_keys.add(_hp_key(params))
            done += 1
            print(f"  [{done}/{grand_total}] {tname}/{model_name}/{feat} "
                    f"{params}  val={row['metric_val']:.4f}  "
                    f"test={row['metric_test']:.4f}  ({row['runtime_s']:.1f}s)")
            if best_trial is None or row["metric_val"] > best_trial["metric_val"]:
                best_trial = row; best_obj = mdl
            _atomic_write_json(out_partial, {
                "task": tname, "model": model_name, "feature": feat,
                "trials": trials,
            })

        out_json = hpo_dir / f"{tname}__{model_name}__{feat}.json"
        _atomic_write_json(out_json, {
            "task": tname, "model": model_name, "feature": feat,
            "best_hyperparams": best_trial["hyperparams"] if best_trial else None,
            "best_metric_val":  best_trial["metric_val"]  if best_trial else None,
            "best_metric_test": best_trial["metric_test"] if best_trial else None,
            "trials": trials,
        })
        out_partial.unlink(missing_ok=True)
        if best_obj is None and best_trial is not None:
            try:
                _, best_obj = _train_torch_streaming(
                    model_name, feat, kind, n_out, best_trial["hyperparams"],
                    tr_ds, va_ds, te_ds, epochs=epochs)
            except Exception as e:
                print(f"  WARN: rematerialise of best model failed: {e}")
        if best_obj is not None:
            tag = f"{tname}_{model_name}_{feat}"
            torch.save({
                "state_dict": best_obj.state_dict(),
                "model_name": model_name,
                "n_out": n_out,
                "in_shape": list(tr_ds[0][0].shape),
                "hyperparams": best_trial["hyperparams"],
            }, models_dir / f"{tag}.pt")

    print("\ndone.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path,
                      default=_REPO / "dataset" / "features.h5")
    p.add_argument("--out", type=Path, default=_REPO / "results")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--seed", type=int, default=SEED,
                      help="Override the global seed for multi-seed runs.")
    args = p.parse_args()
    import ml_pipeline.train as _train
    globals()["SEED"] = args.seed
    _train.SEED = args.seed
    run(args.features, args.out, args.epochs)


if __name__ == "__main__":
    main()
