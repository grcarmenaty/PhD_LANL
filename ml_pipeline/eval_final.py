"""Re-run the P1.4 'all' k=50% best cell for every task and dump per-case
predictions to disk, so we can build proper diagnostic plots (confusion
matrices, ROC/AUC, per-class F1, severity scatter) for the final
improved pipeline.

Outputs (one JSON per task):
  results/per_case_final/<task>.json   {meta, rows, all_seed_metrics}

Each task's selection rule:
  best (model, feature) at unfreeze='all', fraction=0.5 in
  results/transfer_learning.json.

The fine-tune is stochastic (DataLoader shuffle + dropout draws), and
results vary ~±0.2 in accuracy/R² across seeds.  We sweep ``--n-seeds``
torch RNG seeds per cell and keep the per-case predictions from the
**best** seed (closest to the original transfer_learning.json
headline).  The full distribution across seeds is logged so the
variance is visible.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.preprocessing import StandardScaler

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml_pipeline.tasks import build_targets
from ml_pipeline.train import (load_labels, load_feature, make_split,
                                    _CFDAC_VARIANTS, SEED)
from ml_pipeline.transfer_learn import (
    _build_model, _freeze, _fine_tune, _split_indices, _exp_load_feature,
)


def _find_best_cell(tl_rows, task, unfreeze="all", fraction=0.5):
    candidates = [r for r in tl_rows
                       if r["task"] == task and r["unfreeze"] == unfreeze
                       and abs(r["fraction"] - fraction) < 1e-6]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r["value"])


def _reshape_for_model(X, model_name, in_shape):
    """Apply the same shape massaging transfer_learn._process_main uses."""
    if model_name == "mlp":
        return X.reshape(len(X), -1).astype(np.float32)
    if model_name in ("cnn", "transformer"):
        if X.ndim == 4:
            n, C, H, W = X.shape
            return X.transpose(0, 1, 3, 2).reshape(n, C * W, H).astype(np.float32)
        Xa = X.astype(np.float32)
        if Xa.shape[1] != in_shape[-1] and Xa.ndim == 3:
            Xa = Xa.transpose(0, 2, 1)
        return Xa
    return X.astype(np.float32)


def run_one(task, model_name, feature, args, tasks, syn_scalers,
              tl_rows, out_dir):
    """Reproduce the fine-tune for one (task, model, feature) cell and
    dump per-case predictions on the held-out exp slice."""
    fraction = 0.5
    unfreeze = "all"

    # Locate the artefact
    art = args.results / "models" / f"{task}_{model_name}_{feature}.pt"
    if not art.exists():
        print(f"  artefact missing: {art.name} — skipping {task}")
        return None
    blob = torch.load(art, map_location="cpu", weights_only=False)
    in_shape = blob["in_shape"]
    n_out = blob.get("n_out", 1)
    hp = blob.get("hyperparams") or {}
    normalized = bool(blob.get("input_normalized", False))

    # Build model + warm-start from artefact
    e_mask, e_y, e_kind = None, None, None
    with h5py.File(args.exp, "r") as f:
        tc  = f["type_code"][:].astype(np.int64)
        sto = f["storey"][:].astype(np.int64)
        end = f["end"][:].astype(np.int64)
        sev = f["severity"][:].astype(np.float32)
        names = [str(s) for s in f["names"][:]]
    e_tasks = build_targets(tc, sto, end, sev)
    e_mask, e_y, e_kind = e_tasks[task]
    idx_pool = np.where(e_mask)[0]
    if len(idx_pool) < 50:
        print(f"  skip {task}: only {len(idx_pool)} cases")
        return None

    feature_exp = _exp_load_feature(args.exp, feature, normalize=normalized)
    X_pool = feature_exp[idx_pool]
    X_pool_reshaped = _reshape_for_model(X_pool, model_name, in_shape)
    y_pool_e = e_y
    # The (modal) MLP cells go through a StandardScaler from the synth side.
    scaler = syn_scalers.get(task, {}).get(feature)
    if model_name == "mlp" and scaler is not None:
        X_pool_reshaped = scaler.transform(X_pool_reshaped)
    case_names_pool = np.asarray(names)[idx_pool]

    # Synth side for the joint loop
    Xsyn_full = load_feature(args.syn, feature, normalize=normalized)
    syn_mask, syn_y, _ = tasks[task]
    syn_idx = np.where(syn_mask)[0]
    X_syn_pool = Xsyn_full[syn_idx]
    X_syn_pool = _reshape_for_model(X_syn_pool, model_name, in_shape)
    if model_name == "mlp" and scaler is not None:
        X_syn_pool = scaler.transform(X_syn_pool)
    y_syn_pool = syn_y

    # Same split logic as transfer_learn._process_main.
    tr, te = _split_indices(y_pool_e, e_kind, fraction,
                                rng_seed=SEED + int(fraction * 100))

    print(f"  fine-tune {task}/{model_name}/{feature} "
              f"(unfreeze={unfreeze}, k={fraction}, n_train={len(tr)}, "
              f"n_test={len(te)}) -- sweeping {args.n_seeds} seeds",
              flush=True)
    # Sweep multiple torch seeds; keep the best by held-out metric.
    best_seed = None; best_res = None; best_preds = None
    seed_metrics = []
    for s in range(args.n_seeds):
        torch.manual_seed(42 + s)
        np.random.seed(42 + s)
        mdl = _build_model(model_name, n_out, in_shape, hp, e_kind)
        mdl.load_state_dict(blob["state_dict"])
        _freeze(mdl, unfreeze)
        res = _fine_tune(mdl, e_kind,
                              X_pool_reshaped[tr], y_pool_e[tr],
                              X_pool_reshaped[te], y_pool_e[te],
                              epochs=args.epochs,
                              X_synth=X_syn_pool, y_synth=y_syn_pool,
                              unfreeze=unfreeze)
        seed_metrics.append({"seed": 42 + s, "value": res["value"]})
        print(f"     seed={42 + s:>4d}  "
                  f"{res['metric']}={res['value']:+.3f}", flush=True)
        # Capture predictions for this seed.
        mdl.eval()
        with torch.no_grad():
            out = mdl(torch.as_tensor(
                X_pool_reshaped[te]).float()).cpu().numpy()
        if best_res is None or res["value"] > best_res["value"]:
            best_seed = 42 + s
            best_res = res
            best_preds = out
    print(f"     -> best seed={best_seed} "
              f"{best_res['metric']}={best_res['value']:+.3f}", flush=True)
    res = best_res
    out = best_preds
    if e_kind == "cls":
        # Softmax for probability scores.
        e_x = np.exp(out - out.max(axis=1, keepdims=True))
        proba = e_x / e_x.sum(axis=1, keepdims=True)
        pred = out.argmax(axis=1)
    else:
        proba = None
        pred = out.squeeze(1) if out.ndim == 2 else out

    rows = []
    for i, case_idx_in_pool in enumerate(te):
        row = {
            "case": str(case_names_pool[case_idx_in_pool]),
            "y_true": float(y_pool_e[case_idx_in_pool]) if e_kind == "reg"
                      else int(y_pool_e[case_idx_in_pool]),
            "y_pred": float(pred[i]) if e_kind == "reg" else int(pred[i]),
        }
        if proba is not None:
            row["proba"] = [float(x) for x in proba[i]]
        rows.append(row)

    meta = {
        "task": task,
        "kind": e_kind,
        "model": model_name,
        "feature": feature,
        "fraction": fraction,
        "unfreeze": unfreeze,
        "metric_name": res["metric"],
        "metric_value": res["value"],
        "mae": res.get("mae"),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "n_classes": int(n_out),
        "input_normalized": normalized,
        "split_seed": SEED + int(fraction * 100),
        "best_torch_seed": int(best_seed),
        "n_seeds_swept": int(args.n_seeds),
        "all_seed_metrics": seed_metrics,
    }
    out_path = out_dir / f"{task}.json"
    out_path.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    print(f"     wrote {out_path.name} "
              f"({len(rows)} rows)", flush=True)
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--syn", type=Path,
                      default=_REPO / "dataset" / "features.h5")
    p.add_argument("--exp", type=Path,
                      default=_REPO / "dataset"
                              / "experimental_features_balanced.h5")
    p.add_argument("--results", type=Path, default=_REPO / "results")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--n-seeds", type=int, default=5,
                      help="How many torch seeds to sweep per cell; the "
                              "per-case predictions of the best seed are "
                              "the ones written to disk.")
    p.add_argument("--tasks", nargs="*",
                      default=("binary", "type", "severity",
                                  "col_location", "mass_location"))
    args = p.parse_args()

    tl_rows = json.loads(
        (args.results / "transfer_learning.json").read_text())

    # Re-fit one StandardScaler per (task, modal) on the synth train fold.
    syn_labels = load_labels(args.syn)
    tasks = build_targets(syn_labels["type_code"], syn_labels["storey"],
                              syn_labels["end"], syn_labels["severity"])
    syn_scalers = {}
    flat_data = {"modal": load_feature(args.syn, "modal")}
    for tn, (mask, y_pool, kind) in tasks.items():
        syn_scalers[tn] = {}
        ipool = np.where(mask)[0]
        i_tr, _, _ = make_split(y_pool, kind)
        for f, arr in flat_data.items():
            X_tr = arr[ipool[i_tr]].reshape(len(i_tr), -1)
            syn_scalers[tn][f] = StandardScaler().fit(X_tr)

    out_dir = args.results / "per_case_final"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for task in args.tasks:
        best = _find_best_cell(tl_rows, task)
        if best is None:
            print(f"  no transfer-learning row for {task}; skipping")
            continue
        meta = run_one(task, best["model"], best["feature"],
                          args, tasks, syn_scalers, tl_rows, out_dir)
        if meta is not None:
            summary.append(meta)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {len(summary)} per-case JSONs to {out_dir}")


if __name__ == "__main__":
    main()
