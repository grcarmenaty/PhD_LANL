"""Fine-tune every synth-trained Torch model on a fraction of the
**balanced** experimental data and evaluate on the held-out
experimental remainder.

For each ``.pt`` artefact in ``results/models/`` (and
``results/models_indicators/``) the script:

  1. Reads the architecture parameters from the artefact's
     ``in_shape``, ``model_name`` and ``hyperparams`` keys.
  2. Reconstructs the model and loads the synth-trained
     ``state_dict``.
  3. Freezes parameters at one of two depths:
       * "head"     — only the final ``Linear`` layer is trainable;
       * "head_proj" — the last block before global pool is
                         also unfrozen.
  4. Stratifies the balanced experimental data into a
     ``k`` % fine-tune slice + ``(1-k)`` % hold-out slice for
     ``k ∈ {10, 20, 30, 40, 50}`` (seed = 20260511 for repro).
  5. Fine-tunes for ``--epochs`` epochs (default 8) on the slice.
  6. Reports the metric (accuracy / R²) on the hold-out slice.

Outputs:
    results/transfer_learning.json   one row per
                                       (task, model, feature, k, unfreeze)
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
from sklearn.metrics import (
    accuracy_score, mean_absolute_error, r2_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml_pipeline.case_design import TYPE_PRISTINE                       # noqa: E402
from ml_pipeline.features import INDICATOR_NAMES                        # noqa: E402
from ml_pipeline.models import (                                          # noqa: E402
    MLP, Conv1DStack, SmallTransformer, Conv2DStack, Conv3DStack,
)
from ml_pipeline.tasks import build_targets                               # noqa: E402
from ml_pipeline.train import (                                            # noqa: E402
    load_labels, load_feature, make_split, SEED, _CFDAC_VARIANTS,
)

FRACTIONS = (0.10, 0.20, 0.30, 0.40, 0.50)
# P1.4: 'all' unfreezes the backbone too; joint synth+exp mini-batches
# are used in that mode so the backbone gets exp gradients without
# forgetting the synth task entirely.
UNFREEZE_DEPTHS = ("head", "head_proj", "all")
# Weight on the L2 anchor regulariser ||W - W_synth||^2 in 'all' mode.
ANCHOR_LAMBDA = 1e-4
# Synth:exp mini-batch ratio in 'all' mode (5 synth examples per 1 exp).
SYNTH_EXP_RATIO = 5
torch.set_num_threads(4)


def _exp_load_feature(exp_path: Path, name: str) -> np.ndarray:
    """Re-create the model's expected input view from the experimental
    file.  Mirrors `evaluate_full_experimental._exp_load_feature`."""
    if name in _CFDAC_VARIANTS:
        parts, mode = _CFDAC_VARIANTS[name]
        with h5py.File(exp_path, "r") as f:
            layers = [f[p][:] for p in parts]
        x = np.stack(layers, axis=1)
        if mode == "stack3d":
            x = x[:, np.newaxis, ...]
        return x
    with h5py.File(exp_path, "r") as f:
        return f[name][:]


def _build_model(model_name: str, n_out: int, in_shape: list, hp: dict,
                   kind: str) -> nn.Module:
    if model_name == "mlp":
        in_dim = in_shape[-1] if len(in_shape) == 1 else int(np.prod(in_shape))
        return MLP(in_dim=in_dim, n_out=n_out,
                       hidden=tuple(hp.get("hidden", (256, 128, 64))),
                       regression=(kind == "reg"))
    if model_name == "cnn":
        ch = in_shape[1] if len(in_shape) == 2 else in_shape[-1]
        return Conv1DStack(n_channels=ch, n_out=n_out,
                              widths=tuple(hp.get("widths", (32, 64, 128))),
                              kernel_size=int(hp.get("kernel_size", 7)),
                              regression=(kind == "reg"))
    if model_name == "transformer":
        ch = in_shape[1] if len(in_shape) == 2 else in_shape[-1]
        return SmallTransformer(n_channels=ch, n_out=n_out,
                                    d_model=int(hp.get("d_model", 48)),
                                    n_layers=int(hp.get("n_layers", 2)),
                                    regression=(kind == "reg"))
    if model_name == "cnn2d":
        return Conv2DStack(n_channels=in_shape[0], n_out=n_out,
                              widths=tuple(hp.get("widths", (16, 32, 64))),
                              kernel_size=int(hp.get("kernel_size", 5)),
                              regression=(kind == "reg"))
    if model_name == "cnn3d":
        depth = in_shape[1]
        return Conv3DStack(depth=depth, n_out=n_out,
                              widths=tuple(hp.get("widths", (8, 16, 32))),
                              kernel_size=int(hp.get("kernel_size", 3)),
                              regression=(kind == "reg"))
    raise ValueError(model_name)


def _freeze(mdl: nn.Module, depth: str) -> int:
    """Freeze everything, then unfreeze a small slice depending on
    ``depth``.  Returns the number of trainable parameters.

    ``depth='all'`` (P1.4) unfreezes everything; the joint loop adds
    an L2 anchor to the synth-trained weights to prevent catastrophic
    forgetting.
    """
    for p in mdl.parameters():
        p.requires_grad = (depth == "all")
    if depth == "head":
        # Unfreeze only the very last Linear layer in each model.
        # The model factories above end in either nn.Linear or
        # nn.Sequential(... nn.Linear); walk modules in reverse and
        # flip the first Linear we find.
        for m in reversed(list(mdl.modules())):
            if isinstance(m, nn.Linear):
                for p in m.parameters():
                    p.requires_grad = True
                break
    elif depth == "head_proj":
        # Unfreeze the whole `head` sub-module if present, otherwise
        # the last 2 Linear layers.
        if hasattr(mdl, "head") and isinstance(mdl.head, nn.Module):
            for p in mdl.head.parameters():
                p.requires_grad = True
        else:
            linears = [m for m in mdl.modules() if isinstance(m, nn.Linear)]
            for m in linears[-2:]:
                for p in m.parameters():
                    p.requires_grad = True
    return sum(p.numel() for p in mdl.parameters() if p.requires_grad)


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


def _split_indices(y: np.ndarray, kind: str, fraction: float,
                    rng_seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx) for a single split.  Stratified by
    target for classification, simple shuffle for regression."""
    splitter = (StratifiedShuffleSplit if kind == "cls" else ShuffleSplit)(
        n_splits=1, train_size=fraction, random_state=rng_seed
    )
    if kind == "cls":
        idx_train, idx_test = next(splitter.split(np.zeros(len(y)), y))
    else:
        idx_train, idx_test = next(splitter.split(np.zeros(len(y)), y))
    return idx_train, idx_test


def _fine_tune(mdl: nn.Module, kind: str,
                X_train, y_train, X_test, y_test,
                epochs: int = 8, lr: float = 1e-3,
                X_synth: np.ndarray | None = None,
                y_synth: np.ndarray | None = None,
                unfreeze: str = "head",
                ) -> Dict[str, float]:
    """Fine-tune ``mdl`` on (X_train, y_train) and report (X_test, y_test).

    P1.4: when ``unfreeze='all'`` and synth data are passed via
    ``X_synth/y_synth``, each step mixes ``SYNTH_EXP_RATIO`` synth
    examples per exp example and adds an L2 anchor on the synth-
    trained weights so the backbone doesn't forget the synth task.
    """
    Xtr = torch.as_tensor(X_train).float()
    Xte = torch.as_tensor(X_test).float()
    if kind == "cls":
        ytr = torch.as_tensor(y_train).long()
        loss_fn = nn.CrossEntropyLoss()
    else:
        ytr = torch.as_tensor(y_train).float().unsqueeze(1)
        loss_fn = nn.MSELoss()
    trainable = [p for p in mdl.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)

    joint_mode = (unfreeze == "all" and X_synth is not None
                      and y_synth is not None and len(y_synth) > 0)
    # Snapshot anchor weights *before* any training step.
    anchor = ({n: p.detach().clone() for n, p in mdl.named_parameters()
                  if p.requires_grad} if joint_mode else None)

    if joint_mode:
        # Build the synth dataset/loader.  Use the larger batch so the
        # 5:1 synth:exp ratio is preserved on average.
        Xs = torch.as_tensor(X_synth).float()
        if kind == "cls":
            ys = torch.as_tensor(y_synth).long()
        else:
            ys = torch.as_tensor(y_synth).float().unsqueeze(1)
        bs_exp = 32
        bs_synth = bs_exp * SYNTH_EXP_RATIO
        dl_exp   = DataLoader(TensorDataset(Xtr, ytr),
                                  batch_size=bs_exp, shuffle=True)
        dl_synth = DataLoader(TensorDataset(Xs, ys),
                                  batch_size=bs_synth, shuffle=True)
        for ep in range(epochs):
            mdl.train()
            synth_iter = iter(dl_synth)
            for xb, yb in dl_exp:
                try:
                    xs, ysb = next(synth_iter)
                except StopIteration:
                    synth_iter = iter(dl_synth)
                    xs, ysb = next(synth_iter)
                opt.zero_grad()
                loss = loss_fn(mdl(xb), yb) + loss_fn(mdl(xs), ysb)
                if anchor is not None:
                    anchor_loss = 0.0
                    for n, p in mdl.named_parameters():
                        if n in anchor:
                            anchor_loss = anchor_loss + (
                                (p - anchor[n]) ** 2).sum()
                    loss = loss + ANCHOR_LAMBDA * anchor_loss
                loss.backward(); opt.step()
    else:
        dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=64, shuffle=True)
        for ep in range(epochs):
            mdl.train()
            for xb, yb in dl:
                opt.zero_grad(); loss_fn(mdl(xb), yb).backward(); opt.step()
    mdl.eval()
    with torch.no_grad():
        out = mdl(Xte).cpu().numpy()
    if kind == "cls":
        return {"metric": "accuracy",
                "value": float(accuracy_score(y_test, out.argmax(1))),
                "mae": None}
    pred = out.squeeze(1)
    return {"metric": "R2",
            "value": float(r2_score(y_test, pred)),
            "mae": float(mean_absolute_error(y_test, pred))}


def _process_main(args, tasks, syn_scalers, exp_path: Path,
                    syn_labels) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    feat_cache: Dict[str, np.ndarray] = {}
    # P1.4: synth-side feature cache for joint training in unfreeze='all'.
    syn_feat_cache: Dict[str, np.ndarray] = {}
    for art in sorted((args.results / "models").iterdir()):
        if art.suffix != ".pt":
            continue
        tag = art.stem
        task, model_name, feature = _parse_tag(tag, tasks)
        if task is None:
            continue
        blob = torch.load(art, map_location="cpu", weights_only=False)
        in_shape = blob["in_shape"]; n_out = blob.get("n_out", 1)
        hp = blob.get("hyperparams") or {}
        mask, y_pool, kind = tasks[task]
        # Load the experimental feature for this cell.
        if feature not in feat_cache:
            try:
                feat_cache[feature] = _exp_load_feature(exp_path, feature)
            except Exception as e:
                print(f"  skip {tag}: feature {feature}: {e}", flush=True)
                continue
        X_all = feat_cache[feature]
        # MLP needs flat + scaler.
        scaler = syn_scalers.get(task, {}).get(feature)
        # Reconstruct the experimental labels matching the task.
        with h5py.File(exp_path, "r") as f:
            tc  = f["type_code"][:].astype(np.int64)
            sto = f["storey"][:].astype(np.int64)
            end = f["end"][:].astype(np.int64)
            sev = f["severity"][:].astype(np.float32)
        e_tasks = build_targets(tc, sto, end, sev)
        e_mask, e_y, e_kind = e_tasks[task]
        idx_pool = np.where(e_mask)[0]
        y_pool_e = e_y
        X_pool   = X_all[idx_pool]
        n_pool   = len(idx_pool)
        if n_pool < 50:
            print(f"  skip {tag}: only {n_pool} experimental cases", flush=True)
            continue
        for unfreeze in UNFREEZE_DEPTHS:
            for k in FRACTIONS:
                tag_full = f"{tag}__{unfreeze}__k{int(k*100):02d}"
                t0 = time.time()
                try:
                    mdl = _build_model(model_name, n_out, in_shape, hp, e_kind)
                    mdl.load_state_dict(blob["state_dict"])
                    n_trainable = _freeze(mdl, unfreeze)
                except Exception as e:
                    print(f"  FAIL build {tag} {unfreeze} k={k}: {e}",
                              flush=True)
                    continue
                # Build features
                try:
                    if model_name == "mlp":
                        Xf = X_pool.reshape(len(X_pool), -1).astype(np.float32)
                        if scaler is not None:
                            Xf = scaler.transform(Xf)
                        X_use = Xf
                    elif model_name in ("cnn", "transformer"):
                        # Either already (n, C, L) from frf_mag/timeseries
                        # or CFDAC: need _to_seq reshape.
                        if X_pool.ndim == 4:
                            n, C, H, W = X_pool.shape
                            X_use = X_pool.transpose(0, 1, 3, 2).reshape(n, C * W, H).astype(np.float32)
                        else:
                            X_use = X_pool.astype(np.float32)
                            if X_use.shape[1] != in_shape[-1] and X_use.ndim == 3:
                                X_use = X_use.transpose(0, 2, 1)
                    else:
                        X_use = X_pool.astype(np.float32)
                    tr, te = _split_indices(y_pool_e, e_kind, k,
                                                rng_seed=SEED + int(k*100))
                    # P1.4: for unfreeze='all', supply a synth batch so
                    # the joint loop can mix domains and apply the
                    # anchor regulariser.
                    X_synth_use = None; y_synth_use = None
                    if unfreeze == "all":
                        if feature not in syn_feat_cache:
                            try:
                                syn_feat_cache[feature] = load_feature(
                                    args.syn, feature)
                            except Exception:
                                syn_feat_cache[feature] = None
                        Xsyn = syn_feat_cache[feature]
                        if Xsyn is not None:
                            syn_mask, syn_y, _ = tasks[task]
                            syn_idx = np.where(syn_mask)[0]
                            X_synth_pool = Xsyn[syn_idx]
                            # Apply the same shape/scaler treatment as exp.
                            if model_name == "mlp":
                                Xs = X_synth_pool.reshape(
                                    len(X_synth_pool), -1).astype(np.float32)
                                if scaler is not None:
                                    Xs = scaler.transform(Xs)
                                X_synth_use = Xs
                            elif model_name in ("cnn", "transformer"):
                                if X_synth_pool.ndim == 4:
                                    n_, C_, H_, W_ = X_synth_pool.shape
                                    X_synth_use = X_synth_pool.transpose(
                                        0, 1, 3, 2).reshape(
                                        n_, C_ * W_, H_).astype(np.float32)
                                else:
                                    X_synth_use = X_synth_pool.astype(
                                        np.float32)
                                    if (X_synth_use.shape[1] !=
                                            in_shape[-1]
                                            and X_synth_use.ndim == 3):
                                        X_synth_use = (
                                            X_synth_use.transpose(0, 2, 1))
                            else:
                                X_synth_use = X_synth_pool.astype(np.float32)
                            y_synth_use = syn_y
                    row = _fine_tune(mdl, e_kind,
                                          X_use[tr], y_pool_e[tr],
                                          X_use[te], y_pool_e[te],
                                          epochs=args.epochs,
                                          X_synth=X_synth_use,
                                          y_synth=y_synth_use,
                                          unfreeze=unfreeze)
                except Exception as e:
                    print(f"  FAIL fit {tag} {unfreeze} k={k}: {e}",
                              flush=True)
                    continue
                row.update({
                    "tag": tag, "task": task, "model": model_name,
                    "feature": feature, "fraction": k,
                    "unfreeze": unfreeze,
                    "trainable_params": n_trainable,
                    "runtime_s": time.time() - t0,
                })
                rows.append(row)
                print(f"  {tag_full:<55s} {row['metric']:<8s} "
                          f"{row['value']:+.3f}  ({row['runtime_s']:.1f}s)",
                          flush=True)
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--syn", type=Path,
                      default=_REPO / "dataset" / "features.h5")
    p.add_argument("--exp", type=Path,
                      default=_REPO / "dataset"
                              / "experimental_features_balanced.h5")
    p.add_argument("--results", type=Path, default=_REPO / "results")
    p.add_argument("--epochs", type=int, default=8)
    args = p.parse_args()

    syn_labels = load_labels(args.syn)
    tasks = build_targets(syn_labels["type_code"], syn_labels["storey"],
                              syn_labels["end"], syn_labels["severity"])
    # Re-fit one StandardScaler per (task, flat-feature) on the synth
    # train fold so MLP cells see the same input distribution they
    # were trained on.
    syn_scalers: Dict[str, Dict[str, StandardScaler]] = {}
    flat_features = ("modal",)
    flat_data = {f: load_feature(args.syn, f) for f in flat_features}
    for tn, (mask, y_pool, kind) in tasks.items():
        syn_scalers[tn] = {}
        ipool = np.where(mask)[0]
        i_tr, _, _ = make_split(y_pool, kind)
        for f in flat_features:
            X_tr = flat_data[f][ipool[i_tr]].reshape(len(i_tr), -1)
            syn_scalers[tn][f] = StandardScaler().fit(X_tr)

    rows = _process_main(args, tasks, syn_scalers, args.exp, syn_labels)
    (args.results / "transfer_learning.json").write_text(
        json.dumps(rows, indent=2))
    print(f"\nwrote results/transfer_learning.json  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
