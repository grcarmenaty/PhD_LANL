"""Synth-only training of ImageNet-pretrained vision backbones on CFDAC.

Compares the existing cnn2d-on-CFDAC baseline against five general-
purpose vision-model backbones (ResNet50, EfficientNet-B0, ConvNeXt-T,
Swin-T, ViT-B/16) trained on the same 10 000-sample synth dataset.
Training never sees experimental data; the cross-domain metric is
zero-shot on the full 2638-case IQS set.

Output per cell:
  results/models_vision/<task>_<backbone>_<feature>.pt
                   {state_dict, model_name, n_out, in_shape,
                    input_normalized: True, vision_backbone: True,
                    vision_backbone_name: <backbone>}

Aggregate metrics rolled into results/vision_eval.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, r2_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml_pipeline.tasks import build_targets
from ml_pipeline.train import (load_labels, load_feature, make_split,
                                    _CFDAC_VARIANTS)
from ml_pipeline.vision_models import (
    VisionBackbone, VISION_BACKBONES, is_vision_backbone,
)


DEVICE = torch.device("cpu")
torch.set_num_threads(4)


def _cls_or_reg(kind: str) -> str:
    return "reg" if kind == "reg" else "cls"


def _train_one_cell(backbone_name: str, feature: str, task_name: str,
                      mask: np.ndarray, y_pool: np.ndarray, kind: str,
                      syn_path: Path, args) -> Dict:
    """Train a single (task, backbone, feature) cell on synth data only."""
    print(f"  load synth feature: {feature}", flush=True)
    X = load_feature(syn_path, feature)
    if feature not in _CFDAC_VARIANTS:
        raise ValueError(f"vision backbones expect CFDAC features; got {feature!r}")
    # X is already (n, C, H, W) — load_feature does the stacking.
    if X.ndim == 5:
        # 3-D variant (n, 1, D, H, W) -> reshape D into channels.
        X = X.reshape(X.shape[0], X.shape[1] * X.shape[2], X.shape[3], X.shape[4])
    n_channels = int(X.shape[1])
    print(f"    shape={X.shape} channels={n_channels}", flush=True)

    idx_pool = np.where(mask)[0]
    y = y_pool
    if args.subsample is not None and args.subsample < len(idx_pool):
        rng = np.random.default_rng(20260518)
        keep = rng.choice(len(idx_pool), size=args.subsample, replace=False)
        idx_pool = idx_pool[np.sort(keep)]
        y = y_pool[np.sort(keep)] if kind == "reg" else y_pool[np.sort(keep)]
    # Actually keep the alignment: y_pool already aligned to mask-True.
    # Re-derive y from the kept indices.
    if args.subsample is not None and args.subsample < len(np.where(mask)[0]):
        # ipool order matches y_pool order (by construction in tasks)
        full_pool_idx = np.where(mask)[0]
        rng = np.random.default_rng(20260518)
        keep_local = np.sort(rng.choice(len(full_pool_idx),
                                              size=args.subsample, replace=False))
        idx_pool = full_pool_idx[keep_local]
        y = y_pool[keep_local]

    # 70/15/15 split, stratified for cls
    i_tr, i_va, i_te = make_split(y, kind)
    X_tr = X[idx_pool[i_tr]]
    X_va = X[idx_pool[i_va]]
    X_te = X[idx_pool[i_te]]
    y_tr = y[i_tr]; y_va = y[i_va]; y_te = y[i_te]

    n_out = (int(y.max()) + 1) if kind == "cls" else 1
    mdl = VisionBackbone(backbone_name, n_channels=n_channels, n_out=n_out,
                              regression=(kind == "reg"),
                              bounded_output=True, pretrained=True,
                              target_size=args.target_size,
                              channel_adapter=args.channel_adapter
                              ).to(DEVICE)

    if kind == "cls":
        # Tier-1 A1: class-weighted CE defends minorities against the
        # majority-class collapse pattern that produced misleading
        # accuracies in the first vision sweep.
        cls_weight = None
        if args.class_weights == "inverse-freq":
            cls_counts = np.bincount(y_tr, minlength=n_out).astype(np.float32)
            inv = (cls_counts.sum() / np.clip(cls_counts * n_out, 1e-6, None))
            cls_weight = torch.as_tensor(inv).float().to(DEVICE)
            print(f"    class_weights inverse-freq: {cls_weight.tolist()}",
                      flush=True)
        loss_fn = nn.CrossEntropyLoss(weight=cls_weight)
        ytr = torch.as_tensor(y_tr).long()
        yva = torch.as_tensor(y_va).long()
        yte = torch.as_tensor(y_te).long()
    else:
        loss_fn = nn.MSELoss()
        ytr = torch.as_tensor(y_tr).float().unsqueeze(1)
        yva = torch.as_tensor(y_va).float().unsqueeze(1)
        yte = torch.as_tensor(y_te).float().unsqueeze(1)

    Xtr_t = torch.as_tensor(X_tr).float()
    Xva_t = torch.as_tensor(X_va).float()
    Xte_t = torch.as_tensor(X_te).float()
    dl = DataLoader(TensorDataset(Xtr_t, ytr),
                       batch_size=args.batch, shuffle=True)

    # Tier-1 A2: linear-probe → fine-tune.  First `probe_epochs`
    # freeze the backbone (train only the head + channel projector if
    # any).  Remaining epochs unfreeze everything and use a smaller
    # backbone lr (`args.lr * args.backbone_lr_mult`) so the
    # pretrained features aren't blown up by the head's gradient.
    def _set_backbone_trainable(flag: bool):
        for p in mdl.backbone.parameters():
            p.requires_grad = flag
        # head + projector stay trainable always
    _set_backbone_trainable(False)

    head_params = [p for n, p in mdl.named_parameters()
                       if not n.startswith("backbone.")]
    probe_opt = torch.optim.AdamW(head_params, lr=args.lr,
                                          weight_decay=1e-4)
    probe_epochs = min(int(args.probe_epochs), args.epochs)

    t0 = time.time()
    best_val = -np.inf
    best_state = None
    opt = probe_opt
    sched = None
    for ep in range(args.epochs):
        if ep == probe_epochs:
            _set_backbone_trainable(True)
            # Differential lr: backbone lower, head full
            backbone_params = list(mdl.backbone.parameters())
            opt = torch.optim.AdamW(
                [{"params": backbone_params,
                    "lr": args.lr * float(args.backbone_lr_mult)},
                 {"params": head_params, "lr": args.lr}],
                weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, args.epochs - probe_epochs))
            print(f"    epoch {ep + 1}: unfreezing backbone "
                      f"(backbone_lr={args.lr * float(args.backbone_lr_mult):.1e}, "
                      f"head_lr={args.lr:.1e})", flush=True)
        mdl.train()
        for xb, yb in dl:
            opt.zero_grad()
            out = mdl(xb)
            loss_fn(out, yb).backward()
            opt.step()
        if sched is not None:
            sched.step()
        mdl.eval()
        with torch.no_grad():
            out_va = mdl(Xva_t).cpu()
            if kind == "cls":
                pred_va = out_va.argmax(1).numpy()
                # Tier-1 A4: select checkpoint by macro-F1 -- on a
                # balanced val fold this matches accuracy, but it
                # makes the right inductive choice if anything pulls
                # the val distribution off-balance.
                acc = accuracy_score(y_va, pred_va)
                f1m = f1_score(y_va, pred_va,
                                  labels=list(range(n_out)),
                                  average="macro", zero_division=0)
                metric = f1m if args.select_by == "macro_f1" else acc
            else:
                metric = r2_score(y_va, out_va.squeeze(1).numpy())
                acc = float("nan"); f1m = float("nan")
        if kind == "cls":
            print(f"      epoch {ep+1}/{args.epochs}  "
                      f"acc={acc:+.4f} macro-F1={f1m:+.4f}  "
                      f"({time.time()-t0:.0f}s elapsed)", flush=True)
        else:
            print(f"      epoch {ep+1}/{args.epochs}  R2={metric:+.4f} "
                      f"({time.time()-t0:.0f}s elapsed)", flush=True)
        if metric > best_val:
            best_val = metric
            best_state = {k: v.detach().clone()
                              for k, v in mdl.state_dict().items()}

    if best_state is not None:
        mdl.load_state_dict(best_state)
    mdl.eval()
    with torch.no_grad():
        out_te = mdl(Xte_t).cpu()
        if kind == "cls":
            yhat = out_te.argmax(1).numpy()
            test_metric = accuracy_score(y_te, yhat)
            mae = None
        else:
            yhat = out_te.squeeze(1).numpy()
            test_metric = r2_score(y_te, yhat)
            mae = mean_absolute_error(y_te, yhat)

    return {
        "model": mdl,
        "state_dict": mdl.state_dict(),
        "n_out": n_out,
        "in_shape": list(X_tr.shape[1:]),
        "n_channels": n_channels,
        "val_metric": float(best_val),
        "test_metric": float(test_metric),
        "mae": None if mae is None else float(mae),
        "runtime_s": time.time() - t0,
    }


def _exp_eval(backbone_name: str, mdl: nn.Module, feature: str, task: str,
                exp_path: Path, batch: int = 32) -> Dict:
    """Zero-shot evaluation on the full 2638-case experimental set.

    Batches inference so the 224x224 upsample doesn't materialise the
    whole feature tensor at once (2638 * 4 * 224 * 224 * float32 ≈ 2 GB).
    """
    from ml_pipeline.evaluate_full_experimental import _exp_load_feature
    X = _exp_load_feature(exp_path, feature, normalize=True)
    if X.ndim == 5:
        X = X.reshape(X.shape[0], X.shape[1] * X.shape[2],
                        X.shape[3], X.shape[4])
    with h5py.File(exp_path, "r") as f:
        tc = f["type_code"][:].astype(np.int64)
        sto = f["storey"][:].astype(np.int64)
        end = f["end"][:].astype(np.int64)
        sev = f["severity"][:].astype(np.float32)
    e_tasks = build_targets(tc, sto, end, sev)
    e_mask, e_y, e_kind = e_tasks[task]
    idx = np.where(e_mask)[0]
    Xe = X[idx]
    mdl.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(Xe), batch):
            xb = torch.as_tensor(Xe[i:i + batch]).float()
            outs.append(mdl(xb).cpu().numpy())
    out = np.concatenate(outs, axis=0)
    if e_kind == "cls":
        pred = out.argmax(1)
        metric = accuracy_score(e_y, pred)
        return {"metric_name": "accuracy", "value": float(metric),
                  "mae": None, "n": int(len(idx))}
    pred = out.squeeze(1) if out.ndim == 2 else out
    return {"metric_name": "R2",
            "value": float(r2_score(e_y, pred)),
            "mae": float(mean_absolute_error(e_y, pred)),
            "n": int(len(idx))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--syn", type=Path,
                      default=_REPO / "dataset" / "features.h5")
    p.add_argument("--exp", type=Path,
                      default=_REPO / "dataset" / "experimental_features.h5")
    p.add_argument("--out", type=Path, default=_REPO / "results")
    p.add_argument("--backbones", nargs="+",
                      default=list(VISION_BACKBONES.keys()),
                      help="Subset of vision backbones to train.")
    p.add_argument("--features", nargs="+",
                      default=["cfdac_all"],
                      help="Which CFDAC variants to train on.")
    p.add_argument("--tasks", nargs="+",
                      default=["type", "severity", "col_location",
                                  "mass_location"])
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--target-size", type=int, default=224)
    p.add_argument("--subsample", type=int, default=None,
                      help="If set, randomly sub-sample the synth pool "
                              "to this many rows before splitting (eg. "
                              "3000 to keep runs fast).")
    p.add_argument("--force", action="store_true",
                      help="Retrain even if the .pt artefact exists.")
    # ── Tier-1 vision-sweep fixes ───────────────────────────────────────
    p.add_argument("--class-weights",
                      choices=("none", "inverse-freq"),
                      default="inverse-freq",
                      help="Weight the CE loss to defend minorities.")
    p.add_argument("--probe-epochs", type=int, default=2,
                      help="Number of linear-probe epochs (backbone "
                              "frozen) before unfreezing.")
    p.add_argument("--backbone-lr-mult", type=float, default=0.1,
                      help="Backbone lr multiplier after unfreezing "
                              "(differential learning rate).")
    p.add_argument("--channel-adapter",
                      choices=("first_conv_replace", "projector",
                                  "passthrough"),
                      default="projector",
                      help="How to bridge ImageNet 3-ch stem to "
                              "CFDAC's n_channels.")
    p.add_argument("--select-by",
                      choices=("accuracy", "macro_f1"),
                      default="macro_f1",
                      help="Which val metric drives best-epoch "
                              "checkpoint selection.")
    args = p.parse_args()

    print(f"vision sweep: {len(args.backbones)} backbones x "
              f"{len(args.features)} features x {len(args.tasks)} tasks "
              f"= {len(args.backbones)*len(args.features)*len(args.tasks)} cells",
              flush=True)

    syn_labels = load_labels(args.syn)
    tasks = build_targets(syn_labels["type_code"], syn_labels["storey"],
                              syn_labels["end"], syn_labels["severity"])

    models_dir = args.out / "models_vision"
    models_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for backbone in args.backbones:
        for feature in args.features:
            for task in args.tasks:
                tag = f"{task}_{backbone}_{feature}"
                art = models_dir / f"{tag}.pt"
                if art.exists() and not args.force:
                    print(f"  skip {tag} (cached)")
                    # Still load + eval to add row.
                    blob = torch.load(art, map_location="cpu",
                                          weights_only=False)
                    n_out = blob["n_out"]
                    n_channels = blob["n_channels"]
                    # Honour the artefact's recorded channel adapter
                    # so older first_conv_replace artefacts continue
                    # to load even after the default switches.
                    adapter = blob.get("channel_adapter",
                                          "first_conv_replace")
                    mdl = VisionBackbone(backbone, n_channels=n_channels,
                                              n_out=n_out,
                                              regression=(tasks[task][2]=="reg"),
                                              bounded_output=True,
                                              pretrained=False,
                                              channel_adapter=adapter)
                    mdl.load_state_dict(blob["state_dict"])
                else:
                    print(f"== {tag} ==", flush=True)
                    mask, y_pool, kind = tasks[task]
                    out = _train_one_cell(backbone, feature, task,
                                                mask, y_pool, kind,
                                                args.syn, args)
                    mdl = out["model"]
                    blob = {"state_dict": out["state_dict"],
                              "model_name": "vision_backbone",
                              "vision_backbone_name": backbone,
                              "n_out": out["n_out"],
                              "n_channels": out["n_channels"],
                              "in_shape": out["in_shape"],
                              "input_normalized": True,
                              "val_metric": out["val_metric"],
                              "test_metric": out["test_metric"],
                              "mae": out["mae"],
                              "runtime_s": out["runtime_s"],
                              "epochs": args.epochs,
                              "target_size": args.target_size,
                              # Tier-1 settings stamped on the artefact
                              # so reconstruction at eval time matches.
                              "channel_adapter": args.channel_adapter,
                              "class_weights": args.class_weights,
                              "probe_epochs": int(args.probe_epochs),
                              "backbone_lr_mult": float(args.backbone_lr_mult),
                              "select_by": args.select_by}
                    torch.save(blob, art)
                    print(f"    synth val={out['val_metric']:+.3f}  "
                              f"test={out['test_metric']:+.3f}  "
                              f"runtime={out['runtime_s']:.0f}s", flush=True)
                # Cross-domain eval (zero-shot)
                exp_res = _exp_eval(backbone, mdl, feature, task, args.exp)
                row = {
                    "task": task, "backbone": backbone, "feature": feature,
                    "n_channels": blob["n_channels"],
                    "synth_val": float(blob.get("val_metric", float("nan"))),
                    "synth_test": float(blob.get("test_metric", float("nan"))),
                    "exp_metric_name": exp_res["metric_name"],
                    "exp_value": exp_res["value"],
                    "exp_mae": exp_res["mae"],
                    "exp_n": exp_res["n"],
                    "runtime_s": float(blob.get("runtime_s", 0)),
                }
                rows.append(row)
                print(f"    exp zero-shot {exp_res['metric_name']}="
                          f"{exp_res['value']:+.3f}  (n={exp_res['n']})",
                          flush=True)
                # Incremental write
                (args.out / "vision_eval.json").write_text(
                    json.dumps(rows, indent=2))

    print(f"\nwrote {args.out / 'vision_eval.json'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
