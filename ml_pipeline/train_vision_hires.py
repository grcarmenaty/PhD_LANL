"""High-resolution CFDAC vision training (preliminary).

Trains top-3 ImageNet vision backbones on CFDAC computed at the **native
381² resolution** (no decimation), reading FRFs straight off disk. This
is the higher-res arm of the comparison against the 128² streaming sweep
in ``train_vision.py``.

Key differences from train_vision.py:
  * No disk-resident CFDAC; per-batch CFDAC is computed on the fly.
  * The synth subsample is selected FIRST so we only materialise CFDAC
    for those ~1 500 rows (a full 10k × 381² × 2ch × float32 tensor would
    be 11 GB, out of reach on the current container).
  * Per-cell streams a per-case JSON; no .pt files.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml_pipeline.tasks import build_targets
from ml_pipeline.train import load_labels, make_split
import ml_pipeline.train as _train_mod
from ml_pipeline.vision_models import VisionBackbone, VISION_BACKBONES
from ml_pipeline.cfdac_runtime import (
    compute_cfdac_runtime, per_sample_normalize_cfdac)
from ml_pipeline.eval_vision_percase import load_exp_context, write_per_case


DEVICE = torch.device("cpu")
torch.set_num_threads(4)

CHANNEL_MAP = {
    "realimag":  ("real", "imag"),
    "mag":       ("mag",),
    "magphase":  ("mag", "phase"),
    "all":       ("real", "imag", "mag", "phase"),
}


def _train_one(mdl: VisionBackbone, X_tr, y_tr, X_va, y_va,
                kind: str, n_out: int, args, class_weight=None) -> Dict:
    if kind == "cls":
        loss_fn = nn.CrossEntropyLoss(weight=class_weight)
        ytr = torch.as_tensor(y_tr).long()
        yva = torch.as_tensor(y_va).long()
    else:
        loss_fn = nn.MSELoss()
        ytr = torch.as_tensor(y_tr).float().unsqueeze(1)
        yva = torch.as_tensor(y_va).float().unsqueeze(1)
    Xtr_t = torch.as_tensor(X_tr).float()
    Xva_t = torch.as_tensor(X_va).float()
    dl = DataLoader(TensorDataset(Xtr_t, ytr),
                       batch_size=args.batch, shuffle=True)

    # Linear-probe -> fine-tune schedule.
    for p in mdl.backbone.parameters():
        p.requires_grad = False
    head_params = [p for n, p in mdl.named_parameters()
                       if not n.startswith("backbone.")]
    opt = torch.optim.AdamW(head_params, lr=args.lr, weight_decay=1e-4)
    sched = None
    best = -np.inf; best_state = None
    t0 = time.time()
    for ep in range(args.epochs):
        if ep == args.probe_epochs:
            for p in mdl.backbone.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(
                [{"params": list(mdl.backbone.parameters()),
                    "lr": args.lr * args.backbone_lr_mult},
                 {"params": head_params, "lr": args.lr}],
                weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, args.epochs - args.probe_epochs))
            print(f"    epoch {ep+1}: unfreeze (bb_lr="
                  f"{args.lr*args.backbone_lr_mult:.1e}, head_lr={args.lr:.1e})",
                  flush=True)
        mdl.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss_fn(mdl(xb), yb).backward()
            opt.step()
        if sched is not None:
            sched.step()
        mdl.eval()
        with torch.no_grad():
            out_va = mdl(Xva_t).cpu()
            if kind == "cls":
                pred = out_va.argmax(1).numpy()
                metric = f1_score(y_va, pred, labels=list(range(n_out)),
                                   average="macro", zero_division=0)
            else:
                metric = r2_score(y_va, out_va.squeeze(1).numpy())
        print(f"      epoch {ep+1}/{args.epochs}  val={metric:+.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        if metric > best:
            best = float(metric)
            best_state = {k: v.detach().clone() for k, v in mdl.state_dict().items()}
    if best_state is not None:
        mdl.load_state_dict(best_state)
    return {"val": best, "runtime_s": time.time() - t0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--syn", type=Path,
                    default=_REPO / "dataset" / "features.h5")
    p.add_argument("--exp", type=Path,
                    default=_REPO / "dataset" / "experimental_features.h5")
    p.add_argument("--per-case-out", type=Path, required=True)
    p.add_argument("--backbones", nargs="+",
                    default=["convnext_tiny", "resnet50", "vit_b_16"])
    p.add_argument("--tasks", nargs="+",
                    default=["type", "is_bolt", "is_hole", "severity",
                              "col_location", "mass_location", "is_mass",
                              "is_crack", "is_pristine", "binary"])
    p.add_argument("--cfdac-channels", default="realimag",
                    choices=list(CHANNEL_MAP),
                    help="Which CFDAC channels to compute.")
    p.add_argument("--n-target", type=int, default=381,
                    help="CFDAC resolution (≤ 381, the native FRF length).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subsample", type=int, default=1500)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--probe-epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--backbone-lr-mult", type=float, default=0.1)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    # Seed everything.
    import random as _r
    _r.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    _train_mod.SEED = int(args.seed)

    args.per_case_out.mkdir(parents=True, exist_ok=True)
    channels = CHANNEL_MAP[args.cfdac_channels]
    feature_tag = f"cfdac_{args.cfdac_channels}_hires{args.n_target}"

    syn_labels = load_labels(args.syn)
    tasks = build_targets(syn_labels["type_code"], syn_labels["storey"],
                            syn_labels["end"], syn_labels["severity"])

    e_tasks, exp_names = load_exp_context(args.exp)

    # Pre-compute exp CFDAC at hires ONCE (used by every cell).
    print(f"hires-cfdac: precompute exp at {args.n_target}² channels={channels}",
          flush=True)
    t0 = time.time()
    X_exp = compute_cfdac_runtime(args.exp, rows=None,
                                    n_target=args.n_target,
                                    channels=channels)
    X_exp = per_sample_normalize_cfdac(X_exp, channels)
    print(f"  exp shape: {X_exp.shape}  ({time.time()-t0:.0f}s, "
          f"{X_exp.nbytes/1e9:.1f} GB)", flush=True)

    print(f"hires sweep [seed={args.seed}]: {len(args.backbones)} backbones x "
          f"{len(args.tasks)} tasks = {len(args.backbones)*len(args.tasks)} cells",
          flush=True)

    for backbone in args.backbones:
        for task in args.tasks:
            tag = f"{task}_{backbone}_{feature_tag}"
            pc_path = args.per_case_out / f"{tag}.json"
            if pc_path.exists() and not args.force:
                print(f"  skip {tag} (exists)", flush=True)
                continue
            mask, y_pool, kind = tasks[task]
            # Subsample synth pool first, then compute CFDAC only for those rows.
            full_pool_idx = np.where(mask)[0]
            rng = np.random.default_rng(20260518 + args.seed)
            sub_n = min(args.subsample, len(full_pool_idx))
            keep = np.sort(rng.choice(len(full_pool_idx), size=sub_n,
                                       replace=False))
            idx_pool = full_pool_idx[keep]
            y = y_pool[keep]
            print(f"== {tag} ==  loading hires CFDAC for {sub_n} synth rows",
                  flush=True)
            t1 = time.time()
            X = compute_cfdac_runtime(args.syn, rows=idx_pool,
                                        n_target=args.n_target,
                                        channels=channels)
            X = per_sample_normalize_cfdac(X, channels)
            print(f"  synth CFDAC shape={X.shape} "
                  f"({time.time()-t1:.0f}s, {X.nbytes/1e9:.2f} GB)",
                  flush=True)

            i_tr, i_va, i_te = make_split(y, kind)
            X_tr = X[i_tr]; X_va = X[i_va]; X_te = X[i_te]
            y_tr = y[i_tr]; y_va = y[i_va]; y_te = y[i_te]
            n_out = (int(y.max()) + 1) if kind == "cls" else 1

            cls_w = None
            if kind == "cls":
                counts = np.bincount(y_tr, minlength=n_out).astype(np.float32)
                inv = counts.sum() / np.clip(counts * n_out, 1e-6, None)
                cls_w = torch.as_tensor(inv).float()

            mdl = VisionBackbone(backbone, n_channels=len(channels),
                                  n_out=n_out, regression=(kind == "reg"),
                                  bounded_output=True, pretrained=True,
                                  target_size=224,
                                  channel_adapter="timm_in_chans")
            out = _train_one(mdl, X_tr, y_tr, X_va, y_va, kind, n_out,
                              args, class_weight=cls_w)
            # Test-fold metric
            mdl.eval()
            with torch.no_grad():
                yhat = mdl(torch.as_tensor(X_te).float()).cpu()
                if kind == "cls":
                    pred = yhat.argmax(1).numpy()
                    test_metric = accuracy_score(y_te, pred)
                else:
                    pred = yhat.squeeze(1).numpy()
                    test_metric = r2_score(y_te, pred)
            print(f"    synth val={out['val']:+.3f}  test={test_metric:+.3f}  "
                  f"runtime={out['runtime_s']:.0f}s", flush=True)

            # Per-case JSON on exp set (using the precomputed X_exp).
            mask_e, e_y, e_kind = e_tasks[task]
            idx_e = np.where(mask_e)[0]
            Xe = X_exp[idx_e]
            mdl.eval()
            outs = []; probs = []
            with torch.no_grad():
                for i in range(0, len(Xe), args.batch):
                    xb = torch.as_tensor(Xe[i:i + args.batch]).float()
                    raw = mdl(xb).cpu().numpy()
                    outs.append(raw)
                    if e_kind == "cls":
                        e_x = np.exp(raw - raw.max(axis=1, keepdims=True))
                        probs.append(e_x / e_x.sum(axis=1, keepdims=True))
            raw = np.concatenate(outs, axis=0)
            if e_kind == "cls":
                proba = np.concatenate(probs, axis=0)
                pred = raw.argmax(1)
            else:
                pred = raw.squeeze(1) if raw.ndim == 2 else raw
                proba = None
            rows = []
            for i, ix in enumerate(idx_e):
                row = {"case": exp_names[ix],
                        "y_true": int(e_y[i]) if e_kind == "cls" else float(e_y[i]),
                        "y_pred": int(pred[i]) if e_kind == "cls" else float(pred[i])}
                if proba is not None:
                    row["proba"] = [float(p) for p in proba[i]]
                rows.append(row)
            meta = {"task": task, "backbone": backbone, "feature": feature_tag,
                      "kind": e_kind, "n_out": int(n_out),
                      "n_channels": len(channels),
                      "n_target": args.n_target,
                      "synth_test": float(test_metric),
                      "synth_val": float(out["val"]),
                      "input_normalized": True}
            pc_path.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
            print(f"    wrote per-case {tag}.json", flush=True)
            del X, X_tr, X_va, X_te, mdl


if __name__ == "__main__":
    main()
