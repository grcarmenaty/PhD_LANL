"""Preliminary high-resolution CFDAC training: top CFDAC cell per task
re-trained at 1601² resolution (the native exp grid).

For each of the 10 tasks we pick the cell that scored best on its synth
test fold in the existing 128² study (mix of bespoke cnn2d/cfdac and
vision backbones) and re-train that model on CFDAC computed at full
1601² resolution from FRFs at df=0.0625 Hz. Tested on both held-out
synth (in-domain) and the full 2 638-case experimental set (zero-shot).

Reads:
  dataset/features_hires.h5                 (synth, 1601-bin FRFs)
  dataset/experimental_features_hires.h5    (exp,   1601-bin FRFs)
Writes:
  results_hires/per_case/{task}_{model}_{feature}.json
  results_hires/synth_test.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, r2_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO.parent / "pymodal") not in sys.path:
    sys.path.insert(0, str(_REPO.parent / "pymodal"))

from pymodal import utils as pm_utils
from ml_pipeline.tasks import build_targets
from ml_pipeline.train import make_split
import ml_pipeline.train as _train_mod
from ml_pipeline.models import Conv2DStack
from ml_pipeline.vision_models import VisionBackbone, VISION_BACKBONES


DEVICE = torch.device("cpu")
torch.set_num_threads(4)


# Top-CFDAC cell per task, identified from training_metrics.json + completed
# vision per-case JSONs by synth_test ranking. The 'feature' is the canonical
# 2-channel CFDAC (real + imag); for vision cells whose 128² peak was at
# cfdac_mag/cfdac_all we still use real+imag at hires per user spec.
TOP_CELLS: Dict[str, Tuple[str, str]] = {
    # task           : (model,        feature)
    "binary":         ("cnn2d",         "cfdac_realimag"),
    "col_location":   ("cnn2d",         "cfdac_realimag"),
    "mass_location":  ("cnn2d",         "cfdac_realimag"),
    "severity":       ("cnn2d",         "cfdac_realimag"),
    "type":           ("cnn2d",         "cfdac_realimag"),
    "is_bolt":        ("convnext_tiny", "cfdac_realimag"),
    "is_crack":       ("convnext_tiny", "cfdac_realimag"),
    "is_mass":        ("convnext_tiny", "cfdac_realimag"),
    "is_hole":        ("resnet50",      "cfdac_realimag"),
    "is_pristine":    ("resnet50",      "cfdac_realimag"),
}


# ---------------------------------------------------------------------------
# Lazy CFDAC dataset
# ---------------------------------------------------------------------------

class CFDACLazyDataset(Dataset):
    """Holds the complex FRFs for a subset of cases in RAM (cheap) and
    computes the 2-channel (real, imag) CFDAC matrix per sample at fetch
    time. The shared reference FRF is held alongside.

    Memory cost:
      * synth subsample 1 500 × 1601 × 9 × complex64 = 173 MB
      * exp full        2 638 × 1601 × 9 × complex64 = 304 MB
    """

    def __init__(self, H: np.ndarray, H_ref: np.ndarray, y: np.ndarray,
                  channels: tuple[str, ...] = ("real", "imag"),
                  normalize: bool = True):
        # H: (k, n_freq, 9) complex; H_ref: (n_freq, 9) complex
        self.H = H.astype(np.complex64, copy=False)
        self.H_ref = H_ref.astype(np.complex64, copy=False)
        self.y = y
        self.channels = channels
        self.normalize = normalize

    def __len__(self):
        return len(self.H)

    def __getitem__(self, idx):
        c = pm_utils.value_CFDAC(self.H_ref, self.H[idx])  # (n_freq, n_freq) complex64
        out = np.empty((len(self.channels), c.shape[0], c.shape[1]),
                        dtype=np.float32)
        for j, ch in enumerate(self.channels):
            if ch == "real":  out[j] = c.real
            elif ch == "imag": out[j] = c.imag
            elif ch == "mag":  out[j] = np.abs(c)
            elif ch == "phase": out[j] = np.angle(c)
        if self.normalize:
            # Per-sample mean-subtract on real/imag, phase /pi, mag 2x-1 then mean-subtract
            for j, ch in enumerate(self.channels):
                if ch in ("real", "imag"):
                    out[j] -= out[j].mean()
                elif ch == "mag":
                    out[j] = 2.0 * out[j] - 1.0
                    out[j] -= out[j].mean()
                elif ch == "phase":
                    out[j] /= np.float32(np.pi)
        return torch.from_numpy(out), self.y[idx]


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(model_name: str, n_channels: int, n_out: int,
                 kind: str) -> nn.Module:
    if model_name == "cnn2d":
        return Conv2DStack(n_channels=n_channels, n_out=n_out,
                            widths=(16, 32, 64), kernel_size=5,
                            regression=(kind == "reg"))
    if model_name in VISION_BACKBONES:
        return VisionBackbone(model_name, n_channels=n_channels,
                                n_out=n_out, regression=(kind == "reg"),
                                bounded_output=True, pretrained=True,
                                target_size=224, channel_adapter="timm_in_chans")
    raise ValueError(f"unknown model {model_name!r}")


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def _train(mdl, ds_tr, ds_va, args, kind, n_out, class_weight=None):
    dl = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=0)
    if kind == "cls":
        loss_fn = nn.CrossEntropyLoss(weight=class_weight)
    else:
        loss_fn = nn.MSELoss()
    is_vision = hasattr(mdl, "backbone")
    if is_vision:
        for p in mdl.backbone.parameters():
            p.requires_grad = False
        head_params = [p for n, p in mdl.named_parameters()
                            if not n.startswith("backbone.")]
        opt = torch.optim.AdamW(head_params, lr=args.lr, weight_decay=1e-4)
        sched = None
    else:
        opt = torch.optim.AdamW(mdl.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = None
    best = -np.inf; best_state = None
    t0 = time.time()
    for ep in range(args.epochs):
        if is_vision and ep == args.probe_epochs:
            for p in mdl.backbone.parameters():
                p.requires_grad = True
            opt = torch.optim.AdamW(
                [{"params": list(mdl.backbone.parameters()),
                    "lr": args.lr * args.backbone_lr_mult},
                 {"params": head_params, "lr": args.lr}],
                weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, args.epochs - args.probe_epochs))
            print(f"    epoch {ep+1}: unfreezing backbone", flush=True)
        mdl.train()
        for xb, yb in dl:
            opt.zero_grad()
            out = mdl(xb)
            if kind == "cls":
                yb_t = torch.as_tensor(yb).long()
                loss_fn(out, yb_t).backward()
            else:
                yb_t = torch.as_tensor(yb).float().unsqueeze(1)
                loss_fn(out, yb_t).backward()
            opt.step()
        if sched is not None: sched.step()
        # Validation
        mdl.eval()
        preds = []; truths = []
        with torch.no_grad():
            for xb, yb in dl_va:
                out = mdl(xb).cpu().numpy()
                if kind == "cls":
                    preds.append(out.argmax(1))
                else:
                    preds.append(out.squeeze(1) if out.ndim == 2 else out)
                truths.append(np.asarray(yb))
        p = np.concatenate(preds); t = np.concatenate(truths)
        if kind == "cls":
            metric = f1_score(t, p, labels=list(range(n_out)),
                                 average="macro", zero_division=0)
        else:
            metric = r2_score(t, p)
        print(f"      epoch {ep+1}/{args.epochs}  val={metric:+.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        if metric > best:
            best = float(metric)
            best_state = {k: v.detach().clone() for k, v in mdl.state_dict().items()}
    if best_state is not None:
        mdl.load_state_dict(best_state)
    return {"val": best, "runtime_s": time.time() - t0}


def _predict_dataset(mdl, ds, kind, batch=16):
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    mdl.eval()
    preds = []; probs = []
    with torch.no_grad():
        for xb, _ in dl:
            out = mdl(xb).cpu().numpy()
            if kind == "cls":
                preds.append(out.argmax(1))
                e_x = np.exp(out - out.max(axis=1, keepdims=True))
                probs.append(e_x / e_x.sum(axis=1, keepdims=True))
            else:
                preds.append(out.squeeze(1) if out.ndim == 2 else out)
    return np.concatenate(preds), (np.concatenate(probs) if kind == "cls" else None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--syn", type=Path,
                    default=_REPO / "dataset" / "features_hires.h5")
    p.add_argument("--exp", type=Path,
                    default=_REPO / "dataset" / "experimental_features_hires.h5")
    p.add_argument("--out", type=Path, default=_REPO / "results_hires")
    p.add_argument("--tasks", nargs="+", default=list(TOP_CELLS))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subsample", type=int, default=1500)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--probe-epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--backbone-lr-mult", type=float, default=0.1)
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    (a.out / "per_case").mkdir(parents=True, exist_ok=True)

    import random as _r
    _r.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    _train_mod.SEED = int(a.seed)

    # Load synth labels + tasks
    with h5py.File(a.syn, "r") as f:
        tc = f["type_code"][:].astype(np.int64)
        sto = f["storey"][:].astype(np.int64)
        end = f["end"][:].astype(np.int64)
        sev = f["severity"][:].astype(np.float32)
        H_ref_syn = f["reference/frf_complex"][:]  # (1601, 9) complex
    syn_tasks = build_targets(tc, sto, end, sev)
    print(f"syn tasks built: {sorted(syn_tasks)}")

    # Load exp labels + tasks + case names
    with h5py.File(a.exp, "r") as f:
        tc_e = f["type_code"][:].astype(np.int64)
        sto_e = f["storey"][:].astype(np.int64)
        end_e = f["end"][:].astype(np.int64)
        sev_e = f["severity"][:].astype(np.float32)
        names = [str(s) for s in f["names"][:]]
        H_ref_exp = f["reference/frf_complex"][:]  # (1601, 9) complex
    exp_tasks = build_targets(tc_e, sto_e, end_e, sev_e)
    print(f"exp tasks built: {sorted(exp_tasks)}")

    # Pre-cache exp complex FRFs (304 MB) once.
    print("loading exp FRFs into RAM…", flush=True)
    t0 = time.time()
    with h5py.File(a.exp, "r") as f:
        H_exp = (f["frf_real"][:] + 1j * f["frf_imag"][:]).astype(np.complex64)
    print(f"  exp FRFs shape {H_exp.shape}  ({time.time()-t0:.0f}s)", flush=True)

    # Preserve previously-completed synth entries so a per-task invocation
    # doesn't clobber the accumulated synth_test.json (cells are run one at a
    # time for per-cell checkpoint/commit).
    synth_results = {}
    _synth_path = a.out / "synth_test.json"
    if _synth_path.exists():
        try:
            synth_results = json.loads(_synth_path.read_text())
            print(f"loaded {len(synth_results)} prior synth entries")
        except Exception:
            synth_results = {}
    for task in a.tasks:
        if task not in TOP_CELLS:
            print(f"  skip {task}: not in TOP_CELLS"); continue
        model_name, feature = TOP_CELLS[task]
        tag = f"{task}_{model_name}_{feature}_hires1601"
        pc_path = a.out / "per_case" / f"{tag}.json"
        if pc_path.exists() and not a.force:
            print(f"  skip {tag} (exists)"); continue
        kind = syn_tasks[task][2]
        print(f"\n== {tag} ==  kind={kind}", flush=True)

        # Subsample synth pool
        mask, y_pool, _ = syn_tasks[task]
        full_pool_idx = np.where(mask)[0]
        rng = np.random.default_rng(20260518 + a.seed)
        sub_n = min(a.subsample, len(full_pool_idx))
        keep = np.sort(rng.choice(len(full_pool_idx), size=sub_n, replace=False))
        idx_pool = full_pool_idx[keep]
        y = y_pool[keep]
        # Load FRFs for the subsample only
        t1 = time.time()
        with h5py.File(a.syn, "r") as f:
            order = np.argsort(idx_pool)
            sorted_rows = idx_pool[order]
            re_arr = f["frf_real"][sorted_rows]
            im_arr = f["frf_imag"][sorted_rows]
        # restore original order
        H_syn = np.empty_like(re_arr, dtype=np.complex64)
        H_syn[order] = (re_arr + 1j * im_arr).astype(np.complex64)
        print(f"  synth FRFs {H_syn.shape} ({time.time()-t1:.0f}s)", flush=True)

        # 70/15/15 split + class weights
        i_tr, i_va, i_te = make_split(y, kind)
        y_tr, y_va, y_te = y[i_tr], y[i_va], y[i_te]
        n_out = (int(y.max()) + 1) if kind == "cls" else 1
        cls_w = None
        if kind == "cls":
            counts = np.bincount(y_tr, minlength=n_out).astype(np.float32)
            inv = counts.sum() / np.clip(counts * n_out, 1e-6, None)
            cls_w = torch.as_tensor(inv).float()
            print(f"  class weights: {cls_w.tolist()}", flush=True)

        n_channels = 2  # cfdac_realimag
        mdl = build_model(model_name, n_channels=n_channels,
                            n_out=n_out, kind=kind)
        ds_tr = CFDACLazyDataset(H_syn[i_tr], H_ref_syn, y_tr)
        ds_va = CFDACLazyDataset(H_syn[i_va], H_ref_syn, y_va)
        ds_te = CFDACLazyDataset(H_syn[i_te], H_ref_syn, y_te)
        out = _train(mdl, ds_tr, ds_va, a, kind, n_out, class_weight=cls_w)

        # Synth test
        pred_te, _ = _predict_dataset(mdl, ds_te, kind, batch=a.batch)
        if kind == "cls":
            test_metric = accuracy_score(y_te, pred_te)
            test_mf1 = f1_score(y_te, pred_te, labels=list(range(n_out)),
                                  average="macro", zero_division=0)
        else:
            test_metric = r2_score(y_te, pred_te)
            test_mf1 = None
        print(f"  synth val={out['val']:+.3f}  test_metric={test_metric:+.3f}"
              + (f"  macro_f1={test_mf1:+.3f}" if test_mf1 is not None else ""),
              flush=True)

        synth_results[tag] = {
            "task": task, "model": model_name, "feature": feature,
            "n_target": 1601, "kind": kind, "n_out": n_out,
            "synth_val": out["val"],
            "synth_test_metric": float(test_metric),
            "synth_test_macro_f1": (float(test_mf1) if test_mf1 is not None else None),
            "runtime_s": out["runtime_s"],
        }
        (a.out / "synth_test.json").write_text(json.dumps(synth_results, indent=2))

        # Experimental zero-shot eval
        mask_e, e_y, e_kind = exp_tasks[task]
        idx_e = np.where(mask_e)[0]
        H_exp_sub = H_exp[idx_e]
        # y aligned to mask subset (length = sum(mask_e))
        ds_exp = CFDACLazyDataset(H_exp_sub, H_ref_exp, e_y)
        pred_e, prob_e = _predict_dataset(mdl, ds_exp, e_kind, batch=a.batch)
        rows = []
        for i, ix in enumerate(idx_e):
            row = {"case": names[ix],
                    "y_true": int(e_y[i]) if e_kind == "cls" else float(e_y[i]),
                    "y_pred": int(pred_e[i]) if e_kind == "cls" else float(pred_e[i])}
            if prob_e is not None:
                row["proba"] = [float(p) for p in prob_e[i]]
            rows.append(row)
        meta = {"task": task, "model": model_name, "feature": feature,
                 "n_target": 1601, "n_channels": n_channels,
                 "kind": e_kind, "n_out": int(n_out),
                 "synth_test": float(test_metric),
                 "synth_test_macro_f1": (float(test_mf1) if test_mf1 is not None else None),
                 "input_normalized": True}
        pc_path.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
        print(f"  wrote per-case {tag}.json", flush=True)
        del mdl, H_syn, ds_tr, ds_va, ds_te, ds_exp

    print(f"\nDone. synth results: {a.out / 'synth_test.json'}", flush=True)


if __name__ == "__main__":
    main()
