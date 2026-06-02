"""Per-case predictions on the full 2638-case experimental set for
vision cells.

Two entry points:
  * ``write_per_case(...)`` — score one in-memory model and write its
    ``<task>_<backbone>_<feature>.json``. Imported by ``train_vision`` so
    the streaming sweep can produce per-case output without ever leaving
    a large ``.pt`` on disk (the container has very little free space).
  * ``main()`` — batch mode: score every ``.pt`` in a models directory.

Output JSON: {"meta": {...}, "rows": [{case, y_true, y_pred, proba?}, …]}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml_pipeline.tasks import build_targets
from ml_pipeline.vision_models import VisionBackbone, VISION_BACKBONES


def load_exp_context(exp_path: Path):
    """Read the experimental labels + case names once; build the task
    table. Returns (e_tasks, names)."""
    with h5py.File(exp_path, "r") as f:
        tc = f["type_code"][:].astype(np.int64)
        sto = f["storey"][:].astype(np.int64)
        end = f["end"][:].astype(np.int64)
        sev = f["severity"][:].astype(np.float32)
        names = [str(s) for s in f["names"][:]]
    e_tasks = build_targets(tc, sto, end, sev)
    return e_tasks, names


def write_per_case(mdl, task: str, backbone: str, feature: str,
                    n_out: int, n_channels: int, exp_path: Path,
                    e_tasks: dict, names: list, out_dir: Path,
                    synth_test: float = float("nan"),
                    feat_cache: dict | None = None,
                    batch: int = 32) -> Path:
    """Zero-shot score one model on the full experimental set and write
    the per-case JSON. ``feat_cache`` (optional) avoids re-reading the
    exp feature across calls that share a feature."""
    from ml_pipeline.evaluate_full_experimental import _exp_load_feature
    out_dir.mkdir(parents=True, exist_ok=True)
    if feat_cache is not None and feature in feat_cache:
        X = feat_cache[feature]
    else:
        X = _exp_load_feature(exp_path, feature, normalize=True)
        if X.ndim == 5:
            X = X.reshape(X.shape[0], X.shape[1] * X.shape[2],
                            X.shape[3], X.shape[4])
        if feat_cache is not None:
            feat_cache[feature] = X

    mask, e_y, kind = e_tasks[task]
    idx = np.where(mask)[0]
    Xe = X[idx]
    mdl.eval()
    outs, probs = [], []
    with torch.no_grad():
        for i in range(0, len(Xe), batch):
            xb = torch.as_tensor(Xe[i:i + batch]).float()
            out = mdl(xb).cpu().numpy()
            outs.append(out)
            if kind == "cls":
                e_x = np.exp(out - out.max(axis=1, keepdims=True))
                probs.append(e_x / e_x.sum(axis=1, keepdims=True))
    out = np.concatenate(outs, axis=0)
    if kind == "cls":
        proba = np.concatenate(probs, axis=0)
        pred = out.argmax(1)
    else:
        pred = out.squeeze(1) if out.ndim == 2 else out
        proba = None

    rows = []
    # e_y is aligned to the mask-subset (length = sum(mask)); the label is
    # e_y[i] while the case name uses the full-array index ix = idx[i].
    for i, ix in enumerate(idx):
        row = {"case": names[ix],
                 "y_true": int(e_y[i]) if kind == "cls" else float(e_y[i]),
                 "y_pred": int(pred[i]) if kind == "cls" else float(pred[i])}
        if proba is not None:
            row["proba"] = [float(p) for p in proba[i]]
        rows.append(row)
    meta = {"task": task, "backbone": backbone, "feature": feature,
              "kind": kind, "n_out": int(n_out), "n_channels": int(n_channels),
              "synth_test": float(synth_test), "input_normalized": True}
    out_path = out_dir / f"{task}_{backbone}_{feature}.json"
    out_path.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2))
    return out_path


def _parse_tag(tag: str, known_backbones, e_tasks):
    feature = None
    for fc in ("cfdac_realimag", "cfdac_magphase", "cfdac_all",
               "cfdac_real", "cfdac_imag", "cfdac_mag", "cfdac_phase", "cfdac"):
        if tag.endswith("_" + fc):
            feature = fc; break
    if feature is None:
        return None
    rest = tag[: -len(feature) - 1]
    backbone = None
    for bk in known_backbones:
        if rest.endswith("_" + bk):
            backbone = bk; break
    if backbone is None:
        return None
    task = rest[: -len(backbone) - 1]
    if task not in e_tasks:
        return None
    return task, backbone, feature


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path,
                    default=_REPO / "results" / "models_vision")
    ap.add_argument("--exp", type=Path,
                    default=_REPO / "dataset" / "experimental_features.h5")
    ap.add_argument("--out", type=Path,
                    default=_REPO / "results" / "per_case_vision")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    e_tasks, names = load_exp_context(a.exp)
    known_backbones = sorted(VISION_BACKBONES, key=len, reverse=True)
    feat_cache: dict = {}
    for art in sorted(a.models.iterdir()):
        if art.suffix != ".pt":
            continue
        parsed = _parse_tag(art.stem, known_backbones, e_tasks)
        if parsed is None:
            print(f"  skip {art.stem}: cannot parse"); continue
        task, backbone, feature = parsed
        out_path = a.out / f"{task}_{backbone}_{feature}.json"
        if out_path.exists() and not a.force:
            print(f"  skip {art.stem}: exists"); continue
        blob = torch.load(art, map_location="cpu", weights_only=False)
        n_out = blob["n_out"]; n_channels = blob["n_channels"]
        kind = e_tasks[task][2]
        mdl = VisionBackbone(backbone, n_channels=n_channels, n_out=n_out,
                              regression=(kind == "reg"), bounded_output=True,
                              pretrained=False,
                              channel_adapter=blob.get("channel_adapter",
                                                       "timm_in_chans"))
        mdl.load_state_dict(blob["state_dict"])
        write_per_case(mdl, task, backbone, feature, n_out, n_channels,
                        a.exp, e_tasks, names, a.out,
                        synth_test=blob.get("test_metric", float("nan")),
                        feat_cache=feat_cache)
        print(f"  wrote {task}_{backbone}_{feature}.json")


if __name__ == "__main__":
    main()
