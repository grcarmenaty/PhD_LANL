"""Generate per-case predictions on the full 2638-case experimental
set for every cell in results/models_vision/, so we can build
confusion matrices / ROC for the vision sweep.

Outputs: results/per_case_vision/<task>_<backbone>_<feature>.json
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
from ml_pipeline.vision_models import VisionBackbone


def main():
    models_dir = _REPO / "results" / "models_vision"
    exp_path = _REPO / "dataset" / "experimental_features.h5"
    out_dir = _REPO / "results" / "per_case_vision"
    out_dir.mkdir(parents=True, exist_ok=True)

    from ml_pipeline.evaluate_full_experimental import _exp_load_feature
    with h5py.File(exp_path, "r") as f:
        tc = f["type_code"][:].astype(np.int64)
        sto = f["storey"][:].astype(np.int64)
        end = f["end"][:].astype(np.int64)
        sev = f["severity"][:].astype(np.float32)
        names = [str(s) for s in f["names"][:]]
    e_tasks = build_targets(tc, sto, end, sev)
    feat_cache = {}
    for art in sorted(models_dir.iterdir()):
        if art.suffix != ".pt":
            continue
        tag = art.stem  # type_<backbone>_<feature>
        parts = tag.split("_")
        # Backbone names contain underscores -- find feature by suffix
        feature = None
        for f_candidate in ("cfdac_realimag", "cfdac_magphase", "cfdac_all",
                                "cfdac_real", "cfdac_imag", "cfdac_mag",
                                "cfdac_phase"):
            if tag.endswith("_" + f_candidate):
                feature = f_candidate
                break
        if feature is None:
            print(f"  skip {tag}: cannot parse feature"); continue
        rest = tag[: -len(feature) - 1]   # type_<backbone>
        task = rest.split("_")[0]
        backbone = "_".join(rest.split("_")[1:])

        blob = torch.load(art, map_location="cpu", weights_only=False)
        n_out = blob["n_out"]; n_channels = blob["n_channels"]
        kind = e_tasks[task][2]
        adapter = blob.get("channel_adapter", "first_conv_replace")

        mdl = VisionBackbone(backbone, n_channels=n_channels, n_out=n_out,
                                  regression=(kind == "reg"),
                                  bounded_output=True,
                                  pretrained=False,
                                  channel_adapter=adapter)
        mdl.load_state_dict(blob["state_dict"])
        mdl.eval()

        # Load (or cache) the exp feature
        if feature not in feat_cache:
            X = _exp_load_feature(exp_path, feature, normalize=True)
            if X.ndim == 5:
                X = X.reshape(X.shape[0], X.shape[1] * X.shape[2],
                                X.shape[3], X.shape[4])
            feat_cache[feature] = X
        X = feat_cache[feature]

        mask, e_y, _ = e_tasks[task]
        idx = np.where(mask)[0]
        Xe = X[idx]
        outs = []; probs = []
        bs = 32
        with torch.no_grad():
            for i in range(0, len(Xe), bs):
                xb = torch.as_tensor(Xe[i:i + bs]).float()
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
        for i, ix in enumerate(idx):
            row = {"case": names[ix],
                     "y_true": int(e_y[ix]) if kind == "cls" else float(e_y[ix]),
                     "y_pred": int(pred[i])  if kind == "cls" else float(pred[i])}
            if proba is not None:
                row["proba"] = [float(p) for p in proba[i]]
            rows.append(row)
        meta = {"task": task, "backbone": backbone, "feature": feature,
                  "kind": kind, "n_out": int(n_out),
                  "n_channels": int(n_channels),
                  "synth_test": float(blob.get("test_metric", float("nan"))),
                  "input_normalized": True}
        (out_dir / f"{tag}.json").write_text(
            json.dumps({"meta": meta, "rows": rows}, indent=2))
        print(f"  wrote {tag}.json  (n={len(rows)})")


if __name__ == "__main__":
    main()
