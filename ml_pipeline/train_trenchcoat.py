"""Train the 5 binary "is this damage type?" classifiers and aggregate
their per-sample sigmoid outputs into a 5-class type prediction.

Workflow:
  1.  For each k in {is_pristine, is_bolt, is_crack, is_hole, is_mass}:
      train a binary vision-backbone classifier on the full synthetic
      data (positive class is type_code == k, negative class is all
      others -- so naturally imbalanced 2000 vs 8000).
  2.  At inference, run each binary classifier on the held-out
      experimental set; collect 5 sigmoid probabilities per sample.
  3.  Aggregate via argmax (per-sample) for a 5-class prediction; also
      report per-sample uncertainty = 1 - max(prob) and the agreement
      pattern (how many binary classifiers said "yes" for this sample).
  4.  Compare aggregator metrics against the single-model 5-class
      baseline on the same experimental set.

Outputs:
  results/models_vision/<is_*>_<backbone>_<feature>.pt   per-binary artefact
  results/per_case_vision/<is_*>_<backbone>_<feature>.json per-binary preds
  results/trenchcoat_eval.json   aggregator metrics + per-sample row

Usage:
  python -m ml_pipeline.train_trenchcoat \\
      --backbone convnext_tiny --feature cfdac_all --subsample 1500 \\
      --epochs 4 --probe-epochs 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, roc_auc_score,
)

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml_pipeline.tasks import (
    build_targets, BINARY_TYPE_DECOMPOSITION,
)
from ml_pipeline.vision_models import VisionBackbone

TYPE_NAMES = ["Pristine", "Bolt", "Crack", "Hole", "Mass"]
LABEL_TO_TYPE_IDX = {
    "is_pristine": 0, "is_bolt": 1, "is_crack": 2,
    "is_hole":     3, "is_mass": 4,
}


def _train_binary(label: str, backbone: str, feature: str, args) -> Path:
    """Delegate to train_vision.main with the binary task name."""
    import subprocess
    cmd = [
        sys.executable, "-m", "ml_pipeline.train_vision",
        "--backbones", backbone,
        "--features", feature,
        "--tasks", label,
        "--subsample", str(args.subsample),
        "--epochs", str(args.epochs),
        "--batch", str(args.batch),
        "--lr", str(args.lr),
        "--probe-epochs", str(args.probe_epochs),
        "--class-weights", args.class_weights,
        "--channel-adapter", args.channel_adapter,
        "--select-by", "macro_f1",
    ]
    if args.force:
        cmd.append("--force")
    print(f"  $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    return _REPO / "results" / "models_vision" / f"{label}_{backbone}_{feature}.pt"


def _run_inference(art: Path, feature: str, exp_path: Path) -> dict:
    """Load a binary .pt + run inference on the FULL 2638 experimental
    cases.  Returns y_true, y_pred (binary), proba (positive-class
    probability)."""
    from ml_pipeline.evaluate_full_experimental import _exp_load_feature
    blob = torch.load(art, map_location="cpu", weights_only=False)
    n_out = blob["n_out"]; n_channels = blob["n_channels"]
    adapter = blob.get("channel_adapter", "first_conv_replace")
    mdl = VisionBackbone(blob["vision_backbone_name"],
                              n_channels=n_channels, n_out=n_out,
                              regression=False, bounded_output=True,
                              pretrained=False,
                              channel_adapter=adapter)
    mdl.load_state_dict(blob["state_dict"])
    mdl.eval()

    X = _exp_load_feature(exp_path, feature, normalize=True)
    if X.ndim == 5:
        X = X.reshape(X.shape[0], X.shape[1] * X.shape[2],
                        X.shape[3], X.shape[4])
    with h5py.File(exp_path, "r") as f:
        tc = f["type_code"][:].astype(np.int64)
    n = len(tc)

    probas = []
    with torch.no_grad():
        for i in range(0, n, 32):
            xb = torch.as_tensor(X[i:i+32]).float()
            out = mdl(xb).cpu().numpy()
            e_x = np.exp(out - out.max(axis=1, keepdims=True))
            p = e_x / e_x.sum(axis=1, keepdims=True)
            probas.append(p[:, 1])           # P(positive class)
    proba = np.concatenate(probas, axis=0)
    return {
        "type_code": tc,
        "proba_pos": proba,    # per-sample P(this damage type)
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--syn", type=Path,
                      default=_REPO / "dataset" / "features.h5")
    p.add_argument("--exp", type=Path,
                      default=_REPO / "dataset" / "experimental_features.h5")
    p.add_argument("--out", type=Path, default=_REPO / "results")
    p.add_argument("--backbone", type=str, default="convnext_tiny",
                      choices=("resnet50", "efficientnet_b0",
                                  "convnext_tiny", "swin_t", "vit_b_16"))
    p.add_argument("--feature", type=str, default="cfdac_all")
    p.add_argument("--subsample", type=int, default=1500)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--probe-epochs", type=int, default=2)
    p.add_argument("--class-weights", default="inverse-freq")
    p.add_argument("--channel-adapter", default="projector")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    print(f"Trenchcoat: {args.backbone} / {args.feature}, "
              f"5 binary tasks", flush=True)

    artefacts = {}
    for label in BINARY_TYPE_DECOMPOSITION:
        artefacts[label] = _train_binary(label, args.backbone,
                                                args.feature, args)
        print(f"  -> {artefacts[label].name}", flush=True)

    print("\nInference + aggregation on the full 2638-case exp set...",
              flush=True)
    proba_matrix = None
    type_code = None
    for label in BINARY_TYPE_DECOMPOSITION:
        res = _run_inference(artefacts[label], args.feature, args.exp)
        if proba_matrix is None:
            proba_matrix = np.zeros((len(res["type_code"]),
                                              len(BINARY_TYPE_DECOMPOSITION)),
                                              dtype=np.float32)
            type_code = res["type_code"]
        col = LABEL_TO_TYPE_IDX[label]
        proba_matrix[:, col] = res["proba_pos"]

    pred = proba_matrix.argmax(axis=1)
    # Per-sample uncertainty: 1 - max(prob)
    uncertainty = 1.0 - proba_matrix.max(axis=1)
    # Per-sample agreement: number of binaries with prob > 0.5
    binary_yes_count = (proba_matrix > 0.5).sum(axis=1)

    cm = confusion_matrix(type_code, pred, labels=[0, 1, 2, 3, 4])
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    acc = float(accuracy_score(type_code, pred))
    f1m = float(f1_score(type_code, pred, labels=[0, 1, 2, 3, 4],
                                average="macro", zero_division=0))
    f1_per = f1_score(type_code, pred, labels=[0, 1, 2, 3, 4],
                          average=None, zero_division=0).tolist()

    per_class_auc = {}
    for k, label in enumerate(BINARY_TYPE_DECOMPOSITION):
        y_bin = (type_code == k).astype(int)
        try:
            per_class_auc[label] = float(roc_auc_score(y_bin,
                                                              proba_matrix[:, k]))
        except Exception:
            per_class_auc[label] = float("nan")

    print(f"\nAGGREGATOR RESULTS (argmax of 5 binary sigmoids):")
    print(f"  type 5-class accuracy = {acc:.3f}")
    print(f"  type macro-F1         = {f1m:.3f}")
    print(f"  per-class F1: " +
              ", ".join(f"{TYPE_NAMES[i]}={f1_per[i]:.2f}"
                            for i in range(5)))
    print(f"  per-binary AUC: " +
              ", ".join(f"{l[3:]}={per_class_auc[l]:.2f}"
                            for l in BINARY_TYPE_DECOMPOSITION))

    out_path = args.out / "trenchcoat_eval.json"
    out_path.write_text(json.dumps({
        "backbone": args.backbone,
        "feature": args.feature,
        "subsample": args.subsample,
        "epochs": args.epochs,
        "aggregator_accuracy": acc,
        "aggregator_macro_f1": f1m,
        "aggregator_per_class_f1": dict(zip(TYPE_NAMES, f1_per)),
        "per_binary_auc": per_class_auc,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalised": cm_norm.tolist(),
        "type_names": TYPE_NAMES,
        # Per-sample dump for diagnostic plots
        "per_case": [
            {"type_code": int(tc), "pred": int(p),
             "proba": proba_matrix[i].tolist(),
             "uncertainty": float(uncertainty[i]),
             "binary_yes_count": int(binary_yes_count[i])}
            for i, (tc, p) in enumerate(zip(type_code, pred))
        ],
    }, indent=2))
    print(f"\nwrote {out_path.name}")


if __name__ == "__main__":
    main()
