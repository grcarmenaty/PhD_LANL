"""Mass-plate location: logit-averaging ensemble of two complementary cells.

Motivation (iteration-3 council advocate suggestion). Per the seeded
sweep, two cells transfer to the 4-class `mass_location` task with
complementary strengths:

  * `mlp / cfdac_imag`  — macro-F1 0.45, balanced acc 0.51
  * `cnn2d / cfdac_real` — macro-F1 ~0.36 but different per-class errors

Averaging their pre-softmax logits *should* combine the complementary
signal. This script does that combination on the 238 experimental Mass
cases and reports macro-F1, balanced accuracy, and the per-class
confusion matrix versus each component.

No training. Uses the canonical seeded models in
`results/models/mass_location_<model>_<feature>.pt`. Run after
`evaluate_full_experimental.py` has produced its baseline numbers.

Outputs `results/ensemble_mass_location.json` with the per-cell and
ensemble metrics so the report can cite them without re-running.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                                f1_score, accuracy_score)

import sys
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from ml_pipeline.models import MLP, Conv2DStack  # type: ignore  # noqa: E402
from ml_pipeline.tasks import build_targets  # type: ignore  # noqa: E402


def _build_torch_model(blob: dict, n_out: int) -> torch.nn.Module:
    """Reconstruct the torch model from the .pt blob's hyperparams."""
    name = blob["model_name"]
    in_shape = blob["in_shape"]
    hp = blob.get("hyperparams") or {}
    if name == "mlp":
        in_dim = int(np.prod(in_shape))
        return MLP(in_dim=in_dim, n_out=n_out,
                   hidden=tuple(hp.get("hidden", (256, 128, 64))),
                   regression=False, bounded_output=True)
    if name == "cnn2d":
        return Conv2DStack(n_channels=in_shape[0], n_out=n_out,
                           widths=tuple(hp.get("widths", (16, 32, 64))),
                           kernel_size=int(hp.get("kernel_size", 5)),
                           regression=False, bounded_output=True)
    raise ValueError(f"Unsupported model {name}")


def _logits(model_path: Path, X: np.ndarray, n_out: int) -> np.ndarray:
    """Forward inference; return pre-softmax logits, shape (n, n_out)."""
    blob = torch.load(model_path, map_location="cpu", weights_only=False)
    mdl = _build_torch_model(blob, n_out)
    mdl.load_state_dict(blob["state_dict"])
    mdl.eval()
    t = torch.as_tensor(np.asarray(X)).float()
    if blob["model_name"] == "mlp" and t.ndim != 2:
        t = t.flatten(1)
    if blob["model_name"] == "cnn2d" and t.ndim == 3:
        # (n, 128, 128) → (n, 1, 128, 128)
        t = t.unsqueeze(1)
    with torch.no_grad():
        out = mdl(t)
    return out.numpy()


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, tag: str) -> dict:
    return {
        "tag": tag,
        "n": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion": confusion_matrix(y_true, y_pred).tolist(),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp", type=Path,
                   default=_REPO / "dataset" / "experimental_features.h5")
    p.add_argument("--models-dir", type=Path,
                   default=_REPO / "results" / "models")
    p.add_argument("--out", type=Path,
                   default=_REPO / "results" / "ensemble_mass_location.json")
    p.add_argument("--cells", nargs="+", default=[
        "mlp:cfdac_imag",
        "cnn2d:cfdac_real",
    ], help="model:feature pairs to ensemble (default: the two suggested)")
    args = p.parse_args()

    print(f"Ensemble cells: {args.cells}")

    with h5py.File(args.exp, "r") as f:
        type_code = f["type_code"][:].astype(np.int64)
        storey    = f["storey"][:].astype(np.int64)
        end       = f["end"][:].astype(np.int64)
        severity  = f["severity"][:].astype(np.float32)

    tasks = build_targets(type_code, storey, end, severity)
    mask, y_true, kind = tasks["mass_location"]
    assert kind == "cls"
    print(f"mass_location: {len(y_true)} Mass cases, {y_true.max()+1} classes")

    n_out = 4
    cell_logits = []
    cell_results = []
    with h5py.File(args.exp, "r") as f:
        for cell in args.cells:
            model, feat = cell.split(":")
            mp = args.models_dir / f"mass_location_{model}_{feat}.pt"
            if not mp.exists():
                print(f"  skip — model file missing: {mp}")
                continue
            X = f[feat][:][mask]
            print(f"  {cell}: feature {feat} shape {X.shape}")
            log = _logits(mp, X, n_out)
            cell_logits.append((cell, log))
            cell_results.append(_metrics(y_true, log.argmax(1), cell))

    if len(cell_logits) < 2:
        print("Need >= 2 cells to ensemble; aborting.")
        return

    avg = np.mean(np.stack([l for _, l in cell_logits], axis=0), axis=0)
    ens = _metrics(y_true, avg.argmax(1),
                   "ENSEMBLE(" + "+".join(c for c, _ in cell_logits) + ")")
    rows = cell_results + [ens]

    print("\nResults:")
    print(f'{"tag":50s} {"macroF1":>8s} {"balAcc":>8s} {"acc":>8s}')
    for r in rows:
        print(f'{r["tag"]:50s} {r["macro_f1"]:8.3f} {r["balanced_acc"]:8.3f} {r["accuracy"]:8.3f}')

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
