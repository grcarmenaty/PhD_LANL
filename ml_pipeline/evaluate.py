"""Test the trained models against the IQS experimental dataset.

The experimental FRFs live in ``median_frfs.h5`` on the linspace 0–100 Hz
grid (1601 bins).  This script

  1. Parses each experimental case name into our 5-class taxonomy.
  2. Interpolates the experimental FRF onto the same 5–100 Hz band as
     the synthetic ``features.h5``.
  3. Reconstructs a 9-channel time series by multiplying the resampled
     ``H_exp(f)`` by the same deterministic chirp ``F(f)`` and inverse-
     transforming – mirroring the synthetic generation pipeline.
  4. Computes the same feature set (timeseries, frf_mag, modal, indicators).
  5. Runs every trained model on its compatible feature representation
     and writes a metrics report.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml_pipeline.case_design import (   # noqa: E402
    TYPE_PRISTINE, TYPE_BOLT, TYPE_CRACK, TYPE_HOLE, TYPE_MASS,
    SEVERITY_BOUNDS, END_BD, END_AD,
)
from ml_pipeline.features import (   # noqa: E402
    modal_features, indicator_features, INDICATOR_NAMES,
)
from ml_pipeline.generate_dataset import (   # noqa: E402
    make_chirp, N_T, FS, fft_freqs,
)
from ml_pipeline.models import MLP, Conv1DStack, SmallTransformer  # noqa: E402
from ml_pipeline.tasks import TASK_DESCRIPTION  # noqa: E402


# ── Case-name parser ────────────────────────────────────────────────────────
_BOLT_RE  = re.compile(r'(?:d|damage)\s*\(?\s*(\d+)\s*%?\s*\)?\s*(\d)\s*([ab])d', re.I)
# Crack/Hole appear in both "<size>mm <storey><end>" and "<storey><end> <size>mm"
# orders in the IQS labels.
_CRACK_RE_A = re.compile(r'crack\s*(\d+)\s*mm\s*(\d)\s*([ab])d', re.I)
_CRACK_RE_B = re.compile(r'crack\s*(\d)\s*([ab])d\s*(\d+)\s*mm', re.I)
_HOLE_RE_A  = re.compile(r'hole\s*(\d+)\s*mm\s*(\d)\s*([ab])d', re.I)
_HOLE_RE_B  = re.compile(r'hole\s*(\d)\s*([ab])d\s*(\d+)\s*mm', re.I)
_MASS_RE    = re.compile(
    r'mass\s*(base|first\s*floor|second\s*floor|third\s*floor|1f|2f|3f)',
    re.I,
)


def parse_case(label: str) -> List[Dict]:
    """Return one or more operation dicts; primary op (first) drives labels.

    Each op: ``{type_code, storey, end, severity}``.
    """
    label_low = label.lower()
    if "pristine" in label_low:
        return [dict(type_code=TYPE_PRISTINE, storey=-1, end=-1, severity=0.0)]
    ops: List[Dict] = []
    for m in _BOLT_RE.finditer(label_low):
        pct = int(m.group(1)); sty = int(m.group(2)) - 1
        end = END_AD if m.group(3) == 'a' else END_BD
        ops.append(dict(type_code=TYPE_BOLT, storey=sty, end=end,
                          severity=float(pct)))
    for rex, orient in ((_CRACK_RE_A, "size_first"),
                          (_CRACK_RE_B, "loc_first")):
        for m in rex.finditer(label_low):
            if orient == "size_first":
                size = int(m.group(1)); sty = int(m.group(2)) - 1; face = m.group(3)
            else:
                sty = int(m.group(1)) - 1; face = m.group(2); size = int(m.group(3))
            end = END_AD if face == 'a' else END_BD
            ops.append(dict(type_code=TYPE_CRACK, storey=sty, end=end,
                              severity=float(size)))
    for rex, orient in ((_HOLE_RE_A, "size_first"),
                          (_HOLE_RE_B, "loc_first")):
        for m in rex.finditer(label_low):
            if orient == "size_first":
                size = int(m.group(1)); sty = int(m.group(2)) - 1; face = m.group(3)
            else:
                sty = int(m.group(1)) - 1; face = m.group(2); size = int(m.group(3))
            end = END_AD if face == 'a' else END_BD
            ops.append(dict(type_code=TYPE_HOLE, storey=sty, end=end,
                              severity=float(size)))
    for m in _MASS_RE.finditer(label_low):
        tag = m.group(1).lower().replace(" ", "")
        plate = {
            "base": 0,
            "firstfloor": 1, "1f": 1,
            "secondfloor": 2, "2f": 2,
            "thirdfloor": 3, "3f": 3,
        }[tag]
        ops.append(dict(type_code=TYPE_MASS, storey=-1, end=plate, severity=1.2))
    return ops or [dict(type_code=TYPE_PRISTINE, storey=-1, end=-1, severity=0.0)]


def primary_op(label: str) -> Dict:
    """Return the dominant damage op for an experimental case.

    Preference order (chosen to match the strongest physical effect on the
    FRFs): bolt loosening > crack > hole > mass > pristine.
    """
    ops = parse_case(label)
    prio = {TYPE_BOLT: 0, TYPE_CRACK: 1, TYPE_HOLE: 2, TYPE_MASS: 3, TYPE_PRISTINE: 4}
    return sorted(ops, key=lambda d: prio[d["type_code"]])[0]


# ── FRF resampling + time-series reconstruction ─────────────────────────────
def resample_frf(H_exp: np.ndarray, f_src: np.ndarray,
                  f_dst: np.ndarray) -> np.ndarray:
    """Linear interpolation of a complex FRF onto a target frequency grid."""
    out = np.zeros((len(f_dst), H_exp.shape[1]), dtype=np.complex64)
    for c in range(H_exp.shape[1]):
        out[:, c].real = np.interp(f_dst, f_src, H_exp[:, c].real, left=0.0, right=0.0)
        out[:, c].imag = np.interp(f_dst, f_src, H_exp[:, c].imag, left=0.0, right=0.0)
    return out


def synthesize_timeseries(H_full_bins: np.ndarray, chirp: np.ndarray
                            ) -> np.ndarray:
    """Inverse-transform ``H(f) * F(f)`` back to time domain on the rfft grid."""
    F = np.fft.rfft(chirp.astype(np.float64))
    Y = H_full_bins * F[:, None]
    y = np.fft.irfft(Y, n=len(chirp), axis=0)
    return y.astype(np.float32)


# ── Build a "features.h5"-shaped record for experimental cases ──────────────
def build_experimental_features(median_path: Path, features_path: Path
                                  ) -> Dict[str, np.ndarray]:
    with h5py.File(median_path, "r") as f:
        names = [c.decode() for c in f["case_names"][:]]
        H_exp = f["frfs"][:]                     # (n_cases, 1601, 9) complex
        f_src = f["freqs"][:]                    # 0..100 Hz, 1601 bins
    # Flip S2 (ch0) polarity (experimental sensor is mounted inverted).
    H_exp[:, :, 0] *= -1.0
    n_cases = H_exp.shape[0]
    n_ch    = H_exp.shape[2]

    # Match the synthetic feature grid: FFT-bin frequencies on [f_lo, f_hi].
    bins  = fft_freqs(N_T, FS)
    with h5py.File(features_path, "r") as f:
        f_band = f["freqs"][:]
        H_ref  = f["reference/frf_complex"][:]   # (N_F, 9) pristine mean
        f_lo, f_hi = float(f.attrs["f_lo_hz"]), float(f.attrs["f_hi_hz"])
    band_mask = (bins >= f_lo) & (bins <= f_hi)

    t, chirp = make_chirp()

    # Output containers.
    H_band     = np.zeros((n_cases, len(f_band), n_ch), dtype=np.complex64)
    timeseries = np.zeros((n_cases, N_T, n_ch), dtype=np.float32)
    for i in range(n_cases):
        # Resample onto the full rfft grid for time-series reconstruction.
        H_full = resample_frf(H_exp[i], f_src, bins)
        H_band[i] = H_full[band_mask]
        timeseries[i] = synthesize_timeseries(H_full, chirp)

    frf_mag  = np.abs(H_band).astype(np.float32)
    frf_real = H_band.real.astype(np.float32)
    frf_imag = H_band.imag.astype(np.float32)

    modal = np.stack([modal_features(frf_mag[i], f_band) for i in range(n_cases)])
    ind   = np.stack([indicator_features(H_band[i], H_ref) for i in range(n_cases)])

    # Labels from case names.
    type_code = np.zeros(n_cases, dtype=np.int8)
    storey    = np.full(n_cases, -1, dtype=np.int8)
    end       = np.full(n_cases, -1, dtype=np.int8)
    severity  = np.zeros(n_cases, dtype=np.float32)
    for i, name in enumerate(names):
        op = primary_op(name)
        type_code[i] = op["type_code"]
        storey[i]    = op["storey"]
        end[i]       = op["end"]
        severity[i]  = op["severity"]

    return {
        "names":      names,
        "timeseries": timeseries,
        "frf_mag":    frf_mag,
        "frf_real":   frf_real,
        "frf_imag":   frf_imag,
        "modal":      modal,
        "indicators": ind,
        "labels": {
            "type_code": type_code,
            "storey":    storey,
            "end":       end,
            "severity":  severity,
        },
    }


# ── Run trained models on these features ────────────────────────────────────
def _load_torch_model(path: Path, n_channels: int | None, in_dim: int | None,
                       n_out: int) -> torch.nn.Module:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    name = blob["model_name"]
    if name == "mlp":
        mdl = MLP(in_dim=in_dim, n_out=n_out)
    elif name == "cnn":
        mdl = Conv1DStack(n_channels=n_channels, n_out=n_out)
    elif name == "transformer":
        mdl = SmallTransformer(n_channels=n_channels, n_out=n_out)
    else:
        raise ValueError(name)
    mdl.load_state_dict(blob["state_dict"])
    mdl.eval()
    return mdl


def predict(model_path: Path, X: np.ndarray, model_kind: str,
             task_kind: str) -> np.ndarray:
    if model_path.suffix == ".pkl":
        with open(model_path, "rb") as f:
            blob = pickle.load(f)
        mdl    = blob["model"]
        scaler = blob.get("scaler")
        Xf = X.reshape(len(X), -1)
        if scaler is not None:
            Xf = scaler.transform(Xf)
        return mdl.predict(Xf)
    # Torch
    seq = X.ndim == 3
    if seq:
        t = torch.as_tensor(X).float().permute(0, 2, 1)
    else:
        t = torch.as_tensor(X).float()
    blob = torch.load(model_path, map_location="cpu", weights_only=False)
    in_shape = blob["in_shape"]
    name = blob["model_name"]
    n_out = blob["n_out"]
    if name == "mlp":
        in_dim = int(np.prod(in_shape))
        mdl = MLP(in_dim=in_dim, n_out=n_out, regression=(task_kind == "reg"))
        t = t if t.ndim == 2 else t.flatten(1)
    elif name == "cnn":
        mdl = Conv1DStack(n_channels=in_shape[1] if len(in_shape) == 2 else in_shape[-1],
                             n_out=n_out, regression=(task_kind == "reg"))
    else:
        mdl = SmallTransformer(n_channels=in_shape[1] if len(in_shape) == 2 else in_shape[-1],
                                  n_out=n_out, regression=(task_kind == "reg"))
    mdl.load_state_dict(blob["state_dict"])
    mdl.eval()
    with torch.no_grad():
        out = mdl(t)
    if task_kind == "cls":
        return out.argmax(1).numpy()
    return out.squeeze(1).numpy()


def evaluate_all(features_path: Path, median_path: Path,
                  results_dir: Path) -> None:
    print("Building experimental features …")
    exp = build_experimental_features(median_path, features_path)

    # Build targets just like the training pipeline.
    from ml_pipeline.tasks import build_targets
    L = exp["labels"]
    tasks = build_targets(L["type_code"].astype(np.int64),
                            L["storey"].astype(np.int64),
                            L["end"].astype(np.int64),
                            L["severity"].astype(np.float32))

    names_arr = np.array(exp["names"])
    rows: List[Dict] = []
    per_case_rows: List[Dict] = []
    models_dir = results_dir / "models"
    if not models_dir.exists():
        raise FileNotFoundError(f"{models_dir} not found.  Run train.py first.")

    for art in sorted(models_dir.iterdir()):
        tag = art.stem
        # tag format: <task>_<model>_<feature>
        parts = tag.split("_")
        # Some task names have underscores (col_location, mass_location).
        for k in (3, 2):
            cand_task = "_".join(parts[:-k])
            if cand_task in tasks:
                task_name = cand_task
                rest = parts[-k:]
                break
        else:
            print(f"  skip {tag}: cannot parse task")
            continue
        model_name = rest[0]
        feature    = "_".join(rest[1:])

        mask, y_all, kind = tasks[task_name]
        X = exp[feature][mask]
        y = y_all
        names_sub = names_arr[mask]
        if len(X) == 0:
            continue

        try:
            pred = predict(art, X, model_name, kind)
        except Exception as e:
            print(f"  skip {tag}: {e}")
            continue

        if kind == "cls":
            acc = float(accuracy_score(y, pred))
            metric = ("accuracy", acc, None)
        else:
            r2  = float(r2_score(y, pred))
            mae = float(mean_absolute_error(y, pred))
            metric = ("R2", r2, mae)

        rows.append({
            "task":    task_name,
            "model":   model_name,
            "feature": feature,
            "n":       int(len(X)),
            "metric":  metric[0],
            "value":   metric[1],
            "mae":     metric[2],
        })
        for nm, yt, yp in zip(names_sub.tolist(), y.tolist(), pred.tolist()):
            per_case_rows.append({
                "task": task_name, "model": model_name, "feature": feature,
                "case": nm, "y_true": float(yt), "y_pred": float(yp),
            })
        extra = f"  MAE={metric[2]:.3f}" if metric[2] is not None else ""
        print(f"  {tag:<50s}  {metric[0]}={metric[1]:.3f}  (n={len(X)}){extra}")

    out_json = results_dir / "experimental_evaluation.json"
    out_json.write_text(json.dumps(rows, indent=2))
    (results_dir / "experimental_per_case.json").write_text(
        json.dumps(per_case_rows, indent=2))
    print(f"\nWrote {out_json}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--median",   type=Path, default=_REPO / "median_frfs.h5")
    parser.add_argument("--features", type=Path,
                          default=_REPO / "dataset" / "features.h5")
    parser.add_argument("--results",  type=Path, default=_REPO / "results")
    args = parser.parse_args()
    evaluate_all(args.features, args.median, args.results)


if __name__ == "__main__":
    main()
