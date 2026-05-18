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
from ml_pipeline.models import MLP, Conv1DStack, SmallTransformer, Conv2DStack  # noqa: E402
from ml_pipeline.tasks import TASK_DESCRIPTION  # noqa: E402
from pymodal import utils as pm_utils  # noqa: E402


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
        H_exp = f["median_frf"][:]               # (n_cases, 1601, 9) complex
        f_src = f["freq"][:]                     # 0..100 Hz, 1601 bins
    # Flip S2 (ch0) polarity (experimental sensor is mounted inverted).
    H_exp[:, :, 0] *= -1.0
    n_cases = H_exp.shape[0]
    n_ch    = H_exp.shape[2]

    # Match the synthetic feature grid: FFT-bin frequencies on [f_lo, f_hi].
    bins  = fft_freqs(N_T, FS)
    with h5py.File(features_path, "r") as f:
        f_band = f["freqs"][:]
        H_ref_synth = f["reference/frf_complex"][:]   # (N_F, 9) synth pristine mean
        f_lo, f_hi = float(f.attrs["f_lo_hz"]), float(f.attrs["f_hi_hz"])
        has_cfdac = "cfdac_real" in f
        cfdac_n   = int(f.attrs["cfdac_n"]) if has_cfdac else None
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

    # P0.1: build the experimental pristine reference from Pristine
    # cases in this median dataset; fall back to the synth ref if none
    # are present (the legacy 61-case median set may not have one).
    type_codes_pre = np.array(
        [primary_op(name)["type_code"] for name in names], dtype=np.int8
    )
    pristine_idx = np.where(type_codes_pre == TYPE_PRISTINE)[0]
    if pristine_idx.size > 0:
        H_ref = H_band[pristine_idx].mean(axis=0).astype(np.complex64)
        print(f"  using experimental pristine reference ({pristine_idx.size} cases)")
    else:
        H_ref = H_ref_synth
        print("  no Pristine experimental cases; using synth reference")

    frf_mag  = np.abs(H_band).astype(np.float32)
    frf_real = H_band.real.astype(np.float32)
    frf_imag = H_band.imag.astype(np.float32)

    modal = np.stack([modal_features(frf_mag[i], f_band) for i in range(n_cases)])
    ind   = np.stack([indicator_features(H_band[i], H_ref) for i in range(n_cases)])

    # CFDAC at the same resolution as the synthetic dataset.
    cfdac = None
    if has_cfdac:
        from ml_pipeline.cfdac import _decimate
        H_ref_d = _decimate(H_ref, cfdac_n)              # (cfdac_n, 9)
        cfdac = np.zeros((n_cases, 2, cfdac_n, cfdac_n), dtype=np.float32)
        for i in range(n_cases):
            H_d = _decimate(H_band[i], cfdac_n)          # (cfdac_n, 9)
            m   = pm_utils.value_CFDAC(H_ref_d, H_d)     # (cfdac_n, cfdac_n) complex
            cfdac[i, 0] = m.real.astype(np.float32)
            cfdac[i, 1] = m.imag.astype(np.float32)

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

    out = {
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
    if cfdac is not None:
        out["cfdac"] = cfdac
    return out


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
    elif name == "cnn2d":
        mdl = Conv2DStack(n_channels=n_channels, n_out=n_out)
    else:
        raise ValueError(name)
    mdl.load_state_dict(blob["state_dict"])
    mdl.eval()
    return mdl


def predict(model_path: Path, X: np.ndarray, model_kind: str,
             task_kind: str, scaler=None) -> np.ndarray:
    """Run a saved HPO model on ``X``.

    ``scaler``: ``StandardScaler`` previously fit on the train fold for
    flat features (modal / indicators).  Torch ``.pt`` files don't
    embed this so it must be passed in by the caller — see
    ``evaluate_all`` for how the per-task scalers are built.
    """
    if model_path.suffix == ".pkl":
        with open(model_path, "rb") as f:
            blob = pickle.load(f)
        mdl    = blob["model"]
        sk_scaler = blob.get("scaler") or scaler
        Xf = X.reshape(len(X), -1)
        if sk_scaler is not None:
            Xf = sk_scaler.transform(Xf)
        return mdl.predict(Xf)
    # Torch
    blob = torch.load(model_path, map_location="cpu", weights_only=False)
    in_shape = blob["in_shape"]
    name = blob["model_name"]
    n_out = blob["n_out"]
    hp = blob.get("hyperparams") or {}

    Xa = X
    if name == "mlp" and scaler is not None:
        Xa = scaler.transform(X.reshape(len(X), -1))
    t = torch.as_tensor(np.asarray(Xa)).float()
    seq = t.ndim == 3
    if seq:
        t = t.permute(0, 2, 1)

    if name == "mlp":
        in_dim = t.shape[-1] if t.ndim == 2 else int(np.prod(in_shape))
        if t.ndim == 3:
            t = t.flatten(1)
        hidden = tuple(hp.get("hidden", (256, 128, 64)))
        mdl = MLP(in_dim=in_dim, n_out=n_out, hidden=hidden,
                     regression=(task_kind == "reg"))
    elif name == "cnn":
        ch = in_shape[1] if len(in_shape) == 2 else in_shape[-1]
        mdl = Conv1DStack(n_channels=ch, n_out=n_out,
                             widths=tuple(hp.get("widths", (32, 64, 128))),
                             kernel_size=int(hp.get("kernel_size", 7)),
                             regression=(task_kind == "reg"))
    elif name == "transformer":
        ch = in_shape[1] if len(in_shape) == 2 else in_shape[-1]
        mdl = SmallTransformer(n_channels=ch, n_out=n_out,
                                  d_model=int(hp.get("d_model", 48)),
                                  n_layers=int(hp.get("n_layers", 2)),
                                  regression=(task_kind == "reg"))
    elif name == "cnn2d":
        mdl = Conv2DStack(n_channels=in_shape[0], n_out=n_out,
                              widths=tuple(hp.get("widths", (16, 32, 64))),
                              kernel_size=int(hp.get("kernel_size", 5)),
                              regression=(task_kind == "reg"))
    else:
        raise ValueError(name)
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
    from ml_pipeline.train import (load_labels, load_feature, make_split,
                                       FEATURES_FLAT)
    from sklearn.preprocessing import StandardScaler
    L = exp["labels"]
    tasks = build_targets(L["type_code"].astype(np.int64),
                            L["storey"].astype(np.int64),
                            L["end"].astype(np.int64),
                            L["severity"].astype(np.float32))

    # Re-fit each (task, flat-feature) StandardScaler on the synthetic
    # train fold — required because the .pt blobs don't embed it.
    print("Fitting per-task scalers from the synthetic train fold …")
    syn_labels = load_labels(features_path)
    syn_tasks  = build_targets(syn_labels["type_code"], syn_labels["storey"],
                                  syn_labels["end"], syn_labels["severity"])
    scalers: Dict[str, Dict[str, "StandardScaler"]] = {}
    flat_syn = {name: load_feature(features_path, name) for name in FEATURES_FLAT}
    for tn, (sm, sy, sk_kind) in syn_tasks.items():
        scalers[tn] = {}
        ipool = np.where(sm)[0]
        i_tr, _, _ = make_split(sy, sk_kind)
        idx_tr = ipool[i_tr]
        for feat in FEATURES_FLAT:
            X_tr = flat_syn[feat][idx_tr].reshape(len(idx_tr), -1)
            scalers[tn][feat] = StandardScaler().fit(X_tr)

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
            scaler = scalers.get(task_name, {}).get(feature)
            pred = predict(art, X, model_name, kind, scaler=scaler)
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
