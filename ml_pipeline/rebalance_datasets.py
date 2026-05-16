"""Produce location-balanced synth + experimental datasets.

Motivation
==========

The full synthetic dataset is perfectly stratified (333-334 samples per
(type, location) cell, 500 per Mass plate) but the full experimental
dataset is heavily skewed — Crack and Hole have *zero* AD-end samples,
Bolt has zero S3AD samples, and the per-cell counts span 40 – 531.
Mixing these two during cross-domain evaluation makes per-task metrics
incomparable between cells.

This script generates two matched, balanced datasets:

* ``dataset/features_balanced.h5`` — synthetic, restricted to the
  (type, location) cells present in experimental, then per-cell
  rebalanced to ``TARGET_SYNTH`` (500 by default).  Cells with fewer
  than 500 source samples are *augmented* by sampling with
  replacement from the source rows; cells with more than 500 are
  *restricted* by random subsample.  Indicator and CFDAC arrays are
  index-copied — no synthetic perturbation is applied.

* ``dataset/experimental_features_balanced.h5`` — experimental,
  *eliminated* to the same common cells with ``TARGET_EXP`` samples
  per cell (40 by default — the smallest IQS cell size that keeps
  every Mass plate).  Cells with more than 40 samples are randomly
  subsampled; cells with fewer than 40 are excluded.

The 17 common cells:

* Pristine (1)
* Bolt: S1BD, S1AD, S2BD, S2AD, S3BD (5; drop S3AD)
* Crack: S1BD, S2BD, S3BD (3; drop all AD)
* Hole: S1BD, S2BD, S3BD, S3AD (4; drop S1AD, S2AD)
* Mass: Base, F1, F2, F3 (4)

Totals (per-cell × 17 cells):
  ``features_balanced.h5``           : 500 × 17 = 8 500 samples
  ``experimental_features_balanced.h5`` :  40 × 17 =   680 samples
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml_pipeline.case_design import (                                # noqa: E402
    TYPE_PRISTINE, TYPE_BOLT, TYPE_CRACK, TYPE_HOLE, TYPE_MASS,
)

TARGET_SYNTH = 500
TARGET_EXP   = 40
SEED         = 20260511

# (type_code, storey, end) tuples for column damage; (type_code, plate)
# for mass.  Pristine is (TYPE_PRISTINE,).
COMMON_CELLS = (
    (TYPE_PRISTINE,),
    (TYPE_BOLT,  0, 0), (TYPE_BOLT,  0, 1),
    (TYPE_BOLT,  1, 0), (TYPE_BOLT,  1, 1),
    (TYPE_BOLT,  2, 0),                              # drop S3AD
    (TYPE_CRACK, 0, 0), (TYPE_CRACK, 1, 0), (TYPE_CRACK, 2, 0),
    (TYPE_HOLE,  0, 0), (TYPE_HOLE,  1, 0),
    (TYPE_HOLE,  2, 0), (TYPE_HOLE,  2, 1),
    (TYPE_MASS_PLATE := TYPE_MASS, 0),  # plate 0 = Base
    (TYPE_MASS, 1), (TYPE_MASS, 2), (TYPE_MASS, 3),
)


def _cell_indices(tc: np.ndarray, sto: np.ndarray, end: np.ndarray,
                    cell: tuple) -> np.ndarray:
    if cell[0] == TYPE_PRISTINE:
        return np.where(tc == TYPE_PRISTINE)[0]
    if cell[0] == TYPE_MASS:
        plate = cell[1]
        return np.where((tc == TYPE_MASS) & (end == plate))[0]
    t, s, e = cell
    return np.where((tc == t) & (sto == s) & (end == e))[0]


def _select_rows(src_idx: np.ndarray, target: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Subsample (no replacement) if source larger than target;
    bootstrap-augment (with replacement) if smaller; pass-through if equal."""
    if len(src_idx) >= target:
        return rng.choice(src_idx, size=target, replace=False)
    return rng.choice(src_idx, size=target, replace=True)


def _build(in_path: Path, out_path: Path, target: int,
            label_in_group: bool, mode: str) -> None:
    """Read ``in_path`` (synth or exp), build the per-cell-balanced
    output at ``out_path``.

    ``mode`` is "synth" (allow bootstrap augmentation) or "exp"
    (drop cells under ``target``)."""
    rng = np.random.default_rng(SEED)
    with h5py.File(in_path, "r") as f:
        if label_in_group:
            tc  = f["labels"]["type_code"][:]
            sto = f["labels"]["storey"][:]
            end = f["labels"]["end"][:]
            sev = f["labels"]["severity"][:]
            sid = f["labels"]["sample_id"][:]
        else:
            tc  = f["type_code"][:]
            sto = f["storey"][:]
            end = f["end"][:]
            sev = f["severity"][:]
            sid = np.arange(len(tc))
        _LABEL_KEYS = {"type_code", "storey", "end", "severity", "sample_id"}
        feature_keys = [
            k for k in f.keys()
            if k not in ({"labels", "reference"} | _LABEL_KEYS)
        ]
        # Determine sizes for output allocation
        kept_rows: list[np.ndarray] = []
        kept_labels: list[tuple] = []
        for cell in COMMON_CELLS:
            src = _cell_indices(tc, sto, end, cell)
            if mode == "exp" and len(src) < target:
                print(f"  drop cell {cell} (exp count {len(src)} < target {target})")
                continue
            rows = _select_rows(src, target, rng)
            kept_rows.append(rows)
            kept_labels.append(cell)
        rows_all = np.concatenate(kept_rows)
        n_out = len(rows_all)
        print(f"  writing {n_out} rows ({len(kept_labels)} cells × {target})")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(out_path, "w") as g:
            # Labels (flat, matching experimental_features.h5 schema)
            g.create_dataset("type_code", data=tc[rows_all])
            g.create_dataset("storey",    data=sto[rows_all])
            g.create_dataset("end",       data=end[rows_all])
            g.create_dataset("severity",  data=sev[rows_all])
            g.create_dataset("sample_id", data=sid[rows_all]
                              if sid.dtype.kind != "O" else
                              np.array([str(s) for s in sid[rows_all]],
                                          dtype=h5py.string_dtype()))
            # Copy feature arrays
            for k in feature_keys:
                arr = f[k][:]
                if arr.ndim == 0 or arr.shape[0] != len(tc):
                    # e.g. 'freqs' is per-frequency-bin; copy verbatim
                    g.create_dataset(k, data=arr)
                    continue
                if k == "names":
                    g.create_dataset(k, data=np.array(
                        arr[rows_all], dtype=h5py.string_dtype()))
                    continue
                g.create_dataset(k, data=arr[rows_all],
                                  compression="gzip", compression_opts=4)
            # Provenance attributes
            g.attrs["source"]            = str(in_path)
            g.attrs["target_per_cell"]   = target
            g.attrs["balancing_mode"]    = mode
            g.attrs["common_cells_json"] = repr(kept_labels)
    print(f"  wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--synth-in", type=Path,
                      default=_REPO / "dataset" / "features.h5")
    p.add_argument("--exp-in", type=Path,
                      default=_REPO / "dataset" / "experimental_features.h5")
    p.add_argument("--synth-out", type=Path,
                      default=_REPO / "dataset" / "features_balanced.h5")
    p.add_argument("--exp-out", type=Path,
                      default=_REPO / "dataset" / "experimental_features_balanced.h5")
    p.add_argument("--target-synth", type=int, default=TARGET_SYNTH)
    p.add_argument("--target-exp",   type=int, default=TARGET_EXP)
    args = p.parse_args()

    print(f"== synthetic → {args.synth_out.name} (target {args.target_synth}/cell) ==")
    _build(args.synth_in, args.synth_out, args.target_synth,
            label_in_group=True, mode="synth")
    print(f"== experimental → {args.exp_out.name} (target {args.target_exp}/cell) ==")
    _build(args.exp_in, args.exp_out, args.target_exp,
            label_in_group=False, mode="exp")


if __name__ == "__main__":
    main()
