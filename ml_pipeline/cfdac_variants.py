"""Append additional CFDAC representations to ``dataset/features.h5``.

Already present:  ``cfdac_real``, ``cfdac_imag`` (128 × 128 each).
Added here:       ``cfdac_mag``, ``cfdac_phase``.

Models that consume CFDAC then build the multi-channel / Conv3d
inputs at training time by stacking these arrays — no additional
on-disk copies needed.  See ``ml_pipeline/train.py`` ``load_feature``
for the variant naming convention:

    cfdac_real      -> (n, 1, 128, 128)
    cfdac_imag      -> (n, 1, 128, 128)
    cfdac_mag       -> (n, 1, 128, 128)
    cfdac_phase     -> (n, 1, 128, 128)
    cfdac_realimag  -> (n, 2, 128, 128)   (existing — re-used)
    cfdac_magphase  -> (n, 2, 128, 128)
    cfdac_all       -> (n, 4, 128, 128)
    cfdac_3d_realimag -> (n, 1, 2, 128, 128)   Conv3d input
    cfdac_3d_magphase -> (n, 1, 2, 128, 128)
    cfdac_3d_all      -> (n, 1, 4, 128, 128)
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


def append_mag_phase(features_path: Path, batch: int = 256) -> None:
    with h5py.File(features_path, "a") as f:
        n, H, W = f["cfdac_real"].shape
        for name in ("cfdac_mag", "cfdac_phase"):
            if name in f:
                del f[name]
            f.create_dataset(
                name, shape=(n, H, W), dtype=np.float32,
                compression="gzip", compression_opts=4,
                chunks=(1, H, W),
            )
        for i0 in range(0, n, batch):
            i1 = min(i0 + batch, n)
            re = f["cfdac_real"][i0:i1]
            im = f["cfdac_imag"][i0:i1]
            f["cfdac_mag"][i0:i1] = np.sqrt(re ** 2 + im ** 2).astype(np.float32)
            f["cfdac_phase"][i0:i1] = np.arctan2(im, re).astype(np.float32)
            if (i1 % 1000 == 0) or (i1 == n):
                print(f"   .. {i1}/{n}")
        if "reference/cfdac_mag" in f:
            del f["reference/cfdac_mag"]; del f["reference/cfdac_phase"]
        ref_re = f["reference/cfdac_real"][:]
        ref_im = f["reference/cfdac_imag"][:]
        f["reference"].create_dataset(
            "cfdac_mag",
            data=np.sqrt(ref_re ** 2 + ref_im ** 2).astype(np.float32),
        )
        f["reference"].create_dataset(
            "cfdac_phase",
            data=np.arctan2(ref_im, ref_re).astype(np.float32),
        )
    print(f"\nAppended cfdac_mag, cfdac_phase to {features_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path,
                          default=_REPO / "dataset" / "features.h5")
    args = parser.parse_args()
    append_mag_phase(args.features)


if __name__ == "__main__":
    main()
