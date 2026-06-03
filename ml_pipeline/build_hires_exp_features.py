"""Repackage experimental_frfs.h5 (native 1601-bin FRFs) into a
features-style HDF5 matching the schema the training pipeline expects.

Source data already lives at 1601 bins covering 0–100 Hz (df=0.0625 Hz)
— the synth chunks at N_T=4096 produce the matching grid, so the two
domains share the same FRF resolution natively.

We skip the cfdac_* arrays (computed on the fly from FRFs at training
time) and skip the raw timeseries (not used by the hires CFDAC sweep).
Labels and case names are copied from the existing experimental
features file at 128² so the per-case JSON ordering stays identical.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import h5py
import numpy as np

_REPO = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src-frfs", type=Path,
                    default=_REPO / "experimental_frfs.h5")
    p.add_argument("--src-labels", type=Path,
                    default=_REPO / "dataset" / "experimental_features.h5")
    p.add_argument("--out", type=Path,
                    default=_REPO / "dataset" / "experimental_features_hires.h5")
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(a.src_frfs, "r") as src:
        H = src["data"][:]                          # (2638, 1601, 9) complex64
        freqs = src["freq"][:].astype(np.float32)
    n_cases, n_freq, n_ch = H.shape
    print(f"loaded {n_cases} cases × {n_freq} freqs × {n_ch} channels "
          f"({freqs[0]:.3f}–{freqs[-1]:.3f} Hz, df={freqs[1]-freqs[0]:.4f})")

    # Copy labels + case names from the existing experimental features.
    with h5py.File(a.src_labels, "r") as src:
        type_code = src["type_code"][:]
        storey    = src["storey"][:]
        end       = src["end"][:]
        severity  = src["severity"][:]
        names     = src["names"][:]
    assert len(type_code) == n_cases, "label/FRF count mismatch"

    # Pristine reference = mean over pristine cases.
    pristine_mask = (type_code == 0)
    H_ref = H[pristine_mask].mean(axis=0)           # (1601, 9) complex
    print(f"pristine reference: {pristine_mask.sum()} cases, "
          f"H_ref shape {H_ref.shape}")

    with h5py.File(a.out, "w") as out:
        out.create_dataset("freqs", data=freqs)
        out.create_dataset("frf_real",
                            data=H.real.astype(np.float32),
                            compression="gzip", compression_opts=4,
                            chunks=(1, n_freq, n_ch))
        out.create_dataset("frf_imag",
                            data=H.imag.astype(np.float32),
                            compression="gzip", compression_opts=4,
                            chunks=(1, n_freq, n_ch))
        out.create_dataset("frf_mag",
                            data=np.abs(H).astype(np.float32),
                            compression="gzip", compression_opts=4,
                            chunks=(1, n_freq, n_ch))
        out.create_dataset("type_code", data=type_code)
        out.create_dataset("storey",    data=storey)
        out.create_dataset("end",       data=end)
        out.create_dataset("severity",  data=severity)
        out.create_dataset("names",     data=names)
        ref_g = out.create_group("reference")
        ref_g.create_dataset("frf_complex", data=H_ref.astype(np.complex64))
        ref_g.create_dataset("frf_mag", data=np.abs(H_ref).astype(np.float32))
        out.attrs["n_freq"] = n_freq
        out.attrs["f_lo"]   = float(freqs[0])
        out.attrs["f_hi"]   = float(freqs[-1])
    print(f"wrote {a.out}  ({a.out.stat().st_size/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
