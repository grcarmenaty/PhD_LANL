"""Append a (n, 128, 128, 2) CFDAC dataset to ``dataset/features.h5``.

CFDAC is the Complex Frequency-Domain Assurance Criterion of
Pastor & Binda (2012); it is the matricial damage signature behind
the SCI / M2L scalar indicators in `pymodal`.  For each sample we
compute

    CFDAC(i, j) = ( Σ_k H_dmg_k(ω_i) · conj(H_ref_k(ω_j)) )² /
                  ( Σ_k |H_dmg_k(ω_i)|² · Σ_k |H_ref_k(ω_j)|² )

with the *same* reference FRF that ``features.py`` already saved
under ``reference/frf_complex``.  Frequencies are decimated to 128
bins inside the 5–100 Hz band so the resulting tensor has a
manageable size of ``(n, 128, 128, 2)`` (real, imag).
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
if str(_REPO.parent / "pymodal") not in sys.path:
    sys.path.insert(0, str(_REPO.parent / "pymodal"))

from pymodal import utils as pm_utils  # noqa: E402


CFDAC_N = 128


def _decimate(H: np.ndarray, n_target: int) -> np.ndarray:
    """Bin-average `H` from `(n_freq, …)` to `(n_target, …)`."""
    n_freq = H.shape[0]
    edges  = np.linspace(0, n_freq, n_target + 1).astype(int)
    out = np.empty((n_target,) + H.shape[1:], dtype=H.dtype)
    for i in range(n_target):
        lo, hi = edges[i], max(edges[i] + 1, edges[i + 1])
        out[i] = H[lo:hi].mean(axis=0)
    return out


def append_cfdac(features_path: Path,
                  n_target: int = CFDAC_N) -> None:
    with h5py.File(features_path, "r") as f:
        H_ref = f["reference/frf_complex"][:]   # (N_F, 9) complex64
        n_samples = f["frf_real"].shape[0]

    # NOTE: pymodal's ``value_CFDAC`` produces a (rows, rows) matrix from
    # ``rows × cols`` inputs.  For the frequency-domain CFDAC we want
    # (n_freq, n_freq), so we keep H in the natural (n_freq, n_dof)
    # orientation here — NOT the (n_dof, n_freq) ``_as_matrix`` form used
    # by the scalar indicators in ``features.py``.
    H_ref_d = _decimate(H_ref, n_target)        # (n_target, 9)

    with h5py.File(features_path, "a") as f:
        if "cfdac_real" in f:
            del f["cfdac_real"]
            del f["cfdac_imag"]
        d_real = f.create_dataset(
            "cfdac_real", shape=(n_samples, n_target, n_target),
            dtype=np.float32, compression="gzip", compression_opts=4,
            chunks=(1, n_target, n_target),
        )
        d_imag = f.create_dataset(
            "cfdac_imag", shape=(n_samples, n_target, n_target),
            dtype=np.float32, compression="gzip", compression_opts=4,
            chunks=(1, n_target, n_target),
        )
        for i in range(n_samples):
            # Reconstruct the complex FRF for this sample.
            H_re = f["frf_real"][i]          # (N_F, 9)
            H_im = f["frf_imag"][i]
            H_c  = (H_re + 1j * H_im).astype(np.complex64)
            H_d  = _decimate(H_c, n_target)  # (n_target, 9)
            cfdac = pm_utils.value_CFDAC(H_ref_d, H_d)          # (n_target, n_target) complex
            d_real[i] = cfdac.real.astype(np.float32)
            d_imag[i] = cfdac.imag.astype(np.float32)
            if (i + 1) % 500 == 0 or i == n_samples - 1:
                print(f"   .. {i + 1}/{n_samples}")
        # Reference CFDAC (CFDAC of the pristine mean against itself).
        ref_cfdac = pm_utils.value_CFDAC(H_ref_d, H_ref_d)
        if "reference/cfdac_real" in f:
            del f["reference/cfdac_real"]
            del f["reference/cfdac_imag"]
        f["reference"].create_dataset("cfdac_real",
                                         data=ref_cfdac.real.astype(np.float32))
        f["reference"].create_dataset("cfdac_imag",
                                         data=ref_cfdac.imag.astype(np.float32))
        f.attrs["cfdac_n"] = n_target

    print(f"\nWrote CFDAC datasets → {features_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path,
                          default=_REPO / "dataset" / "features.h5")
    parser.add_argument("--n", type=int, default=CFDAC_N)
    args = parser.parse_args()
    append_cfdac(args.features, args.n)


if __name__ == "__main__":
    main()
