"""Pre-compute every feature representation for the full 2638-case
IQS experimental dataset and save to ``dataset/experimental_features.h5``.

Mirrors the synthetic ``features.h5`` schema so downstream training,
evaluation and plot scripts can read either file with the same code
path.

Inputs:
    experimental_frfs.h5    (n_cases, 1601, 9) complex64
        + cases attribute, freq dataset, labels dataset

Outputs:
    dataset/experimental_features.h5
        names           (n_cases,) string
        type_code       (n_cases,) int8
        storey          (n_cases,) int8
        end             (n_cases,) int8
        severity        (n_cases,) float32
        timeseries      (n_cases, 1024, 9)   float32
        frf_mag         (n_cases, N_F, 9)    float32
        frf_real        (n_cases, N_F, 9)    float32
        frf_imag        (n_cases, N_F, 9)    float32
        modal           (n_cases, 81)        float32
        indicators      (n_cases, 22)        float32
        cfdac_real      (n_cases, 128, 128)  float32
        cfdac_imag      (n_cases, 128, 128)  float32
        cfdac_mag       (n_cases, 128, 128)  float32
        cfdac_phase     (n_cases, 128, 128)  float32
        freqs           (N_F,)               float32
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO.parent / "pymodal") not in sys.path:
    sys.path.insert(0, str(_REPO.parent / "pymodal"))

from pymodal import utils as pm_utils                              # noqa: E402

from ml_pipeline.evaluate import (                                    # noqa: E402
    primary_op, resample_frf, synthesize_timeseries,
)
from ml_pipeline.features import (                                    # noqa: E402
    modal_features, indicator_features, INDICATOR_NAMES,
)
from ml_pipeline.generate_dataset import (                            # noqa: E402
    make_chirp, N_T, FS, fft_freqs,
)
from ml_pipeline.cfdac import _decimate                                # noqa: E402
from ml_pipeline.case_design import TYPE_PRISTINE                       # noqa: E402


def build(exp_frfs_path: Path, syn_features_path: Path,
            out_path: Path, batch: int = 64) -> None:
    with h5py.File(exp_frfs_path, "r") as f:
        H_exp_full = f["data"][:]                  # (n, 1601, 9) complex64
        f_src      = f["freq"][:]
        case_names = json.loads(f.attrs["case_names_json"])
    n_cases, n_src_freq, n_ch = H_exp_full.shape
    print(f"loaded {n_cases} experimental cases, {n_src_freq} freqs, "
            f"{n_ch} channels")

    # Flip S2 (channel 0) — sensor mounted inverted in the IQS protocol.
    H_exp_full[:, :, 0] *= -1.0

    bins = fft_freqs(N_T, FS)
    with h5py.File(syn_features_path, "r") as f:
        f_band = f["freqs"][:]
        H_ref_synth = f["reference/frf_complex"][:]
        f_lo   = float(f.attrs["f_lo_hz"]); f_hi = float(f.attrs["f_hi_hz"])
        cfdac_n = int(f.attrs["cfdac_n"])
    band_mask = (bins >= f_lo) & (bins <= f_hi)
    n_freq    = int(band_mask.sum())

    # P0.1: build an EXPERIMENTAL pristine reference by averaging the
    # complex band-FRFs of every case labelled Pristine in the IQS data.
    # Using the synth pristine mean for CFDAC/indicators on exp data
    # injects the full synth-vs-exp domain shift into the features
    # themselves; the experimental ref removes that bias.
    pristine_codes = np.array(
        [primary_op(name)["type_code"] for name in case_names], dtype=np.int8
    )
    pristine_idx = np.where(pristine_codes == TYPE_PRISTINE)[0]
    if pristine_idx.size == 0:
        print("WARNING: no Pristine experimental cases; falling back to synth ref")
        H_ref = H_ref_synth
    else:
        print(f"computing experimental pristine reference from "
              f"{pristine_idx.size} cases")
        H_ref = np.zeros((n_freq, n_ch), dtype=np.complex64)
        for j in pristine_idx:
            H_full_j = resample_frf(H_exp_full[j], f_src, bins)
            H_ref += H_full_j[band_mask]
        H_ref = (H_ref / float(pristine_idx.size)).astype(np.complex64)

    H_ref_d = _decimate(H_ref, cfdac_n)            # (cfdac_n, 9) complex64
    t_axis, chirp = make_chirp()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as out:
        # Strings + scalars first.
        dt = h5py.string_dtype("utf-8")
        out.create_dataset("names", data=np.array(case_names, dtype=object),
                              dtype=dt)
        type_code = np.zeros(n_cases, dtype=np.int8)
        storey    = np.full(n_cases, -1, dtype=np.int8)
        end       = np.full(n_cases, -1, dtype=np.int8)
        severity  = np.zeros(n_cases, dtype=np.float32)
        for i, name in enumerate(case_names):
            op = primary_op(name)
            type_code[i] = op["type_code"]
            storey[i]    = op["storey"]
            end[i]       = op["end"]
            severity[i]  = op["severity"]
        out.create_dataset("type_code", data=type_code)
        out.create_dataset("storey",    data=storey)
        out.create_dataset("end",       data=end)
        out.create_dataset("severity",  data=severity)
        out.create_dataset("freqs",     data=f_band.astype(np.float32))

        ts = out.create_dataset(
            "timeseries", shape=(n_cases, N_T, n_ch), dtype=np.float32,
            compression="gzip", compression_opts=4, chunks=(1, N_T, n_ch))
        fmag = out.create_dataset(
            "frf_mag", shape=(n_cases, n_freq, n_ch), dtype=np.float32,
            compression="gzip", compression_opts=4, chunks=(1, n_freq, n_ch))
        freal = out.create_dataset(
            "frf_real", shape=(n_cases, n_freq, n_ch), dtype=np.float32,
            compression="gzip", compression_opts=4, chunks=(1, n_freq, n_ch))
        fimag = out.create_dataset(
            "frf_imag", shape=(n_cases, n_freq, n_ch), dtype=np.float32,
            compression="gzip", compression_opts=4, chunks=(1, n_freq, n_ch))
        modal_d = out.create_dataset(
            "modal", shape=(n_cases, 81), dtype=np.float32,
            compression="gzip", compression_opts=4)
        ind_d = out.create_dataset(
            "indicators", shape=(n_cases, 22), dtype=np.float32,
            compression="gzip", compression_opts=4)
        cre = out.create_dataset(
            "cfdac_real", shape=(n_cases, cfdac_n, cfdac_n),
            dtype=np.float32, compression="gzip", compression_opts=4,
            chunks=(1, cfdac_n, cfdac_n))
        cim = out.create_dataset(
            "cfdac_imag", shape=(n_cases, cfdac_n, cfdac_n),
            dtype=np.float32, compression="gzip", compression_opts=4,
            chunks=(1, cfdac_n, cfdac_n))
        cmag = out.create_dataset(
            "cfdac_mag", shape=(n_cases, cfdac_n, cfdac_n),
            dtype=np.float32, compression="gzip", compression_opts=4,
            chunks=(1, cfdac_n, cfdac_n))
        cph = out.create_dataset(
            "cfdac_phase", shape=(n_cases, cfdac_n, cfdac_n),
            dtype=np.float32, compression="gzip", compression_opts=4,
            chunks=(1, cfdac_n, cfdac_n))

        for i in range(n_cases):
            H_full = resample_frf(H_exp_full[i], f_src, bins)         # (N, 9) complex
            H_band = H_full[band_mask]                                # (N_F, 9)
            ts[i]    = synthesize_timeseries(H_full, chirp)
            fmag[i]  = np.abs(H_band).astype(np.float32)
            freal[i] = H_band.real.astype(np.float32)
            fimag[i] = H_band.imag.astype(np.float32)
            modal_d[i] = modal_features(fmag[i], f_band)
            ind_d[i]   = indicator_features(H_band, H_ref)

            H_d = _decimate(H_band, cfdac_n)                          # (cfdac_n, 9)
            cf  = pm_utils.value_CFDAC(H_ref_d, H_d)                  # (cfdac_n, cfdac_n) cplx
            cre[i]  = cf.real.astype(np.float32)
            cim[i]  = cf.imag.astype(np.float32)
            cmag[i] = np.sqrt(cf.real ** 2 + cf.imag ** 2).astype(np.float32)
            cph[i]  = np.arctan2(cf.imag, cf.real).astype(np.float32)
            if (i + 1) % 100 == 0 or i == n_cases - 1:
                print(f"   .. {i + 1}/{n_cases}")

        # P0.1: persist BOTH references so downstream consumers can
        # choose, and so the file documents which ref the indicators /
        # CFDAC inside it were computed against.
        ref_grp = out.create_group("reference")
        ref_grp.create_dataset("frf_complex",       data=H_ref)
        ref_grp.create_dataset("frf_complex_synth", data=H_ref_synth)
        ref_grp.attrs["used_for_features"] = "frf_complex"  # = experimental pristine mean
        ref_grp.attrs["n_pristine_cases"]  = int(pristine_idx.size)

        out.attrs["n_channels"] = n_ch
        out.attrs["f_lo_hz"]    = f_lo
        out.attrs["f_hi_hz"]    = f_hi
        out.attrs["cfdac_n"]    = cfdac_n
        out.attrs["indicator_names"] = ",".join(INDICATOR_NAMES)

    print(f"\nWrote {out_path}  "
            f"({out_path.stat().st_size / 1024 / 1024:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=Path,
                          default=_REPO / "experimental_frfs.h5")
    parser.add_argument("--syn", type=Path,
                          default=_REPO / "dataset" / "features.h5")
    parser.add_argument("--out", type=Path,
                          default=_REPO / "dataset" / "experimental_features.h5")
    args = parser.parse_args()
    build(args.exp, args.syn, args.out)


if __name__ == "__main__":
    main()
