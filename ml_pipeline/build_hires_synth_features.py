"""Build hires synth features (FRFs only) from the N_T=4096 chunks.

Reads dataset_hires/chunk_*.h5 (regenerated at 16 s simulation length so
the native FFT grid is df=0.0625 Hz, matching exp), computes FRFs by
FFT(signals) / FFT(excitation), crops to 0–100 Hz to align with the
experimental grid exactly (1601 bins), and writes:

  - frf_real, frf_imag, frf_mag    (10000, 1601, 9) float32
  - freqs                          (1601,) float32
  - reference/frf_complex          (1601, 9) complex64   (pristine mean)
  - type_code, storey, end, severity, params           (10000,) labels

Skips timeseries, modal, indicators — not needed for the hires CFDAC
preliminary; cuts the file from ~3 GB to ~1.7 GB.
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


def _frf_from_chunk(f, freq_mask):
    sig = f["signals"][:]           # (n, N_T, 9)
    exc = f["excitation"][:]        # (N_T,)
    fft_sig = np.fft.rfft(sig, axis=1)         # (n, N_F, 9)
    fft_exc = np.fft.rfft(exc)                  # (N_F,)
    # Drop the zero-freq bin to avoid divide-by-zero noise, replace with
    # tiny epsilon. We keep f=0 in the output (constant zero band) by
    # masking it out via freq_mask anyway -- but the chirp has very low
    # energy at DC so the division is numerically unstable.
    fft_exc[0] = fft_exc[0] if abs(fft_exc[0]) > 1e-12 else 1e-12
    H = fft_sig / fft_exc[None, :, None]        # (n, N_F, 9) complex
    H = H[:, freq_mask, :]                       # crop to band
    return H


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", type=Path, default=_REPO / "dataset_hires")
    p.add_argument("--out", type=Path,
                    default=_REPO / "dataset" / "features_hires.h5")
    p.add_argument("--f-lo", type=float, default=0.0)
    p.add_argument("--f-hi", type=float, default=100.0)
    a = p.parse_args()

    chunk_paths = sorted(a.chunks.glob("chunk_*.h5"))
    if not chunk_paths:
        raise FileNotFoundError(f"no chunks in {a.chunks}")
    print(f"  {len(chunk_paths)} chunks in {a.chunks}")

    with h5py.File(chunk_paths[0], "r") as f:
        time = f["time"][:]
        n_t = time.shape[0]
        n_ch = f["signals"].shape[2]
    fs = 1.0 / float(time[1] - time[0])
    freqs = np.fft.rfftfreq(n_t, d=1.0 / fs).astype(np.float32)
    band_mask = (freqs >= a.f_lo) & (freqs <= a.f_hi)
    freqs_band = freqs[band_mask]
    n_freq = int(band_mask.sum())
    print(f"  band: {freqs_band[0]:.3f}–{freqs_band[-1]:.3f} Hz, "
          f"{n_freq} bins, df={freqs_band[1]-freqs_band[0]:.4f}")

    n_total = 0
    for p_ in chunk_paths:
        with h5py.File(p_, "r") as f:
            n_total += f["signals"].shape[0]
    print(f"  {n_total} total samples")

    # Pristine reference: mean FRF across pristine samples (type_code == 0).
    print("  building pristine reference …")
    pristine_sum = None; n_pristine = 0
    for p_ in chunk_paths:
        with h5py.File(p_, "r") as f:
            tc = f["labels/type_code"][:]
            mask = (tc == 0)
            if not mask.any():
                continue
            H = _frf_from_chunk(f, band_mask)        # (n, 1601, 9)
            H_p = H[mask]
            if pristine_sum is None:
                pristine_sum = H_p.sum(axis=0)
            else:
                pristine_sum = pristine_sum + H_p.sum(axis=0)
            n_pristine += mask.sum()
    H_ref = (pristine_sum / n_pristine).astype(np.complex64)
    print(f"  reference from {n_pristine} pristine samples, shape={H_ref.shape}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(a.out, "w") as out:
        out.create_dataset("freqs", data=freqs_band)
        out.create_dataset(
            "frf_real", shape=(n_total, n_freq, n_ch),
            dtype=np.float32, compression="gzip", compression_opts=4,
            chunks=(1, n_freq, n_ch))
        out.create_dataset(
            "frf_imag", shape=(n_total, n_freq, n_ch),
            dtype=np.float32, compression="gzip", compression_opts=4,
            chunks=(1, n_freq, n_ch))
        out.create_dataset(
            "frf_mag", shape=(n_total, n_freq, n_ch),
            dtype=np.float32, compression="gzip", compression_opts=4,
            chunks=(1, n_freq, n_ch))
        ref_g = out.create_group("reference")
        ref_g.create_dataset("frf_complex", data=H_ref)
        ref_g.create_dataset("frf_mag",     data=np.abs(H_ref).astype(np.float32))

        # Label arrays preallocated; filled while iterating.
        labels_g = out.create_group("labels")
        type_code_d = labels_g.create_dataset("type_code", shape=(n_total,),
                                                dtype=np.int8)
        storey_d    = labels_g.create_dataset("storey", shape=(n_total,),
                                                dtype=np.int8)
        end_d       = labels_g.create_dataset("end", shape=(n_total,),
                                                dtype=np.int8)
        severity_d  = labels_g.create_dataset("severity", shape=(n_total,),
                                                dtype=np.float32)
        # top-level convenience aliases (the train pipeline reads these names)
        # we'll fill them via the labels arrays after the loop

        i0 = 0
        for ci, p_ in enumerate(chunk_paths):
            with h5py.File(p_, "r") as f:
                H = _frf_from_chunk(f, band_mask)         # (n, 1601, 9) complex
                n = H.shape[0]
                out["frf_real"][i0:i0 + n] = H.real.astype(np.float32)
                out["frf_imag"][i0:i0 + n] = H.imag.astype(np.float32)
                out["frf_mag"] [i0:i0 + n] = np.abs(H).astype(np.float32)
                type_code_d[i0:i0 + n] = f["labels/type_code"][:]
                storey_d   [i0:i0 + n] = f["labels/storey"][:]
                end_d      [i0:i0 + n] = f["labels/end"][:]
                severity_d [i0:i0 + n] = f["labels/severity"][:]
                i0 += n
            if (ci + 1) % 10 == 0 or ci + 1 == len(chunk_paths):
                print(f"   .. chunk {ci+1}/{len(chunk_paths)}  ({i0}/{n_total})")
        # Top-level aliases so existing label loaders work
        out.create_dataset("type_code", data=type_code_d[:])
        out.create_dataset("storey",    data=storey_d[:])
        out.create_dataset("end",       data=end_d[:])
        out.create_dataset("severity",  data=severity_d[:])
        out.attrs["n_freq"] = n_freq
        out.attrs["f_lo"]   = float(freqs_band[0])
        out.attrs["f_hi"]   = float(freqs_band[-1])

    sz = a.out.stat().st_size / 1e9
    print(f"wrote {a.out}  ({sz:.2f} GB)")


if __name__ == "__main__":
    main()
