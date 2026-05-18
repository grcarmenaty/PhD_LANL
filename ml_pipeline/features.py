"""Feature extraction from the synthetic time-series dataset.

Several vibrational representations are computed per sample and saved
to a single HDF5 file so the downstream training scripts can index any
of them by sample id:

  * ``timeseries``      – raw ``(N_T, 9)`` acceleration (passed through)
  * ``frf_mag``         – ``(N_F, 9)`` magnitude of accelerance H1
  * ``frf_complex``     – ``(N_F, 9)`` complex H1 (real/imag stored as 2 channels)
  * ``modal``           – natural frequencies, peak amplitudes, band energy
                          (a fixed-length scalar feature vector)
  * ``indicators``      – pymodal damage indicators vs a pristine reference
                          (SCI, unsigned_SCI, DRQ, AIGAC, FRFRMS, FRFSF,
                          FRFSM, ODS_diff, r2_imag) + summary stats of
                          1-D indicators (RVAC, GAC, M2L)

The reference FRF used by ``indicators`` is the mean of all Pristine
samples (averaged on the unit FFT-bin grid) – computed once and stored
under ``/reference/frf`` for reproducibility.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO.parent / "pymodal") not in sys.path:
    sys.path.insert(0, str(_REPO.parent / "pymodal"))

import pymodal as pm  # noqa: E402
from pymodal import utils as pm_utils  # noqa: E402

from ml_pipeline.case_design import TYPE_PRISTINE  # noqa: E402


# ── frequency grid configuration (must match generate_dataset.py) ───────────
def get_freq_band(time: np.ndarray, signals_shape: tuple,
                   f_lo: float = 2.0, f_hi: float = 100.0
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """Return the FFT-bin grid and a boolean mask selecting [f_lo, f_hi]."""
    n_t = signals_shape[1]
    fs  = 1.0 / float(time[1] - time[0])
    freqs = np.fft.rfftfreq(n_t, d=1.0 / fs)
    mask  = (freqs >= f_lo) & (freqs <= f_hi)
    return freqs, mask


def compute_frfs(signals: np.ndarray, excitation: np.ndarray,
                  freq_mask: np.ndarray) -> np.ndarray:
    """Compute the H1-style FRF per sample.

    ``signals``    : (n, N_T, 9) float
    ``excitation`` : (N_T,) float
    Returns        : (n, N_F, 9) complex64  where N_F = mask.sum().
    """
    n, n_t, n_ch = signals.shape
    Y = np.fft.rfft(signals.astype(np.float64), axis=1)        # (n, N_F, 9)
    X = np.fft.rfft(excitation.astype(np.float64))             # (N_F,)
    # H1 = Sxy / Sxx ; with a deterministic input this reduces to Y / X.
    # P0.5: guard against near-zero excitation bins (DC and below the
    # chirp's lowest frequency).  Without this, lowering CHIRP_F_LO below
    # ~1 Hz introduces NaNs/Infs that propagate silently through every
    # downstream feature.
    X_safe = np.where(np.abs(X) > 1e-12, X, 1e-12)
    H = Y / X_safe[None, :, None]
    return H[:, freq_mask, :].astype(np.complex64)


# ── modal-style fixed-length features ───────────────────────────────────────
def modal_features(H_mag: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """A compact fixed-length feature vector from |H(f)|.

    Concatenates, for each of 9 channels:
      [peak1_freq, peak1_amp, peak2_freq, peak2_amp, peak3_freq, peak3_amp,
       mean_log_amp, std_log_amp, band_energy]
    so length = 9 * 9 = 81.

    Peaks are found by ``argpartition`` of |H|.
    """
    n_freq, n_ch = H_mag.shape
    feats = []
    for c in range(n_ch):
        amp = H_mag[:, c]
        log_amp = np.log10(np.clip(amp, 1e-12, None))
        # Top-3 peaks
        top_idx = np.argpartition(-amp, 3)[:3]
        top_idx = top_idx[np.argsort(-amp[top_idx])]
        for k in range(3):
            if k < len(top_idx):
                feats.append(float(freqs[top_idx[k]]))
                feats.append(float(log_amp[top_idx[k]]))
            else:
                feats.append(0.0)
                feats.append(0.0)
        feats.append(float(np.mean(log_amp)))
        feats.append(float(np.std(log_amp)))
        feats.append(float(np.sum(amp ** 2)))
    return np.asarray(feats, dtype=np.float32)


# ── damage-indicator features ───────────────────────────────────────────────
def _as_dof_by_freq(H: np.ndarray) -> np.ndarray:
    """pymodal convention: ``(n_dof, n_freq)`` real or complex."""
    # H here is (N_F, 9). Transpose to (9, N_F).
    return H.T


def indicator_features(H_complex: np.ndarray,
                        H_ref_complex: np.ndarray) -> np.ndarray:
    """Compute all pymodal damage indicators against a reference FRF.

    Returns a fixed-length scalar feature vector.
    """
    ref = _as_dof_by_freq(H_ref_complex)
    dmg = _as_dof_by_freq(H_complex)

    cfdac_ref = np.abs(pm_utils.value_CFDAC(ref, ref))
    cfdac_dmg = np.abs(pm_utils.value_CFDAC(ref, dmg))
    cfdac_dmg_signed = pm_utils.value_CFDAC(ref, dmg)

    feats: List[float] = []
    feats.append(float(pm_utils.SCI(cfdac_ref, cfdac_dmg)))
    feats.append(float(pm_utils.unsigned_SCI(cfdac_ref, cfdac_dmg)))

    rvac = np.real(pm_utils.value_RVAC(ref, dmg))
    gac  = np.real(pm_utils.value_GAC(ref, dmg))
    feats.append(float(pm_utils.DRQ(rvac)))
    feats.append(float(pm_utils.AIGAC(gac)))

    feats.append(float(pm_utils.FRFRMS(ref, dmg)))
    feats.append(float(pm_utils.FRFSF(ref, dmg)))
    feats.append(float(pm_utils.FRFSM(ref, dmg, 6.0)))
    feats.append(float(pm_utils.ODS_diff(ref, dmg)))
    feats.append(float(pm_utils.r2_imag(ref, dmg)))

    # Summary stats of 1-D indicators (RVAC, GAC, M2L).
    for vec in (rvac, gac):
        feats.append(float(np.mean(vec)))
        feats.append(float(np.std(vec)))
        feats.append(float(np.min(vec)))
        feats.append(float(np.max(vec)))

    m2l = np.real(pm_utils.M2L(cfdac_dmg_signed))
    feats.append(float(np.mean(m2l)))
    feats.append(float(np.std(m2l)))
    feats.append(float(np.min(m2l)))
    feats.append(float(np.max(m2l)))
    feats.append(float(np.sum(np.abs(m2l))))

    return np.asarray(feats, dtype=np.float32)


INDICATOR_NAMES = [
    "SCI", "unsigned_SCI", "DRQ", "AIGAC",
    "FRFRMS", "FRFSF", "FRFSM_6dB", "ODS_diff", "r2_imag",
    "RVAC_mean", "RVAC_std", "RVAC_min", "RVAC_max",
    "GAC_mean",  "GAC_std",  "GAC_min",  "GAC_max",
    "M2L_mean",  "M2L_std",  "M2L_min",  "M2L_max", "M2L_abs_sum",
]


# ── reference FRF (pristine mean) ───────────────────────────────────────────
def build_pristine_reference(chunk_paths: List[Path],
                              freq_mask: np.ndarray) -> np.ndarray:
    """Average all Pristine samples' complex FRFs across the dataset."""
    accum: np.ndarray | None = None
    count = 0
    for path in chunk_paths:
        with h5py.File(path, "r") as f:
            tc = f["labels/type_code"][:]
            keep = np.where(tc == TYPE_PRISTINE)[0]
            if keep.size == 0:
                continue
            signals = f["signals"][keep]                  # (k, N_T, 9)
            exc     = f["excitation"][:]
            H = compute_frfs(signals, exc, freq_mask)     # (k, N_F, 9)
        if accum is None:
            accum = np.zeros(H.shape[1:], dtype=np.complex128)
        accum += H.sum(axis=0).astype(np.complex128)
        count += H.shape[0]
    if accum is None or count == 0:
        raise RuntimeError("No Pristine samples found to build the reference.")
    return (accum / count).astype(np.complex64)


# ── main extraction routine ─────────────────────────────────────────────────
def extract_all(dataset_dir: Path, out_path: Path,
                 f_lo: float = 5.0, f_hi: float = 100.0) -> None:
    chunk_paths = sorted(dataset_dir.glob("chunk_*.h5"))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunks in {dataset_dir}")

    # Inspect chunk 0 to learn shape and build freq grid.
    with h5py.File(chunk_paths[0], "r") as f:
        time    = f["time"][:]
        sig0    = f["signals"][0:1]
        signals_shape = (1, sig0.shape[1], sig0.shape[2])
    freqs, mask = get_freq_band(time, signals_shape, f_lo, f_hi)
    freqs_band = freqs[mask]
    n_freq_band = int(mask.sum())
    n_ch = sig0.shape[2]
    print(f"  band freqs: {freqs_band[0]:.2f} .. {freqs_band[-1]:.2f} Hz "
          f"({n_freq_band} bins)")

    # Build reference (pristine mean).
    print("  building pristine reference …")
    H_ref = build_pristine_reference(chunk_paths, mask)

    # Pre-count total samples.
    n_total = 0
    for p in chunk_paths:
        with h5py.File(p, "r") as f:
            n_total += f["signals"].shape[0]
    print(f"  {n_total} samples across {len(chunk_paths)} chunks")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as out:
        out.create_dataset("freqs", data=freqs_band.astype(np.float32))
        ref_g = out.create_group("reference")
        ref_g.create_dataset("frf_complex", data=H_ref)
        ref_g.create_dataset("frf_mag",     data=np.abs(H_ref))

        timeseries_d  = out.create_dataset(
            "timeseries", shape=(n_total, signals_shape[1], n_ch),
            dtype=np.float32, compression="gzip", compression_opts=4,
            chunks=(1, signals_shape[1], n_ch),
        )
        frf_mag_d = out.create_dataset(
            "frf_mag", shape=(n_total, n_freq_band, n_ch),
            dtype=np.float32, compression="gzip", compression_opts=4,
            chunks=(1, n_freq_band, n_ch),
        )
        frf_real_d = out.create_dataset(
            "frf_real", shape=(n_total, n_freq_band, n_ch),
            dtype=np.float32, compression="gzip", compression_opts=4,
            chunks=(1, n_freq_band, n_ch),
        )
        frf_imag_d = out.create_dataset(
            "frf_imag", shape=(n_total, n_freq_band, n_ch),
            dtype=np.float32, compression="gzip", compression_opts=4,
            chunks=(1, n_freq_band, n_ch),
        )

        # Modal-style feature length is fixed by modal_features() definition.
        # We'll discover it on the first sample.
        modal_len: int | None = None
        ind_len:   int | None = None
        modal_d:   h5py.Dataset | None = None
        ind_d:     h5py.Dataset | None = None

        labels_g = out.create_group("labels")
        type_d     = labels_g.create_dataset("type_code", shape=(n_total,), dtype=np.int8)
        storey_d   = labels_g.create_dataset("storey",    shape=(n_total,), dtype=np.int8)
        end_d      = labels_g.create_dataset("end",       shape=(n_total,), dtype=np.int8)
        severity_d = labels_g.create_dataset("severity",  shape=(n_total,), dtype=np.float32)
        sample_id  = labels_g.create_dataset("sample_id", shape=(n_total,), dtype=np.int32)

        i = 0
        for path in chunk_paths:
            with h5py.File(path, "r") as f:
                signals = f["signals"][:]
                exc     = f["excitation"][:]
                tc      = f["labels/type_code"][:]
                sty     = f["labels/storey"][:]
                end     = f["labels/end"][:]
                sev     = f["labels/severity"][:]
                sid     = f["labels/sample_id"][:]
            n = signals.shape[0]
            H_complex = compute_frfs(signals, exc, mask)        # (n, N_F, 9)
            H_mag     = np.abs(H_complex).astype(np.float32)

            for j in range(n):
                timeseries_d[i + j] = signals[j]
                frf_mag_d[i + j]    = H_mag[j]
                frf_real_d[i + j]   = H_complex[j].real.astype(np.float32)
                frf_imag_d[i + j]   = H_complex[j].imag.astype(np.float32)
                mf = modal_features(H_mag[j], freqs_band)
                if modal_d is None:
                    modal_len = mf.shape[0]
                    modal_d = out.create_dataset(
                        "modal", shape=(n_total, modal_len), dtype=np.float32,
                        compression="gzip", compression_opts=4,
                    )
                modal_d[i + j] = mf

                ind_feat = indicator_features(H_complex[j], H_ref)
                if ind_d is None:
                    ind_len = ind_feat.shape[0]
                    ind_d = out.create_dataset(
                        "indicators", shape=(n_total, ind_len), dtype=np.float32,
                        compression="gzip", compression_opts=4,
                    )
                ind_d[i + j] = ind_feat

            type_d[i:i+n]     = tc
            storey_d[i:i+n]   = sty
            end_d[i:i+n]      = end
            severity_d[i:i+n] = sev
            sample_id[i:i+n]  = sid
            i += n
            print(f"   .. {i}/{n_total}")

        out.attrs["indicator_names"] = ",".join(INDICATOR_NAMES)
        out.attrs["n_channels"] = n_ch
        out.attrs["f_lo_hz"] = f_lo
        out.attrs["f_hi_hz"] = f_hi

    print(f"Wrote features → {out_path}  ({out_path.stat().st_size/1024/1024:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=_REPO / "dataset")
    parser.add_argument("--out",     type=Path, default=_REPO / "dataset" / "features.h5")
    parser.add_argument("--f-lo",    type=float, default=5.0)
    parser.add_argument("--f-hi",    type=float, default=100.0)
    args = parser.parse_args()
    extract_all(args.dataset, args.out, args.f_lo, args.f_hi)


if __name__ == "__main__":
    main()
