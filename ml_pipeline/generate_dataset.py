"""Generate the 10000-sample synthetic time-series dataset.

For every sample the script
  1. Draws a SampleParams record (case_design + variation).
  2. Builds the perturbed BuildingGeometry.
  3. Computes the 9-sensor accelerance FRF on the FFT-bin grid.
  4. Multiplies by the FFT of a deterministic chirp force F(t) and
     inverse-transforms to obtain the 9-channel time response y(t).
  5. Appends (signals, labels, params) to the current chunk file; a
     new chunk is opened once the on-disk size approaches 20 MB.

The chirp excitation is deterministic and identical for every sample
(no noise per the project spec) — variability across samples is
entirely physical (material, geometry, joint and damage parameters).
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

from ml_pipeline.case_design import (   # noqa: E402
    expand_samples, TYPE_NAMES, SEVERITY_UNITS,
)
from ml_pipeline.variation import (   # noqa: E402
    SampleParams, sample_params, geometry_from_params,
)
from reduced_model_semirigid import compute_frf_matrix  # noqa: E402
sys.path.insert(0, str(_REPO))
from model_3sbb import _input_position, _sensor_positions  # noqa: E402


# ── Time-series acquisition spec ────────────────────────────────────────────
FS         = 256.0     # Hz   (sampling frequency)
N_T        = 1024      # samples per signal
T_DURATION = N_T / FS  # 4.0 s
N_CHANNELS = 9
N_INPUTS   = 1

# Chirp parameters
CHIRP_F_LO    = 2.0     # Hz
CHIRP_F_HI    = 100.0   # Hz
CHIRP_AMPL_N  = 1.0     # N

# Chunking
MAX_CHUNK_MB     = 18.0
DEFAULT_SEED     = 20260511
DEFAULT_PER_TYPE = 2000

# ── Deterministic chirp excitation ──────────────────────────────────────────
def make_chirp(n_t: int = N_T, fs: float = FS,
                f_lo: float = CHIRP_F_LO,
                f_hi: float = CHIRP_F_HI,
                amplitude: float = CHIRP_AMPL_N) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic linear chirp force F(t) and its time axis."""
    t = np.arange(n_t, dtype=np.float64) / fs
    T = n_t / fs
    k = (f_hi - f_lo) / T
    phase = 2.0 * np.pi * (f_lo * t + 0.5 * k * t * t)
    f = amplitude * np.sin(phase)
    return t.astype(np.float32), f.astype(np.float32)


def fft_freqs(n_t: int = N_T, fs: float = FS) -> np.ndarray:
    """Single-sided positive-frequency grid matching ``np.fft.rfft``."""
    return np.fft.rfftfreq(n_t, d=1.0 / fs)


# ── One-sample time-domain response ─────────────────────────────────────────
def simulate_sample(params: SampleParams,
                     chirp: np.ndarray,
                     freq_array: np.ndarray) -> np.ndarray:
    """Return the 9-channel acceleration response time series.

    ``freq_array`` must be the FFT-bin grid (`np.fft.rfftfreq(N_T, 1/FS)`).
    Output shape ``(N_T, 9)`` float32.
    """
    geom = geometry_from_params(params)
    damping = getattr(geom, 'damping_modes', None)
    if damping is None or np.size(damping) == 0:
        damping = geom.damping
    H = compute_frf_matrix(freq_array,
                            _input_position(geom),
                            _sensor_positions(geom),
                            geom, damping=damping)
    # H shape: (n_freq, 9, 1)
    F = np.fft.rfft(chirp.astype(np.float64))         # (n_freq,)
    Y = H[:, :, 0] * F[:, None]                        # (n_freq, 9)
    y = np.fft.irfft(Y, n=len(chirp), axis=0)          # (N_T, 9)
    return y.astype(np.float32)


# ── HDF5 chunk writer ───────────────────────────────────────────────────────
class ChunkWriter:
    """Write samples into successive HDF5 chunk files, capping each chunk's
    on-disk size near ``MAX_CHUNK_MB``.
    """

    SCHEMA_VERSION = 1

    def __init__(self, out_dir: Path, time_axis: np.ndarray,
                  excitation: np.ndarray, freq_array: np.ndarray,
                  max_mb: float = MAX_CHUNK_MB):
        self.out_dir   = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.time_axis = time_axis
        self.excitation = excitation
        self.freq_array = freq_array
        # Bytes per sample (approx; compression typically reduces this).
        per_sample = (N_T * N_CHANNELS * 4) + 64  # signals + labels overhead
        self.max_samples = max(50,
                                int((max_mb * 1024 * 1024) / per_sample))
        self.chunk_idx  = 0
        self.buf_signals: list[np.ndarray] = []
        self.buf_params:  list[SampleParams] = []
        self.total = 0

    def add(self, signals: np.ndarray, p: SampleParams) -> None:
        self.buf_signals.append(signals)
        self.buf_params.append(p)
        if len(self.buf_signals) >= self.max_samples:
            self.flush()

    def flush(self) -> None:
        if not self.buf_signals:
            return
        path = self.out_dir / f"chunk_{self.chunk_idx:04d}.h5"
        n = len(self.buf_signals)
        signals = np.stack(self.buf_signals, axis=0)   # (n, N_T, 9)
        with h5py.File(path, "w") as f:
            f.create_dataset(
                "signals", data=signals,
                compression="gzip", compression_opts=4, shuffle=True,
                chunks=(1, N_T, N_CHANNELS),
            )
            f.create_dataset("time",       data=self.time_axis)
            f.create_dataset("excitation", data=self.excitation)
            f.create_dataset("freqs",      data=self.freq_array.astype(np.float32))

            grp = f.create_group("labels")
            grp.create_dataset("sample_id",  data=np.array(
                [p.sample_id for p in self.buf_params], dtype=np.int32))
            grp.create_dataset("type_code",  data=np.array(
                [p.type_code for p in self.buf_params], dtype=np.int8))
            grp.create_dataset("storey",     data=np.array(
                [p.storey for p in self.buf_params], dtype=np.int8))
            grp.create_dataset("end",        data=np.array(
                [p.end for p in self.buf_params], dtype=np.int8))
            grp.create_dataset("severity",   data=np.array(
                [p.severity for p in self.buf_params], dtype=np.float32))

            pgrp = f.create_group("params")
            for key in (
                "young_factor", "density_factor", "jsr_factor",
                "damping_factor", "plate_lx_factor", "plate_ly_factor",
                "plate_lz_factor", "col_lx_factor", "col_ly_factor",
                "base_extra_mass_dkg", "plate_extra_mass_dkg_fl1",
                "plate_extra_mass_dkg_fl2", "plate_extra_mass_dkg_fl3",
            ):
                pgrp.create_dataset(key, data=np.array(
                    [getattr(p, key) for p in self.buf_params],
                    dtype=np.float32))

            f.attrs["units_signals"]   = "(m/s^2)/N * N = m/s^2"
            f.attrs["units_time"]      = "second"
            f.attrs["units_excitation"]= "newton"
            f.attrs["fs_hz"]           = FS
            f.attrs["n_t"]             = N_T
            f.attrs["sensors"]         = "S2,S5,S6,S7,S8,S11,S12,S13,S14"
            f.attrs["chirp_f_lo_hz"]   = CHIRP_F_LO
            f.attrs["chirp_f_hi_hz"]   = CHIRP_F_HI
            f.attrs["schema_version"]  = self.SCHEMA_VERSION

        self.total += n
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"   wrote {path.name}  ({n} samples, {size_mb:.2f} MB,  "
              f"running total: {self.total})")
        self.buf_signals.clear()
        self.buf_params.clear()
        self.chunk_idx += 1


# ── Manifest writer ─────────────────────────────────────────────────────────
def write_manifest(out_dir: Path, n_total: int,
                    per_type: int, seed: int) -> None:
    manifest = {
        "n_samples":       n_total,
        "per_type":        per_type,
        "fs_hz":           FS,
        "n_t":             N_T,
        "n_channels":      N_CHANNELS,
        "n_inputs":        N_INPUTS,
        "chirp_f_lo_hz":   CHIRP_F_LO,
        "chirp_f_hi_hz":   CHIRP_F_HI,
        "rng_seed":        seed,
        "type_codes":      TYPE_NAMES,
        "severity_units":  SEVERITY_UNITS,
        "schema": {
            "signals":    "(n, n_t, n_channels) float32  acceleration m/s^2",
            "time":       "(n_t,)  float32  seconds",
            "excitation": "(n_t,)  float32  newton (shared chirp)",
            "freqs":      "(n_t//2+1,) float32  Hz, rfft bins",
            "labels/sample_id": "(n,) int32",
            "labels/type_code": "(n,) int8  0=Pristine, 1=Bolt, 2=Crack, 3=Hole, 4=Mass",
            "labels/storey":    "(n,) int8  -1=N/A else 0..2",
            "labels/end":       "(n,) int8  -1=N/A | 0=BD,1=AD (bolt/crack/hole) | 0..3 plate (mass)",
            "labels/severity":  "(n,) float32  units depend on type",
            "params/*":         "per-sample physical jitter factors",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


# ── Main driver ─────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=_REPO / "dataset")
    parser.add_argument("--per-type", type=int, default=DEFAULT_PER_TYPE)
    parser.add_argument("--seed",     type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-mb",   type=float, default=MAX_CHUNK_MB)
    parser.add_argument("--limit",    type=int, default=None,
                          help="If set, only generate the first N samples (smoke test).")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    time_axis, chirp = make_chirp()
    freq_array       = fft_freqs()

    schedule = expand_samples(args.per_type)
    # Shuffle the schedule so each chunk contains a class-balanced mix.
    order = np.arange(len(schedule))
    rng_shuf = np.random.default_rng(args.seed + 1)
    rng_shuf.shuffle(order)
    schedule = [schedule[i] for i in order]
    if args.limit:
        schedule = schedule[: args.limit]

    writer = ChunkWriter(args.out, time_axis, chirp, freq_array,
                          max_mb=args.max_mb)

    n_total = len(schedule)
    print(f"Generating {n_total} samples → {args.out}")
    print(f"  fs={FS} Hz, N_t={N_T}, T={T_DURATION:.2f} s, "
          f"chirp {CHIRP_F_LO}–{CHIRP_F_HI} Hz, chunk max {args.max_mb} MB")

    report_every = max(50, n_total // 50)
    for i, (sid, loc) in enumerate(schedule):
        params = sample_params(sid, loc, rng)
        y      = simulate_sample(params, chirp, freq_array)
        writer.add(y, params)
        if (i + 1) % report_every == 0:
            print(f"   .. {i+1}/{n_total}")
    writer.flush()

    write_manifest(args.out, writer.total, args.per_type, args.seed)
    print(f"\nDone. {writer.total} samples in {writer.chunk_idx} chunks → {args.out}")


if __name__ == "__main__":
    main()
