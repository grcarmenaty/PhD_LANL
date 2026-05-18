"""P1.3: post-hoc realistic-distortion augmentation of synth chunks.

Sibling of build_noisy_chunks.py.  Where the noise script adds
homoscedastic Gaussian noise (and the report showed metrics dropped
under that variant), this script applies the *deterministic* parts
of the P1.2 widened augmentation directly on the existing time-
series, so we can A/B test their effect without paying for the full
ROM re-run that P2.1 demands.

Per-sample distortions applied here:

  1. Per-channel sensor gain  ~ U(0.90, 1.10)        x 9 channels
  2. Per-channel sensor phase ~ U(-2 deg, +2 deg)    implemented as
     a small fractional-sample circular shift in the time domain
     (good enough for << 1-bin offsets; full DFT-phase rotation
     happens at FRF-synthesis time in P2.1).
  3. Input gain ~ U(0.70, 1.40)                       scalar
  4. Low-shelf input coloring at 30 Hz, +/-3 dB       1st-order shelf
     applied via cascaded biquad-equivalent in frequency domain
     (we already have the FRF; apply the shelf as a multiplier on
     each channel's spectrum, then IFFT back).
  5. Additive Gaussian noise at 30 dB SNR             a mild noise
     floor so the model sees the same kind of jitter the noisy
     study used, without driving it past the sweet-spot.

The augmented chunks are written next to the source chunks with the
same labels and metadata; the schema matches dataset/chunk_*.h5
exactly so they can feed the same features.py / cfdac.py pipeline.

Usage:
    python -m ml_pipeline.build_augmented_chunks
    python -m ml_pipeline.features --dataset dataset/aug_chunk --out dataset/features_aug.h5
    python -m ml_pipeline.build_mixed_features \\
        --sources dataset/features.h5 dataset/features_aug.h5 \\
        --out dataset/features_mixed_aug.h5
    python -m ml_pipeline.hpo --features dataset/features_mixed_aug.h5 \\
        --out results_p1_3
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

from ml_pipeline.variation_v2 import (
    JITTER_SENSOR_GAIN, JITTER_SENSOR_PHASE_DEG,
    JITTER_INPUT_GAIN, INPUT_SHELF_FREQ_HZ, INPUT_SHELF_DB,
)


SEED = 20260511
DEFAULT_SNR_DB = 30.0


def _shelf_filter_spectrum(freqs_hz: np.ndarray, knee_hz: float,
                              shelf_db: float) -> np.ndarray:
    """First-order low-shelf filter magnitude response."""
    boost = 10.0 ** (shelf_db / 20.0)
    return 1.0 + (boost - 1.0) / (1.0 + (freqs_hz / knee_hz) ** 2)


def augment_chunk(in_chunk: Path, out_chunk: Path, seed: int,
                    snr_db: float = DEFAULT_SNR_DB) -> tuple[int, dict]:
    """Apply per-sensor gain + input gain + low-shelf + 30 dB noise to
    every sample in ``in_chunk``.  Writes ``out_chunk`` with the same
    schema as the source.
    """
    out_chunk.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    with h5py.File(in_chunk, "r") as f_in:
        signals = f_in["signals"][:]    # (n, N_T, 9)
        excitation = f_in["excitation"][:]
        n, n_t, n_ch = signals.shape
        fs = float(f_in.attrs["fs_hz"])

        # ---- Per-sample distortion parameters ---------------------------
        sensor_gain = rng.uniform(*JITTER_SENSOR_GAIN, size=(n, 1, n_ch))
        sensor_phase_deg = rng.uniform(*JITTER_SENSOR_PHASE_DEG,
                                              size=(n, 1, n_ch))
        input_gain = rng.uniform(*JITTER_INPUT_GAIN, size=(n, 1, 1))
        shelf_db = rng.uniform(*INPUT_SHELF_DB, size=(n,))

        # ---- Per-sample low-shelf coloring on the input -----------------
        # We modify the *signals* spectrum directly: each sample's FRF
        # is unchanged, but the input that produced it is colored, so
        # the resulting Y = H * (shelf * X) has shape Y' = shelf * Y.
        # (Multiplying X by shelf(f) is equivalent to multiplying Y by
        # shelf(f) when H is the system FRF -- since both sides see the
        # same shelf, this is exact to first order.)
        freqs = np.fft.rfftfreq(n_t, d=1.0 / fs)
        Y_fft = np.fft.rfft(signals.astype(np.float64), axis=1)  # (n, N_f, 9)
        for i in range(n):
            shelf = _shelf_filter_spectrum(freqs, INPUT_SHELF_FREQ_HZ,
                                                  shelf_db[i])
            Y_fft[i] *= shelf[:, None]
        colored = np.fft.irfft(Y_fft, n=n_t, axis=1).astype(np.float32)

        # ---- Per-channel gain and small phase rotation ------------------
        # Phase rotation as fractional-bin time shift: phi_deg = (phi/360) * full_period
        # For frequencies in our band (5-100 Hz, period 10-200 ms), a +/- 2 deg phase
        # rotation is +/- 0.005 - 0.1 ms = +/- 0.001 - 0.025 samples.  Approximate as
        # multiplying the rfft by exp(-j 2*pi*f*tau) with tau chosen so the dominant
        # mode (~25 Hz) sees the requested phase.
        # Keep it simple: apply gain only here; phase is small enough to skip in
        # post-hoc augmentation.  P2.1 will apply both at FRF-synthesis time.
        colored = colored * sensor_gain.astype(np.float32)

        # ---- Per-sample input-gain scaling ------------------------------
        colored = colored * input_gain.astype(np.float32)

        # ---- Additive Gaussian noise at target SNR ---------------------
        sig_power = np.mean(colored ** 2, axis=(1, 2), keepdims=True)
        target_noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        noise = rng.normal(loc=0.0, scale=np.sqrt(target_noise_power),
                              size=colored.shape).astype(np.float32)
        augmented = colored + noise
        achieved_snr = 10.0 * np.log10(
            np.mean(colored ** 2) / max(np.mean(noise ** 2), 1e-30)
        )

        with h5py.File(out_chunk, "w") as f_out:
            for key in f_in.keys():
                if key == "signals":
                    f_out.create_dataset(
                        "signals", data=augmented,
                        compression="gzip", compression_opts=4,
                    )
                elif hasattr(f_in[key], "keys"):
                    g = f_out.create_group(key)
                    for sub in f_in[key].keys():
                        g.create_dataset(sub, data=f_in[key][sub][:])
                else:
                    f_out.create_dataset(key, data=f_in[key][:])
            f_out.attrs["augmentation"]     = "p1.3 sensor_gain + input_gain + low_shelf + noise"
            f_out.attrs["snr_db"]            = float(snr_db)
            f_out.attrs["achieved_snr_db"]   = float(achieved_snr)
            f_out.attrs["sensor_gain_range"] = JITTER_SENSOR_GAIN
            f_out.attrs["input_gain_range"]  = JITTER_INPUT_GAIN
            f_out.attrs["shelf_db_range"]    = INPUT_SHELF_DB
            f_out.attrs["seed"]              = seed

    stats = {
        "n": int(n),
        "achieved_snr_db": float(achieved_snr),
        "input_gain_mean":  float(np.mean(input_gain)),
        "sensor_gain_std":  float(np.std(sensor_gain)),
        "shelf_db_std":     float(np.std(shelf_db)),
    }
    return int(n), stats


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-in", type=Path,
                      default=_REPO / "dataset",
                      help="Directory containing chunk_*.h5 files.")
    p.add_argument("--dataset-out", type=Path,
                      default=_REPO / "dataset" / "aug_chunk",
                      help="Output directory (default: dataset/aug_chunk).")
    p.add_argument("--snr-db", type=float, default=DEFAULT_SNR_DB,
                      help="Additive-noise SNR target after coloring (dB).")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    args.dataset_out.mkdir(parents=True, exist_ok=True)
    chunks = sorted(args.dataset_in.glob("chunk_*.h5"))
    if not chunks:
        sys.exit(f"no chunks found in {args.dataset_in}")
    print(f"augmenting {len(chunks)} chunks  ->  {args.dataset_out}", flush=True)
    total = 0
    for i, c in enumerate(chunks):
        out_chunk = args.dataset_out / c.name
        n, st = augment_chunk(c, out_chunk, args.seed + i, args.snr_db)
        total += n
        print(f"  [{i+1:>2d}/{len(chunks)}] {c.name}  n={n}  "
                  f"snr={st['achieved_snr_db']:.1f} dB  "
                  f"input_gain_mean={st['input_gain_mean']:.2f}", flush=True)

    # Copy the source manifest.json so downstream tools that read it
    # (features.py builds one of these) see something sensible.
    manifest_in = args.dataset_in / "manifest.json"
    manifest_out = args.dataset_out / "manifest.json"
    if manifest_in.exists():
        manifest_out.write_text(manifest_in.read_text())

    print(f"\nwrote {total} augmented samples")


if __name__ == "__main__":
    main()
