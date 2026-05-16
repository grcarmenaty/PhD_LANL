"""Add Gaussian noise to the synthetic time series and re-derive every
downstream feature.

Pipeline mirrors ``features.py``:

  1.  For every chunk in ``dataset/chunk_*.h5`` load the
      ``(N_samples, 1024, 9)`` time series.
  2.  Compute the per-sample signal power and scale Gaussian noise to
      hit the target SNR (in dB) — `noise_std = sqrt(P_signal / 10**(SNR/10))`.
  3.  Add the noise to the timeseries, write the noisy chunk to
      ``dataset/chunk_<NN>_noisy_<SNR>dB.h5``.

The downstream feature-extraction script (``features.py``) is then
re-run pointing at the noisy chunk directory, producing a parallel
``features_noisy_<SNR>dB.h5``.  The CFDAC variants and indicator
regression artefacts are re-extracted from that file by the
existing scripts.

Noise is **additive Gaussian, per-sample SNR-controlled, applied
independently per channel**.  The 9 sensors see independent noise so
the FRF estimator cannot average it away across sensors.

Usage:
    python ml_pipeline/build_noisy_chunks.py --snr-db 20
    python ml_pipeline/features.py --dataset dataset/noisy_20dB \\
        --out dataset/features_noisy_20dB.h5
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

_REPO = Path(__file__).resolve().parent.parent

SEED = 20260511


def add_noise_to_chunk(in_chunk: Path, out_chunk: Path, snr_db: float,
                        seed: int) -> tuple[int, float]:
    """Write a noisy copy of ``in_chunk`` to ``out_chunk``.  Returns
    `(n_samples, achieved_snr_db)`."""
    out_chunk.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    with h5py.File(in_chunk, "r") as f_in:
        signals = f_in["signals"][:]    # (n, 1024, 9)
        # Power per sample, averaged over time × channels.
        sig_power = np.mean(signals ** 2, axis=(1, 2), keepdims=True)
        target_noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        noise = rng.normal(
            loc=0.0,
            scale=np.sqrt(target_noise_power),
            size=signals.shape,
        ).astype(signals.dtype)
        noisy = signals + noise
        achieved = 10.0 * np.log10(
            np.mean(signals ** 2) / np.mean(noise ** 2)
        )
        with h5py.File(out_chunk, "w") as f_out:
            for key in f_in.keys():
                if key == "signals":
                    f_out.create_dataset(
                        "signals", data=noisy,
                        compression="gzip", compression_opts=4,
                    )
                elif hasattr(f_in[key], "keys"):
                    # group
                    g = f_out.create_group(key)
                    for sub in f_in[key].keys():
                        g.create_dataset(sub, data=f_in[key][sub][:])
                else:
                    f_out.create_dataset(key, data=f_in[key][:])
            f_out.attrs["snr_db"]            = snr_db
            f_out.attrs["achieved_snr_db"]   = float(achieved)
            f_out.attrs["seed"]              = seed
        return len(signals), float(achieved)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--snr-db", type=float, required=True,
                      help="Target signal-to-noise ratio in dB.")
    p.add_argument("--dataset-in", type=Path,
                      default=_REPO / "dataset",
                      help="Directory containing chunk_*.h5 files.")
    p.add_argument("--dataset-out", type=Path, default=None,
                      help="Output directory (default: "
                              "dataset/noisy_<SNR>dB).")
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    out_dir = args.dataset_out
    if out_dir is None:
        out_dir = args.dataset_in / f"noisy_{int(round(args.snr_db))}dB"
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks = sorted(args.dataset_in.glob("chunk_*.h5"))
    if not chunks:
        sys.exit(f"no chunks found in {args.dataset_in}")
    print(f"adding noise at {args.snr_db:.1f} dB to {len(chunks)} chunks "
              f"→ {out_dir}", flush=True)
    total = 0; achieved = []
    for i, c in enumerate(chunks):
        out_chunk = out_dir / c.name
        n, ach = add_noise_to_chunk(c, out_chunk, args.snr_db,
                                          args.seed + i)
        total += n
        achieved.append(ach)
        print(f"  [{i+1:>2d}/{len(chunks)}] {c.name}  n={n}  "
                  f"achieved SNR={ach:.2f} dB", flush=True)

    print(f"\nwrote {total} noisy samples; mean achieved SNR = "
              f"{np.mean(achieved):.2f} dB")


if __name__ == "__main__":
    main()
