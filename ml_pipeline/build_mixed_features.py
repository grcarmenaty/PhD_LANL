"""Build a HDF5 Virtual Dataset that concatenates the six features files
(clean + 35/25/20/15/10 dB noisy) into one logical view.

The virtual file is a thin set of pointers — disk usage is ~KB regardless
of how big the sources are.  Downstream training scripts open it like a
normal HDF5 file via ``h5py.File(...)`` and the reads transparently slab
into the appropriate source.

Usage:
    python ml_pipeline/build_mixed_features.py \\
        --sources dataset/features.h5 \\
                  dataset/features_noisy_35dB.h5 \\
                  dataset/features_noisy_25dB.h5 \\
                  dataset/features_noisy_20dB.h5 \\
                  dataset/features_noisy_15dB.h5 \\
                  dataset/features_noisy_10dB.h5 \\
        --out dataset/features_mixed.h5

Datasets virtualised (per-sample, n grows with each source):
    cfdac_real, cfdac_imag, cfdac_mag, cfdac_phase
    frf_real, frf_imag, frf_mag
    timeseries, modal, indicators
    labels/{sample_id, storey, end, severity, type_code}

Datasets copied (not virtualised, small):
    freqs (per-frequency-bin vector, same in every source)
    reference/* (single-row references, taken from first source)
Attributes (cfdac_n, f_lo_hz, f_hi_hz, n_channels, indicator_names) copied
from the first source.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


PER_SAMPLE_DATASETS = (
    "cfdac_real",
    "cfdac_imag",
    "cfdac_mag",
    "cfdac_phase",
    "frf_real",
    "frf_imag",
    "frf_mag",
    "timeseries",
    "modal",
    "indicators",
)
PER_SAMPLE_LABELS = ("sample_id", "storey", "end", "severity", "type_code")


def _check_sources(sources: list[Path]) -> list[int]:
    counts: list[int] = []
    shapes: dict[str, tuple] = {}
    for s in sources:
        with h5py.File(s, "r") as f:
            n = f["timeseries"].shape[0]
            counts.append(n)
            for k in PER_SAMPLE_DATASETS:
                if k not in f:
                    raise SystemExit(f"{s}: missing dataset {k!r}")
                shape_after_first = f[k].shape[1:]
                if k in shapes and shapes[k] != shape_after_first:
                    raise SystemExit(
                        f"{s}: {k} shape {f[k].shape} differs from "
                        f"first source {shapes[k]}")
                shapes[k] = shape_after_first
            for k in PER_SAMPLE_LABELS:
                if f"labels/{k}" not in f:
                    raise SystemExit(f"{s}: missing labels/{k}")
            print(f"  ok {s.name}: n={n}", flush=True)
    return counts


def build_mixed(sources: list[Path], out_path: Path) -> None:
    print(f"checking {len(sources)} source files...", flush=True)
    counts = _check_sources(sources)
    total = sum(counts)
    print(f"total samples: {total}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    with h5py.File(sources[0], "r") as f0:
        attrs = dict(f0.attrs)
        freqs = f0["freqs"][:]
        ref_keys = list(f0["reference"].keys())
        ref_data = {k: f0["reference"][k][:] for k in ref_keys}
        per_sample_meta = {
            k: (f0[k].shape[1:], f0[k].dtype) for k in PER_SAMPLE_DATASETS
        }
        per_label_meta = {
            k: f0[f"labels/{k}"].dtype for k in PER_SAMPLE_LABELS
        }

    with h5py.File(out_path, "w") as out:
        for k, v in attrs.items():
            out.attrs[k] = v
        out.attrs["mixed_sources"] = ";".join(str(s) for s in sources)
        out.attrs["mixed_counts"] = ",".join(str(c) for c in counts)

        out.create_dataset("freqs", data=freqs)

        ref_grp = out.create_group("reference")
        for k, v in ref_data.items():
            ref_grp.create_dataset(k, data=v)

        labels_grp = out.create_group("labels")
        for k in PER_SAMPLE_LABELS:
            dtype = per_label_meta[k]
            layout = h5py.VirtualLayout(shape=(total,), dtype=dtype)
            offset = 0
            for s, n in zip(sources, counts):
                vsource = h5py.VirtualSource(
                    str(s), f"labels/{k}", shape=(n,), dtype=dtype
                )
                layout[offset:offset + n] = vsource
                offset += n
            labels_grp.create_virtual_dataset(k, layout)
            print(f"  labels/{k}: virtualised over {len(sources)} sources",
                      flush=True)

        for k in PER_SAMPLE_DATASETS:
            sample_shape, dtype = per_sample_meta[k]
            full_shape = (total,) + sample_shape
            layout = h5py.VirtualLayout(shape=full_shape, dtype=dtype)
            offset = 0
            for s, n in zip(sources, counts):
                src_shape = (n,) + sample_shape
                vsource = h5py.VirtualSource(
                    str(s), k, shape=src_shape, dtype=dtype
                )
                layout[offset:offset + n] = vsource
                offset += n
            out.create_virtual_dataset(k, layout)
            print(f"  {k}: virtualised, shape={full_shape}", flush=True)

    print(f"\nwrote {out_path}  ({out_path.stat().st_size} bytes)",
              flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sources", type=Path, nargs="+", required=True,
                      help="Six features files in concatenation order: "
                              "clean first, then descending SNR.")
    p.add_argument("--out", type=Path,
                      default=Path(__file__).resolve().parent.parent
                                  / "dataset" / "features_mixed.h5")
    args = p.parse_args()
    build_mixed(args.sources, args.out)


if __name__ == "__main__":
    main()
