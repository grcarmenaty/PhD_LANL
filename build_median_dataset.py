"""build_median_dataset.py — collapse the 2 638-measurement IQS dataset to
per-experiment medians.

The original ``experimental_frfs.h5`` carries every individual H1 measurement
(shape ``(2638, 1601, 9)`` complex64).  For everything the calibration and
correlation review need, only one summary FRF per experimental case is
required.  This script produces ``experimental_frfs_median.h5`` with shape
``(n_cases, 1601, 9)`` — typically ~12 MB — small enough to commit directly
to the repo so the Colab notebook needs no external downloads.

Median definition
-----------------
For each case we save **two** summaries because both are needed downstream:

* ``mag_median``      — ``median(|H|)``                  (real, float32)
* ``complex_median``  — ``median(real(H)) + 1j·median(imag(H))``  (complex64)

The complex elementwise median is the natural extension of the scalar median
to complex numbers and preserves the relative phase between sensors that the
mode-shape and CFDAC analysis depends on.

Usage
-----
    python build_median_dataset.py SOURCE.h5 DEST.h5

If ``SOURCE.h5`` is omitted, the script tries the public Drive link.  In
Colab, ``gdown`` is preinstalled and the download takes ~30 s.
"""
from __future__ import annotations
import sys
import json
import time
from pathlib import Path

import numpy as np
import h5py

PUBLIC_DRIVE_ID = "158Ruje1CnU0zHWD2_rrIxvO0gEztNpHp"


def _download_via_gdown(target: Path) -> None:
    import subprocess
    print(f"Downloading from Drive id={PUBLIC_DRIVE_ID} → {target}")
    subprocess.run(
        ["gdown", PUBLIC_DRIVE_ID, "-O", str(target)],
        check=True,
    )


def build(src_h5: Path, dst_h5: Path) -> None:
    print(f"Reading {src_h5} ({src_h5.stat().st_size/1e6:.1f} MB) …")
    t0 = time.time()
    with h5py.File(src_h5, "r") as f:
        freq    = np.array(f["freq"])
        data    = np.array(f["data"])
        labels  = np.array(f["labels"])
        cases   = [c.decode() if isinstance(c, bytes) else str(c)
                   for c in np.array(f["cases"])]
        lmap_js = f.attrs.get("label_map_json", "{}")
    print(f"  loaded in {time.time()-t0:.1f}s   raw shape={data.shape}")

    label2case = json.loads(lmap_js) if isinstance(lmap_js, str) else {}
    # Build label → case_name lookup that survives even without label_map_json
    if not label2case:
        label2case = {int(lab): nm for lab, nm in zip(labels, cases)}
    else:
        label2case = {int(v): k for k, v in label2case.items()}

    unique_labels = np.unique(labels)
    n_cases = len(unique_labels)
    n_freq  = data.shape[1]
    n_chan  = data.shape[2]
    print(f"  {n_cases} unique cases, {n_freq} freq pts, {n_chan} channels")

    mag_median  = np.zeros((n_cases, n_freq, n_chan), dtype=np.float32)
    cplx_median = np.zeros((n_cases, n_freq, n_chan), dtype=np.complex64)
    case_names  = []
    counts      = np.zeros(n_cases, dtype=np.int32)

    for i, lab in enumerate(unique_labels):
        mask  = labels == lab
        H     = data[mask]                                  # (n_meas, Nf, 9) complex
        counts[i]      = int(mask.sum())
        case_names.append(label2case.get(int(lab), f"label_{int(lab)}"))
        mag_median[i]  = np.median(np.abs(H), axis=0).astype(np.float32)
        cplx_median[i] = (np.median(H.real, axis=0)
                           + 1j * np.median(H.imag, axis=0)).astype(np.complex64)

    print(f"\nWriting {dst_h5} …")
    case_names_b = np.array(case_names, dtype=h5py.string_dtype("utf-8"))
    with h5py.File(dst_h5, "w") as f:
        f.create_dataset("freq",           data=freq.astype(np.float32))
        f.create_dataset("mag_median",     data=mag_median,  compression="gzip", compression_opts=4)
        f.create_dataset("complex_median", data=cplx_median, compression="gzip", compression_opts=4)
        f.create_dataset("case_names",     data=case_names_b)
        f.create_dataset("case_counts",    data=counts)
        f.attrs["label_map_json"] = json.dumps(
            {nm: int(lab) for lab, nm in zip(unique_labels, case_names)})
        f.attrs["source_n_meas"]  = int(data.shape[0])
    print(f"  wrote {dst_h5} ({dst_h5.stat().st_size/1e6:.2f} MB)")
    print(f"\nCases by measurement count (top 5):")
    for nm, ct in sorted(zip(case_names, counts), key=lambda x: -x[1])[:5]:
        print(f"  {ct:4d}  {nm}")


def main():
    here = Path(__file__).parent.resolve()
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "experimental_frfs.h5"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else here / "experimental_frfs_median.h5"

    if not src.exists():
        print(f"{src} not found — fetching from public Drive link …")
        _download_via_gdown(src)

    build(src, dst)


if __name__ == "__main__":
    main()
