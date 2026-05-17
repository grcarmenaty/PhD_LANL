"""Build pymodal-compatible spectra from the HBTA experimental HDF5.

Windows the 100 Hz acceleration data into 1024-sample chunks; for each
window computes the rfft output spectrum and the H1 input-output FRF
(using the on-shaker AS reference sensor); writes a pymodal frf
collection plus chunk-format HDF5 files with damage-state labels.

Source HDF5 schema:
  50 top-level records named MVS_<P1|P2>_<UDS|DS1..8>_<SM|NM>_<Y|Z>_NN.
  Each record holds a Group `acceleration/<sensor>/<axis>` of float64
  time-series at 100 Hz, ~622 s long (62 147 samples). Sensors: 18
  arch (AG01-18), 40 deck (AL01-40), 1 shaker reference (AS), plus
  strain gauges (SB/SC).

Damage classes encoded as integers: 0 = UDS (undamaged), 1..8 = DS1..DS8.

Usage:
  python build_hbta_pymodal.py             # full build (~3 min)
  python build_hbta_pymodal.py --smoke 6   # 6 records, smoke test
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np

# pymodal-compatible writer; falls back to plain HDF5 if pymodal unavailable
try:
    from pymodal import frf as PMF
    HAVE_PYMODAL = True
except Exception:
    HAVE_PYMODAL = False


# ── Geometry constants from the dataset's sensor-layout PDF ──────────
BRIDGE_LENGTH = 35.0       # m (X axis, midspan at X=0)
BRIDGE_WIDTH  = 4.5        # m (Y axis)

# 18 arch accelerometers, 40 deck accelerometers, 15 strain gauges,
# plus AS on the shaker. The dataset stores y and z component datasets
# (depending on the test, only one may be populated).
ARCH_SENSORS = [f"AG{i:02d}" for i in range(1, 19)]
DECK_SENSORS = [f"AL{i:02d}" for i in range(1, 41)]
STRAIN_BOTTOM = [f"SB{i:02d}" for i in range(1, 9)]
STRAIN_CROSS  = [f"SC{i:02d}" for i in range(1, 8)]
SHAKER_SENSOR = "AS"

# Channel selection: 10 arch accelerometers (5 north + 5 south, picking
# alternating joints along the span) + 2 representative deck stringers.
# Edit this list to expand to more sensors if needed.
SENSOR_SELECTION = [
    "AG02", "AG04", "AG05", "AG06", "AG08",  # north arch (top girder joints)
    "AG11", "AG13", "AG14", "AG15", "AG17",  # south arch (top girder joints)
    "AL10", "AL26",                          # deck stringers, two representative
]
N_CH = len(SENSOR_SELECTION)

# Class label encoding: 0 = UDS, 1..8 = DS1..DS8
def _class_code(damage_state: str) -> int:
    if damage_state == "UDS":
        return 0
    if damage_state.startswith("DS"):
        return int(damage_state[2:])
    raise ValueError(f"unknown damage_state {damage_state}")


# ── Data shape parameters ────────────────────────────────────────────
FS         = 100.0         # Hz, from dataset attr "sample_rate"
N_T        = 1024          # window length (samples) -> 10.24 s window
N_F        = N_T // 2 + 1  # 513 rfft bins
DF         = FS / N_T      # 0.09766 Hz frequency resolution
F_MAX      = FS / 2.0      # 50 Hz Nyquist (paper notes 100 Hz is resampled from 400 Hz)

# Window targets — balanced across the 9 damage classes
TARGETS_TOTAL = 9000        # total windows
TARGETS_PER_CLASS = TARGETS_TOTAL // 9     # 9 classes, 1000 each


# ── Logging setup ────────────────────────────────────────────────────
def setup_logging(name: str, log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                              datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_dir / f"{name}.log", mode="w")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    return log


# ── Sensor channel reader ────────────────────────────────────────────
def read_sensor_axes(group: h5py.Group, sensor_name: str) -> Dict[str, np.ndarray]:
    """Return a dict {axis: signal} for the requested sensor under
    group['acceleration']. Some sensors only carry one axis (e.g. AS
    when shaker drove laterally only)."""
    grp = group["acceleration"][sensor_name]
    return {ax: np.asarray(grp[ax]) for ax in grp.keys()
            if hasattr(grp[ax], "shape")}


def pick_sensor_channel(grp_acc: h5py.Group, sensor_name: str,
                        mvs_direction_label: str) -> np.ndarray:
    """Pick a single time-series for the named sensor that matches the
    shaker excitation direction. Falls back to whichever axis exists.

    `mvs_direction_label` is the record attr e.g.
    "Horizontal lateral (global CSYS Y-direction)" or "Vertical (Z)".
    """
    axes = read_sensor_axes(grp_acc.parent, sensor_name)
    if not axes:
        raise SystemExit(f"sensor {sensor_name} has no time-series axes")
    if "y" in mvs_direction_label.lower() and "y" in axes:
        return axes["y"]
    if "z" in mvs_direction_label.lower() and "z" in axes:
        return axes["z"]
    # Fall back to a deterministic axis order
    for ax in ("z", "y", "x"):
        if ax in axes:
            return axes[ax]
    raise SystemExit(f"sensor {sensor_name}: no usable axis in {list(axes)}")


# ── Per-record processing ────────────────────────────────────────────
def process_record(h5: h5py.File, record_name: str,
                   sensors: List[str], shaker: str,
                   log: logging.Logger) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Window one HBTA record and return (signals, output_specs, frfs, meta).

    signals  : (n_w, N_T, N_CH) float32  — time-domain acceleration windows
    output_specs : (n_w, N_F, N_CH) complex64 — rfft of windowed output
    frfs     : (n_w, N_F, N_CH) complex64 — H1 estimator output / input
                   (set to NaN if shaker channel not available)
    meta     : per-record metadata dict
    """
    g = h5[record_name]
    attrs = dict(g.attrs)
    direction = str(attrs.get("mvs_direction", ""))

    # Pull each output sensor + shaker as 1D arrays in matching direction
    series = []
    for s in sensors:
        series.append(pick_sensor_channel(g["acceleration"], s, direction))
    output_data = np.stack(series, axis=1)             # (n_samples, N_CH)

    # Shaker (AS) reference acceleration — measured input
    try:
        shaker_series = pick_sensor_channel(g["acceleration"], shaker, direction)
        have_input = True
    except SystemExit:
        shaker_series = None
        have_input = False

    n_samples = output_data.shape[0]
    n_windows = n_samples // N_T
    if n_windows == 0:
        log.warning(f"  {record_name}: not enough samples ({n_samples}) for one window")
        return (np.zeros((0, N_T, N_CH), np.float32),
                np.zeros((0, N_F, N_CH), np.complex64),
                np.zeros((0, N_F, N_CH), np.complex64),
                attrs)

    # Reshape into non-overlapping windows
    cut = n_windows * N_T
    win = output_data[:cut].reshape(n_windows, N_T, N_CH).astype(np.float32)
    # Hann window for the FFT to reduce leakage
    hann = np.hanning(N_T).astype(np.float32)[None, :, None]
    spec_out = np.fft.rfft(win * hann, axis=1).astype(np.complex64)  # (n_w, N_F, N_CH)

    if have_input:
        shaker_win = shaker_series[:cut].reshape(n_windows, N_T).astype(np.float32)
        spec_in = np.fft.rfft(shaker_win * hann[..., 0], axis=1)     # (n_w, N_F)
        # H1 = S_xy / S_xx  (Welch single-window approximation)
        Sxx = (spec_in.conj() * spec_in).real          # (n_w, N_F)
        Sxy = (spec_in[..., None].conj() * spec_out)   # (n_w, N_F, N_CH)
        with np.errstate(divide="ignore", invalid="ignore"):
            frf = (Sxy / np.maximum(Sxx, 1e-30)[..., None]).astype(np.complex64)
    else:
        frf = np.full((n_windows, N_F, N_CH), np.nan + 1j*np.nan,
                       dtype=np.complex64)

    log.info(f"  {record_name}: {n_windows} windows, "
             f"have_input={have_input}, direction={direction[:8]}")
    return win, spec_out, frf, attrs


# ── Class scheduling — pick records and window counts per class ──────
def schedule(h5: h5py.File) -> Dict[int, List[str]]:
    by_class: Dict[int, List[str]] = defaultdict(list)
    for name in h5:
        # parse damage_state from the record name to avoid loading the attr
        parts = name.split("_")
        ds = parts[2]
        cls = _class_code(ds)
        by_class[cls].append(name)
    return dict(by_class)


# ── Main writer ──────────────────────────────────────────────────────
def write_chunked_dataset(out_dir: Path,
                           signals: np.ndarray, specs: np.ndarray,
                           frfs: np.ndarray, labels: np.ndarray,
                           records: List[str], log: logging.Logger) -> Path:
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = 256
    n = signals.shape[0]
    manifest = []
    for i, start in enumerate(range(0, n, chunk_size)):
        end = min(start + chunk_size, n)
        p = chunks_dir / f"chunk_{i:04d}.h5"
        with h5py.File(p, "w") as f:
            f.create_dataset("signals", data=signals[start:end],
                              compression="gzip", compression_opts=4)
            f.create_dataset("spec_output", data=specs[start:end],
                              compression="gzip", compression_opts=4)
            f.create_dataset("frf_H1", data=frfs[start:end],
                              compression="gzip", compression_opts=4)
            f.create_dataset("labels/class_code", data=labels[start:end])
            src_arr = np.array([records[k].encode() for k in range(start, end)])
            f.create_dataset("labels/source_record", data=src_arr)
            f.attrs["fs_hz"]  = FS
            f.attrs["n_t"]    = N_T
            f.attrs["sensors"] = np.array(SENSOR_SELECTION, dtype=h5py.string_dtype())
            f.attrs["class_legend"] = "0=UDS  1-8=DS1..DS8"
        log.info(f"  wrote {p.name}  n={end-start}  "
                 f"size={p.stat().st_size/1e6:.1f}MB")
        manifest.append(dict(file=p.name, n=int(end - start)))
    (chunks_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return chunks_dir


def write_pymodal_collection(spec_per_class: Dict[int, np.ndarray],
                              out_path: Path, log: logging.Logger) -> Path:
    """Write a pymodal frf collection containing the band-restricted
    output spectra. Falls back to a plain HDF5 if pymodal is missing."""
    if not HAVE_PYMODAL:
        log.warning("pymodal not available; writing plain HDF5 collection")
        with h5py.File(out_path, "w") as f:
            for cls, arr in spec_per_class.items():
                f.create_dataset(f"class_{cls}/spec_output", data=arr,
                                   compression="gzip", compression_opts=4)
            f.attrs["fs_hz"]  = FS
            f.attrs["n_t"]    = N_T
            f.attrs["class_legend"] = "0=UDS 1-8=DS1..DS8"
        return out_path

    # pymodal flat layout: one measurement per window, labels = class code
    measurements, names, labels = [], [], []
    for cls in sorted(spec_per_class):
        for i, m in enumerate(spec_per_class[cls]):
            measurements.append(m)                          # (N_F, N_CH)
            names.append(f"class{cls:01d}_{i:05d}")
            labels.append(float(cls))
    coll = PMF(
        measurements=measurements,
        freq_resolution=DF,
        measurements_units="meter / second ** 2",
        freq_units="hertz",
        method="SIMO",
        names=names,
        labels=labels,
        path=out_path,
    )
    coll.close(keep=True)
    log.info(f"  wrote pymodal collection {out_path.name}  "
             f"size={out_path.stat().st_size/1e6:.1f}MB")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent.parent
    parser.add_argument("--input", type=Path,
                          default=here / "raw" / "data_100Hz.h5")
    parser.add_argument("--out-dir", type=Path,
                          default=here / "output")
    parser.add_argument("--smoke", type=int, default=None,
                          help="Smoke test: process only N records")
    parser.add_argument("--per-class", type=int, default=TARGETS_PER_CLASS,
                          help="Max windows kept per class")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = here / "logs"
    log = setup_logging("build_hbta", log_dir)
    log.info(f"input  = {args.input}")
    log.info(f"output = {args.out_dir}")
    log.info(f"sensors ({N_CH}): {SENSOR_SELECTION}")
    log.info(f"per-class target = {args.per_class}")

    h5 = h5py.File(args.input, "r")
    records_by_class = schedule(h5)
    log.info("records per class:")
    for c in sorted(records_by_class):
        log.info(f"  class {c}: {len(records_by_class[c])} records")

    all_sig, all_spec, all_frf, all_lab, all_src = [], [], [], [], []
    spec_per_class: Dict[int, List[np.ndarray]] = defaultdict(list)
    n_processed = 0
    t0 = time.time()

    for cls in sorted(records_by_class):
        recs = records_by_class[cls]
        if args.smoke is not None:
            recs = recs[: max(1, args.smoke // len(records_by_class))]
        for r in recs:
            sig, spec, frf, attrs = process_record(
                h5, r, SENSOR_SELECTION, SHAKER_SENSOR, log)
            # Truncate to keep balance
            kept = 0
            for k in range(sig.shape[0]):
                if len(spec_per_class[cls]) >= args.per_class:
                    break
                all_sig.append(sig[k])
                all_spec.append(spec[k])
                all_frf.append(frf[k])
                all_lab.append(cls)
                all_src.append(r)
                spec_per_class[cls].append(spec[k])
                kept += 1
            log.info(f"    kept {kept} windows for class {cls} "
                     f"(running total = {len(spec_per_class[cls])})")
            n_processed += 1
            if args.smoke is not None and n_processed >= args.smoke:
                break
        if args.smoke is not None and n_processed >= args.smoke:
            break

    h5.close()

    if not all_sig:
        log.error("no windows produced")
        return

    log.info(f"\nstacking {len(all_sig)} windows…")
    signals = np.stack(all_sig).astype(np.float32)
    specs   = np.stack(all_spec).astype(np.complex64)
    frfs    = np.stack(all_frf).astype(np.complex64)
    labels  = np.array(all_lab, dtype=np.int32)
    sources = list(all_src)

    log.info(f"signals = {signals.shape}  specs = {specs.shape}  "
             f"frfs = {frfs.shape}  labels = {labels.shape}")

    # ── Outputs ───────────────────────────────────────────────────
    pymodal_path = args.out_dir / "hbta_collection.h5"
    spec_per_class_arr = {c: np.stack(v) for c, v in spec_per_class.items()}
    write_pymodal_collection(spec_per_class_arr, pymodal_path, log)

    write_chunked_dataset(args.out_dir, signals, specs, frfs, labels, sources, log)

    elapsed = time.time() - t0
    log.info(f"\n=== SUMMARY ===")
    log.info(f"  total windows: {labels.size}")
    log.info(f"  class counts : "
             f"{dict(sorted([(int(c), int(np.sum(labels==c))) for c in np.unique(labels)]))}")
    log.info(f"  wall time    : {elapsed:.1f} s")
    log.info(f"  pymodal      : {pymodal_path}")
    log.info(f"  chunks dir   : {args.out_dir / 'chunks'}")


if __name__ == "__main__":
    main()
