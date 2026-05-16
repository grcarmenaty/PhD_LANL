"""Convert the Flossgraben Bridge dataset (https://doi.org/10.25532/OPARA-767)
into a pymodal frf collection.

Source: ~273 acceleration recordings (600 s @ 1 kHz, 56 channels) across three
states — Reference (pristine), Field 4 (39 t mass perturbation in span 4),
Field 3 (mass perturbation in span 3). Output-only (traffic excitation).

Output: 10 000-sample pymodal frf collection, class-balanced across the three
states. Each sample is a 4 s window (n_t = 1024 @ fs = 256 Hz) reduced to 9
sensor channels and FFT-ed to a complex (513, 9, 1) spectrum.

Strategy: stream each remote zip via HTTP Range (no full-zip on disk), fetch
one h5 measurement at a time into RAM, extract channels, resample, segment,
FFT, append to in-memory list, free the h5 bytes. Peak disk usage stays
below 2 GB.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import remotezip
import scipy.signal

import pymodal
from pymodal import frf as PMF


# ── Dataset spec ────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "output"
CATALOG_DIR = HERE.parent / "catalogue"
LOG_DIR = HERE.parent / "logs"

# Acquisition (target — matches the 3SBB pipeline)
FS_TARGET = 256                     # Hz
N_T = 1024                          # samples per window  → 4.0 s
N_F = N_T // 2 + 1                  # 513 bins
DF = FS_TARGET / N_T                # 0.25 Hz

# Original recording is 1 kHz; resample 1000 → 256 via 32/125 poly
RS_UP, RS_DOWN = 32, 125

# Parallel fetch
FETCH_WORKERS_DEFAULT = 6

# 9 channels: 7 reported (one per span) + 2 extras near damage (Field 3 & 4)
# Indices in RecBuff are 1-indexed for accel channels (row 0 is the timestamp).
# Catalogue reports Ch3,11,19,27,35,43,51 (one per span). We add Ch21 and Ch29
# near the damaged spans for better spatial resolution.
SENSOR_CHANNELS = [3, 11, 19, 21, 27, 29, 35, 43, 51]   # 1-indexed
N_CH = len(SENSOR_CHANNELS)

# Class targets — balanced 10 k
TARGETS = {
    "reference": 3334,
    "field3":    3333,
    "field4":    3333,
}

# Class encoding (also stored as labels in pymodal frf)
#   0.0 = Reference (Pristine)
#   1.0 = Field 3 damaged (Mass at span 3)
#   2.0 = Field 4 damaged (Mass at span 4)
CLASS_LABEL = {"reference": 0.0, "field3": 1.0, "field4": 2.0}

# 3SBB-compatible labels (mirrored into HDF5 attached metadata for downstream use)
#   type_code: 0 = Pristine, 4 = Mass
#   end:       -1 = N/A,  encodes plate idx for mass → use span idx 3 or 4
#   storey:    -1 (not meaningful for a girder bridge)
#   severity:  0 for Pristine, 39 000 kg for damaged
TYPE_CODE = {"reference": 0, "field3": 4, "field4": 4}
END_CODE  = {"reference": -1, "field3": 3, "field4": 4}
SEVERITY  = {"reference": 0.0, "field3": 39_000.0, "field4": 39_000.0}

# OPARA download URLs
ZIP_URLS = {
    "reference": "https://opara.zih.tu-dresden.de/bitstreams/251229e6-dd64-47e1-976b-4b1b22aaaef6/download",
    "field4":    "https://opara.zih.tu-dresden.de/bitstreams/966be526-37a6-4f0d-a89a-68a0e0ae5f07/download",
    "field3":    "https://opara.zih.tu-dresden.de/bitstreams/b7229686-68f9-4d02-89dc-07ee0dca79e2/download",
}

CATALOG_FILES = {
    "reference": "Catalogue-Reference.csv",
    "field4":    "Catalogue-Field4.csv",
    "field3":    "Catalogue-Field3.csv",
}


# ── Logging ─────────────────────────────────────────────────────────────────
def setup_logging(name: str) -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("flossgraben")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                             datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); log.addHandler(sh)
    fh = logging.FileHandler(LOG_DIR / f"{name}.log"); fh.setFormatter(fmt); log.addHandler(fh)
    return log


# ── Catalogue → per-measurement environmental metadata ──────────────────────
def load_catalogue(state: str) -> Dict[str, Dict]:
    """Return mapping filename → environmental metadata dict for one state."""
    path = CATALOG_DIR / CATALOG_FILES[state]
    out: Dict[str, Dict] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["Name"]
            # Skip rows whose date is N/A or project differs from Reference/Field
            try:
                out[name] = {
                    "date":          row["Date"],
                    "duration_s":    float(row["Duration"]),
                    "surface_temp_c": float(row["Surface Temperature(°C)"]),
                    "indoor_temp_c": float(row["Indoor Temperature(℃)"] or "nan"),
                    "outdoor_temp_c": float(row["Outdoor Temperature(℃)"] or "nan"),
                    "wind_ms":       float(row["Wind(m/s)"] or "nan"),
                    "wind_dir_deg":  float(row["Wind Direction(°)"] or "nan"),
                    "dM_kg":         float(row["dM"]),
                    "dM_pos":        float(row["dM_Pos"]),
                }
            except Exception:
                continue
    return out


# ── Time-series → spectra ───────────────────────────────────────────────────
def resample_and_window(rec: np.ndarray, windows_per_meas: int,
                         rng: np.random.Generator) -> np.ndarray:
    """rec: (n_orig=600000, n_ch=9) float32 raw acceleration @ 1 kHz.

    Returns (n_windows, N_T, N_CH) float32 acceleration @ 256 Hz, in m/s².
    Windows are non-overlapping (or minimally overlapping if too many requested)
    and selected from random positions inside the resampled signal.
    """
    # Resample 1000 Hz → 256 Hz along axis 0
    rs = scipy.signal.resample_poly(rec, up=RS_UP, down=RS_DOWN, axis=0).astype(np.float32)
    n_total = rs.shape[0]                              # 153 600
    max_non_overlap = n_total // N_T                   # 150
    if windows_per_meas <= max_non_overlap:
        # Pick evenly-spaced non-overlapping start indices, then jitter a bit
        starts = np.linspace(0, n_total - N_T, windows_per_meas, dtype=int)
    else:
        # Slight overlap: pick evenly across whole signal
        starts = np.linspace(0, n_total - N_T, windows_per_meas, dtype=int)
    out = np.stack([rs[s:s + N_T] for s in starts], axis=0)
    return out                                        # (n_windows, N_T, N_CH)


def windows_to_spectra(windows: np.ndarray) -> np.ndarray:
    """(n_windows, N_T, N_CH) float32 → (n_windows, N_F, N_CH, 1) complex64.

    Treats every window as the response to a virtual unity-input excitation.
    Output is the rfft of the windowed (Hann) acceleration, complex.
    """
    win = np.hanning(N_T).astype(np.float32)
    # Apply window along time axis
    ww = windows * win[None, :, None]
    # rfft along time axis
    spec = np.fft.rfft(ww.astype(np.float64), axis=1)     # (n, N_F, N_CH)
    spec = spec.astype(np.complex64)
    return spec[:, :, :, None]                            # (n, N_F, N_CH, 1)


# ── Per-zip streaming pipeline ──────────────────────────────────────────────
def process_zip(state: str, target_samples: int, log: logging.Logger,
                 max_files: int | None = None
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    """Stream one OPARA zip, build (signals, spectra, label, meta) arrays.

    Returns:
        signals  : (target_samples, N_T, N_CH) float32
        spectra  : (target_samples, N_F, N_CH, 1) complex64
        meta     : list of per-sample dicts (env conditions + source file)
    """
    url = ZIP_URLS[state]
    catalog = load_catalogue(state)

    log.info(f"[{state}] opening remote zip via HTTP Range")
    t0 = time.time()
    # First pass: discover entries via a brief connection.
    with remotezip.RemoteZip(url) as rz_idx:
        h5_entries = [n for n in rz_idx.namelist() if n.endswith(".h5")]
    log.info(f"[{state}] {len(h5_entries)} h5 entries discovered")
    n_files = len(h5_entries) if max_files is None else min(max_files, len(h5_entries))
    files = h5_entries[:n_files]

    per_file = int(np.ceil(target_samples / n_files))
    log.info(f"[{state}] plan: {per_file} windows per measurement × {n_files} files "
             f"→ ~{per_file * n_files} samples (target {target_samples})")

    # Parallel fetching: thread-local RemoteZip handles. We open one connection
    # per worker. The server (Dataverse-like) supports ~8 concurrent HTTP/1.1
    # range requests comfortably; we cap at FETCH_WORKERS.
    tlocal = threading.local()
    def _local_rz():
        if not hasattr(tlocal, "rz"):
            tlocal.rz = remotezip.RemoteZip(url)
        return tlocal.rz

    sig_list: List[np.ndarray] = [None] * len(files)
    spec_list: List[np.ndarray] = [None] * len(files)
    meta_list: List[List[Dict]] = [None] * len(files)
    rng = np.random.default_rng(20260515 + hash(state) % 1000)

    # Capacity per file
    needs = []
    remaining = target_samples
    for i in range(len(files)):
        n = min(per_file, remaining)
        needs.append(n)
        remaining -= n
        if remaining <= 0:
            break
    n_active = len(needs)
    files_active = files[:n_active]

    def fetch_and_process(idx: int) -> Tuple[int, float, float, float, str]:
        entry = files_active[idx]
        fname = Path(entry).name
        t1 = time.time()
        rz = _local_rz()
        try:
            raw = rz.read(entry)
        except Exception as e:
            return idx, 0.0, 0.0, 0.0, f"FETCH_ERR:{e}"
        t_fetch = time.time() - t1

        t2 = time.time()
        try:
            with h5py.File(io.BytesIO(raw), "r") as f:
                rb = f["RecBuff"]
                # Rows: 0=timestamp, 1..56=accel, 57=surface T, 58..113=BitStates
                accel = np.empty((rb.shape[1], N_CH), dtype=np.float32)
                for j, ch in enumerate(SENSOR_CHANNELS):
                    accel[:, j] = rb[ch, :].astype(np.float32)
        except Exception as e:
            del raw
            return idx, t_fetch, 0.0, 0.0, f"READ_ERR:{e}"
        del raw
        t_read = time.time() - t2

        t3 = time.time()
        need = needs[idx]
        wins = resample_and_window(accel, need, rng)
        specs = windows_to_spectra(wins)
        t_proc = time.time() - t3

        env = catalog.get(fname, {})
        meta = []
        for w in range(need):
            meta.append({
                "source_file": fname,
                "window_idx":  w,
                "state":       state,
                "type_code":   TYPE_CODE[state],
                "end":         END_CODE[state],
                "storey":      -1,
                "severity":    SEVERITY[state],
                **env,
            })
        sig_list[idx] = wins
        spec_list[idx] = specs
        meta_list[idx] = meta
        return idx, t_fetch, t_read, t_proc, fname

    workers = FETCH_WORKERS_DEFAULT
    log.info(f"[{state}] launching {workers} parallel fetch workers for {n_active} files")
    completed = 0
    failures = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(fetch_and_process, i) for i in range(n_active)]
        for fut in as_completed(futures):
            idx, t_fetch, t_read, t_proc, status = fut.result()
            completed += 1
            elapsed = time.time() - t0
            if status.startswith("FETCH_ERR") or status.startswith("READ_ERR"):
                failures += 1
                log.warning(f"[{state}] [{completed:3d}/{n_active}] FAILED idx={idx}: {status}  elapsed={elapsed:.0f}s")
            else:
                log.info(f"[{state}] [{completed:3d}/{n_active}]  {status}  "
                          f"fetch={t_fetch:4.1f}s  read={t_read:4.1f}s  proc={t_proc:4.1f}s  "
                          f"elapsed={elapsed:.0f}s")

    # Drop failures and concatenate
    sig_list = [s for s in sig_list if s is not None]
    spec_list = [s for s in spec_list if s is not None]
    meta_list_flat: List[Dict] = []
    for ml in meta_list:
        if ml is not None:
            meta_list_flat.extend(ml)

    if not sig_list:
        raise RuntimeError(f"[{state}] no files processed successfully")
    signals = np.concatenate(sig_list, axis=0)[:target_samples]
    spectra = np.concatenate(spec_list, axis=0)[:target_samples]
    meta = meta_list_flat[:target_samples]
    log.info(f"[{state}] done: {signals.shape[0]} samples (target {target_samples}, "
              f"failures={failures}) in {time.time()-t0:.0f}s")
    return signals, spectra, meta


# ── Pymodal frf collection assembly ────────────────────────────────────────
def write_pymodal_collection(spectra_per_state: Dict[str, np.ndarray],
                              meta_per_state: Dict[str, List[Dict]],
                              path: Path, log: logging.Logger) -> Path:
    """Build a single pymodal frf collection from all per-state spectra.

    The collection has one item per sample. Each item is shape (N_F, N_CH, 1)
    complex64. Labels are floats: 0=Reference, 1=Field3, 2=Field4.
    """
    path = Path(path)
    if path.exists():
        path.unlink()

    measurements: List[np.ndarray] = []
    labels: List[float] = []
    names: List[str] = []
    for state in ("reference", "field3", "field4"):
        spec_arr = spectra_per_state[state]          # (n, N_F, N_CH, 1)
        meta = meta_per_state[state]
        for i in range(spec_arr.shape[0]):
            measurements.append(spec_arr[i])         # (N_F, N_CH, 1)
            labels.append(CLASS_LABEL[state])
            names.append(f"{state}_{i:05d}")

    log.info(f"writing pymodal frf collection: {len(measurements)} items → {path}")
    coll = PMF(
        measurements=measurements,
        freq_resolution=DF,
        measurements_units="meter / second ** 2",
        freq_units="hertz",
        method="SIMO",
        names=names,
        labels=labels,
        path=path,
    )
    # Attach per-sample metadata as auxiliary HDF5 group
    coll.close(keep=True)
    with h5py.File(path, "a") as f:
        grp = f.require_group("aux_metadata")
        keys = ["source_file", "window_idx", "state", "type_code", "end",
                "storey", "severity", "dM_kg", "dM_pos",
                "surface_temp_c", "outdoor_temp_c", "wind_ms"]
        all_meta = [m for state in ("reference", "field3", "field4") for m in meta_per_state[state]]
        for k in keys:
            vals = [m.get(k, "" if isinstance(m.get(k), str) else float("nan")) for m in all_meta]
            if k in ("source_file", "state"):
                vals_s = np.array([str(v) for v in vals], dtype=h5py.string_dtype())
                grp.create_dataset(k, data=vals_s)
            elif k in ("window_idx", "type_code", "end", "storey"):
                grp.create_dataset(k, data=np.array(vals, dtype=np.int32))
            else:
                grp.create_dataset(k, data=np.array(vals, dtype=np.float32))
        f.attrs["dataset"] = "Flossgraben Bridge — OPARA-767"
        f.attrs["sensor_channels_1indexed"] = np.array(SENSOR_CHANNELS, dtype=np.int32)
        f.attrs["fs_hz"] = float(FS_TARGET)
        f.attrs["n_t"]   = int(N_T)
        f.attrs["class_legend"] = "0=Reference 1=Field3 2=Field4"
    log.info(f"  → wrote {path.stat().st_size/1e6:.1f} MB")
    return path


def write_chunk_format(signals_per_state: Dict[str, np.ndarray],
                        meta_per_state: Dict[str, List[Dict]],
                        out_dir: Path, log: logging.Logger,
                        chunk_size: int = 256) -> Path:
    """Write 3SBB-pipeline-compatible chunked HDF5 dataset.

    Each chunk file mirrors the schema generate_dataset.py produces:
        signals    (n, N_T, N_CH) float32
        time       (N_T,)         float32
        excitation (N_T,)         float32   (zero — no controlled input)
        freqs      (N_F,)         float32
        labels/sample_id          int32
        labels/type_code          int8
        labels/end                int8
        labels/storey             int8
        labels/severity           float32
        params/source_file        S
        params/dM_kg              float32
        params/surface_temp_c     float32
        params/wind_ms            float32
        attrs: fs_hz, n_t, sensors, …
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("chunk_*.h5"):
        old.unlink()

    # Build interleaved order so each chunk is class-balanced
    rng = np.random.default_rng(20260515)
    indices: List[Tuple[str, int]] = []
    for state in ("reference", "field3", "field4"):
        n = signals_per_state[state].shape[0]
        indices.extend((state, i) for i in range(n))
    rng.shuffle(indices)

    time_axis = (np.arange(N_T, dtype=np.float32) / FS_TARGET)
    excitation = np.zeros(N_T, dtype=np.float32)
    freqs = np.fft.rfftfreq(N_T, d=1.0 / FS_TARGET).astype(np.float32)

    sample_id = 0
    chunk_idx = 0
    n_total = len(indices)
    for start in range(0, n_total, chunk_size):
        sel = indices[start:start + chunk_size]
        n = len(sel)
        signals = np.empty((n, N_T, N_CH), dtype=np.float32)
        type_code = np.empty(n, dtype=np.int8)
        end_code  = np.empty(n, dtype=np.int8)
        storey    = np.empty(n, dtype=np.int8)
        severity  = np.empty(n, dtype=np.float32)
        sample_ids = np.empty(n, dtype=np.int32)
        source_file = np.empty(n, dtype=h5py.string_dtype())
        dM = np.empty(n, dtype=np.float32)
        surf_T = np.empty(n, dtype=np.float32)
        wind = np.empty(n, dtype=np.float32)

        for i, (state, idx) in enumerate(sel):
            signals[i] = signals_per_state[state][idx]
            m = meta_per_state[state][idx]
            type_code[i] = m["type_code"]
            end_code[i]  = m["end"]
            storey[i]    = m["storey"]
            severity[i]  = m["severity"]
            sample_ids[i] = sample_id
            source_file[i] = m.get("source_file", "")
            dM[i]      = float(m.get("dM_kg", 0.0))
            surf_T[i]  = float(m.get("surface_temp_c", np.nan))
            wind[i]    = float(m.get("wind_ms", np.nan))
            sample_id += 1

        path = out_dir / f"chunk_{chunk_idx:04d}.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("signals", data=signals,
                              compression="gzip", compression_opts=4,
                              shuffle=True, chunks=(1, N_T, N_CH))
            f.create_dataset("time", data=time_axis)
            f.create_dataset("excitation", data=excitation)
            f.create_dataset("freqs", data=freqs)
            g = f.create_group("labels")
            g.create_dataset("sample_id", data=sample_ids)
            g.create_dataset("type_code", data=type_code)
            g.create_dataset("end",       data=end_code)
            g.create_dataset("storey",    data=storey)
            g.create_dataset("severity",  data=severity)
            p = f.create_group("params")
            p.create_dataset("source_file",    data=source_file)
            p.create_dataset("dM_kg",          data=dM)
            p.create_dataset("surface_temp_c", data=surf_T)
            p.create_dataset("wind_ms",        data=wind)
            f.attrs["units_signals"]    = "m/s^2"
            f.attrs["units_time"]       = "second"
            f.attrs["units_excitation"] = "newton (placeholder; output-only)"
            f.attrs["fs_hz"]            = float(FS_TARGET)
            f.attrs["n_t"]              = int(N_T)
            f.attrs["sensors"]          = ",".join(f"Ch{c}" for c in SENSOR_CHANNELS)
            f.attrs["schema_version"]   = 1
            f.attrs["dataset"]          = "Flossgraben Bridge — OPARA-767"
            f.attrs["class_legend"]     = "type_code=0 Pristine, 4 Mass; end=-1 N/A, 3 Field3, 4 Field4"
        sz = path.stat().st_size / 1e6
        log.info(f"  wrote {path.name}  n={n}  size={sz:.1f}MB")
        chunk_idx += 1

    # Manifest
    manifest = {
        "dataset":      "Flossgraben Bridge — OPARA-767",
        "n_samples":    n_total,
        "n_chunks":     chunk_idx,
        "fs_hz":        FS_TARGET,
        "n_t":          N_T,
        "n_channels":   N_CH,
        "n_inputs":     1,
        "sensors_1indexed": SENSOR_CHANNELS,
        "type_codes":   {"0": "Pristine", "4": "Mass"},
        "end_codes":    {"-1": "N/A", "3": "Field3", "4": "Field4"},
        "schema": {
            "signals":    "(n, n_t, n_channels) float32 acceleration m/s^2",
            "time":       "(n_t,) float32 seconds",
            "excitation": "(n_t,) float32 newton — ZERO (output-only)",
            "freqs":      "(n_t//2+1,) float32 Hz rfft bins",
            "labels/sample_id": "(n,) int32",
            "labels/type_code": "(n,) int8 0=Pristine 4=Mass",
            "labels/end":       "(n,) int8 -1=N/A 3=Field3 4=Field4",
            "labels/storey":    "(n,) int8 -1=N/A (girder bridge)",
            "labels/severity":  "(n,) float32 kg of added mass",
            "params/*":         "per-measurement environmental metadata",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info(f"manifest written → {out_dir / 'manifest.json'}")
    return out_dir


# ── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    global FETCH_WORKERS_DEFAULT
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--limit-files", type=int, default=None,
                          help="Smoke test: cap number of h5 files per state.")
    parser.add_argument("--targets", type=int, nargs=3, default=None,
                          metavar=("REF", "FIELD3", "FIELD4"),
                          help="Override per-state sample targets.")
    parser.add_argument("--workers", type=int, default=FETCH_WORKERS_DEFAULT,
                          help="Parallel HTTP fetch workers per state.")
    args = parser.parse_args()
    FETCH_WORKERS_DEFAULT = args.workers

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    if args.targets:
        TARGETS["reference"], TARGETS["field3"], TARGETS["field4"] = args.targets

    log = setup_logging("build_flossgraben")
    log.info(f"pymodal {pymodal.__version__}  →  output {out_dir}")
    log.info(f"targets: {TARGETS}")
    log.info(f"sensors (1-indexed): {SENSOR_CHANNELS}")

    signals_per: Dict[str, np.ndarray] = {}
    spectra_per: Dict[str, np.ndarray] = {}
    meta_per: Dict[str, List[Dict]] = {}

    for state in ("reference", "field3", "field4"):
        sig, spec, meta = process_zip(state, TARGETS[state], log,
                                        max_files=args.limit_files)
        signals_per[state] = sig
        spectra_per[state] = spec
        meta_per[state] = meta

    # Pymodal collection (clean)
    clean_pm = out_dir / "flossgraben_collection.h5"
    write_pymodal_collection(spectra_per, meta_per, clean_pm, log)

    # Chunk-format dataset (3SBB-compatible)
    chunks_dir = out_dir / "chunks"
    write_chunk_format(signals_per, meta_per, chunks_dir, log)

    # Summary
    log.info("\n=== SUMMARY ===")
    log.info(f"  pymodal clean : {clean_pm}  ({clean_pm.stat().st_size/1e6:.1f} MB)")
    log.info(f"  chunks dir    : {chunks_dir}  ({len(list(chunks_dir.glob('chunk_*.h5')))} files)")
    log.info(f"  manifest      : {chunks_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
