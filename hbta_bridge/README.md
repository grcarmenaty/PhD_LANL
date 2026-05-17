# Hell Bridge Test Arena (HBTA) — SHM benchmark

A pymodal-format experimental dataset and FEM modelling workspace for
the Hell Bridge Test Arena: a decommissioned 35 m × 4.5 m bow-string
steel-truss railway bridge in Hell, Norway, instrumented as a
full-scale damage-detection benchmark.

## What's in the dataset

* **18 arch accelerometers** (AG01–AG18) on the upper truss chord and
  joints, north and south sides
* **40 deck accelerometers** (AL01–AL40) along the four longitudinal
  stringer lines under the deck
* **15 strain gauges** (SB01–SB08 bottom chord, SC01–SC07 cross
  girders)
* **1 reference accelerometer** (AS) on the modal vibration shaker —
  this is the **measured input**, so true input-output FRFs are
  computable
* **2 shaker positions** (P1 at midspan, P2 off-centre) × 2 excitation
  directions (Y horizontal, Z vertical) × 9 structural states (UDS
  undamaged + DS1–DS8 progressive damage) = **50 records**, each a
  ~622 s sine-sweep at 100 Hz

Sampling is 100 Hz (resampled from 400 Hz inside the published
dataset); Nyquist is 50 Hz.

* **Dataset DOI:** `10.5281/zenodo.10507957`
* **Paper:** Svendsen et al., *J. Civil Structural Health Monitoring*
  2022, https://link.springer.com/article/10.1007/s13349-021-00530-8
* **License:** CC BY 4.0

## Layout

```
hbta_bridge/
├── raw/                          (data_100Hz.h5 — 2.5 GB, gitignored)
│   └── sensor_layout.pdf         (committed, from the Zenodo record)
├── scripts/
│   └── build_hbta_pymodal.py     (windowing + FRF builder)
├── output/                       (collection + chunks, gitignored)
│   ├── hbta_collection.h5
│   ├── chunks/
│   └── diagnostic_frf_per_class.png
├── logs/                         (gitignored)
└── model.md                      (FEM model specification)
```

## Loader output

`build_hbta_pymodal.py` reads `raw/data_100Hz.h5`, slices each record
into non-overlapping 1024-sample (10.24 s) windows, and for each window
computes three quantities:

* **`signals`** — `(N_T=1024, N_CH)` float32 time-domain window
* **`spec_output`** — `(N_F=513, N_CH)` complex64, `rfft` of the Hann-
  windowed acceleration
* **`frf_H1`** — `(N_F=513, N_CH)` complex64, the H1 estimator
  `S_xy / S_xx` using the AS reference as input

The current channel selection is 12 sensors (10 arch joints + 2
representative deck stringers); see `SENSOR_SELECTION` in the loader.
Outputs are written to `output/hbta_collection.h5` (pymodal-format) plus
chunked HDF5 files (`output/chunks/chunk_XXXX.h5`) for batched
consumers.

Class encoding: `0 = UDS` (undamaged reference), `1…8 = DS1…DS8`.

## Running

```bash
# Place the 2.5 GB H5 in raw/ first; loader expects raw/data_100Hz.h5
python hbta_bridge/scripts/build_hbta_pymodal.py             # full build (~3 min)
python hbta_bridge/scripts/build_hbta_pymodal.py --smoke 6   # quick test, 6 records
```

## FEM model

See `model.md` for the Salome + Code_Aster FEM specification, damage-
scenario mapping, and calibration pipeline.
