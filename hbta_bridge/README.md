# Hell Bridge Test Arena (HBTA) — SHM benchmark

Replacement for the Flossgraben dataset. **In progress.**

## What this is

35 m × 4.5 m bow-string steel-truss bridge (decommissioned, Hell,
Norway). 22 progressive damage states across 4 realistic damage types,
all driven by a **modal vibration shaker** with measured input force —
unlike Flossgraben which is output-only / traffic-excited. 18 arch
accelerometers + 40 deck accelerometers + 15 strain gauges.

* Dataset DOI: `10.5281/zenodo.10507957`
* Paper: Svendsen et al., *J. Civil Structural Health Monitoring* 2022,
  https://link.springer.com/article/10.1007/s13349-021-00530-8
* Sampling: 100 Hz (Nyquist 50 Hz)
* License: CC BY 4.0

## Why this and not Flossgraben

Three Flossgraben pain points that HBTA fixes:

1. **Measured input force** — true input-output FRFs computable, so the
   3SBB-style CFDAC SCI metric ceases to be hypersensitive to peak
   smearing.
2. **22 discrete damage scenarios** instead of 2 mass perturbations —
   enables proper damage-state discrimination.
3. **Documented geometry** — span length, cross-section, sensor
   coordinates all in the sensor-layout PDF (see `raw/sensor_layout.pdf`).
   No more guessing the deck cross-section.

## Layout (planned, mirrors `flossgraben_bridge/`)

```
hbta_bridge/
├── raw/                       (data_100Hz.h5 — 2.5 GB; not committed)
│   └── sensor_layout.pdf      (committed)
├── scripts/
│   └── build_hbta_pymodal.py  (loader; to be written)
├── output/                    (chunked pymodal files; not committed)
└── logs/                      (download / processing logs; not committed)
```
