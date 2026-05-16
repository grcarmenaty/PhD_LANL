"""v2 calibration: bending + torsion, per-band damping, sensor offset.

Knobs (10):
  E_bend [GPa]          deck effective Young's modulus
  I_yy [m^4]            deck vertical bending inertia
  GJ [N·m²]             deck torsional rigidity
  rho_Ip [kg·m]         deck linear polar inertia
  z_lo, z_mid, z_hi     damping ratios in 3 frequency bands
  y_off [m²]            (sensor east offset) × (force eccentricity)
  f_cut [Hz]            traffic-input PSD low-pass corner
  log10(k_pier_rot)     bending rotational pier spring
"""
from __future__ import annotations
import os
# Single-threaded BLAS — when scipy.optimize uses workers=-1 (multiprocess)
# multi-threaded BLAS causes catastrophic thread oversubscription
# (workers × threads = 16+ threads fighting on 4 cores), turning a 160 ms
# eigh into a 2.7 s eigh. Must be set BEFORE importing numpy/scipy.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["OMP_NUM_THREADS"]      = "1"

import json, sys, time
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import differential_evolution

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import beam_fem_v2 as bf2

EXP_H5 = HERE.parent / "output" / "flossgraben_collection.h5"
FS, N_T, N_F = 256.0, 1024, 513
FREQ_GRID = np.arange(N_F) * (FS / N_T)
BAND_LO, BAND_HI = 0.5, 25.0
BAND = (FREQ_GRID >= BAND_LO) & (FREQ_GRID <= BAND_HI)

F1_EXP = 1.75
W_FREQ = 2.0
N_PER_CLASS = 500


def cfdac(H):
    inner = H.conj() @ H.T
    d = np.real(np.diag(inner)).copy(); d[d < 1e-30] = 1e-30
    return (np.abs(inner) ** 2) / np.outer(d, d)


def sci(C1, C2):
    a = C1.ravel() - C1.mean(); b = C2.ravel() - C2.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float((a @ b) ** 2 / d ** 2) if d > 0 else 0.0


def load_experimental_cfdac():
    name_by_class = {0: "reference", 1: "field3", 2: "field4"}
    out = {v: [] for v in name_by_class.values()}
    with h5py.File(EXP_H5, "r") as f:
        meas = f["measurements"]
        for name in meas:
            if name == "_axes": continue
            label = int(round(float(meas[name]["label"][()])))
            sc = name_by_class.get(label)
            if sc is None or len(out[sc]) >= N_PER_CLASS:
                continue
            out[sc].append(np.asarray(meas[name]["data"]))
            if all(len(v) >= N_PER_CLASS for v in out.values()):
                break
    cache = {}
    for sc, lst in out.items():
        arr = np.stack(lst)
        med = np.median(np.abs(arr[..., 0]).astype(np.float64), axis=0)
        H_band = med[BAND].astype(np.complex128)
        cache[sc] = dict(H=H_band, C=cfdac(H_band))
    return cache


# Module-level cache populated at import time so worker subprocesses
# (whether forked or spawned) see a fully populated cache after they
# import this module.
_EXP_CACHE: dict = load_experimental_cfdac()


def loss(x):
    """Module-level loss function (picklable for workers=-1 parallel DE).

    Safety: reject parameter sets that produce sub-Hz modes (degenerate)
    or non-finite spectra, return a constant penalty so DE can move on.
    """
    (e_gpa, I_yy, log_GJ, log_rho_Ip,
     z_lo, z_mid, z_hi, y_off, f_cut, log_krot) = x
    params = dict(
        E_bend=e_gpa * 1e9,
        I_yy=I_yy,
        mu=2500.0 * 9.4,
        GJ=10.0 ** log_GJ,
        rho_Ip=10.0 ** log_rho_Ip,
        z_lo=z_lo, z_mid=z_mid, z_hi=z_hi,
        y_off=y_off, f_cut=f_cut,
        k_pier_rot=10.0 ** log_krot,
        k_pier_tors=1e12,
    )
    sci_list = []
    f1 = 0.0
    for sc in ("reference", "field3", "field4"):
        try:
            Y, f1_, fb, ft = bf2.auto_spectrum_v2(sc, params, FREQ_GRID, BAND)
        except Exception:
            return 10.0
        if not np.all(np.isfinite(Y)):
            return 10.0
        # Reject degenerate modal structure
        if fb[0] < 0.2 or ft[0] < 0.2:
            return 10.0
        if sc == "reference":
            f1 = f1_
        H_band = Y.astype(np.complex128)
        sci_list.append(sci(_EXP_CACHE[sc]["C"], cfdac(H_band)))
    mean_sci = float(np.mean(sci_list))
    f1_dev = ((f1 - F1_EXP) / F1_EXP) ** 2
    score = mean_sci - W_FREQ * f1_dev
    return -score


def loss_verbose(x, log_state):
    """Wrapper that logs improvements but isn't called by parallel DE."""
    val = loss(x)
    log_state["n"] += 1
    if -val > log_state["best"]:
        log_state["best"] = -val
        print(f"  [{log_state['n']:4d} t={time.time()-log_state['t0']:6.1f}s] "
              f"score={-val:.4f}  x={np.array2string(x, precision=3)}")
    return val


def main():
    print(f"Experimental CFDAC cache loaded at import (3 scenarios)", flush=True)
    bounds = [
        (5.0,  45.0),
        (3.0,  40.0),
        (8.0,  12.5),
        (3.0,   6.5),
        (0.005, 0.20),
        (0.005, 0.25),
        (0.010, 0.30),
        (-15.0, 15.0),
        ( 1.0,  30.0),
        ( 0.0,  11.0),
    ]
    log_state = {"n": 0, "best": -np.inf, "t0": time.time()}
    print("Starting DE…", flush=True)

    def cb(xk, convergence):
        v = loss(xk)
        log_state["n"] += 1
        marker = " *" if -v > log_state["best"] else "  "
        if -v > log_state["best"]:
            log_state["best"] = -v
        print(f"  [gen {log_state['n']:3d} t={time.time()-log_state['t0']:6.1f}s] "
              f"score={-v:.4f}{marker} conv={convergence:.3g} "
              f"x={np.array2string(xk, precision=3, separator=',')}",
              flush=True)

    # Parallel: single-thread BLAS (env above) prevents oversubscription.
    # popsize=15 × 10 = 150 individuals; 80 generations × 150 evals
    # at 160 ms / 4 cores ≈ 12 minutes wall time.
    res = differential_evolution(
        loss, bounds, seed=20260516, maxiter=80, popsize=15,
        tol=1e-4, workers=-1, polish=False, disp=False,
        updating='deferred', callback=cb)
    print(f"\nDone. Best score = {-res.fun:.4f}, n_evals={res.nfev}")
    summary = {
        "best_score": float(-res.fun),
        "best_params": {
            "E_GPa": float(res.x[0]), "I_yy": float(res.x[1]),
            "log_GJ": float(res.x[2]), "log_rho_Ip": float(res.x[3]),
            "z_lo": float(res.x[4]), "z_mid": float(res.x[5]),
            "z_hi": float(res.x[6]), "y_off": float(res.x[7]),
            "f_cut": float(res.x[8]), "log_k_pier_rot": float(res.x[9]),
        },
        "iterations": int(res.nit), "n_evals": int(res.nfev),
    }
    (HERE / "best_params_v2.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote {HERE / 'best_params_v2.json'}")


if __name__ == "__main__":
    main()
