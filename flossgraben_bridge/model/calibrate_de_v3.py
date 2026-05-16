"""v3 calibration: 13 knobs (10 from v2 + pier-v compliance,
pier-torsional spring, sensor-x offset).

Best parameters from v2 used as the centre of seed population.
"""
from __future__ import annotations
import os
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
import beam_fem_v3 as bf3

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


_EXP_CACHE: dict = load_experimental_cfdac()


def loss(x):
    (e_gpa, I_yy, log_GJ, log_rho_Ip,
     z_lo, z_mid, z_hi, y_off, f_cut,
     log_k_pier_v, log_k_pier_rot, log_k_pier_tors,
     dx_sensor) = x
    params = dict(
        E_bend=e_gpa * 1e9,
        I_yy=I_yy,
        mu=2500.0 * 9.4,
        GJ=10.0 ** log_GJ,
        rho_Ip=10.0 ** log_rho_Ip,
        z_lo=z_lo, z_mid=z_mid, z_hi=z_hi,
        y_off=y_off, f_cut=f_cut,
        k_pier_v=10.0 ** log_k_pier_v,
        k_pier_rot=10.0 ** log_k_pier_rot,
        k_pier_tors=10.0 ** log_k_pier_tors,
        dx_sensor=dx_sensor,
    )
    sci_list = []
    f1 = 0.0
    for sc in ("reference", "field3", "field4"):
        try:
            Y, f1_, fb, ft = bf3.auto_spectrum_v3(sc, params, FREQ_GRID, BAND)
        except Exception:
            return 10.0
        if not np.all(np.isfinite(Y)):
            return 10.0
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


def main():
    print("Cache loaded", flush=True)
    bounds = [
        (5.0,  45.0),       # E_bend [GPa]
        (3.0,  40.0),       # I_yy   [m^4]
        (8.0,  12.5),       # log10(GJ)
        (3.0,   6.5),       # log10(rho·Ip)
        (0.005, 0.30),      # z_lo
        (0.005, 0.30),      # z_mid
        (0.010, 0.30),      # z_hi
        (-15.0, 15.0),      # y_off
        ( 1.0,  30.0),      # f_cut
        ( 7.0,  12.0),      # log10(k_pier_v) — N/m
        ( 0.0,  11.0),      # log10(k_pier_rot) — Nm/rad
        ( 6.0,  12.0),      # log10(k_pier_tors) — Nm/rad
        (-15.0,15.0),       # dx_sensor [m]
    ]
    log_state = {"n": 0, "best": -np.inf, "t0": time.time()}
    print("Starting DE v3 (13 knobs)…", flush=True)

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

    res = differential_evolution(
        loss, bounds, seed=20260516, maxiter=120, popsize=12,
        tol=1e-4, workers=-1, polish=False, disp=False,
        updating='deferred', callback=cb)
    print(f"\nDone. Best score = {-res.fun:.4f}, n_evals={res.nfev}", flush=True)
    summary = {
        "best_score": float(-res.fun),
        "best_params": {
            "E_GPa": float(res.x[0]),
            "I_yy": float(res.x[1]),
            "log_GJ": float(res.x[2]),
            "log_rho_Ip": float(res.x[3]),
            "z_lo": float(res.x[4]),
            "z_mid": float(res.x[5]),
            "z_hi": float(res.x[6]),
            "y_off": float(res.x[7]),
            "f_cut": float(res.x[8]),
            "log_k_pier_v": float(res.x[9]),
            "log_k_pier_rot": float(res.x[10]),
            "log_k_pier_tors": float(res.x[11]),
            "dx_sensor": float(res.x[12]),
        },
        "iterations": int(res.nit), "n_evals": int(res.nfev),
    }
    (HERE / "best_params_v3.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote best_params_v3.json", flush=True)


if __name__ == "__main__":
    main()
