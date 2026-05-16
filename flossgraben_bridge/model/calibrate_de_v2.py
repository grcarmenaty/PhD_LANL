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


def make_loss(exp_cache):
    counter = {"n": 0}
    best = {"score": -np.inf, "x": None, "sci": None}
    t0 = time.time()
    def loss(x):
        (e_gpa, I_yy, log_GJ, log_rho_Ip,
         z_lo, z_mid, z_hi, y_off, f_cut, log_krot) = x
        params = dict(
            E_bend=e_gpa * 1e9,
            I_yy=I_yy,
            mu=2500.0 * 9.4,                # locked μ (deck mass per metre)
            GJ=10.0 ** log_GJ,
            rho_Ip=10.0 ** log_rho_Ip,
            z_lo=z_lo, z_mid=z_mid, z_hi=z_hi,
            y_off=y_off, f_cut=f_cut,
            k_pier_rot=10.0 ** log_krot,
            k_pier_tors=1e12,               # treat torsion as pinned at piers
        )
        sci_list = []
        f1 = 0.0
        for sc in ("reference", "field3", "field4"):
            try:
                Y, f1_, _, _ = bf2.auto_spectrum_v2(sc, params, FREQ_GRID, BAND)
            except Exception:
                return 10.0
            if sc == "reference":
                f1 = f1_
            H_band = Y.astype(np.complex128)
            sci_list.append(sci(exp_cache[sc]["C"], cfdac(H_band)))
        mean_sci = float(np.mean(sci_list))
        f1_dev = ((f1 - F1_EXP) / F1_EXP) ** 2
        score = mean_sci - W_FREQ * f1_dev
        counter["n"] += 1
        if score > best["score"]:
            best["score"] = score; best["x"] = x.copy(); best["sci"] = sci_list[:]
            print(f"  [eval {counter['n']:4d} t={time.time()-t0:6.1f}s] "
                  f"SCI={mean_sci:.4f} (R={sci_list[0]:.3f} "
                  f"F3={sci_list[1]:.3f} F4={sci_list[2]:.3f}) "
                  f"f1={f1:.3f} | "
                  f"E={e_gpa:.1f} I={I_yy:.2f} GJ=1e{log_GJ:.1f} "
                  f"ρIp=1e{log_rho_Ip:.1f} z=[{z_lo:.3f},{z_mid:.3f},{z_hi:.3f}] "
                  f"y={y_off:.2f} fc={f_cut:.1f} krot=1e{log_krot:.1f}")
        return -score
    return loss, best


def main():
    print("Loading experimental CFDAC cache…")
    exp_cache = load_experimental_cfdac()
    loss, best = make_loss(exp_cache)
    bounds = [
        (5.0,  45.0),       # E_bend [GPa]
        (3.0,  40.0),       # I_yy   [m^4]
        (8.0,  12.5),       # log10(GJ) — 1e8 .. 3e12 N·m²
        (3.0,   6.5),       # log10(rho·Ip) — 1e3 .. 3e6 kg·m
        (0.005, 0.20),      # z_lo  (0-3 Hz)
        (0.005, 0.25),      # z_mid (3-10 Hz)
        (0.010, 0.30),      # z_hi  (10-25 Hz)
        (-15.0, 15.0),      # y_off [m²]
        ( 1.0,  30.0),      # f_cut
        ( 0.0,  11.0),      # log10(k_pier_rot)
    ]
    print("Starting DE…")
    res = differential_evolution(
        loss, bounds, seed=20260516, maxiter=80, popsize=20,
        tol=1e-4, workers=-1, polish=True, disp=False, updating='deferred')
    print(f"\nDone. Best score = {-res.fun:.4f}")
    print("Best params:", dict(zip(
        ["E_GPa", "I_yy", "logGJ", "logrhoIp",
         "z_lo", "z_mid", "z_hi", "y_off", "f_cut", "logkrot"],
        [float(v) for v in res.x])))
    summary = {
        "best_score": float(-res.fun),
        "best_sci": best["sci"],
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
