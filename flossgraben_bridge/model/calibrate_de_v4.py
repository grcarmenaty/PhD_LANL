"""v4 calibration — smoothed-CFDAC SCI (OMA-appropriate metric).

For input-output FRFs (3SBB-style) the modal peaks are sharp and aligned
across windows, so CFDAC's frequency-precision sensitivity is a feature.
For OMA data with stochastic traffic excitation, the per-window
acceleration spectra are broadband averages where modal peaks blur into
bands. Sharp-peak CFDAC penalises even 1-bin (0.25 Hz) modal
misalignment, capping SCI in the 0.5 range regardless of how good the
model is.

This calibrator computes CFDAC on Gaussian-smoothed log-magnitude
spectra (σ = 2 bins = 0.5 Hz). The metric still requires correct
modal STRUCTURE (right number of bands, right spatial patterns) but
tolerates the ~1 Hz frequency uncertainty inherent to OMA peak picking.
Both experimental and model spectra get the same smoothing kernel so
the comparison is consistent.
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
from scipy.ndimage import gaussian_filter1d

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import beam_fem_v3 as bf3

EXP_H5 = HERE.parent / "output" / "flossgraben_collection.h5"
FS, N_T, N_F = 256.0, 1024, 513
FREQ_GRID = np.arange(N_F) * (FS / N_T)
BAND_LO, BAND_HI = 0.5, 25.0
BAND = (FREQ_GRID >= BAND_LO) & (FREQ_GRID <= BAND_HI)

F1_EXP = 1.75
W_FREQ = 1.0
N_PER_CLASS = 500

# Smoothing kernel (Gaussian on log-magnitude, σ in bins of 0.25 Hz)
SIGMA_BINS = 2.0


def smooth_logmag(spec_mag: np.ndarray) -> np.ndarray:
    """Gaussian smooth of log10 magnitude along frequency axis."""
    return gaussian_filter1d(np.log10(spec_mag + 1e-30), sigma=SIGMA_BINS, axis=0)


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
        mag = np.median(np.abs(arr[..., 0]).astype(np.float64), axis=0)
        mag_band = mag[BAND]
        # Smoothed log-magnitude as the "H" we feed CFDAC
        H_smooth = smooth_logmag(mag_band).astype(np.complex128)
        # Recenter so DC offset doesn't dominate
        H_smooth = H_smooth - H_smooth.mean(axis=0, keepdims=True)
        cache[sc] = dict(H_raw=mag_band, H=H_smooth, C=cfdac(H_smooth))
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
        H_smooth = smooth_logmag(Y).astype(np.complex128)
        H_smooth = H_smooth - H_smooth.mean(axis=0, keepdims=True)
        sci_list.append(sci(_EXP_CACHE[sc]["C"], cfdac(H_smooth)))
    mean_sci = float(np.mean(sci_list))
    f1_dev = ((f1 - F1_EXP) / F1_EXP) ** 2
    score = mean_sci - W_FREQ * f1_dev
    return -score


def main():
    print(f"v4 cache loaded (smoothed-logmag CFDAC, σ={SIGMA_BINS} bins)",
          flush=True)
    bounds = [
        (5.0,  45.0),
        (3.0,  40.0),
        (8.0,  12.5),
        (3.0,   6.5),
        (0.005, 0.30),
        (0.005, 0.30),
        (0.010, 0.30),
        (-15.0, 15.0),
        ( 1.0,  30.0),
        ( 7.0,  12.0),
        ( 0.0,  11.0),
        ( 6.0,  12.0),
        (-15.0, 15.0),
    ]
    log_state = {"n": 0, "best": -np.inf, "t0": time.time()}
    print("Starting DE v4 …", flush=True)

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
        loss, bounds, seed=20260516, maxiter=100, popsize=12,
        tol=1e-4, workers=-1, polish=False, disp=False,
        updating='deferred', callback=cb)
    print(f"\nDone. Best score = {-res.fun:.4f}, n_evals={res.nfev}", flush=True)

    # Also compute raw-magnitude SCI on the best params as a secondary metric
    print("\nSecondary metrics on best params:", flush=True)
    (e_gpa, I_yy, log_GJ, log_rho_Ip,
     z_lo, z_mid, z_hi, y_off, f_cut,
     log_k_pier_v, log_k_pier_rot, log_k_pier_tors,
     dx_sensor) = res.x
    params = dict(
        E_bend=e_gpa * 1e9, I_yy=I_yy, mu=2500.0 * 9.4,
        GJ=10.0 ** log_GJ, rho_Ip=10.0 ** log_rho_Ip,
        z_lo=z_lo, z_mid=z_mid, z_hi=z_hi,
        y_off=y_off, f_cut=f_cut,
        k_pier_v=10.0 ** log_k_pier_v,
        k_pier_rot=10.0 ** log_k_pier_rot,
        k_pier_tors=10.0 ** log_k_pier_tors,
        dx_sensor=dx_sensor)
    raw_scis, smooth_scis = [], []
    for sc in ("reference", "field3", "field4"):
        Y, _, _, _ = bf3.auto_spectrum_v3(sc, params, FREQ_GRID, BAND)
        # Smoothed (matches loss)
        Hs = smooth_logmag(Y).astype(np.complex128)
        Hs = Hs - Hs.mean(axis=0, keepdims=True)
        smooth_scis.append(sci(_EXP_CACHE[sc]["C"], cfdac(Hs)))
        # Raw |H| (3SBB-style)
        H_raw = Y.astype(np.complex128)
        E_raw = _EXP_CACHE[sc]["H_raw"].astype(np.complex128)
        raw_scis.append(sci(cfdac(E_raw), cfdac(H_raw)))
        print(f"  {sc:>10s}  smooth-SCI = {smooth_scis[-1]:.4f}   "
              f"raw-SCI = {raw_scis[-1]:.4f}", flush=True)

    summary = {
        "best_score_smooth": float(-res.fun),
        "smooth_sci": {sc: smooth_scis[i] for i, sc in enumerate(
            ("reference", "field3", "field4"))},
        "raw_sci": {sc: raw_scis[i] for i, sc in enumerate(
            ("reference", "field3", "field4"))},
        "best_params": {
            "E_GPa": float(res.x[0]), "I_yy": float(res.x[1]),
            "log_GJ": float(res.x[2]), "log_rho_Ip": float(res.x[3]),
            "z_lo": float(res.x[4]), "z_mid": float(res.x[5]),
            "z_hi": float(res.x[6]), "y_off": float(res.x[7]),
            "f_cut": float(res.x[8]),
            "log_k_pier_v": float(res.x[9]),
            "log_k_pier_rot": float(res.x[10]),
            "log_k_pier_tors": float(res.x[11]),
            "dx_sensor": float(res.x[12]),
        },
        "iterations": int(res.nit), "n_evals": int(res.nfev),
        "smoothing_sigma_bins": SIGMA_BINS,
    }
    (HERE / "best_params_v4.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote best_params_v4.json", flush=True)


if __name__ == "__main__":
    main()
