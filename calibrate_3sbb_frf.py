"""Full-FRF calibration of the LANL 3SBB reduced model against Pristine data.

Objective
---------
Minimise the mean squared log-magnitude FRF error over Y-sensor channels
in the 10–80 Hz band, plus a soft frequency-residual penalty term.

Optimisation variables
----------------------
x[0]  log10(JSR)         [0.30, 1.50]
x[1]  base_extra_mass    [0, 30] kg
x[2]  cf_s1              [0.5, 3.0]
x[3]  cf_s2              [0.5, 3.0]
x[4]  cf_s3              [0.5, 3.0]
x[5]  damping            [0.001, 0.05]

Key insight: allowing cf > 1 lets the effective story stiffness exceed k_ff,
which is physically equivalent to accounting for contact stiffness or other
stiffening mechanisms not captured in the slender-column model.
"""

from __future__ import annotations
import sys
import os
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.signal import find_peaks
import h5py

# ── local imports ────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE))
from reduced_model_semirigid import BuildingGeometry, modes, compute_frf_matrix
import params as P

# ── load experimental Pristine FRFs ─────────────────────────────────────────
HDF5_PATH = _HERE / "experimental_frfs.h5"
PRISTINE_LABEL = 59

# Y-floor sensors: ch1=S5=fl3, ch2=S6=fl2, ch3=S7=fl1
# Sensor positions (x, y, z) in the model coordinate system
# Rail direction = Y → base DOF is Y-translation
# Floor sensors measure Y-acceleration at mid-plate positions
FLOOR_Y_CHS = [1, 2, 3]   # indices into the 9-channel FRF matrix (0-based)

print("Loading experimental FRFs …", flush=True)
with h5py.File(HDF5_PATH, 'r') as f:
    freq_exp = np.array(f['freq'])   # (1601,)
    data     = np.array(f['data'])   # (2638, 1601, 9), complex
    labels   = np.array(f['labels'])

idx_p  = np.where(labels == PRISTINE_LABEL)[0]
H_raw  = data[idx_p]                          # (N_p, 1601, 9)
H_mean = np.mean(np.abs(H_raw), axis=0)       # (1601, 9) — mean magnitude
df     = freq_exp[1] - freq_exp[0]
print(f"  Pristine: {len(idx_p)} measurements,  Δf={df:.4f} Hz")

# ── frequency band for FRF matching ─────────────────────────────────────────
F_LO, F_HI = 10.0, 80.0
mask_fit = (freq_exp >= F_LO) & (freq_exp <= F_HI)
freq_fit = freq_exp[mask_fit]                  # reduced freq array
H_exp_fit = H_mean[mask_fit, :]                # (N_fit, 9)
H_exp_y   = H_exp_fit[:, FLOOR_Y_CHS]         # (N_fit, 3) — Y-floor channels
logH_exp  = np.log10(H_exp_y + 1e-12)

print(f"  Fit band: {F_LO}–{F_HI} Hz  ({mask_fit.sum()} points)")

# ── identify experimental resonances (peak-pick on S7 = ch3 = fl1Y) ─────────
ch_pk = 3   # S7 = ch3 (0-based index 3)
H_pk  = H_mean[mask_fit, ch_pk]
peaks, props = find_peaks(H_pk, distance=40, prominence=H_pk.max() * 0.05)
top3 = np.argsort(props['prominences'])[-3:]
top3 = peaks[top3[np.argsort(freq_fit[peaks[top3]])]]
f_res = freq_fit[top3]
print(f"\nExperimental Y-mode resonances: {f_res}")

# ── sensor input/output DOF vectors ─────────────────────────────────────────
# Base plate input: Y-direction at plate centroid, z=plate_lz/2
geom_nom = BuildingGeometry()
cx       = geom_nom.plate_lx / 2.0
cy       = geom_nom.plate_ly / 2.0
z0       = geom_nom.plate_lz / 2.0                        # base plate centre z
h1       = geom_nom.storey_height + z0                    # fl1 centre z
h2       = 2.0 * geom_nom.storey_height + z0
h3       = 3.0 * geom_nom.storey_height + z0

INPUT_PT  = ((cx, cy, z0), 'Y')                           # shaker on base, Y
OUTPUT_PTS = [
    ((cx, cy, h3), 'Y'),   # ch1  S5 fl3 Y
    ((cx, cy, h2), 'Y'),   # ch2  S6 fl2 Y
    ((cx, cy, h1), 'Y'),   # ch3  S7 fl1 Y
]

# ── build model from optimisation vector ─────────────────────────────────────
def geom_from_x(x):
    jsr    = 10.0 ** x[0]
    m_xtra = x[1]
    cf1, cf2, cf3 = x[2], x[3], x[4]
    damp   = x[5]
    geom = BuildingGeometry(
        joint_stiffness_ratio = jsr,
        damping               = damp,
        screw_mass_per_joint  = 0.0,
        base_extra_mass       = m_xtra,
    )
    geom.column_factor = np.array([[cf1, cf1, cf1, cf1],
                                   [cf2, cf2, cf2, cf2],
                                   [cf3, cf3, cf3, cf3]], dtype=float)
    return geom


# ── weights ───────────────────────────────────────────────────────────────────
W_FRF  = 1.0
W_FREQ = 5.0   # soft penalty to keep resonances close

def objective(x):
    # hard bounds guard
    if x[1] < 0.0 or x[5] < 5e-4 or any(xi < 0.3 for xi in x[2:5]):
        return 1e9

    geom = geom_from_x(x)
    try:
        H_mod = compute_frf_matrix(
            freq_fit,
            inputs  = [INPUT_PT],
            outputs = OUTPUT_PTS,
            geom    = geom,
            damping = x[5],
        )   # (N_fit, 3, 1) complex
    except Exception:
        return 1e9

    H_mod_mag = np.abs(H_mod[:, :, 0])          # (N_fit, 3)
    logH_mod  = np.log10(H_mod_mag + 1e-12)

    # 1. log-FRF error
    L_frf = W_FRF * np.mean((logH_mod - logH_exp) ** 2)

    # 2. soft resonance penalty using model peak frequencies
    try:
        f_all = modes(geom)[0]
        f_el  = np.sort(f_all[f_all > 0.5])
        if len(f_el) >= 3:
            f3 = f_el[:3]
            rel = (f3 - f_res) / f_res
            L_freq = W_FREQ * np.sum(rel ** 2)
        else:
            L_freq = 1e4
    except Exception:
        L_freq = 1e4

    return float(L_frf + L_freq)


# ── bounds ────────────────────────────────────────────────────────────────────
bounds = [
    (0.30, 1.50),   # log10(JSR): JSR from 2 to 30
    (0.0,  30.0),   # base_extra_mass (kg)
    (0.5,  3.0),    # cf_s1
    (0.5,  3.0),    # cf_s2
    (0.5,  3.0),    # cf_s3
    (0.001, 0.05),  # damping
]

# ── initial guess ─────────────────────────────────────────────────────────────
x0 = np.array([np.log10(7.9), 4.0, 1.0, 1.0, 1.0, 0.005])
print(f"\nInitial cost: {objective(x0):.4f}")

# ── Stage 1: differential evolution ──────────────────────────────────────────
print("\n── Stage 1: differential evolution ─────────────────────────────")
res_de = differential_evolution(
    objective, bounds, seed=42,
    maxiter=400, popsize=20, tol=1e-7,
    mutation=(0.5, 1.5), recombination=0.9,
    workers=1, disp=True, polish=False,
    updating='deferred',
)
x1 = res_de.x
print(f"\nDE result  cost={res_de.fun:.6f}")
print(f"  JSR={10**x1[0]:.3f}  m_xtra={x1[1]:.3f} kg")
print(f"  cf=[{x1[2]:.3f},{x1[3]:.3f},{x1[4]:.3f}]  damp={x1[5]:.5f}")

# ── Stage 2: Nelder-Mead polish ───────────────────────────────────────────────
print("\n── Stage 2: Nelder-Mead polish ──────────────────────────────────")
res_nm = minimize(objective, x1, method='Nelder-Mead',
                  options={'maxiter': 30000, 'xatol': 1e-9, 'fatol': 1e-9,
                           'adaptive': True, 'disp': True})
x_opt = res_nm.x

# ── report ────────────────────────────────────────────────────────────────────
geom_opt = geom_from_x(x_opt)
f_all    = modes(geom_opt)[0]
f_el     = np.sort(f_all[f_all > 0.5])
f_mod3   = f_el[:3]

PARAM_NAMES = ['log10_jsr', 'base_extra_mass_kg', 'cf_s1', 'cf_s2', 'cf_s3', 'damping']
print(f"\n  NM result  cost={res_nm.fun:.6f}")
print(f"\n  OPTIMAL PARAMETERS:")
for name, val in zip(PARAM_NAMES, x_opt):
    print(f"    {name:30s} = {val:.6f}")
print(f"\n  Derived:")
print(f"    JSR             = {10**x_opt[0]:.4f}")
print(f"    base_extra_mass = {x_opt[1]:.4f} kg")
for s, cf in enumerate(x_opt[2:5], 1):
    print(f"    story {s}: cf={cf:.4f}  k_rel={cf**4:.4f}")
print(f"    damping         = {x_opt[5]:.5f}")

print(f"\n  FREQUENCY COMPARISON:")
print(f"  {'Mode':>4}  {'f_exp':>10}  {'f_mod':>10}  {'err%':>8}")
for j in range(3):
    err = (f_mod3[j] - f_res[j]) / f_res[j] * 100
    print(f"  {j+1:>4}  {f_res[j]:>10.4f}  {f_mod3[j]:>10.4f}  {err:>8.3f}%")

# ── save ──────────────────────────────────────────────────────────────────────
out_file = _HERE / "calibration_result_frf.npz"
np.savez(out_file,
         x_opt=x_opt, f_mod=f_mod3, f_res=f_res,
         param_names=np.array(PARAM_NAMES))
print(f"\nResults saved → {out_file}")
print("\n[calibrate_3sbb_frf.py] Done.")
