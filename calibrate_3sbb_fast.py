"""Fast FRF + frequency calibration — designed to complete in ~35 s.

Runs DE(maxiter=80, popsize=12) then Nelder-Mead polish.
Saves calibration_result_frf.npz with optimal parameters.
"""
from __future__ import annotations
import sys
import os
from pathlib import Path
import time

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.signal import find_peaks
from scipy.linalg import eigh
import h5py

_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE))
from reduced_model_semirigid import (BuildingGeometry, stiffness_matrix,
                                      mass_matrix, modes, compute_frf_matrix)
import params as P

# ── load experimental data ────────────────────────────────────────────────────
HDF5_PATH = _HERE / "experimental_frfs.h5"
with h5py.File(HDF5_PATH, 'r') as f:
    freq_exp = np.array(f['freq'])
    data     = np.array(f['data'])
    labels   = np.array(f['labels'])

idx_p  = np.where(labels == 59)[0]
H_raw  = data[idx_p]
H_mean = np.mean(np.abs(H_raw), axis=0)   # (1601, 9)
df     = freq_exp[1] - freq_exp[0]

# ── fit band ─────────────────────────────────────────────────────────────────
mask = (freq_exp >= 10.0) & (freq_exp <= 80.0)
# subsample 4× to speed objective (still 280 pts)
idx_fit = np.where(mask)[0][::4]
freq_fit  = freq_exp[idx_fit]
H_exp_y   = H_mean[np.ix_(idx_fit, [1, 2, 3])]  # fl3, fl2, fl1
logH_exp  = np.log(H_exp_y + 1e-12)

# ── peak-pick resonances ──────────────────────────────────────────────────────
freq_all  = freq_exp[mask]
H_pk      = H_mean[mask, 3]   # ch3=S7=fl1
peaks, pr = find_peaks(H_pk, distance=40, prominence=H_pk.max()*0.05)
top3 = peaks[np.argsort(pr['prominences'])[-3:]]
top3 = top3[np.argsort(freq_all[top3])]
f_res = freq_all[top3]
print(f"Experimental resonances: {f_res}")

# ── geometry helpers ──────────────────────────────────────────────────────────
geom_nom = BuildingGeometry()
cx = geom_nom.plate_lx / 2.0
cy = geom_nom.plate_ly / 2.0
z0 = geom_nom.plate_lz / 2.0
h1 = geom_nom.storey_height + z0
h2 = 2.0 * geom_nom.storey_height + z0
h3 = 3.0 * geom_nom.storey_height + z0

# pre-build DOF vectors (don't change with x)
from reduced_model_semirigid import point_to_dof_vector
def _dof(pt, dirn, g=geom_nom):
    return point_to_dof_vector(pt, dirn, g)

B_col = _dof((cx, cy, z0), 'Y', geom_nom)   # base input
C_fl3 = _dof((cx, cy, h3), 'Y', geom_nom)
C_fl2 = _dof((cx, cy, h2), 'Y', geom_nom)
C_fl1 = _dof((cx, cy, h1), 'Y', geom_nom)

def geom_from_x(x):
    geom = BuildingGeometry(
        joint_stiffness_ratio = 10.0 ** x[0],
        damping               = x[5],
        screw_mass_per_joint  = 0.0,
        base_extra_mass       = x[1],
    )
    cf1, cf2, cf3 = x[2], x[3], x[4]
    geom.column_factor = np.array([[cf1]*4, [cf2]*4, [cf3]*4])
    return geom

OMEGA = 2.0 * np.pi * freq_fit
OMEGA2 = OMEGA ** 2

def fast_frf(geom):
    """Compute (N_fit, 3) accelerance FRF for the 3 Y-floor outputs."""
    K  = stiffness_matrix(geom)
    M_ = mass_matrix(geom)
    w2, V = eigh(K, M_)
    w2 = np.clip(w2, 0.0, None)
    wn = np.sqrt(w2)    # rad/s
    z  = geom.damping

    # modal participation
    b = V.T @ B_col   # (n_dof,)
    c3 = V.T @ C_fl3
    c2 = V.T @ C_fl2
    c1 = V.T @ C_fl1

    is_el = wn > 1e-3 * 2 * np.pi
    H = np.zeros((len(OMEGA), 3), dtype=complex)
    for r in np.where(is_el)[0]:
        den = (wn[r]**2 - OMEGA2) + 2j * z * wn[r] * OMEGA
        ker = -OMEGA2 / den    # (N_fit,)
        H[:, 0] += c3[r] * b[r] * ker
        H[:, 1] += c2[r] * b[r] * ker
        H[:, 2] += c1[r] * b[r] * ker

    return np.abs(H) * 1e3   # m/s²/N → mm/s²/N

W_FRF  = 1.0
W_FREQ = 8.0

_call_count = [0]
_t0 = [time.time()]

def objective(x):
    _call_count[0] += 1
    if x[1] < 0.0 or x[5] < 5e-4 or any(xi < 0.3 for xi in x[2:5]):
        return 1e9
    geom = geom_from_x(x)
    try:
        H_mod = fast_frf(geom)
    except Exception:
        return 1e9
    logH_mod = np.log(H_mod + 1e-12)
    L_frf = W_FRF * np.mean((logH_mod - logH_exp) ** 2)

    # frequency penalty
    try:
        K  = stiffness_matrix(geom)
        M_ = mass_matrix(geom)
        w2, _ = eigh(K, M_)
        f_all = np.sqrt(np.clip(w2, 0, None)) / (2*np.pi)
        f_el  = np.sort(f_all[f_all > 0.5])[:3]
        if len(f_el) == 3:
            L_freq = W_FREQ * np.sum(((f_el - f_res) / f_res) ** 2)
        else:
            L_freq = 1e4
    except Exception:
        L_freq = 1e4

    return float(L_frf + L_freq)


bounds = [
    (0.30, 1.50),   # log10(JSR)
    (0.0,  30.0),   # base_extra_mass
    (0.5,  3.0),    # cf_s1
    (0.5,  3.0),    # cf_s2
    (0.5,  3.0),    # cf_s3
    (0.001, 0.05),  # damping
]

x0 = np.array([np.log10(7.9), 4.0, 1.0, 1.0, 1.0, 0.005])
c0 = objective(x0)
print(f"Initial cost: {c0:.4f}")
print(f"Starting DE (popsize=12, maxiter=100) …")

t0 = time.time()
res_de = differential_evolution(
    objective, bounds, seed=42,
    maxiter=100, popsize=12, tol=1e-6,
    mutation=(0.5, 1.5), recombination=0.9,
    workers=1, disp=True, polish=False,
    updating='deferred',
)
print(f"DE done in {time.time()-t0:.1f}s  cost={res_de.fun:.6f}")

x1 = res_de.x
print(f"  JSR={10**x1[0]:.3f}  m_xtra={x1[1]:.3f} kg  "
      f"cf=[{x1[2]:.3f},{x1[3]:.3f},{x1[4]:.3f}]  damp={x1[5]:.5f}")

print(f"\nNelder-Mead polish …")
t1 = time.time()
res_nm = minimize(objective, x1, method='Nelder-Mead',
                  options={'maxiter': 20000, 'xatol': 1e-9, 'fatol': 1e-9,
                           'adaptive': True, 'disp': True})
print(f"NM done in {time.time()-t1:.1f}s  cost={res_nm.fun:.6f}")

x_opt = res_nm.x
geom_opt = geom_from_x(x_opt)
K  = stiffness_matrix(geom_opt)
M_ = mass_matrix(geom_opt)
w2, V = eigh(K, M_)
f_all = np.sqrt(np.clip(w2, 0, None)) / (2*np.pi)
f_el  = np.sort(f_all[f_all > 0.5])

PNAMES = ['log10_jsr', 'base_extra_mass_kg', 'cf_s1', 'cf_s2', 'cf_s3', 'damping']
print(f"\n{'='*55}")
print(f"OPTIMAL PARAMETERS:")
for nm, v in zip(PNAMES, x_opt):
    print(f"  {nm:30s} = {v:.6f}")
print(f"\nDerived:")
print(f"  JSR             = {10**x_opt[0]:.4f}")
print(f"  base_extra_mass = {x_opt[1]:.4f} kg")
for s, cf in enumerate(x_opt[2:5], 1):
    print(f"  story {s}: cf={cf:.4f}  k_rel={cf**4:.4f}")

print(f"\nFREQUENCY COMPARISON (first {min(6,len(f_el))} elastic modes):")
print(f"  {'Mode':>4}  {'f_mod':>10}")
for j, fj in enumerate(f_el[:6]):
    tag = ""
    if j < 3:
        err = (fj - f_res[j]) / f_res[j] * 100
        tag = f"  exp={f_res[j]:.3f}  err={err:.2f}%"
    print(f"  {j+1:>4}  {fj:>10.4f}{tag}")

# mode shapes
print("\nY-mode shapes (fl3, fl2, fl1) normalised to fl3:")
for j in range(min(3, len(f_el))):
    idx_ev = np.where(f_all > 0.5)[0][j]
    v = V[:, idx_ev]
    fl3, fl2, fl1 = v[8], v[5], v[2]
    ref = fl3 if abs(fl3) > 1e-12 else 1.0
    print(f"  Mode {j+1}: [{fl3/ref:+.4f}, {fl2/ref:+.4f}, {fl1/ref:+.4f}]")

out = _HERE / "calibration_result_frf.npz"
np.savez(out, x_opt=x_opt, f_mod=f_el[:3], f_res=f_res,
         param_names=np.array(PNAMES))
print(f"\nSaved → {out}")
print(f"Total calls: {_call_count[0]}  elapsed: {time.time()-_t0[0]:.1f}s")
