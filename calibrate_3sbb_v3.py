"""Corrected 3SBB calibration (v3):
  - Targets Y-mode frequencies via shaker-participation ordering
  - Per-mode damping: measured values assigned to Y-modes by participation rank,
    default 1% for all other modes
  - FRF log-magnitude error over Y-floor channels 10-80 Hz

Optimisation variables (5):
  x[0]  log10(JSR)        [0.30, 1.50]
  x[1]  base_extra_mass   [0, 30] kg
  x[2]  cf_s1             [0.5, 3.0]
  x[3]  cf_s2             [0.5, 3.0]
  x[4]  cf_s3             [0.5, 3.0]
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.signal import find_peaks
import h5py

_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE))
from reduced_model_semirigid import (BuildingGeometry, modes,
                                      compute_frf_matrix, point_to_dof_vector)
import params as P

# ── Experimental data ────────────────────────────────────────────────────────
print("Loading experimental FRFs …", flush=True)
with h5py.File(_HERE / "experimental_frfs.h5", 'r') as f:
    freq_exp = np.array(f['freq'])
    data     = np.array(f['data'])
    labels   = np.array(f['labels'])

idx_p  = np.where(labels == 59)[0]       # label 59 = Pristine
H_raw  = data[idx_p]
H_mean = np.mean(np.abs(H_raw), axis=0)  # (1601, 9)
print(f"  Pristine: {len(idx_p)} measurements")

# Half-power damping for the 3 experimental Y-modes
def half_power_zeta(H, freq, pk_idx):
    half  = H[pk_idx] / np.sqrt(2)
    left  = np.where((freq < freq[pk_idx]) & (H < half))[0]
    right = np.where((freq > freq[pk_idx]) & (H < half))[0]
    if len(left) and len(right):
        return (freq[right[0]] - freq[left[-1]]) / (2 * freq[pk_idx])
    return 0.01

ch3 = H_mean[:, 3]  # S7 = fl1-Y, all 3 Y-modes visible
peaks, _ = find_peaks(ch3, distance=40, prominence=ch3.max()*0.05)
F_EXP  = freq_exp[peaks[:3]]
ZETA3  = np.array([half_power_zeta(ch3, freq_exp, pk) for pk in peaks[:3]])
print(f"Experimental Y-modes:  {F_EXP} Hz")
print(f"Experimental zeta:     {ZETA3*100} %")

ZETA_DEFAULT = ZETA3[1]   # ~0.6% — use for all non-Y modes

# ── Fit band ─────────────────────────────────────────────────────────────────
F_LO, F_HI = 10.0, 80.0
mask_fit  = (freq_exp >= F_LO) & (freq_exp <= F_HI)
freq_fit  = freq_exp[mask_fit]
FLOOR_Y_CHS = [1, 2, 3]
H_exp_y  = H_mean[mask_fit][:, FLOOR_Y_CHS]
logH_exp = np.log10(H_exp_y + 1e-12)

# ── Nominal geometry helpers ──────────────────────────────────────────────────
geom_nom = BuildingGeometry()
cx  = geom_nom.plate_lx / 2.0
z0  = geom_nom.plate_z_centre(0)
z1  = geom_nom.plate_z_centre(1)
z2  = geom_nom.plate_z_centre(2)
z3  = geom_nom.plate_z_centre(3)
PL  = geom_nom.plate_ly
xl  = geom_nom.col_lx / 2.0

INPUT_PT   = ((cx, 0.0, z0), 'Y')
OUTPUT_PTS = [
    ((xl, PL, z3), 'Y'),   # ch1 S5 fl3
    ((xl, PL, z2), 'Y'),   # ch2 S6 fl2
    ((xl, PL, z1), 'Y'),   # ch3 S7 fl1
]

def geom_from_x(x):
    g = BuildingGeometry(
        joint_stiffness_ratio = 10.0 ** x[0],
        damping               = float(ZETA3[0]),
        base_extra_mass       = float(x[1]),
    )
    g.column_factor = np.array([[x[2]]*4, [x[3]]*4, [x[4]]*4], dtype=float)
    return g

def participation_damping(geom):
    """Full 9-element damping array (elastic modes sorted by frequency).
    The 3 most-excited modes by base-Y shaker get ZETA3 values;
    all others get ZETA_DEFAULT.
    """
    freqs_all, V, _ = modes(geom)
    b    = point_to_dof_vector((cx, 0.0, z0), 'Y', geom)
    bphi = np.abs(V.T @ b)

    elastic     = freqs_all > 0.5
    n_el        = elastic.sum()
    ep_elastic  = bphi[elastic]           # participations for elastic modes

    # Identify Y-modes by highest participation, then sort by frequency
    # (avoids tie-breaking issues when two modes have equal participation)
    part_order    = np.argsort(-ep_elastic)
    y_mode_idxs   = np.sort(part_order[:3])   # frequency-ordered Y-mode indices

    zeta = np.full(n_el, ZETA_DEFAULT)
    for rank, idx in enumerate(y_mode_idxs):
        zeta[idx] = ZETA3[rank]    # assign in ascending frequency order
    return zeta

def y_mode_freqs(geom):
    freqs_all, V, _ = modes(geom)
    b    = point_to_dof_vector((cx, 0.0, z0), 'Y', geom)
    bphi = np.abs(V.T @ b)
    elastic = freqs_all > 0.5
    ef = freqs_all[elastic]
    ep = bphi[elastic]
    order = np.argsort(-ep)
    return np.sort(ef[order[:3]])

# ── Objective ─────────────────────────────────────────────────────────────────
W_FRF  = 1.0
W_FREQ = 8.0

def objective(x):
    if x[1] < 0.0 or any(xi < 0.3 for xi in x[2:5]):
        return 1e9
    geom = geom_from_x(x)
    try:
        zeta = participation_damping(geom)
        H_mod = compute_frf_matrix(
            freq_fit, inputs=[INPUT_PT], outputs=OUTPUT_PTS,
            geom=geom, damping=zeta,
        )
    except Exception:
        return 1e9

    logH_mod = np.log10(np.abs(H_mod[:, :, 0]) + 1e-12)
    L_frf  = W_FRF  * np.mean((logH_mod - logH_exp) ** 2)

    try:
        f_y3 = y_mode_freqs(geom)
        rel  = (f_y3 - F_EXP) / F_EXP
        L_freq = W_FREQ * np.sum(rel ** 2)
    except Exception:
        return 1e9

    return float(L_frf + L_freq)

# ── Bounds & initial guess ────────────────────────────────────────────────────
bounds = [
    (0.30, 1.50),
    (0.0,  30.0),
    (0.5,  3.0),
    (0.5,  3.0),
    (0.5,  3.0),
]
x0 = np.array([np.log10(7.9), 6.5, 1.23, 1.02, 1.14])
print(f"\nInitial cost: {objective(x0):.4f}")
print(f"Initial Y-mode freqs: {y_mode_freqs(geom_from_x(x0))}")

# ── DE ────────────────────────────────────────────────────────────────────────
print("\n── Stage 1: Differential evolution ─────────────────────────")
res_de = differential_evolution(
    objective, bounds, seed=42,
    maxiter=600, popsize=20, tol=1e-8,
    mutation=(0.5, 1.5), recombination=0.9,
    workers=1, disp=True, polish=False,
    updating='deferred', x0=x0,
)
x1 = res_de.x
print(f"\nDE cost={res_de.fun:.6f}  Y-modes={y_mode_freqs(geom_from_x(x1))}")

# ── Nelder-Mead polish ────────────────────────────────────────────────────────
print("\n── Stage 2: Nelder-Mead polish ──────────────────────────────")
res_nm = minimize(objective, x1, method='Nelder-Mead',
                  options={'maxiter':40000,'xatol':1e-9,'fatol':1e-9,
                           'adaptive':True,'disp':True})
x_opt = res_nm.x

# ── Report ────────────────────────────────────────────────────────────────────
geom_opt  = geom_from_x(x_opt)
f_y_opt   = y_mode_freqs(geom_opt)
zeta_full = participation_damping(geom_opt)
print(f"\nOptimal cost: {res_nm.fun:.6f}")
print(f"JSR={10**x_opt[0]:.4f}  m_extra={x_opt[1]:.3f} kg")
print(f"cf=[{x_opt[2]:.4f}, {x_opt[3]:.4f}, {x_opt[4]:.4f}]")
print(f"\nFREQUENCY COMPARISON:")
for j in range(3):
    err = (f_y_opt[j] - F_EXP[j]) / F_EXP[j] * 100
    print(f"  Y-mode {j+1}: exp={F_EXP[j]:.3f}  mod={f_y_opt[j]:.3f}  err={err:+.2f}%")
print(f"\nDAMPING (9 elastic modes, frequency order):")
for j, z in enumerate(zeta_full):
    print(f"  mode {j+1}: {z*100:.3f}%")

# ── Save ──────────────────────────────────────────────────────────────────────
out = _HERE / "calibration_result.npz"
np.savez(out,
         jsr            = 10**x_opt[0],
         base_extra_mass= x_opt[1],
         cf_s1=x_opt[2], cf_s2=x_opt[3], cf_s3=x_opt[4],
         damping        = float(zeta_full[0]),    # scalar fallback
         damping_modes  = zeta_full,              # 9-element array
         f_cal=f_y_opt, f_exp=F_EXP,
         phi_cal=np.eye(3), phi_exp=np.eye(3),   # placeholder
         param_names    = np.array(['jsr','base_extra_mass_kg',
                                    'cf_s1','cf_s2','cf_s3']),
         k_story1=0.0, k_story2=0.0, k_story3=0.0,  # recomputed after
)
print(f"\nSaved → {out}")
