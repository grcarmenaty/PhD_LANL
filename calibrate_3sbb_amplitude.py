"""Amplitude-aware calibration for the LANL 3SBB reduced-order model.

After the unit-handling fix in the notebook, the dominant residual between
model and experiment is per-mode amplitude.  This script extends
``calibrate_3sbb_frf.py`` by jointly fitting:

  * per-mode damping ratios  ζ_1, ζ_2, ζ_3
  * aluminium density (2680–2810 kg/m³ — covers 5052/6061/2024 alloys)
  * aluminium Young's modulus (68.9–73 GPa)
  * plate thickness (±5 % manufacturing tolerance)
  * base extra mass and per-story column factors
  * joint-stiffness ratio JSR

against the **log-magnitude FRF** measured at all 9 sensors over 5–80 Hz,
using the experimental Pristine median as the target.

Notes
-----
* The shaker excitation is a sine sweep, but FRF objectives are independent
  of the drive waveform — only input *location and direction* matter, and
  those are already encoded in ``model_3sbb._input_position``.  We therefore
  do NOT calibrate any excitation parameter.
* Damping is parameterised per-mode (3 ratios for the 3 Y-modes); higher
  modes inherit the last value.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import differential_evolution, minimize

_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE))

from reduced_model_semirigid import BuildingGeometry, compute_frf_matrix, modes
import params as P
from model_3sbb import _input_position, _sensor_positions

EXP_H5 = _HERE / "median_frfs.h5"
F_LO, F_HI = 5.0, 80.0
N_MODES_FIT = 3

# ── load experimental Pristine median ────────────────────────────────────────
with h5py.File(EXP_H5, 'r') as f:
    cn = [c.decode() for c in f['case_names'][:]]
    p = cn.index('Pristine')
    exp_freq = f['freq'][:]
    H_exp_all = f['median_frf'][p]              # (1601, 9), complex64

mask = (exp_freq >= F_LO) & (exp_freq <= F_HI)
freq_fit = exp_freq[mask]
H_exp = np.abs(H_exp_all[mask])                  # (N_fit, 9), magnitude
log_exp = np.log10(H_exp + 1e-12)

print(f"Fit band {F_LO}–{F_HI} Hz, {mask.sum()} points, 9 sensors")

# ── parameter vector layout ──────────────────────────────────────────────────
# 0   log10(JSR)
# 1   base_extra_mass (kg)
# 2-4 cf_s1, cf_s2, cf_s3   per-story column factor
# 5-7 zeta_1, zeta_2, zeta_3 per-mode damping
# 8   density (kg/m^3)
# 9   Young modulus (GPa, x1e9)
# 10  plate_lz (m)

PARAM_NAMES = [
    'log10_jsr', 'base_extra_mass_kg',
    'cf_s1', 'cf_s2', 'cf_s3',
    'zeta_1', 'zeta_2', 'zeta_3',
    'density_kgm3', 'young_GPa', 'plate_lz_m',
]

BOUNDS = [
    (0.30, 1.50),     # JSR ∈ [2, 31]
    (0.0,  20.0),     # base extra mass (kg)
    (0.5,  3.0),      # cf_s1
    (0.5,  3.0),      # cf_s2
    (0.5,  3.0),      # cf_s3
    (0.005, 0.08),    # zeta_1
    (0.005, 0.08),    # zeta_2
    (0.005, 0.08),    # zeta_3
    (2680.0, 2810.0), # alloy density window
    (68.9,  73.0),    # alloy modulus window (GPa)
    (0.024, 0.0265),  # plate thickness ±~5%
]


def geom_from_x(x):
    g = BuildingGeometry(
        joint_stiffness_ratio = 10.0 ** x[0],
        damping               = 0.01,
        screw_mass_per_joint  = 0.0,
        base_extra_mass       = x[1],
    )
    g.column_factor = np.array([[x[2]] * 4, [x[3]] * 4, [x[4]] * 4], dtype=float)
    g.density  = float(x[8])
    g.young    = float(x[9]) * 1e9
    g.plate_lz = float(x[10])
    return g


def model_FRF(geom, zetas):
    """Compute |H| at the 9 sensors for the supplied geometry and damping."""
    inputs  = _input_position(geom)
    outputs = _sensor_positions(geom)
    H = compute_frf_matrix(freq_fit, inputs, outputs, geom, damping=zetas)
    return np.abs(H[:, :, 0])                    # (N_fit, 9)


def objective(x):
    if any(np.isnan(x)):
        return 1e9
    zetas = x[5:8]
    geom = geom_from_x(x)
    try:
        H_mod = model_FRF(geom, zetas)
    except Exception:
        return 1e9

    log_mod = np.log10(H_mod + 1e-12)
    L_frf = np.mean((log_mod - log_exp) ** 2)

    # soft frequency penalty: keep first 3 modes near experiment
    f_ref = np.array([20.94, 49.94, 68.19])      # IQS pristine modes
    f_all = modes(geom)[0]
    f_el = np.sort(f_all[f_all > 0.5])[:3]
    if len(f_el) < 3:
        return 1e9
    L_freq = np.sum(((f_el - f_ref) / f_ref) ** 2)

    return float(L_frf + 0.5 * L_freq)


def main():
    # initial guess: load existing calibration
    cal_file = _HERE / 'calibration_result.npz'
    if cal_file.exists():
        cal = np.load(cal_file)
        x0 = np.array([
            np.log10(float(cal['jsr'])),
            float(cal['base_extra_mass']),
            float(cal['cf_s1']), float(cal['cf_s2']), float(cal['cf_s3']),
            float(cal['damping']), float(cal['damping']), float(cal['damping']),
            P.ALU_RHO, P.ALU_E / 1e9, P.PLATE_LZ,
        ])
    else:
        x0 = np.array([np.log10(7.9), 4.0, 1.0, 1.0, 1.0,
                       0.02, 0.02, 0.02, 2700.0, 68.9, 0.0254])
    print(f"\nInitial cost: {objective(x0):.6f}")

    print("\n── Stage 1: differential evolution ─────────────────────────────")
    res_de = differential_evolution(
        objective, BOUNDS, seed=42,
        maxiter=300, popsize=18, tol=1e-7,
        mutation=(0.5, 1.5), recombination=0.9,
        workers=1, disp=True, polish=False,
        updating='deferred', x0=x0,
    )
    print(f"\nDE cost = {res_de.fun:.6f}")

    print("\n── Stage 2: Nelder-Mead polish ─────────────────────────────────")
    res_nm = minimize(objective, res_de.x, method='Nelder-Mead',
                      options={'maxiter': 30000, 'xatol': 1e-8,
                               'fatol': 1e-9, 'adaptive': True, 'disp': True})
    x_opt = res_nm.x

    print("\nOptimised parameters:")
    for n, v in zip(PARAM_NAMES, x_opt):
        print(f"  {n:22s} = {v:.6g}")

    # save in a layout backwards-compatible with the existing pipeline
    np.savez(_HERE / 'calibration_result.npz',
             jsr             = 10.0 ** x_opt[0],
             base_extra_mass = x_opt[1],
             cf_s1           = x_opt[2],
             cf_s2           = x_opt[3],
             cf_s3           = x_opt[4],
             damping         = x_opt[5],
             damping_modes   = x_opt[5:8],
             density_kgm3    = x_opt[8],
             young_GPa       = x_opt[9],
             plate_lz_m      = x_opt[10],
             param_names     = np.array(PARAM_NAMES))
    print("\nSaved → calibration_result.npz")


if __name__ == '__main__':
    main()
