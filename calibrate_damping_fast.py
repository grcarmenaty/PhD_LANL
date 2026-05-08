"""Fit per-elastic-mode damping ζ_r against the experimental Pristine median.

Uses scipy.optimize.minimize on the log-magnitude FRF residual averaged
across the 9 sensors over 5–80 Hz.  Because the eigenvalue decomposition
returns Y, X, and torsion modes interleaved by frequency (the 3SBB has
3 Y modes plus X/θ modes between them), per-mode damping must be
optimised in mode-index order rather than in Y-mode order.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import minimize

from reduced_model_semirigid import compute_frf_matrix, modes
from model_3sbb import _input_position, _sensor_positions
from damage_scenarios import _pristine_geom

_HERE  = Path(__file__).parent.resolve()
EXP_H5 = _HERE / 'median_frfs.h5'

# Frequency band for the fit
F_LO, F_HI = 5.0, 80.0
# Number of elastic modes to assign individual damping ratios to
N_MODES = 9


def main():
    with h5py.File(EXP_H5, 'r') as f:
        cn = [c.decode() for c in f['case_names'][:]]
        H_exp = f['median_frf'][cn.index('Pristine')]
        f_exp = f['freq'][:]

    mask    = (f_exp >= F_LO) & (f_exp <= F_HI)
    f_fit   = f_exp[mask]
    H_target = np.abs(H_exp[mask])                    # (N_fit, 9)
    log_target = np.log10(H_target + 1e-12)

    g       = _pristine_geom()
    inputs  = _input_position(g)
    outputs = _sensor_positions(g)
    f_modes, _, _ = modes(g)
    print(f'Elastic mode frequencies: {f_modes[1:N_MODES+1]}')

    cal_old  = np.load(_HERE / 'calibration_result.npz')
    zeta_old = (cal_old['damping_modes']
                if 'damping_modes' in cal_old.files
                else np.full(N_MODES, float(cal_old['damping'])))
    zeta_old = np.asarray(zeta_old, dtype=float)
    if zeta_old.size < N_MODES:
        zeta_old = np.concatenate([zeta_old,
                                    np.full(N_MODES - zeta_old.size, zeta_old[-1])])

    def obj(log10_zeta):
        zeta = 10.0 ** log10_zeta
        if np.any(zeta < 1e-4) or np.any(zeta > 0.30):
            return 1e9
        H = compute_frf_matrix(f_fit, inputs, outputs, g, damping=zeta)[:, :, 0]
        log_mod = np.log10(np.abs(H) + 1e-12)
        return float(np.mean((log_mod - log_target) ** 2))

    x0  = np.log10(np.clip(zeta_old[:N_MODES], 5e-4, 0.2))
    print(f'Initial log-FRF MSE = {obj(x0):.4f}')
    res = minimize(obj, x0, method='Nelder-Mead',
                   options={'xatol': 1e-4, 'fatol': 1e-5,
                            'maxiter': 4000, 'adaptive': True, 'disp': True})
    zeta_new = 10.0 ** res.x

    print(f'\nOptimised damping (per elastic mode, lowest first):')
    for i in range(N_MODES):
        print(f'  mode {i:2d}  f≈{f_modes[i+1]:6.2f} Hz  '
              f'ζ {zeta_old[i]:.4f} -> {zeta_new[i]:.4f}')
    print(f'\nFinal log-FRF MSE = {res.fun:.4f}')

    out = {k: cal_old[k] for k in cal_old.files}
    out['damping_modes'] = zeta_new
    out['damping']       = float(zeta_new[0])
    np.savez(_HERE / 'calibration_result.npz', **out)
    print('Saved -> calibration_result.npz')


if __name__ == '__main__':
    main()
