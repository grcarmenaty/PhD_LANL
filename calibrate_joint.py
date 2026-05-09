"""Joint calibration of cf_si, JSR, base mass and per-mode damping.

Optimises log-magnitude FRF on all 9 sensors over 5–80 Hz with:
  - bounded L-BFGS-B (Nelder-Mead would escape damping bounds and hide
    frequency errors behind over-damped peaks).
  - peak-weighted residuals (each frequency weighted by exp |H| so
    peaks count more than valleys).
  - hard penalty on the 3 Y-dominant mode frequencies (identified by
    base Y-translation eigenvector amplitude, not by sort order, so the
    X / torsion modes between Y modes don't fool the optimiser).
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

F_LO, F_HI   = 5.0, 80.0
N_DAMP_MODES = 9
F_REF        = np.array([20.94, 49.94, 68.19])      # exp Y-mode peaks
W_FREQ       = 50.0                                  # heavy penalty


def _y_dominant_3(geom):
    """Return the frequencies of the 3 modes with largest base-Y eigenvector
    component (DOF 0).  This robustly picks the Y modes regardless of the
    X / torsion modes interleaved between them.
    """
    freqs, V, _ = modes(geom)
    score = (V[0] ** 2).copy()
    score[freqs < 0.5] = -1.0     # exclude rigid-body
    top3 = np.argsort(score)[-3:]
    f3 = np.sort(freqs[top3])
    return f3


def main():
    with h5py.File(EXP_H5, 'r') as f:
        cn = [c.decode() for c in f['case_names'][:]]
        H_exp = f['median_frf'][cn.index('Pristine')]
        f_exp = f['freq'][:]

    mask  = (f_exp >= F_LO) & (f_exp <= F_HI)
    f_fit = f_exp[mask]
    H_fit = np.abs(H_exp[mask])                                  # (Nf, 9)
    log_target = np.log10(H_fit + 1e-12)
    # Per-frequency, per-sensor weight: emphasise peaks.
    w = (H_fit / H_fit.max(axis=0, keepdims=True)).clip(min=0.05)
    w = w / w.mean()

    cal0 = np.load(_HERE / 'calibration_result.npz')

    def x_to_geom_and_zeta(x):
        g = _pristine_geom()
        g.joint_stiffness_ratio = 10.0 ** x[0]
        g.base_extra_mass        = float(x[1])
        g.column_factor          = np.array([[x[2]] * 4,
                                              [x[3]] * 4,
                                              [x[4]] * 4], dtype=float)
        zeta = x[5:5 + N_DAMP_MODES]
        return g, zeta

    inputs  = _input_position(_pristine_geom())
    outputs = _sensor_positions(_pristine_geom())

    def obj(x):
        g, zeta = x_to_geom_and_zeta(x)
        try:
            f3 = _y_dominant_3(g)
            H = compute_frf_matrix(f_fit, inputs, outputs, g,
                                   damping=zeta)[:, :, 0]
        except Exception:
            return 1e9
        log_mod = np.log10(np.abs(H) + 1e-12)
        L_frf  = float(np.mean(w * (log_mod - log_target) ** 2))
        L_freq = float(np.sum(((f3 - F_REF) / F_REF) ** 2))
        return L_frf + W_FREQ * L_freq

    z0 = np.full(N_DAMP_MODES, 0.012)
    z0[0] = 0.04
    x0 = np.array([
        np.log10(float(cal0['jsr'])),
        float(cal0['base_extra_mass']),
        float(cal0['cf_s1']), float(cal0['cf_s2']), float(cal0['cf_s3']),
        *z0,
    ])
    print(f'Initial cost: {obj(x0):.5f}')
    print(f'Initial Y-modes: {_y_dominant_3(x_to_geom_and_zeta(x0)[0])}')

    bounds = ([(0.5, 1.5), (0.0, 8.0)]
              + [(0.7, 1.6)] * 3
              + [(0.005, 0.05)] * N_DAMP_MODES)

    rng  = np.random.default_rng(0)
    best = None
    for trial in range(6):
        if trial == 0:
            x_init = x0.copy()
        else:
            pert = rng.uniform(-0.15, 0.15, size=x0.size) * np.array(
                [b[1] - b[0] for b in bounds])
            x_init = np.clip(x0 + pert,
                             [b[0] for b in bounds],
                             [b[1] for b in bounds])
        r = minimize(obj, x_init, method='L-BFGS-B', bounds=bounds,
                     options={'maxiter': 600, 'ftol': 1e-10})
        f3 = _y_dominant_3(x_to_geom_and_zeta(r.x)[0])
        print(f'  trial {trial}: cost {r.fun:.5f}  Y-modes={f3}')
        if best is None or r.fun < best.fun:
            best = r

    res = best
    g, zeta = x_to_geom_and_zeta(res.x)
    f3 = _y_dominant_3(g)
    print(f'\nBest cost: {res.fun:.5f}')
    print(f'Y-mode frequencies: {f3}  (target {F_REF})')
    print(f'JSR = {g.joint_stiffness_ratio:.3f}, '
          f'base_mass = {g.base_extra_mass:.3f}, '
          f'cf = {g.column_factor[:, 0]}')
    print(f'damping_modes = {zeta}')

    out = {k: cal0[k] for k in cal0.files}
    out['jsr']             = g.joint_stiffness_ratio
    out['base_extra_mass'] = g.base_extra_mass
    out['cf_s1']           = g.column_factor[0, 0]
    out['cf_s2']           = g.column_factor[1, 0]
    out['cf_s3']           = g.column_factor[2, 0]
    out['damping']         = float(zeta[0])
    out['damping_modes']   = zeta
    np.savez(_HERE / 'calibration_result.npz', **out)
    print('Saved -> calibration_result.npz')


if __name__ == '__main__':
    main()
