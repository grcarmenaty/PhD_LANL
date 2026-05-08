"""SCI-direct calibration v2 with frequency anchoring and per-plate mass.

Key fix vs v1: maximising raw SCI lets the optimiser drift the modal
frequencies away from the experimental peaks (mode 1 was driven from
21 Hz to 29 Hz in v1 because the CFDAC diagonal of 1's dominates the
Pearson metric and tolerates frequency translation).  v2 keeps SCI as
the primary objective but adds a soft penalty `W_FREQ * Σ ((f_r -
f_ref_r)/f_ref_r)^2` on the 3 Y-dominant mode frequencies.  Y modes
are identified by base-Y eigenvector amplitude rather than sort
order, so X / torsion modes between Y modes don't fool the optimiser.

New free parameters
-------------------
log10(JSR)              JSR ∈ [3, 32]
plate_extra_mass[0..3]  per-plate added mass, kg   ∈ [0, 8]
cf_s1, cf_s2, cf_s3                                 ∈ [0.7, 1.6]

The optimiser maximises mean SCI minus the frequency penalty over
six anchor cases (Pristine + a spread of damage and mass cases).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import minimize

from reduced_model_semirigid import compute_frf_matrix, modes
from model_3sbb import _input_position, _sensor_positions
from damage_scenarios import _pristine_geom, geometry_for_case

_HERE  = Path(__file__).parent.resolve()
EXP_H5 = _HERE / 'median_frfs.h5'

ANCHOR_CASES = [
    'Pristine',
    'D(11%) 1BD',
    'D(50%) 1BD',
    'Damage (85%) 1BD',
    'Mass First Floor',
    'Hole 4mm 1BD',
]
F_LO, F_HI = 5.0, 100.0
DF_CFDAC   = 1.0
F_REF      = np.array([20.94, 49.94, 68.19])    # exp Y-mode peaks (Pristine)
W_FREQ     = 4.0                                # frequency-penalty weight


def _band_indices(freq, f_lo=F_LO, f_hi=F_HI, step=DF_CFDAC):
    target = np.arange(f_lo, f_hi + 1e-9, step)
    return np.array([int(np.argmin(np.abs(freq - t))) for t in target])


def _cfdac(H):
    inner = H.conj() @ H.T
    d = np.real(np.diag(inner)).copy()
    d[d < 1e-30] = 1e-30
    return (np.abs(inner) ** 2) / np.outer(d, d)


def _sci(C1, C2):
    a = C1.ravel() - C1.mean()
    b = C2.ravel() - C2.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float((a @ b) ** 2 / d ** 2) if d > 0 else 0.0


def _y_dominant_3(geom):
    freqs, V, _ = modes(geom)
    score = (V[0] ** 2).copy()
    score[freqs < 0.5] = -1.0
    top3 = np.argsort(score)[-3:]
    return np.sort(freqs[top3])


def main():
    with h5py.File(EXP_H5, 'r') as f:
        cn = [c.decode() for c in f['case_names'][:]]
        H_exp_all = f['median_frf'][:]
        f_exp = f['freq'][:]

    band_idx = _band_indices(f_exp)
    band_freqs = f_exp[band_idx]
    print(f'CFDAC band: {band_freqs[0]:.0f}-{band_freqs[-1]:.0f} Hz, '
          f'{len(band_freqs)} freqs')

    exp_cfdacs = []
    for nm in ANCHOR_CASES:
        if nm not in cn:
            raise RuntimeError(f'anchor {nm!r} not in median_frfs.h5')
        Hb = H_exp_all[cn.index(nm)][band_idx]
        exp_cfdacs.append(_cfdac(Hb))

    cal0 = np.load(_HERE / 'calibration_result.npz')
    zeta_old = (cal0['damping_modes']
                if 'damping_modes' in cal0.files
                else np.full(9, float(cal0['damping'])))

    inputs  = _input_position(_pristine_geom())
    outputs = _sensor_positions(_pristine_geom())
    f_grid = band_freqs.astype(float)

    cf_pristine = np.array([float(cal0['cf_s1']),
                             float(cal0['cf_s2']),
                             float(cal0['cf_s3'])])

    def case_geom(name, x):
        g = geometry_for_case(name)
        g.joint_stiffness_ratio = 10.0 ** x[0]
        g.base_extra_mass       = float(x[1])
        for k in range(3):
            g.plate_extra_mass[k + 1] += float(x[2 + k])
        scales = np.array([x[5], x[6], x[7]])
        for s in range(3):
            g.column_factor[s, :] *= scales[s] / cf_pristine[s]
        return g

    def obj(x):
        if any(np.isnan(x)):
            return 1e9
        # Y-mode frequencies are case-independent (depend only on g_pristine
        # before damage).  Compute once with the trial pristine geometry.
        g_pristine = case_geom('Pristine', x)
        try:
            f3 = _y_dominant_3(g_pristine)
        except Exception:
            return 1e9
        L_freq = float(np.sum(((f3 - F_REF) / F_REF) ** 2))

        sci_sum = 0.0
        for k, name in enumerate(ANCHOR_CASES):
            g = case_geom(name, x)
            try:
                H = compute_frf_matrix(f_grid, inputs, outputs, g,
                                       damping=zeta_old)[:, :, 0]
            except Exception:
                return 1e9
            sci_sum += _sci(exp_cfdacs[k], _cfdac(H))
        mean_sci = sci_sum / len(ANCHOR_CASES)
        return -mean_sci + W_FREQ * L_freq

    x0 = np.array([
        np.log10(float(cal0['jsr'])),
        float(cal0['base_extra_mass']),
        0.0, 0.0, 0.0,                      # plate extras for fl1/2/3
        float(cal0['cf_s1']), float(cal0['cf_s2']), float(cal0['cf_s3']),
    ])
    print(f'Initial cost: {obj(x0):.5f}')
    f3_init = _y_dominant_3(case_geom('Pristine', x0))
    print(f'Initial Y-modes: {f3_init}')

    bounds = [(0.5, 1.5), (0.0, 15.0),
              (0.0, 8.0), (0.0, 8.0), (0.0, 8.0),
              (0.7, 2.0), (0.7, 2.0), (0.7, 2.0)]

    rng = np.random.default_rng(2025)
    best = None
    for trial in range(8):
        if trial == 0:
            x_init = x0.copy()
        else:
            pert = rng.uniform(-0.2, 0.2, x0.size) * np.array(
                [b[1] - b[0] for b in bounds])
            x_init = np.clip(x0 + pert,
                             [b[0] for b in bounds],
                             [b[1] for b in bounds])
        r = minimize(obj, x_init, method='L-BFGS-B', bounds=bounds,
                     options={'maxiter': 200, 'ftol': 1e-9})
        f3 = _y_dominant_3(case_geom('Pristine', r.x))
        # Approximate SCI part of the cost (cost = -mean_sci + W*L_freq):
        L_freq = float(np.sum(((f3 - F_REF) / F_REF) ** 2))
        mean_sci_pure = -r.fun + W_FREQ * L_freq
        print(f'  trial {trial}: cost={r.fun:.4f}  mean_SCI≈{mean_sci_pure:.4f}'
              f'  f={f3.round(2)}')
        if best is None or r.fun < best.fun:
            best = r

    res = best
    f3 = _y_dominant_3(case_geom('Pristine', res.x))
    print(f'\nBest cost: {res.fun:.4f}')
    print(f'Y-mode frequencies: {f3}  (target {F_REF})')
    print(f'  log10(JSR) = {res.x[0]:.3f}  -> JSR = {10**res.x[0]:.3f}')
    print(f'  base_extra_mass = {res.x[1]:.3f}')
    print(f'  plate_extra_mass = [_, {res.x[2]:.3f}, {res.x[3]:.3f}, {res.x[4]:.3f}]')
    print(f'  cf = [{res.x[5]:.3f}, {res.x[6]:.3f}, {res.x[7]:.3f}]')

    out = {k: cal0[k] for k in cal0.files}
    out['jsr']             = 10 ** res.x[0]
    out['base_extra_mass'] = res.x[1]
    out['plate_extra_mass_fl1'] = res.x[2]
    out['plate_extra_mass_fl2'] = res.x[3]
    out['plate_extra_mass_fl3'] = res.x[4]
    out['cf_s1']           = res.x[5]
    out['cf_s2']           = res.x[6]
    out['cf_s3']           = res.x[7]
    np.savez(_HERE / 'calibration_result.npz', **out)
    print('Saved -> calibration_result.npz')


if __name__ == '__main__':
    main()
