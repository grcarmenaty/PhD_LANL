"""SCI-direct calibration of the LANL 3SBB reduced-order model.

Optimisation target: **mean SCI between experimental and synthetic
CFDAC matrices over a representative subset of IQS cases.**  CFDAC and
SCI are amplitude-invariant (CFDAC normalises by `|H(f_i)|·|H(f_j)|`),
so this objective is sensitive to *mode shapes and resonance
frequencies* but indifferent to overall accelerance level.  Damping
therefore drops out of the calibration entirely.

Free parameters
---------------
log10(JSR)         — joint stiffness ratio
base_extra_mass    — kg, shaker / fixture mass on the base plate
cf_s1, cf_s2, cf_s3 — per-storey column scale factors

Bounds
------
JSR ∈ [3, 32], base_extra_mass ∈ [0, 8] kg, cf_si ∈ [0.7, 1.6].

Objective
---------
The optimiser maximises the **mean SCI** over the four "anchor" cases
listed below.  Each anchor case has a corresponding synthetic
geometry built by ``damage_scenarios.geometry_for_case``; the per-mode
damping is left at its current calibrated value (it doesn't matter for
SCI but keeps the model output sane for the FRF section).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import minimize

from reduced_model_semirigid import compute_frf_matrix
from model_3sbb import _input_position, _sensor_positions
from damage_scenarios import _pristine_geom

_HERE  = Path(__file__).parent.resolve()
EXP_H5 = _HERE / 'median_frfs.h5'

ANCHOR_CASES = [
    'Pristine',
    'D(11%) 1BD',
    'Damage (85%) 1BD',
    'Mass First Floor',
]
F_LO, F_HI = 5.0, 80.0
DF_CFDAC   = 1.0


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


def main():
    # Load experimental anchor CFDACs
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
        Hb = H_exp_all[cn.index(nm)][band_idx]            # (Nb, 9)
        exp_cfdacs.append(_cfdac(Hb))

    # Calibration variables
    cal0 = np.load(_HERE / 'calibration_result.npz')
    zeta_old = (cal0['damping_modes']
                if 'damping_modes' in cal0.files
                else np.full(9, float(cal0['damping'])))

    # We will rebuild the geometry from scratch for each anchor case using
    # damage_scenarios.geometry_for_case, then *override* the calibrated
    # parameters with the trial vector x.  This way the per-case damage
    # markup (column factors, plate masses) is preserved.
    from damage_scenarios import geometry_for_case
    inputs  = _input_position(_pristine_geom())
    outputs = _sensor_positions(_pristine_geom())

    # Only need synthetic FRFs at the band frequencies — much faster than
    # the full 1601-point grid used for plotting.
    f_grid = band_freqs.astype(float)

    def case_geom_with_x(name, x):
        g = geometry_for_case(name)
        g.joint_stiffness_ratio = 10.0 ** x[0]
        g.base_extra_mass = float(x[1])
        # Multiplicative scaling: preserve the per-storey damage mark-up
        # already applied by damage_scenarios.geometry_for_case.
        scales = np.array([x[2], x[3], x[4]])
        # Reset to "pristine cf × damage scale" pattern: divide out the
        # pristine cf used by damage_scenarios, multiply by trial cf.
        cf_pristine = np.array([float(cal0['cf_s1']),
                                 float(cal0['cf_s2']),
                                 float(cal0['cf_s3'])])
        for s in range(3):
            g.column_factor[s, :] *= scales[s] / cf_pristine[s]
        return g

    def obj(x):
        if any(np.isnan(x)):
            return 1e9
        sci_sum = 0.0
        for k, name in enumerate(ANCHOR_CASES):
            g = case_geom_with_x(name, x)
            try:
                H = compute_frf_matrix(f_grid, inputs, outputs, g,
                                       damping=zeta_old)[:, :, 0]
            except Exception:
                return 1e9
            sci_sum += _sci(exp_cfdacs[k], _cfdac(H))
        return -sci_sum / len(ANCHOR_CASES)

    x0 = np.array([
        np.log10(float(cal0['jsr'])),
        float(cal0['base_extra_mass']),
        float(cal0['cf_s1']), float(cal0['cf_s2']), float(cal0['cf_s3']),
    ])
    print(f'Initial -mean SCI = {obj(x0):.4f}')

    bounds = [(0.5, 1.5), (0.0, 8.0),
              (0.7, 1.6), (0.7, 1.6), (0.7, 1.6)]

    rng = np.random.default_rng(11)
    best = None
    for trial in range(8):
        x_init = (x0.copy() if trial == 0
                  else np.clip(x0 + rng.uniform(-0.2, 0.2, x0.size)
                                * np.array([b[1] - b[0] for b in bounds]),
                                [b[0] for b in bounds],
                                [b[1] for b in bounds]))
        r = minimize(obj, x_init, method='L-BFGS-B', bounds=bounds,
                     options={'maxiter': 200, 'ftol': 1e-9})
        score = -r.fun
        print(f'  trial {trial}: mean SCI = {score:.4f}'
              + ('  (new best)' if best is None or r.fun < best.fun else ''))
        if best is None or r.fun < best.fun:
            best = r

    res = best
    print(f'\nBest mean SCI on anchors: {-res.fun:.4f}')
    print(f'  log10(JSR) = {res.x[0]:.4f}  -> JSR = {10**res.x[0]:.3f}')
    print(f'  base_extra_mass = {res.x[1]:.4f}')
    print(f'  cf = [{res.x[2]:.4f}, {res.x[3]:.4f}, {res.x[4]:.4f}]')

    out = {k: cal0[k] for k in cal0.files}
    out['jsr']             = 10 ** res.x[0]
    out['base_extra_mass'] = res.x[1]
    out['cf_s1']           = res.x[2]
    out['cf_s2']           = res.x[3]
    out['cf_s3']           = res.x[4]
    np.savez(_HERE / 'calibration_result.npz', **out)
    print(f'Saved -> calibration_result.npz')


if __name__ == '__main__':
    main()
