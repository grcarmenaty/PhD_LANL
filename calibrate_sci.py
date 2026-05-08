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
W_FREQ     = 12.0                               # frequency-penalty weight

# Single-band high-frequency-rise penalty.  The dual-band attempt with
# a heavier weight on 95-115 Hz did not improve overall SCI: a tuned
# attachment in set 2 always introduces an anti-resonance just below
# its peak that re-disturbs the high-frequency region it was supposed
# to fill.  Keep set 2 in the model (so it can be re-enabled when a
# better sub-model is added) but only run a single 78-115 Hz penalty
# at modest weight.
F_RISE1_LO, F_RISE1_HI = 78.0, 115.0
F_RISE2_LO, F_RISE2_HI = 95.0, 115.0   # unused now
RISE_CHANNELS = [1, 5]                  # S5, S11 (floor 3 +Y face)
W_RISE1 = 0.12
W_RISE2 = 0.0                          # disabled

# Free-parameter layout
#   x[0]    log10(JSR)
#   x[1]    base_extra_mass (kg)
#   x[2..4] plate_extra_mass for floors 1, 2, 3
#   x[5..7] cf_s1, cf_s2, cf_s3
#   x[8]    plate_flex_freq_hz   (set 1, shared)
#   x[9..11] plate_flex_mass  per floor 1, 2, 3  (kg)
#   x[12]   plate_flex2_freq_hz  (set 2, higher resonance)
#   x[13..15] plate_flex2_mass per floor 1, 2, 3 (kg)


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

    # Pristine experimental |H| in the two rise windows.
    pristine_idx = cn.index('Pristine')
    rise_mask1 = (f_exp >= F_RISE1_LO) & (f_exp <= F_RISE1_HI)
    f_rise1    = f_exp[rise_mask1]
    log_exp_rise1 = np.log10(np.abs(H_exp_all[pristine_idx][rise_mask1][:,
                               RISE_CHANNELS]) + 1e-12)
    rise_mask2 = (f_exp >= F_RISE2_LO) & (f_exp <= F_RISE2_HI)
    f_rise2    = f_exp[rise_mask2]
    log_exp_rise2 = np.log10(np.abs(H_exp_all[pristine_idx][rise_mask2][:,
                               RISE_CHANNELS]) + 1e-12)
    print(f'Rise band 1: {F_RISE1_LO}-{F_RISE1_HI} Hz, w={W_RISE1}, '
          f'{rise_mask1.sum()} freqs')
    print(f'Rise band 2: {F_RISE2_LO}-{F_RISE2_HI} Hz, w={W_RISE2}, '
          f'{rise_mask2.sum()} freqs')

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
        g.plate_flex_freq_hz = float(x[8])
        g.plate_flex_mass_per_floor = np.array(
            [float(x[9]), float(x[10]), float(x[11])], dtype=float)
        g.plate_flex2_freq_hz = float(x[12])
        g.plate_flex2_mass_per_floor = np.array(
            [float(x[13]), float(x[14]), float(x[15])], dtype=float)
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

        # Two-band high-frequency rise penalty.  Each band penalises
        # log-shortfall (model below exp) on the floor-3 channels.
        try:
            H_r1 = compute_frf_matrix(f_rise1, inputs, outputs, g_pristine,
                                       damping=zeta_old)[:, :, 0]
            H_r2 = compute_frf_matrix(f_rise2, inputs, outputs, g_pristine,
                                       damping=zeta_old)[:, :, 0]
        except Exception:
            return 1e9
        sh1 = np.maximum(log_exp_rise1
                         - np.log10(np.abs(H_r1[:, RISE_CHANNELS]) + 1e-12),
                         0.0)
        sh2 = np.maximum(log_exp_rise2
                         - np.log10(np.abs(H_r2[:, RISE_CHANNELS]) + 1e-12),
                         0.0)
        L_rise1 = float(np.mean(sh1 ** 2))
        L_rise2 = float(np.mean(sh2 ** 2))

        return -mean_sci + W_FREQ * L_freq + W_RISE1 * L_rise1 + W_RISE2 * L_rise2

    x0 = np.array([
        np.log10(float(cal0['jsr'])),
        float(cal0['base_extra_mass']),
        float(cal0.get('plate_extra_mass_fl1', 0.0))
            if 'plate_extra_mass_fl1' in cal0.files else 0.0,
        float(cal0.get('plate_extra_mass_fl2', 0.0))
            if 'plate_extra_mass_fl2' in cal0.files else 0.0,
        float(cal0.get('plate_extra_mass_fl3', 0.0))
            if 'plate_extra_mass_fl3' in cal0.files else 0.0,
        float(cal0['cf_s1']), float(cal0['cf_s2']), float(cal0['cf_s3']),
        float(cal0['plate_flex_freq_hz'])
            if 'plate_flex_freq_hz' in cal0.files else 100.0,
        # per-floor flex masses (set 1)
        float(cal0['plate_flex_mass_fl1'])
            if 'plate_flex_mass_fl1' in cal0.files else 0.0,
        float(cal0['plate_flex_mass_fl2'])
            if 'plate_flex_mass_fl2' in cal0.files else 0.0,
        float(cal0['plate_flex_mass_fl3'])
            if 'plate_flex_mass_fl3' in cal0.files else 0.5,
        # set 2: higher frequency
        float(cal0['plate_flex2_freq_hz'])
            if 'plate_flex2_freq_hz' in cal0.files else 110.0,
        float(cal0['plate_flex2_mass_fl1'])
            if 'plate_flex2_mass_fl1' in cal0.files else 0.0,
        float(cal0['plate_flex2_mass_fl2'])
            if 'plate_flex2_mass_fl2' in cal0.files else 0.0,
        float(cal0['plate_flex2_mass_fl3'])
            if 'plate_flex2_mass_fl3' in cal0.files else 0.5,
    ])
    print(f'Initial cost: {obj(x0):.5f}')
    f3_init = _y_dominant_3(case_geom('Pristine', x0))
    print(f'Initial Y-modes: {f3_init}')

    bounds = [(0.5, 1.5), (0.0, 15.0),
              (0.0, 8.0), (0.0, 8.0), (0.0, 8.0),
              (0.7, 2.0), (0.7, 2.0), (0.7, 2.0),
              (60.0, 95.0),                              # 1st flex freq
              (0.0, 3.0), (0.0, 3.0), (0.0, 3.0),        # 1st flex masses
              (95.0, 160.0),                             # 2nd flex freq
              (0.0, 3.0), (0.0, 3.0), (0.0, 3.0)]        # 2nd flex masses

    # Multi-start strategy.  Several seeds deliberately put non-zero mass on
    # set 2 with f_flex2 ~ 115 Hz so the optimiser explores using set 2 to
    # capture the experimental rise *past* 95 Hz.  A tuned attachment makes
    # a peak just above its anti-resonance, so seeding ω_flex2 ≈ 115 Hz
    # places the rising left flank of that peak inside the 95-110 Hz band.
    seeds = [x0.copy()]
    s1 = x0.copy(); s1[12] = 115.0; s1[15] = 1.5
    seeds.append(s1)
    s2 = x0.copy(); s2[12] = 130.0; s2[14] = 0.6; s2[15] = 1.5
    seeds.append(s2)
    s3 = x0.copy(); s3[8]  = 80.0;  s3[11] = 1.5
    s3[12] = 105.0; s3[15] = 1.0
    seeds.append(s3)
    rng = np.random.default_rng(2025)
    while len(seeds) < 10:
        pert = rng.uniform(-0.2, 0.2, x0.size) * np.array(
            [b[1] - b[0] for b in bounds])
        seeds.append(np.clip(x0 + pert,
                              [b[0] for b in bounds],
                              [b[1] for b in bounds]))

    best = None
    for trial, x_init in enumerate(seeds):
        r = minimize(obj, x_init, method='L-BFGS-B', bounds=bounds,
                     options={'maxiter': 400, 'ftol': 1e-10})
        f3 = _y_dominant_3(case_geom('Pristine', r.x))
        L_freq = float(np.sum(((f3 - F_REF) / F_REF) ** 2))
        mean_sci_pure = -r.fun + W_FREQ * L_freq
        flex2_active = bool(any(r.x[13:16] > 0.05))
        print(f'  trial {trial}: cost={r.fun:.4f}  mean_SCI≈{mean_sci_pure:.4f}'
              f'  f={f3.round(2)}  flex2={flex2_active}')
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
    print(f'  plate_flex set1: freq = {res.x[8]:.2f} Hz   '
          f'masses fl1/2/3 = {res.x[9]:.3f}/{res.x[10]:.3f}/{res.x[11]:.3f} kg')
    print(f'  plate_flex set2: freq = {res.x[12]:.2f} Hz   '
          f'masses fl1/2/3 = {res.x[13]:.3f}/{res.x[14]:.3f}/{res.x[15]:.3f} kg')

    out = {k: cal0[k] for k in cal0.files}
    out['jsr']             = 10 ** res.x[0]
    out['base_extra_mass'] = res.x[1]
    out['plate_extra_mass_fl1'] = res.x[2]
    out['plate_extra_mass_fl2'] = res.x[3]
    out['plate_extra_mass_fl3'] = res.x[4]
    out['cf_s1']           = res.x[5]
    out['cf_s2']           = res.x[6]
    out['cf_s3']           = res.x[7]
    out['plate_flex_freq_hz'] = res.x[8]
    out['plate_flex_mass_fl1'] = res.x[9]
    out['plate_flex_mass_fl2'] = res.x[10]
    out['plate_flex_mass_fl3'] = res.x[11]
    out['plate_flex2_freq_hz'] = res.x[12]
    out['plate_flex2_mass_fl1'] = res.x[13]
    out['plate_flex2_mass_fl2'] = res.x[14]
    out['plate_flex2_mass_fl3'] = res.x[15]
    out['plate_flex_mass'] = float(max(res.x[9], res.x[10], res.x[11]))
    np.savez(_HERE / 'calibration_result.npz', **out)
    print('Saved -> calibration_result.npz')


if __name__ == '__main__':
    main()
