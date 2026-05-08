"""Per-mode damping fit to match experimental peak amplitudes.

Run AFTER ``calibrate_sci.py``.  SCI is amplitude-invariant so this pass
does not affect the SCI scoreboard, but it makes the FRF view in the
notebook look right (peak heights in `|H|` panel agree with experiment).

Procedure
---------
1. Compute model FRF on Pristine geometry with damping=0.02 (uniform).
2. For each Y-dominant mode, find the model peak amplitude and the
   corresponding experimental peak.  Damping ratio at peak satisfies
   `|H_peak| ∝ 1 / ζ_r`, so update is `ζ_r ← ζ_r · (|H_mod| / |H_exp|)`.
3. Iterate twice for self-consistency.
4. Bound ζ to `[0.005, 0.06]` to keep physics sensible.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from scipy.signal import find_peaks

from reduced_model_semirigid import compute_frf_matrix, modes
from model_3sbb import _input_position, _sensor_positions
from damage_scenarios import _pristine_geom

_HERE  = Path(__file__).parent.resolve()
EXP_H5 = _HERE / 'median_frfs.h5'
FLOOR_CHS = [1, 2, 3, 5, 6, 7]    # S5..S13 floor sensors


def _y_dominant_indices(geom, n=3):
    """Return (elastic-mode indices, frequencies) of the *n* Y-dominant modes.

    The damping array passed to ``compute_frf_matrix`` is indexed over
    elastic modes only (rigid modes are excluded), so we have to subtract
    the rigid-mode count from full-eigenvalue indices.
    """
    freqs, V, _ = modes(geom)
    n_rigid = int((freqs < 1e-3).sum())
    score = (V[0] ** 2).copy()
    score[freqs < 0.5] = -1.0
    top_full = np.argsort(score)[-n:]
    top_full = top_full[np.argsort(freqs[top_full])]
    elastic_idx = top_full - n_rigid           # convert to elastic-array index
    return elastic_idx, freqs[top_full]


def main():
    with h5py.File(EXP_H5, 'r') as f:
        cn = [c.decode() for c in f['case_names'][:]]
        H_exp = f['median_frf'][cn.index('Pristine')]
        f_exp = f['freq'][:]

    g = _pristine_geom()
    f_grid = np.linspace(0.0, 100.0, 4001)        # fine grid for sharp peaks
    cal_old = np.load(_HERE / 'calibration_result.npz')
    zeta = (cal_old['damping_modes']
            if 'damping_modes' in cal_old.files
            else np.full(9, 0.02)).astype(float).copy()
    zeta = np.where((zeta > 0.005) & (zeta < 0.06), zeta, 0.02)

    y_idx, y_freqs = _y_dominant_indices(g, 3)
    print(f'Y-dominant mode indices: {y_idx}')
    print(f'Y-dominant frequencies:  {y_freqs}')

    # Iterate twice for self-consistency
    for it in range(2):
        H_mod = compute_frf_matrix(f_grid, _input_position(g),
                                    _sensor_positions(g), g,
                                    damping=zeta)[:, :, 0]
        for j, fr in enumerate(y_freqs):
            ifx = np.abs(f_grid - fr) < 2.0
            iex = np.abs(f_exp - fr) < 2.0
            amp_mod = np.mean([np.abs(H_mod[ifx, ch]).max()
                               for ch in FLOOR_CHS])
            amp_exp = np.mean([np.abs(H_exp[iex, ch]).max()
                               for ch in FLOOR_CHS])
            ratio = amp_mod / amp_exp if amp_exp > 0 else 1.0
            zeta[y_idx[j]] = float(np.clip(zeta[y_idx[j]] * ratio, 0.005, 0.06))
            print(f'  iter {it} mode {j+1} (idx {y_idx[j]}): '
                  f'f={fr:.2f} amp_mod/exp={ratio:5.2f}  '
                  f'zeta -> {zeta[y_idx[j]]:.4f}')

    out = {k: cal_old[k] for k in cal_old.files}
    out['damping']       = float(zeta[y_idx[0]])
    out['damping_modes'] = zeta
    np.savez(_HERE / 'calibration_result.npz', **out)
    print(f'\nFinal damping_modes (first 6): {zeta[:6]}')
    print('Saved -> calibration_result.npz')


if __name__ == '__main__':
    main()
