"""Regenerate ``synthetic_frfs.h5`` 1:1 against ``median_frfs.h5``.

For every experimental case name, the script builds a calibrated
geometry via ``damage_scenarios.geometry_for_case`` and computes the
9-sensor accelerance FRF.  The synthetic file therefore stores the same
case names in the same order as the experimental file, eliminating any
matching ambiguity in downstream analysis.

Output metadata
---------------
attrs['units']   = '(m/s^2)/N'
attrs['model']   = '3SBB reduced-order shear-building (semi-rigid joints)'
attrs['sensors'] = 'S2,S5,S6,S7,S8,S11,S12,S13,S14 (Y-direction)'
attrs['input']   = 'Base plate, mid -Y face, Y-direction (shaker)'

The shaker waveform (sine sweep at LANL) does not enter the FRF
because the FRF is a transfer function — only the input location and
direction matter, both of which are fixed.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from damage_scenarios import geometry_for_case
from model_3sbb import FREQ_ARRAY, _input_position, _sensor_positions
from reduced_model_semirigid import compute_frf_matrix

_HERE = Path(__file__).parent.resolve()
EXP_H5 = _HERE / 'median_frfs.h5'
OUT    = _HERE / 'synthetic_frfs.h5'


def main():
    with h5py.File(EXP_H5, 'r') as f:
        case_names = [c.decode() for c in f['case_names'][:]]

    n_cases = len(case_names)
    n_freq  = len(FREQ_ARRAY)
    n_sens  = 9
    H_all   = np.zeros((n_cases, n_freq, n_sens), dtype=np.complex128)

    for i, name in enumerate(case_names):
        geom    = geometry_for_case(name)
        damping = getattr(geom, 'damping_modes', None)
        if damping is None or np.size(damping) == 0:
            damping = geom.damping
        H = compute_frf_matrix(FREQ_ARRAY,
                               _input_position(geom),
                               _sensor_positions(geom),
                               geom, damping=damping)
        H_all[i] = H[:, :, 0]
        if i % 10 == 0 or i == n_cases - 1:
            print(f'  [{i+1:2d}/{n_cases}] {name}')

    with h5py.File(OUT, 'w') as f:
        f.create_dataset('frfs',       data=H_all)
        f.create_dataset('freqs',      data=FREQ_ARRAY.astype(np.float32))
        f.create_dataset('case_names',
                         data=np.array([n.encode() for n in case_names]))
        f.attrs['units']   = '(m/s^2)/N'
        f.attrs['model']   = '3SBB reduced-order shear-building (semi-rigid joints)'
        f.attrs['sensors'] = 'S2,S5,S6,S7,S8,S11,S12,S13,S14 (Y-direction)'
        f.attrs['input']   = 'Base plate, mid -Y face, Y-direction (shaker)'
    print(f'\nWrote {OUT}  ({n_cases} cases × {n_freq} freqs × {n_sens} sensors)')


if __name__ == '__main__':
    main()
