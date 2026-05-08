"""Regenerate ``synthetic_frfs.h5`` from the calibrated reduced-order model.

The file is written with explicit metadata so downstream consumers (the
notebook in particular) do not have to guess units:

  attrs['units']   = '(m/s^2)/N'
  attrs['model']   = '3SBB reduced-order shear-building (semi-rigid joints)'
  attrs['sensors'] = 'S2,S5,S6,S7,S8,S11,S12,S13,S14 (Y-direction)'
  attrs['input']   = 'Base plate, mid -Y face, Y-direction (shaker)'

Why no excitation parameter?  An FRF is a transfer function, so the temporal
shape of the drive signal (sine sweep at LANL) does not enter the model —
only the input location and direction do, which are encoded in
``model_3sbb._input_position``.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from damage_scenarios import REPRESENTATIVE_CASES
from model_3sbb import (FREQ_ARRAY, _input_position, _sensor_positions)
from reduced_model_semirigid import compute_frf_matrix

OUT_FILE = Path(__file__).parent / 'synthetic_frfs.h5'


def main():
    case_names = list(REPRESENTATIVE_CASES.keys())
    n_cases = len(case_names)
    n_freq  = len(FREQ_ARRAY)
    n_sens  = 9

    H_all = np.zeros((n_cases, n_freq, n_sens), dtype=np.complex128)
    for i, (name, geom) in enumerate(REPRESENTATIVE_CASES.items()):
        damping = getattr(geom, 'damping_modes', None)
        if damping is None:
            damping = geom.damping
        inputs  = _input_position(geom)
        outputs = _sensor_positions(geom)
        H = compute_frf_matrix(FREQ_ARRAY, inputs, outputs, geom, damping=damping)
        H_all[i] = H[:, :, 0]
        print(f'  [{i+1:2d}/{n_cases}] {name}')

    with h5py.File(OUT_FILE, 'w') as f:
        f.create_dataset('frfs', data=H_all)
        f.create_dataset('freqs', data=FREQ_ARRAY.astype(np.float32))
        f.create_dataset('case_names',
                         data=np.array([n.encode() for n in case_names]))
        f.attrs['units']   = '(m/s^2)/N'
        f.attrs['model']   = '3SBB reduced-order shear-building (semi-rigid joints)'
        f.attrs['sensors'] = 'S2,S5,S6,S7,S8,S11,S12,S13,S14 (Y-direction)'
        f.attrs['input']   = 'Base plate, mid -Y face, Y-direction (shaker)'
    print(f'\nWrote {OUT_FILE}  ({n_cases} cases × {n_freq} freqs × {n_sens} sensors)')


if __name__ == '__main__':
    main()
