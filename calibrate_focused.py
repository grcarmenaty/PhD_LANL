"""Focused random-search per-case fitter for the worst N cases.

The iterative grid fitter plateaus around mean SCI 0.94 because the
remaining bottom cases all need parameter combinations that grow the
search space exponentially.  This script burns more compute per case
(20 000 random samples each) over a *bigger* parameter space — every
override key the project supports — and keeps only improvements.
"""
from __future__ import annotations

import re
from pathlib import Path

import h5py
import numpy as np

from reduced_model_semirigid import compute_frf_matrix
from model_3sbb import _input_position, _sensor_positions, FREQ_ARRAY
from damage_scenarios import geometry_for_case
import case_overrides
import calibrate_per_case as cpc

_HERE = Path(__file__).parent.resolve()
N_TRIALS = 20_000


def big_grid_for(case_name: str):
    """Return list of (key, [values]) — full parameter superset."""
    s = case_name.lower()
    grid: dict = {}
    JSR = [0.3, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0]
    DAMP = [0.5, 0.7, 1.0, 1.4, 2.0]
    CF = [0.92, 0.96, 1.00, 1.04, 1.08]
    MASS = [-0.6, 0.0, 0.6, 1.2]

    bd = sorted({int(m.group(1)) for m in re.finditer(r'(\d)\s*bd', s)})
    ad = sorted({int(m.group(1)) for m in re.finditer(r'(\d)\s*ad', s)})
    affected_storeys = set(bd + ad) or {1, 2, 3}

    for sty in (1, 2, 3):
        grid[f'mul_cf_s{sty}'] = CF
    for sty in affected_storeys:
        if sty in bd:
            grid[f'mul_jsr_storey_{sty}_bot'] = JSR
            for c in (0, 1, 2, 3):
                grid[f'mul_jsr_storey_{sty}_bot_corner_{c}'] = [0.3, 1.0, 3.0]
        if sty in ad:
            grid[f'mul_jsr_storey_{sty}_top'] = JSR
            for c in (0, 1, 2, 3):
                grid[f'mul_jsr_storey_{sty}_top_corner_{c}'] = [0.3, 1.0, 3.0]

    for r in (0, 1, 3):
        grid[f'mul_damping_mode_{r}'] = DAMP

    if re.search(r'mass\s*base', s):
        grid['add_plate_extra_mass_0'] = MASS
    if re.search(r'mass\s*(?:1f|first)', s):
        grid['add_plate_extra_mass_1'] = MASS
    if re.search(r'mass\s*second', s):
        grid['add_plate_extra_mass_2'] = MASS
    if re.search(r'mass\s*third', s):
        grid['add_plate_extra_mass_3'] = MASS

    if 'pristine' in s:
        grid['set_plate_flex_freq_hz'] = [60, 70, 80, 90, 95, 105, 115, 125, 140]
        # Pristine session outliers may need totally different per-storey
        # symmetric stiffness multipliers — give a much wider sweep.
        for sty in (1, 2, 3):
            grid[f'mul_cf_s{sty}'] = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]
        # And per-mode damping range
        for r in (0, 1, 3):
            grid[f'mul_damping_mode_{r}'] = [0.3, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0]
    return grid


# Targeted hard-case rerun with very large budget
def hard_cases():
    return ['Pristine (26/1/2021)', 'Pristine (27/1/2021)',
            'Pristine (5/2/2021)', 'Pristine (8/2/2021)',
            'D(11%) 1BD', 'D(85%) 2BD', 'D (11%) 2BD',
            'Hole 6mm 2BD']


def main(targets=None, n_trials=N_TRIALS):
    with h5py.File(_HERE / 'median_frfs.h5', 'r') as f:
        cn = [c.decode() for c in f['case_names'][:]]
        H_exp = f['median_frf'][:]
        f_exp = f['freq'][:]

    band_idx     = cpc._band_indices(f_exp)
    band_idx_syn = cpc._band_indices(FREQ_ARRAY)
    exp_cfdacs = [cpc._cfdac(H_exp[i][band_idx]) for i in range(len(cn))]

    # Pick worst N cases by current SCI if targets is None
    if targets is None:
        cur = []
        for i, nm in enumerate(cn):
            ov = case_overrides.CASE_OVERRIDES.get(nm, {})
            s = cpc._eval_sci(nm, ov, exp_cfdacs[i], band_idx_syn)
            cur.append((s, nm))
        cur.sort()
        targets = [n for s, n in cur if s < 0.90][:8]
        print(f'Targets (worst 8 cases below 0.90):')
        for nm in targets:
            print(f'  {nm}')

    for nm in targets:
        if nm not in cn:
            print(f'  {nm}: not in dataset')
            continue
        i_exp = cn.index(nm)
        # Current best baseline
        prev_ov = dict(case_overrides.CASE_OVERRIDES.get(nm, {}))
        baseline = cpc._eval_sci(nm, prev_ov, exp_cfdacs[i_exp], band_idx_syn)

        grid = big_grid_for(nm)
        keys = list(grid.keys())
        values = [grid[k] for k in keys]
        rng = np.random.default_rng(hash(nm) & 0xffff)

        best_sci = baseline
        best_ov  = prev_ov

        for trial in range(n_trials):
            combo = tuple(rng.choice(v) for v in values)
            ov = {k: float(v) for k, v in zip(keys, combo)
                  if abs(float(v) - 1.0) > 1e-9 or k.startswith(('add_', 'set_'))}
            try:
                s = cpc._eval_sci(nm, ov, exp_cfdacs[i_exp], band_idx_syn)
            except Exception:
                continue
            if s > best_sci:
                best_sci = s
                best_ov  = ov

        delta = best_sci - baseline
        if delta > 0.002:
            case_overrides.CASE_OVERRIDES[nm] = dict(best_ov)
            cpc.persist_overrides()
            marker = ' ✓'
        else:
            case_overrides.CASE_OVERRIDES[nm] = prev_ov
            marker = ''
        print(f'  {nm:<55s}  {baseline:.3f} -> {best_sci:.3f}  Δ={delta:+.3f}{marker}')


if __name__ == '__main__':
    main()
