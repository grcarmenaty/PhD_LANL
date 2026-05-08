"""Iterative per-case override fitter.

Targets every experimental case whose SCI is below a threshold and
runs a grid search over a small set of override knobs to improve it.
The grid is keyed off the case label: bolt-damage cases sweep per-end
JSR multipliers per affected storey, mass cases also sweep extra
plate mass, Pristine variants sweep per-storey column-factor tweaks
(asymmetric measurements).

Each pass:
  1. Builds the synthetic FRF for every case using the current
     `case_overrides.CASE_OVERRIDES`.
  2. Computes SCI vs experiment on the 5–100 Hz CFDAC band.
  3. For every case below ``threshold``:
        - Build a grid based on the case label.
        - Evaluate every combination, keep the best.
        - Persist into ``case_overrides.CASE_OVERRIDES``.
  4. Repeat until no case-level improvement larger than ``min_delta``.
"""
from __future__ import annotations

import itertools
import re
from pathlib import Path

import h5py
import numpy as np

from reduced_model_semirigid import compute_frf_matrix
from model_3sbb import _input_position, _sensor_positions, FREQ_ARRAY
from damage_scenarios import geometry_for_case
import case_overrides

_HERE = Path(__file__).parent.resolve()
F_LO, F_HI = 5.0, 100.0
DF_CFDAC   = 1.0
THRESHOLD  = 0.95
MIN_DELTA  = 0.005


def _band_indices(freq):
    target = np.arange(F_LO, F_HI + 1e-9, DF_CFDAC)
    return np.array([int(np.argmin(np.abs(freq - t))) for t in target])


def _cfdac(H):
    inner = H.conj() @ H.T
    d = np.real(np.diag(inner)).copy()
    d[d < 1e-30] = 1e-30
    return (np.abs(inner) ** 2) / np.outer(d, d)


def _sci(C1, C2):
    a = C1.ravel() - C1.mean(); b = C2.ravel() - C2.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float((a @ b) ** 2 / d ** 2) if d > 0 else 0.0


_BOLT_BD_RE = re.compile(r'(\d)\s*bd', re.I)
_BOLT_AD_RE = re.compile(r'(\d)\s*ad', re.I)
_CRACK_RE   = re.compile(r'crack.*?(\d)\s*[ab]d', re.I)
_HOLE_RE    = re.compile(r'hole.*?(\d)\s*[ab]d', re.I)


def _grid_for(case_name: str):
    """Heuristic grid for *case_name*, keyed off label content."""
    s = case_name.lower()
    grid: dict = {}

    # Per-end JSR multipliers for any BD / AD storey mentioned
    JSR_GRID = [0.3, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0]
    bd_storeys = sorted({int(m.group(1)) for m in _BOLT_BD_RE.finditer(s)})
    ad_storeys = sorted({int(m.group(1)) for m in _BOLT_AD_RE.finditer(s)})
    for sty in bd_storeys:
        grid[f'mul_jsr_storey_{sty}_bot'] = JSR_GRID
    for sty in ad_storeys:
        grid[f'mul_jsr_storey_{sty}_top'] = JSR_GRID

    # Crack / hole storeys: also sweep per-storey cf
    crack_storeys = sorted({int(m.group(1)) for m in _CRACK_RE.finditer(s)})
    hole_storeys  = sorted({int(m.group(1)) for m in _HOLE_RE.finditer(s)})
    for sty in set(crack_storeys + hole_storeys):
        grid.setdefault(f'mul_cf_s{sty}', [0.92, 0.96, 1.00, 1.04, 1.08])

    # Pristine variants: per-storey cf tweaks (sessions vary)
    if 'pristine' in s and s != 'pristine':
        grid['mul_cf_s1'] = [0.94, 0.97, 1.00, 1.03, 1.06]
        grid['mul_cf_s2'] = [0.94, 0.97, 1.00, 1.03, 1.06]
        grid['mul_cf_s3'] = [0.94, 0.97, 1.00, 1.03, 1.06]

    # Mass cases: sweep extra plate mass on the affected plate
    if re.search(r'mass\s*base', s):
        grid.setdefault('add_plate_extra_mass_0', [-0.6, 0.0, 0.6, 1.2])
    if re.search(r'mass\s*(?:1f|first)', s):
        grid.setdefault('add_plate_extra_mass_1', [-0.6, 0.0, 0.6, 1.2])
    if re.search(r'mass\s*second', s):
        grid.setdefault('add_plate_extra_mass_2', [-0.6, 0.0, 0.6, 1.2])
    if re.search(r'mass\s*third', s):
        grid.setdefault('add_plate_extra_mass_3', [-0.6, 0.0, 0.6, 1.2])

    return grid


def _eval_sci(case_name, override, exp_cfdac, band_idx_syn):
    case_overrides.CASE_OVERRIDES[case_name] = override
    g = geometry_for_case(case_name)
    damping = getattr(g, 'damping_modes', g.damping)
    H = compute_frf_matrix(FREQ_ARRAY,
                            _input_position(g),
                            _sensor_positions(g),
                            g, damping=damping)[:, :, 0]
    return _sci(exp_cfdac, _cfdac(H[band_idx_syn]))


def persist_overrides():
    """Re-write case_overrides.py CASE_OVERRIDES dict literal in place."""
    out = _HERE / 'case_overrides.py'
    src = out.read_text()
    new_table = "CASE_OVERRIDES: dict = {\n"
    for k in sorted(case_overrides.CASE_OVERRIDES):
        v = case_overrides.CASE_OVERRIDES[k]
        if not v:
            continue
        new_table += f"    {k!r}: {{\n"
        for kk, vv in v.items():
            new_table += f"        {kk!r}: {vv},\n"
        new_table += "    },\n"
    new_table += "}"
    src = re.sub(r'CASE_OVERRIDES: dict = \{.*?\n\}',
                  new_table, src, count=1, flags=re.S)
    out.write_text(src)


def main(threshold=THRESHOLD, min_delta=MIN_DELTA, max_passes=4):
    with h5py.File(_HERE / 'median_frfs.h5', 'r') as f:
        cn = [c.decode() for c in f['case_names'][:]]
        H_exp = f['median_frf'][:]
        f_exp = f['freq'][:]

    band_idx     = _band_indices(f_exp)
    band_idx_syn = _band_indices(FREQ_ARRAY)
    exp_cfdacs = [_cfdac(H_exp[i][band_idx]) for i in range(len(cn))]

    for pass_no in range(max_passes):
        # Recompute current SCI for every case
        cur = []
        for i, nm in enumerate(cn):
            ov = case_overrides.CASE_OVERRIDES.get(nm, {})
            s = _eval_sci(nm, ov, exp_cfdacs[i], band_idx_syn)
            cur.append((s, nm))
        cur.sort()
        below = [(s, n) for s, n in cur if s < threshold]
        print(f'\n=== pass {pass_no} : {len(below)} cases below {threshold:.2f} ===')

        improvements = 0
        for baseline_sci, nm in below:
            grid = _grid_for(nm)
            if not grid:
                continue
            i_exp = cn.index(nm)
            keys, values = list(grid.keys()), [grid[k] for k in grid.keys()]
            # Cartesian product can blow up — cap at 4000 evals per case
            n_total = 1
            for v in values: n_total *= len(v)
            if n_total > 4000:
                # Random sample instead
                rng = np.random.default_rng(pass_no * 17 + i_exp)
                combos = [tuple(rng.choice(v) for v in values) for _ in range(4000)]
            else:
                combos = list(itertools.product(*values))

            best_sci = baseline_sci
            best_ov  = case_overrides.CASE_OVERRIDES.get(nm, {})
            for combo in combos:
                ov = {k: float(v) for k, v in zip(keys, combo)
                      if abs(float(v) - 1.0) > 1e-9 or k.startswith('add_')}
                s = _eval_sci(nm, ov, exp_cfdacs[i_exp], band_idx_syn)
                if s > best_sci:
                    best_sci = s
                    best_ov  = ov
            delta = best_sci - baseline_sci
            if delta > min_delta:
                case_overrides.CASE_OVERRIDES[nm] = best_ov
                improvements += 1
                marker = ' ✓'
            else:
                # restore previous override to avoid regression
                case_overrides.CASE_OVERRIDES[nm] = case_overrides.CASE_OVERRIDES.get(nm, {})
                marker = ''
            print(f'  {nm:<55s}  {baseline_sci:.3f} -> {best_sci:.3f}  Δ={delta:+.3f}{marker}')

        persist_overrides()
        print(f'pass {pass_no}: {improvements} improved')
        if improvements == 0:
            break

    # Final scoreboard
    final = []
    for i, nm in enumerate(cn):
        ov = case_overrides.CASE_OVERRIDES.get(nm, {})
        s = _eval_sci(nm, ov, exp_cfdacs[i], band_idx_syn)
        final.append((s, nm))
    final.sort()
    print('\nFinal scoreboard:')
    print(f'  mean = {np.mean([s for s,_ in final]):.3f}')
    print(f'  median = {np.median([s for s,_ in final]):.3f}')
    print(f'  >= 0.95 : {sum(1 for s,_ in final if s>=0.95)} / {len(final)}')
    print(f'  >= 0.90 : {sum(1 for s,_ in final if s>=0.90)} / {len(final)}')
    print(f'  Bottom 5:')
    for s, n in final[:5]:
        print(f'    {s:.3f}  {n}')


if __name__ == '__main__':
    main()
