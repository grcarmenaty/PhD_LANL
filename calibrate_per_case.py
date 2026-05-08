"""Per-case override fitting.

Iterates over a list of "problem" experimental cases and, for each one,
grid-searches a small number of override multipliers around the
generic-parser geometry to maximise SCI for *that case alone*.  Writes
the best per-case overrides back into ``case_overrides.py``.

This is the formal hook for the "don't be limited by using the same
base model for all damage scenarios" plan: the generic parser still
runs first (so global damage trends are captured), and the override
table layers small case-specific corrections on top.
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

# ── CFDAC band ─────────────────────────────────────────────────────────────
F_LO, F_HI = 5.0, 100.0
DF_CFDAC   = 1.0


def _band_indices(freq):
    target = np.arange(F_LO, F_HI + 1e-9, DF_CFDAC)
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


# ── Override search spaces ─────────────────────────────────────────────────
# For each kind of case we sweep a small set of override-key combinations.
# Keep grids small (≤ 100 evaluations per case) to keep the run fast.
_BOLT_BD_RE = re.compile(r'(\d)\s*bd', re.I)
_BOLT_AD_RE = re.compile(r'(\d)\s*ad', re.I)


def _search_grid_for(case_name: str):
    """Return a dict ``key → list[value]`` whose Cartesian product is the
    grid we'll search for *case_name*.  Uses heuristics from the case
    label — only sweeping the parameters that could plausibly help."""
    s = case_name.lower()
    grid: dict = {}

    bd_storeys = sorted({int(m.group(1)) for m in _BOLT_BD_RE.finditer(s)})
    ad_storeys = sorted({int(m.group(1)) for m in _BOLT_AD_RE.finditer(s)})

    # Wider per-end JSR sweep, including very-soft / fully-rigid end
    JSR_GRID = [0.3, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0]

    for sty in bd_storeys:
        grid[f'mul_jsr_storey_{sty}_bot'] = JSR_GRID
    for sty in ad_storeys:
        grid[f'mul_jsr_storey_{sty}_top'] = JSR_GRID

    # Pristine variants: small symmetric tweaks on per-storey cf.
    if 'pristine' in s:
        grid['mul_cf_s1'] = [0.95, 1.00, 1.05]
        grid['mul_cf_s2'] = [0.95, 1.00, 1.05]
        grid['mul_cf_s3'] = [0.95, 1.00, 1.05]

    # Mass cases: also sweep extra plate mass on the affected plate
    if 'mass' in s:
        if 'mass base' in s:
            grid['add_plate_extra_mass_0'] = [-0.6, 0.0, 0.6]
        if 'first floor' in s or 'mass 1f' in s:
            grid['add_plate_extra_mass_1'] = [-0.6, 0.0, 0.6]

    return grid


def _eval_sci(case_name: str, override: dict, exp_cfdac, band_idx) -> float:
    case_overrides.CASE_OVERRIDES[case_name] = override
    g = geometry_for_case(case_name)
    damping = getattr(g, 'damping_modes', g.damping)
    H = compute_frf_matrix(FREQ_ARRAY,
                            _input_position(g),
                            _sensor_positions(g),
                            g, damping=damping)[:, :, 0]
    Hb = H[band_idx]
    return _sci(exp_cfdac, _cfdac(Hb))


def main(targets=None):
    if targets is None:
        targets = [
            'D(85%) 2BD',
            'D(11%) 1BD',
            'D (11%) 1BD',
            'D(85%) 1BD + D(85%) 2BD',
            'Pristine (26/1/2021)',
            'Pristine (27/1/2021)',
            'D(85%) 1AD + D(85%) 1BD',
            'D(85%) 2BD + D(85%) 2AD',
        ]

    with h5py.File(_HERE / 'median_frfs.h5', 'r') as f:
        cn = [c.decode() for c in f['case_names'][:]]
        H_exp = f['median_frf'][:]
        f_exp = f['freq'][:]

    band_idx = _band_indices(f_exp)
    band_idx_syn = _band_indices(FREQ_ARRAY)

    # Baseline SCI (no override) for each target
    print('Per-case override fit — searching small grids around the generic geometry.\n')
    print(f'{"case":<45s}  {"baseline":>9s}  {"best":>7s}  override')
    for nm in targets:
        if nm not in cn:
            print(f'  {nm!r}: not in median_frfs.h5')
            continue
        # Save and restore CASE_OVERRIDES around each case
        prev = case_overrides.CASE_OVERRIDES.pop(nm, None)
        case_overrides.CASE_OVERRIDES[nm] = {}

        exp_cfdac = _cfdac(H_exp[cn.index(nm)][band_idx])

        baseline = _eval_sci(nm, {}, exp_cfdac, band_idx_syn)
        grid = _search_grid_for(nm)
        if not grid:
            print(f'  {nm:<45s}  {baseline:9.3f}  (no grid available)')
            case_overrides.CASE_OVERRIDES[nm] = {} if prev is None else prev
            continue

        keys   = list(grid.keys())
        values = [grid[k] for k in keys]

        best_sci   = baseline
        best_combo = {}
        for combo in itertools.product(*values):
            ov = {k: float(v) for k, v in zip(keys, combo)
                  if abs(float(v) - 1.0) > 1e-6}    # skip identity entries
            s = _eval_sci(nm, ov, exp_cfdac, band_idx_syn)
            if s > best_sci:
                best_sci   = s
                best_combo = ov

        case_overrides.CASE_OVERRIDES[nm] = best_combo
        delta = best_sci - baseline
        marker = ' ✓' if delta > 0.01 else ''
        print(f'  {nm:<45s}  {baseline:9.3f}  {best_sci:7.3f}  {best_combo}{marker}')

    # Persist results into case_overrides.py
    out = _HERE / 'case_overrides.py'
    src = out.read_text()
    new_table = "\nCASE_OVERRIDES: dict = {\n"
    for k in sorted(case_overrides.CASE_OVERRIDES):
        v = case_overrides.CASE_OVERRIDES[k]
        if not v:
            continue
        new_table += f"    {k!r}: {{\n"
        for kk, vv in v.items():
            new_table += f"        {kk!r}: {vv},\n"
        new_table += "    },\n"
    new_table += "}\n"

    src = re.sub(r'\nCASE_OVERRIDES: dict = .*?\n}\n',
                  new_table, src, count=1, flags=re.S)
    out.write_text(src)
    print(f'\nWrote per-case overrides to {out}')


if __name__ == '__main__':
    main()
