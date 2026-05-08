"""Damage scenario definitions for the LANL 3SBB reduced-order model.

Pristine geometry uses the empirically calibrated parameters loaded from
``calibration_result.npz``.  Damage is modelled as a reduction in the
*effective story stiffness* of the affected story, applied multiplicatively
on top of the calibrated column factors.

Physical basis
--------------
In the 3SBB benchmark, "D(X%) nBD" denotes bolt-torque loosening at the
connection between floors n and n−1 (story n).  Experimental FRF data shows
that bolt loosening produces small but measurable stiffness changes:

  • Frequency shifts in mode 1 are ≤ 1 Hz for all 1BD cases (bolt at story 1
    barely affects mode 1 due to the base rail constraint).
  • Modes 2 and 3 shift noticeably for 2BD / 3BD cases (inter-floor bolts
    have greater leverage on the higher modes).
  • D(85%) 2BD: mode 2 drops from 50.0 → 42.0 Hz  (≈ 29 % stiffness reduction)
  • D(50%) 2BD: mode 2 drops from 50.0 → 47.4 Hz  (≈ 10 % stiffness reduction)
  • Crack / hole damage produces small but detectable stiffness reductions
    of 3–6 %.

Story indexing
--------------
  1BD  →  storey index 0  (columns connecting base plate to floor 1)
  2BD  →  storey index 1  (columns connecting floor 1 to floor 2)
  3BD  →  storey index 2  (columns connecting floor 2 to floor 3)

Damage model
------------
For damage level D(p%) on storey s:
  column_factor[s, :] *= k_ratio[p] ** 0.25
because stiffness ∝ column_factor ** 4, so applying (k_ratio) ** 0.25
to the factor yields the correct story-stiffness ratio k_ratio.

Empirically calibrated k_ratio values (from experimental freq. shifts):
  D(11%): k_ratio = 0.94  (6 % story stiffness reduction)
  D(20%): k_ratio = 0.91  (9 % reduction)
  D(50%): k_ratio = 0.86  (14 % reduction)
  D(85%): k_ratio = 0.72  (28 % reduction)

Crack / Hole k_ratio values (from experimental mode-3 shifts):
  Crack 5 mm: k_ratio = 0.96   (4 % reduction per story)
  Crack 8 mm: k_ratio = 0.94   (6 % reduction)
  Hole  4 mm: k_ratio = 0.98   (2 % reduction)
  Hole  6 mm: k_ratio = 0.97   (3 % reduction)
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional
import re

import numpy as np

_HERE = Path(__file__).parent.resolve()
_EXAMPLES = _HERE.parent / "pymodal" / "examples" / "los_alamos_3story"
for _p in [str(_HERE), str(_EXAMPLES)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from reduced_model_semirigid import BuildingGeometry
import params as P

# ── Load calibrated pristine parameters ──────────────────────────────────────
_CAL_FILE = _HERE / "calibration_result.npz"
if _CAL_FILE.exists():
    _cal      = np.load(_CAL_FILE)
    _JSR      = float(_cal['jsr'])
    _M_EXTRA  = float(_cal['base_extra_mass'])
    _DAMPING  = float(_cal['damping'])
    _CF_CAL   = np.array([[float(_cal['cf_s1'])] * 4,
                           [float(_cal['cf_s2'])] * 4,
                           [float(_cal['cf_s3'])] * 4], dtype=float)
    # Per-mode damping if available
    _DAMPING_MODES = (np.array(_cal['damping_modes'])
                      if 'damping_modes' in _cal.files else None)
else:
    import warnings
    warnings.warn("calibration_result.npz not found; using nominal JSR=7.9009")
    _JSR            = 7.9009
    _M_EXTRA        = 0.0
    _DAMPING        = 0.005
    _CF_CAL         = np.ones((3, 4), dtype=float)
    _DAMPING_MODES  = None

# ── Story stiffness ratio lookup (empirically calibrated) ─────────────────────
# Key: damage percentage string → story-stiffness ratio k_damaged / k_pristine
_K_RATIO = {
    '5':  0.97,   # small crack / hole reference
    '11': 0.94,
    '20': 0.91,
    '50': 0.86,
    '85': 0.72,
}
# Crack / Hole story stiffness ratios
_CRACK_5MM = 0.96
_CRACK_8MM = 0.94
_HOLE_4MM  = 0.98
_HOLE_6MM  = 0.97


def _pristine_geom(**kw) -> BuildingGeometry:
    """Calibrated pristine BuildingGeometry (no damage)."""
    g = BuildingGeometry(
        joint_stiffness_ratio = _JSR,
        damping               = _DAMPING,
        base_extra_mass       = _M_EXTRA,
        **kw,
    )
    g.column_factor = _CF_CAL.copy()
    if _DAMPING_MODES is not None:
        g.damping_modes = _DAMPING_MODES
    return g


def _damaged_geom(storey: int, k_ratio: float) -> BuildingGeometry:
    """Calibrated geometry with stiffness ratio k_ratio on all 4 columns of *storey*."""
    g = _pristine_geom()
    cf_mult = float(k_ratio) ** 0.25   # stiffness ∝ cf**4
    g.column_factor[storey, :] *= cf_mult
    return g


def geometry_for_case(case_name: str) -> BuildingGeometry:
    """Return a calibrated BuildingGeometry for the given IQS case string."""
    name = case_name.strip()

    # Pristine
    if re.search(r'Pristine', name, re.I):
        return _pristine_geom()

    # Crack
    m = re.search(r'Crack\s+(\d+)mm\s+(\d)BD', name, re.I)
    if m:
        k = _CRACK_8MM if int(m.group(1)) >= 8 else _CRACK_5MM
        return _damaged_geom(int(m.group(2)) - 1, k)

    # Hole
    m = re.search(r'Hole\s+(\d+)mm\s+(\d)[AB]D', name, re.I)
    if m:
        k = _HOLE_6MM if int(m.group(1)) >= 6 else _HOLE_4MM
        return _damaged_geom(int(m.group(2)) - 1, k)

    # General bolt-thinning damage (D / Damage), one or two stories
    dm = re.search(r'(?:D(?:amage)?\s*\(?\s*(\d+)\s*%?\s*\)?)', name, re.I)
    bd_list = re.findall(r'(\d)BD', name, re.I)
    ad_list = re.findall(r'(\d)AD', name, re.I)
    if dm:
        pct = dm.group(1)
        k = _K_RATIO.get(pct, max(0.70, 1.0 - float(pct) / 100.0 * 0.5))
        g = _pristine_geom()
        for s in bd_list:
            g.column_factor[int(s) - 1, :] *= k ** 0.25
        for s in ad_list:
            g.column_factor[int(s) - 1, :] *= k ** 0.25
        return g

    # Mass-only scenarios (not modelled)
    if re.search(r'Mass', name, re.I):
        return _pristine_geom()

    print(f"  [WARNING] Unrecognised case '{name}', returning pristine geometry.")
    return _pristine_geom()


# ── Representative subset ─────────────────────────────────────────────────────
REPRESENTATIVE_CASES = {
    'Pristine':       geometry_for_case('Pristine'),
    'D(11%) 1BD':     geometry_for_case('D (11%) 1BD'),
    'D(11%) 2BD':     geometry_for_case('D (11%) 2BD'),
    'D(11%) 3BD':     geometry_for_case('D (11%) 3BD'),
    'D(20%) 1BD':     geometry_for_case('D (20% 1BD)'),
    'D(20%) 2BD':     geometry_for_case('D (20% 2BD)'),
    'D(20%) 3BD':     geometry_for_case('D (20% 3BD)'),
    'D(50%) 1BD':     geometry_for_case('D(50%) 1BD'),
    'D(50%) 2BD':     geometry_for_case('D(50%) 2BD'),
    'D(85%) 1BD':     geometry_for_case('Damage (85%) 1BD'),
    'D(85%) 2BD':     geometry_for_case('D(85%) 2BD'),
    'D(85%) 3BD':     geometry_for_case('D(85%) 3BD'),
    'D(85%) 1BD+2BD': geometry_for_case('D(85%) 1BD + D(85%) 2BD'),
    'Crack 5mm 1BD':  geometry_for_case('Crack 5mm 1BD'),
    'Crack 8mm 1BD':  geometry_for_case('Crack 8mm 1BD'),
    'Hole 4mm 1BD':   geometry_for_case('Hole 4mm 1BD'),
    'Hole 6mm 1BD':   geometry_for_case('Hole 6mm 1BD'),
}
