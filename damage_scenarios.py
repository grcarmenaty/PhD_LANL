"""Damage-scenario factory for the LANL 3SBB reduced-order model.

Given any IQS label (e.g. ``"D(85%) 1BD + D(85%) 2BD + Mass First Floor"``),
``geometry_for_case`` returns a ``BuildingGeometry`` that the FRF computation
can consume.  Every component of the label is parsed and applied additively
on top of the calibrated pristine geometry.

Supported label fragments
-------------------------
``Pristine``                                          → no damage
``D(X%) nBD`` / ``Damage (X%) nBD`` / ``D (X% nBD)``  → bolt loosening,
                                                        bottom of column at storey *n*
``D(X%) nAD``                                         → bolt loosening,
                                                        top of column at storey *n*
``Crack X mm nBD`` / ``Crack X mm nAD``               → through-thickness crack
``Hole X mm nBD`` / ``Hole X mm nAD``                 → drilled hole on column
``Mass Base`` / ``Mass First/Second/Third Floor``     → 1.2 kg test weight
                                                        on the named plate
``A + B``                                             → both damages applied

Storey indexing
~~~~~~~~~~~~~~~
``1BD`` → storey 0 (columns base→floor 1)
``2BD`` → storey 1 (columns floor 1→floor 2)
``3BD`` → storey 2 (columns floor 2→floor 3)
``AD`` and ``BD`` are the top and bottom ends of the column.  In a 1-D
shear-frame model both contribute to the *same* storey shear stiffness, so
they are applied identically here; the AD/BD distinction will matter once
rocking DOFs are added (model improvement task).

Effective story-stiffness ratios (calibrated empirically against IQS data)
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

_HERE = Path(__file__).parent.resolve()
_EXAMPLES = _HERE.parent / "pymodal" / "examples" / "los_alamos_3story"
for _p in (str(_HERE), str(_EXAMPLES)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from reduced_model_semirigid import BuildingGeometry  # noqa: E402
import params as P                                    # noqa: E402
from case_overrides import apply_overrides            # noqa: E402

# ── Calibrated pristine parameters (loaded from calibration_result.npz) ─────
_CAL_FILE = _HERE / "calibration_result.npz"
if _CAL_FILE.exists():
    _cal = np.load(_CAL_FILE)
    _JSR     = float(_cal['jsr'])
    _M_BASE  = float(_cal['base_extra_mass'])
    _DAMP    = float(_cal['damping'])
    _CF_CAL  = np.array([[float(_cal['cf_s1'])] * 4,
                         [float(_cal['cf_s2'])] * 4,
                         [float(_cal['cf_s3'])] * 4], dtype=float)
    _DAMP_MODES = (np.array(_cal['damping_modes'])
                   if 'damping_modes' in _cal.files else None)
else:
    import warnings
    warnings.warn("calibration_result.npz not found; using nominal parameters.")
    _JSR     = 7.9009
    _M_BASE  = 0.0
    _DAMP    = 0.005
    _CF_CAL  = np.ones((3, 4), dtype=float)
    _DAMP_MODES = None

# ── Damage stiffness-ratio lookup (k_damaged / k_pristine for the storey) ──
_BOLT_K_RATIO = {
    11: 0.94,    # ~6 % stiffness reduction (legacy cf-only path)
    20: 0.91,
    50: 0.86,
    85: 0.72,
}

# Per-end JSR ratio for bolt damage: J_damaged / J_pristine at the loose end.
# Combined with the asymmetric semi-rigid formula, single-end (AD-only or
# BD-only) damage produces *less* effective stiffness reduction than damage
# at both ends of the same storey — exactly what the experimental
# `D(X%) 1AD + D(X%) 1BD` cases show vs `D(X%) 1BD` alone.
_BOLT_JSR_RATIO = {
    11: 0.85,
    20: 0.70,
    50: 0.55,
    85: 0.39,
}
_CRACK_K_RATIO = {
    5: 0.96,
    8: 0.94,
}
_HOLE_K_RATIO = {
    4: 0.98,
    6: 0.97,
}
# IQS test weight (a brass block strapped on top of a plate)
_TEST_MASS_KG = 1.2

# ── Tokenisation ─────────────────────────────────────────────────────────────
# A token is a single damage component (split on the '+' separator).
_BOLT_RE  = re.compile(
    r'(?:d|damage)\s*\(?\s*(?P<pct>\d+)\s*%?\s*(?:\)|\s)?\s*'
    r'(?:\(\s*)?(?P<sty>\d)\s*(?P<face>[ab])d\)?',
    re.I,
)
_CRACK_RE = re.compile(
    r'crack\s*'
    r'(?:'
    r'(?P<size1>\d+)\s*mm\s*(?P<sty1>\d)\s*(?P<face1>[ab])d'
    r'|'
    r'(?P<sty2>\d)\s*(?P<face2>[ab])d\s*(?P<size2>\d+)\s*mm'
    r')',
    re.I,
)
_HOLE_RE  = re.compile(
    r'hole\s*(?P<size>\d+)\s*mm\s*(?P<sty>\d)\s*(?P<face>[ab])d',
    re.I,
)
_MASS_RE  = re.compile(
    r'mass\s*(?:'
    r'(?P<base>base)|'
    r'(?:1f|first\s*floor)|'
    r'(?:second\s*floor)|'
    r'(?:third\s*floor)'
    r')',
    re.I,
)


def _split_tokens(label: str) -> List[str]:
    """Split a multi-component damage label on '+' / ',' separators."""
    return [t.strip() for t in re.split(r'\s*[+,]\s*', label) if t.strip()]


def _parse_token(tok: str) -> List[Tuple]:
    """Parse one token into a list of damage operations.

    Each op is one of:
      ('bolt', storey, face, pct)
      ('crack', storey, face, size_mm)
      ('hole',  storey, face, size_mm)
      ('mass',  plate_index)

    storey, plate_index are 0-based; face ∈ {'a', 'b'}.
    """
    ops: List[Tuple] = []

    for m in _BOLT_RE.finditer(tok):
        ops.append(('bolt', int(m['sty']) - 1, m['face'].lower(), int(m['pct'])))

    for m in _CRACK_RE.finditer(tok):
        if m['size1'] is not None:
            size = int(m['size1']); sty = int(m['sty1']); face = m['face1']
        else:
            size = int(m['size2']); sty = int(m['sty2']); face = m['face2']
        ops.append(('crack', sty - 1, face.lower(), size))

    for m in _HOLE_RE.finditer(tok):
        ops.append(('hole', int(m['sty']) - 1, m['face'].lower(),
                    int(m['size'])))

    if re.search(r'mass\s*base', tok, re.I):
        ops.append(('mass', 0))
    if re.search(r'mass\s*(?:1f|first\s*floor)', tok, re.I):
        ops.append(('mass', 1))
    if re.search(r'mass\s*second\s*floor', tok, re.I):
        ops.append(('mass', 2))
    if re.search(r'mass\s*third\s*floor', tok, re.I):
        ops.append(('mass', 3))

    return ops


def parse_label(label: str) -> List[Tuple]:
    """Parse a full IQS label into a flat list of damage operations."""
    if re.search(r'pristine', label, re.I):
        return []
    ops: List[Tuple] = []
    for tok in _split_tokens(label):
        ops.extend(_parse_token(tok))
    return ops


# ── Geometry construction ───────────────────────────────────────────────────
def _pristine_geom() -> BuildingGeometry:
    g = BuildingGeometry(
        joint_stiffness_ratio = _JSR,
        damping               = _DAMP,
        base_extra_mass       = _M_BASE,
    )
    g.column_factor = _CF_CAL.copy()
    if _DAMP_MODES is not None:
        g.damping_modes = _DAMP_MODES
    # Calibrated per-plate extra masses (independent of test weights, which
    # are applied additively below by ``geometry_for_case``).
    if _CAL_FILE.exists():
        cal = np.load(_CAL_FILE)
        for k, key in enumerate(['plate_extra_mass_fl1',
                                  'plate_extra_mass_fl2',
                                  'plate_extra_mass_fl3'], start=1):
            if key in cal.files:
                g.plate_extra_mass[k] += float(cal[key])
        if 'plate_flex_freq_hz' in cal.files:
            g.plate_flex_freq_hz = float(cal['plate_flex_freq_hz'])
        per_floor = []
        for key in ('plate_flex_mass_fl1', 'plate_flex_mass_fl2',
                    'plate_flex_mass_fl3'):
            if key in cal.files:
                per_floor.append(float(cal[key]))
            else:
                per_floor.append(0.0)
        if any(m > 0 for m in per_floor):
            g.plate_flex_mass_per_floor = np.array(per_floor, dtype=float)
        elif 'plate_flex_mass' in cal.files:
            g.plate_flex_mass = float(cal['plate_flex_mass'])
        # Second flex set
        if 'plate_flex2_freq_hz' in cal.files:
            g.plate_flex2_freq_hz = float(cal['plate_flex2_freq_hz'])
        per_floor2 = []
        for key in ('plate_flex2_mass_fl1', 'plate_flex2_mass_fl2',
                    'plate_flex2_mass_fl3'):
            if key in cal.files:
                per_floor2.append(float(cal[key]))
            else:
                per_floor2.append(0.0)
        if any(m > 0 for m in per_floor2):
            g.plate_flex2_mass_per_floor = np.array(per_floor2, dtype=float)
    return g


def _ensure_per_end_jsr(g: BuildingGeometry) -> np.ndarray:
    """Initialise ``g.joint_stiffness_per_end`` from the scalar JSR if
    not already populated.  Shape = ``(n_stories, 4, 2)`` with last axis
    ``[bottom_end, top_end]``.
    """
    if g.joint_stiffness_per_end is None:
        g.joint_stiffness_per_end = np.full((g.n_stories, 4, 2),
                                              float(g.joint_stiffness_ratio),
                                              dtype=float)
    return g.joint_stiffness_per_end


def geometry_for_case(case_name: str) -> BuildingGeometry:
    """Return a calibrated BuildingGeometry for any IQS case label.

    Bolt damage now reduces the joint-stiffness ratio at the *specific
    end* affected by the looseness (`BD` → bottom end, `AD` → top end),
    using ``_BOLT_JSR_RATIO`` and the asymmetric semi-rigid formula.
    Cracks and holes still act through ``column_factor`` because they
    physically reduce the column section, not the joint stiffness.
    Masses still accumulate on ``plate_extra_mass``.
    """
    g = _pristine_geom()
    ops = parse_label(case_name)
    if not ops:
        return g

    for op in ops:
        kind = op[0]
        if kind == 'bolt':
            _, sty, face, pct = op
            jsr_ratio = _BOLT_JSR_RATIO.get(
                pct, max(0.05, 1.0 - pct / 100.0 * 0.6))
            end_idx   = 1 if face.lower() == 'a' else 0  # BD -> 0 (bot), AD -> 1 (top)
            per_end   = _ensure_per_end_jsr(g)
            per_end[sty, :, end_idx] *= jsr_ratio
        elif kind == 'crack':
            _, sty, face, size = op
            ratio = _CRACK_K_RATIO.get(size, 0.94)
            cols = _columns_for_face(face)
            g.column_factor[sty, cols] *= ratio ** 0.25
        elif kind == 'hole':
            _, sty, face, size = op
            ratio = _HOLE_K_RATIO.get(size, 0.97)
            cols = _columns_for_face(face)
            g.column_factor[sty, cols] *= ratio ** 0.25
        elif kind == 'mass':
            _, plate = op
            if 0 <= plate <= 3:
                g.plate_extra_mass[plate] += _TEST_MASS_KG

    # Apply any per-case overrides registered in case_overrides.CASE_OVERRIDES.
    apply_overrides(g, case_name)
    return g


def _columns_for_face(face: str) -> List[int]:
    """Return column indices on a given face.

    The 4 columns are at corners (xl,−Y), (xh,−Y), (xl,+Y), (xh,+Y).
    'A' (above) and 'B' (below) refer to the +Y / -Y faces of the building
    in the IQS lab convention.  Returning all 4 columns models the symmetric
    case (both faces affected).  When the model gains rocking DOFs the
    A/B distinction will give different stiffnesses.
    """
    # Until rocking DOFs are added: A and B both reduce all 4 columns.
    return [0, 1, 2, 3]
