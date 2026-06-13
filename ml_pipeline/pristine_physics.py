"""Pristine-anchored, first-principles damage submodels.

The default synthetic generator (``ml_pipeline.variation``) builds every
sample on top of the *pristine-calibrated* baseline geometry, but it sizes the
damage with lookup tables whose anchors (bolt 11→0.85 / 20→0.70 / 50→0.55 /
85→0.39, crack 5→0.96 / 8→0.94, hole 4→0.98 / 6→0.97) were **fitted to the
damaged experimental FRFs**.  In other words, the damage magnitudes already
"know" the test set.

This module provides the honest counterpart: damage magnitudes derived purely
from the pristine geometry plus textbook mechanics, with **no information taken
from any damaged measurement**.  Each submodel is anchored only at the pristine
state (severity 0 → ratio 1) and is otherwise first-principles:

  * **Bolt loosening** — the per-end joint-stiffness ratio (JSR) scales with the
    remaining bolt-preload fraction.  Loosening to ``percent`` removes that
    fraction of the clamping force, and the rotational stiffness of a bolted
    lap joint is, to first order, proportional to the contact preload:
        ``jsr_ratio = 1 - percent/100``.

  * **Crack** — a through-thickness edge crack of surface length ``a`` along the
    wide face (``col_lx``) removes that strip from the section.  Y-direction
    bending uses ``I_xx = col_lx * col_ly**3 / 12`` which is *linear* in the
    wide dimension, so the bending-stiffness ratio is
        ``I_remaining / I_full = (col_lx - a) / col_lx``.

  * **Hole** — a circular hole of diameter ``phi`` drilled through the column,
    centred on the section, removes its own centroidal second moment
    ``pi*phi**4/64`` from ``I_xx``:
        ``1 - (pi*phi**4/64) / (col_lx*col_ly**3/12)``.

  * **Mass** — the test weight severity is already a physical mass in kg
    (a known strapped-on block); it needs no fitting and is applied directly.

The damage is injected through exactly the same model plumbing the calibrated
generator uses (``joint_stiffness_per_end`` for bolts; ``column_factor`` raised
to the 1/4 power for cracks/holes, which the ``factor**4`` stiffness law turns
back into the intended stiffness ratio; additive ``plate_extra_mass`` for
mass).  Only the *source* of the ratio changes — the structural application is
identical — so any difference downstream is attributable solely to dropping the
damaged-data-fitted magnitudes.

Nothing here mutates the calibrated modules; ``geometry_from_params_pristine``
is a drop-in replacement for ``variation.geometry_from_params``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from reduced_model_semirigid import BuildingGeometry  # noqa: E402
# After importing reduced_model_semirigid, sys.path[0] is pymodal's example
# dir (it self-inserts).  Put the PhD_LANL repo back at the head so the local
# damage_scenarios.py wins.
sys.path.insert(0, str(_REPO))
from damage_scenarios import _pristine_geom            # noqa: E402

import params as P                                     # noqa: E402
from ml_pipeline.variation import SampleParams         # noqa: E402
from ml_pipeline.case_design import (                  # noqa: E402
    TYPE_BOLT, TYPE_CRACK, TYPE_HOLE, TYPE_MASS, END_AD,
)


# ── First-principles damage submodels (pristine-anchored, no fitted tables) ──
def bolt_jsr_ratio(percent: float) -> float:
    """Per-end JSR multiplier for a bolt loosened to ``percent`` severity.

    Rotational stiffness of the bolted joint scales with the remaining
    preload fraction; anchored only at pristine (0 % → 1.0).  Floored at 0.05
    to keep the joint from going exactly pinned/negative.
    """
    return float(np.clip(1.0 - percent / 100.0, 0.05, 1.0))


def crack_ratio(size_mm: float,
                col_lx: float = P.COL_LX,
                col_ly: float = P.COL_LY) -> float:
    """Y-bending stiffness ratio for a through-thickness crack of length
    ``size_mm`` along the wide face.  ``I_xx`` is linear in ``col_lx``, so the
    ratio is the remaining-width fraction.  Geometry only — no damaged data.
    """
    a = size_mm * 1e-3                       # mm → m
    return float(np.clip((col_lx - a) / col_lx, 0.05, 1.0))


def hole_ratio(diameter_mm: float,
               col_lx: float = P.COL_LX,
               col_ly: float = P.COL_LY) -> float:
    """Y-bending stiffness ratio for a centred circular hole of diameter
    ``diameter_mm``.  Removes the hole's centroidal second moment
    ``pi*phi**4/64`` from ``I_xx = col_lx*col_ly**3/12``.  Geometry only.
    """
    phi = diameter_mm * 1e-3                  # mm → m
    I_full = col_lx * col_ly ** 3 / 12.0
    I_hole = np.pi * phi ** 4 / 64.0
    return float(np.clip(1.0 - I_hole / I_full, 0.05, 1.0))


# ── geometry construction from params (pristine-anchored damage) ─────────────
def _columns_all() -> list[int]:
    return [0, 1, 2, 3]


def geometry_from_params_pristine(p: SampleParams) -> BuildingGeometry:
    """Drop-in replacement for ``variation.geometry_from_params``.

    Identical baseline + physical jitter; damage magnitudes come from the
    first-principles submodels above instead of the damaged-data-fitted tables.
    """
    g = _pristine_geom()

    # Material / geometry jitter (same as the calibrated generator).
    g.young            *= p.young_factor
    g.density          *= p.density_factor
    g.damping          *= p.damping_factor
    g.plate_lx         *= p.plate_lx_factor
    g.plate_ly         *= p.plate_ly_factor
    g.plate_lz         *= p.plate_lz_factor
    g.col_lx           *= p.col_lx_factor
    g.col_ly           *= p.col_ly_factor
    g.base_extra_mass  += p.base_extra_mass_dkg

    if g.joint_stiffness_per_end is None:
        g.joint_stiffness_per_end = np.full((g.n_stories, 4, 2),
                                             float(g.joint_stiffness_ratio),
                                             dtype=float)
    g.joint_stiffness_per_end *= p.jsr_factor

    g.plate_extra_mass[1] += p.plate_extra_mass_dkg_fl1
    g.plate_extra_mass[2] += p.plate_extra_mass_dkg_fl2
    g.plate_extra_mass[3] += p.plate_extra_mass_dkg_fl3

    # Damage on top of the jittered baseline — pristine-anchored physics.
    if p.type_code == TYPE_BOLT:
        r = bolt_jsr_ratio(p.severity)
        end_idx = 1 if p.end == END_AD else 0
        g.joint_stiffness_per_end[p.storey, :, end_idx] *= r
    elif p.type_code == TYPE_CRACK:
        r = crack_ratio(p.severity, g.col_lx, g.col_ly)
        g.column_factor[p.storey, _columns_all()] *= r ** 0.25
    elif p.type_code == TYPE_HOLE:
        r = hole_ratio(p.severity, g.col_lx, g.col_ly)
        g.column_factor[p.storey, _columns_all()] *= r ** 0.25
    elif p.type_code == TYPE_MASS:
        plate = int(p.end)        # 0..3
        if 0 <= plate <= 3:
            g.plate_extra_mass[plate] += p.severity
    # Pristine: nothing extra.

    return g
