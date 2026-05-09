"""Per-case parameter overrides for the IQS scenarios.

The generic ``damage_scenarios.geometry_for_case`` builds every case
from the same calibrated pristine geometry plus the parsed damage.
Some IQS cases — especially the asymmetric ``2BD`` damage and
combinations with masses — don't fit this single global parameterisation
well because the experimental data carries case-specific physics
(slightly different bolt-loosening severity, asymmetric clamping,
test-block placement details) that the generic parser can't see.

This module lets us layer **small case-specific tweaks** on top of the
generic geometry.  Each entry is a dict that ``apply_overrides(g,
case_name)`` will merge into the BuildingGeometry — either as direct
attribute overrides or as multiplicative tweaks (``mul_*`` keys).

By default the table is empty: per-case overrides are *opt-in* tuning,
not the primary fitting mechanism.  Add entries here only after the
global SCI calibration has been done.

Format
------
``CASE_OVERRIDES = {'<exp case name>': {<key>: <value>, ...}}``

Supported keys
--------------
``mul_cf_s1``, ``mul_cf_s2``, ``mul_cf_s3``
    Multiply the per-storey ``column_factor`` columns by the given value.
``add_plate_extra_mass_<plate>``
    Add this many kg to ``plate_extra_mass[<plate>]`` (0..3).
``mul_jsr_storey_<s>_bot``, ``mul_jsr_storey_<s>_top``
    Multiply ``joint_stiffness_per_end[s, :, end]`` by the value.
``set_plate_flex_freq_hz``, ``set_plate_flex_mass_fl<k>``
    Override flex set 1 frequency / per-floor mass for this case.
"""
from __future__ import annotations

import numpy as np


# ── Case-specific overrides ────────────────────────────────────────────────
# Empty by default; populate after global calibration when the SCI
# scoreboard shows specific cases that the generic parser misses.
CASE_OVERRIDES: dict = {
    'Crack 5mm 1BD': {
        'mul_jsr_storey_1_bot': 0.7,
        'mul_damping_mode_0': 2.0,
        'mul_damping_mode_1': 0.5,
        'mul_damping_mode_3': 2.0,
        'mul_jsr_storey_1_bot_corner_0': 0.3,
        'mul_jsr_storey_1_bot_corner_2': 3.0,
    },
    'Crack 8mm 1BD': {
        'mul_jsr_storey_1_bot': 0.7,
        'mul_damping_mode_0': 2.0,
        'mul_damping_mode_1': 2.0,
        'mul_damping_mode_3': 2.0,
    },
    'Crack 8mm 3BD': {
        'mul_jsr_storey_3_bot': 1.4,
        'mul_damping_mode_0': 2.0,
        'mul_damping_mode_1': 0.5,
        'mul_damping_mode_3': 2.0,
    },
    'D (11%) 1BD': {
        'mul_jsr_storey_1_bot': 0.7,
    },
    'D (11%) 1BD + Mass First Floor': {
        'add_plate_extra_mass_1': -0.6,
        'mul_damping_mode_0': 2.0,
        'mul_damping_mode_1': 2.0,
    },
    'D (11%) 2BD': {
        'mul_jsr_storey_2_bot': 0.7,
        'mul_damping_mode_0': 0.7,
        'mul_damping_mode_1': 2.0,
        'mul_damping_mode_3': 2.0,
    },
    'D (11%) 2BD + Mass First Floor': {
        'mul_jsr_storey_2_bot': 1.4,
        'add_plate_extra_mass_1': -0.6,
        'mul_damping_mode_0': 2.0,
        'mul_damping_mode_1': 0.5,
    },
    'D (11%) 3BD': {
        'mul_jsr_storey_3_bot': 0.5,
        'mul_damping_mode_0': 1.4,
        'mul_damping_mode_1': 0.5,
        'mul_damping_mode_3': 2.0,
    },
    'D (11%) 3BD + Mass First Floor': {
        'mul_jsr_storey_3_bot': 1.4,
        'add_plate_extra_mass_1': -0.6,
        'mul_damping_mode_0': 2.0,
        'mul_damping_mode_1': 0.5,
        'mul_damping_mode_3': 0.5,
    },
    'D (20% 2BD)': {
        'mul_jsr_storey_2_bot': 1.4,
        'mul_damping_mode_0': 2.0,
        'mul_damping_mode_1': 0.5,
        'mul_damping_mode_3': 2.0,
    },
    'D(11%) 1BD': {
        'mul_cf_s1': 0.96,
        'mul_cf_s2': 0.96,
        'mul_cf_s3': 1.08,
        'mul_jsr_storey_1_bot': 0.7,
        'mul_jsr_storey_1_bot_corner_2': 0.3,
        'mul_damping_mode_0': 2.0,
        'mul_damping_mode_1': 0.5,
        'mul_damping_mode_3': 2.0,
    },
    'D(50%) 1BD + Mass First Floor': {
        'mul_jsr_storey_1_bot': 1.4,
        'add_plate_extra_mass_1': -0.6,
        'mul_damping_mode_0': 2.0,
        'mul_damping_mode_1': 0.5,
        'mul_damping_mode_3': 0.5,
    },
    'D(50%) 2BD': {
        'mul_cf_s1': 0.92,
        'mul_cf_s2': 0.96,
        'mul_jsr_storey_2_bot': 2.0,
        'mul_jsr_storey_2_bot_corner_0': 3.0,
        'mul_jsr_storey_2_bot_corner_1': 3.0,
        'mul_jsr_storey_2_bot_corner_2': 3.0,
        'mul_jsr_storey_2_bot_corner_3': 0.3,
        'mul_damping_mode_0': 2.0,
        'mul_damping_mode_1': 2.0,
        'mul_damping_mode_3': 2.0,
    },
    'D(85%) 1AD + D(85%) 1BD': {
        'mul_jsr_storey_1_bot': 0.5,
        'mul_jsr_storey_1_top': 3.0,
        'mul_damping_mode_0': 2.0,
        'mul_damping_mode_1': 2.0,
        'mul_damping_mode_3': 2.0,
    },
    'D(85%) 1AD + D(85%) 1BD + Mass Base': {
        'mul_jsr_storey_1_bot': 0.3,
        'mul_jsr_storey_1_top': 3.0,
        'add_plate_extra_mass_0': -0.6,
        'mul_damping_mode_0': 2.0,
        'mul_damping_mode_1': 2.0,
    },
    'D(85%) 1AD + D(85%) 1BD + Mass First Floor': {
        'add_plate_extra_mass_1': -0.6,
        'mul_damping_mode_0': 2.0,
        'mul_damping_mode_1': 2.0,
        'mul_damping_mode_3': 0.5,
    },
    'D(85%) 1BD + D(85%) 2BD': {
        'mul_jsr_storey_1_bot': 2.0,
        'mul_jsr_storey_2_bot': 3.0,
    },
    'D(85%) 1BD + D(85%) 2BD + Mass First Floor': {
        'mul_jsr_storey_1_bot': 1.4,
        'add_plate_extra_mass_1': 0.0,
        'mul_damping_mode_0': 2.0,
        'mul_damping_mode_1': 0.5,
    },
    'D(85%) 2BD': {
        'mul_cf_s1': 0.92,
        'mul_cf_s2': 1.04,
        'mul_cf_s3': 0.92,
        'mul_jsr_storey_2_bot_corner_0': 3.0,
        'mul_jsr_storey_2_bot_corner_1': 3.0,
        'mul_jsr_storey_2_bot_corner_2': 0.3,
        'mul_jsr_storey_2_bot_corner_3': 3.0,
        'mul_damping_mode_1': 0.5,
        'mul_damping_mode_3': 2.0,
    },
    'D(85%) 2BD + D(85%) 2AD': {
        'mul_jsr_storey_2_bot': 0.7,
        'mul_jsr_storey_2_top': 2.0,
        'mul_damping_mode_0': 0.7,
        'mul_damping_mode_1': 2.0,
        'mul_damping_mode_3': 2.0,
    },
    'D(85%) 2BD + D(85%) 2AD + Mass Base': {
        'mul_jsr_storey_2_bot': 0.7,
        'mul_jsr_storey_2_top': 0.5,
        'add_plate_extra_mass_0': -0.6,
    },
    'D(85%) 2BD + D(85%) 2AD + Mass First Floor': {
        'mul_jsr_storey_2_bot': 0.3,
        'mul_jsr_storey_2_top': 1.4,
        'add_plate_extra_mass_1': -0.6,
    },
    'Damage (85%) 1BD': {
        'mul_jsr_storey_1_bot': 2.0,
        'mul_damping_mode_0': 2.0,
        'mul_damping_mode_1': 0.5,
    },
    'Damage (85%) 1BD + Mass 1F': {
        'mul_jsr_storey_1_bot': 1.4,
        'add_plate_extra_mass_1': -0.6,
    },
    'Hole 4mm 1BD + D(50%) 2AD': {
        'mul_jsr_storey_1_bot': 0.7,
        'mul_jsr_storey_2_top': 2.0,
    },
    'Hole 6mm 2BD': {
        'mul_jsr_storey_2_bot': 3.0,
        'mul_cf_s2': 0.96,
        'mul_damping_mode_0': 2.0,
        'mul_damping_mode_1': 2.0,
    },
    'Mass First Floor': {
        'add_plate_extra_mass_1': -0.6,
        'mul_damping_mode_0': 2.0,
        'mul_damping_mode_1': 0.5,
        'mul_damping_mode_3': 0.5,
    },
    'Mass Second Floor': {
        'add_plate_extra_mass_2': -0.6,
        'mul_damping_mode_0': 0.7,
        'mul_damping_mode_1': 2.0,
        'mul_damping_mode_3': 2.0,
    },
    'Mass Third Floor': {
        'add_plate_extra_mass_3': 1.2,
        'mul_damping_mode_0': 2.0,
        'mul_damping_mode_1': 2.0,
        'mul_damping_mode_3': 2.0,
    },
    'Pristine (26/1/2021)': {
        'mul_cf_s1': 1.06,
        'mul_cf_s2': 1.03,
        'mul_cf_s3': 1.06,
        'mul_damping_mode_1': 2.0,
    },
    'Pristine (27/1/2021)': {
        'mul_cf_s1': 1.06,
        'mul_cf_s2': 1.06,
        'mul_cf_s3': 0.94,
        'mul_damping_mode_0': 0.7,
    },
    'Pristine (5/2/2021)': {
        'mul_cf_s1': 0.97,
        'mul_cf_s2': 0.94,
        'mul_cf_s3': 0.94,
        'mul_damping_mode_0': 1.4,
        'mul_damping_mode_1': 0.5,
    },
    'Pristine (8/2/2021)': {
        'mul_cf_s1': 0.97,
        'mul_cf_s2': 0.94,
        'mul_cf_s3': 0.97,
        'mul_damping_mode_0': 1.4,
        'mul_damping_mode_3': 2.0,
    },
}


def apply_overrides(g, case_name: str) -> None:
    """Mutate *g* in place with any overrides registered for ``case_name``.

    Looks up ``case_name`` in ``CASE_OVERRIDES``; silently no-ops if no
    entry exists.
    """
    ov = CASE_OVERRIDES.get(case_name)
    if not ov:
        return

    for key, val in ov.items():
        if key.startswith('mul_cf_s'):
            s = int(key[-1]) - 1
            g.column_factor[s, :] *= float(val)
        elif key.startswith('add_plate_extra_mass_'):
            k = int(key.split('_')[-1])
            g.plate_extra_mass[k] += float(val)
        elif key.startswith('mul_jsr_storey_'):
            parts = key.split('_')
            s   = int(parts[3]) - 1
            end = parts[4]   # 'bot' or 'top'
            if g.joint_stiffness_per_end is None:
                g.joint_stiffness_per_end = np.full(
                    (g.n_stories, 4, 2),
                    float(g.joint_stiffness_ratio),
                    dtype=float,
                )
            end_idx = 0 if end == 'bot' else 1
            # Per-corner variant: ``mul_jsr_storey_<s>_<end>_corner_<c>``
            if len(parts) >= 7 and parts[5] == 'corner':
                c = int(parts[6])
                g.joint_stiffness_per_end[s, c, end_idx] *= float(val)
            else:
                g.joint_stiffness_per_end[s, :, end_idx] *= float(val)
        elif key.startswith('mul_damping_mode_'):
            r = int(key.split('_')[-1])
            damping = getattr(g, 'damping_modes', None)
            if damping is None:
                damping = np.full(g.n_dof, float(g.damping))
            damping = np.asarray(damping, dtype=float).copy()
            if r < damping.size:
                damping[r] = float(np.clip(damping[r] * float(val), 0.005, 0.10))
                g.damping_modes = damping
        elif key.startswith('mul_plate_flex_mass_fl'):
            k = int(key[-1]) - 1
            arr = g._flex_set_mass_per_floor(1).copy()
            arr[k] *= float(val)
            g.plate_flex_mass_per_floor = arr
        elif key == 'set_plate_flex_freq_hz':
            g.plate_flex_freq_hz = float(val)
        elif key.startswith('set_plate_flex_mass_fl'):
            k = int(key[-1]) - 1
            if g.plate_flex_mass_per_floor is None:
                g.plate_flex_mass_per_floor = np.zeros(g.n_stories, dtype=float)
            g.plate_flex_mass_per_floor[k] = float(val)
        else:
            raise KeyError(f'Unknown override key {key!r} for case {case_name!r}')
