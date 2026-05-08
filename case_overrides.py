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
    'D(11%) 1BD': {
        'mul_jsr_storey_1_bot': 0.5,
    },
    'D(85%) 1BD + D(85%) 2BD': {
        'mul_jsr_storey_1_bot': 2.0,
        'mul_jsr_storey_2_bot': 2.0,
    },
    'D(85%) 1AD + D(85%) 1BD': {
        'mul_jsr_storey_1_bot': 0.7,
        'mul_jsr_storey_1_top': 2.0,
    },
    'D(85%) 2BD + D(85%) 2AD': {
        'mul_jsr_storey_2_bot': 0.7,
        'mul_jsr_storey_2_top': 2.0,
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
            g.joint_stiffness_per_end[s, :, end_idx] *= float(val)
        elif key == 'set_plate_flex_freq_hz':
            g.plate_flex_freq_hz = float(val)
        elif key.startswith('set_plate_flex_mass_fl'):
            k = int(key[-1]) - 1
            if g.plate_flex_mass_per_floor is None:
                g.plate_flex_mass_per_floor = np.zeros(g.n_stories, dtype=float)
            g.plate_flex_mass_per_floor[k] = float(val)
        else:
            raise KeyError(f'Unknown override key {key!r} for case {case_name!r}')
