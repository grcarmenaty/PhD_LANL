"""V2a: v1 domain randomisation + v2 asymmetric Crack/Hole damage.

Ablation of the v2 chunk regeneration (`variation_v2.py`), which
bundled two changes:

  1. **P1.2** — widened domain randomisation (per-end JSR ×24,
     per-mode damping ×9, per-channel gain/phase ×9 each, input
     gain + low-shelf).
  2. **P2.1 / P2.2** — asymmetric Crack/Hole damage (single end pair
     instead of all 4 corners).

The v2 chunk regeneration FAILED the pre-registered floor test
(`is_hole/mlp/modal` BA 0.661 → 0.500), but the regression cannot be
attributed cleanly to either change. **v2a isolates change (2): it
uses v1's domain randomisation and v1's SampleParams record, but
v2's asymmetric-damage geometry.**

This file is identical to `variation.py` except for the two damage
application lines (`column_factor[..., _columns_all()]` →
`column_factor[..., corners]` with the exponent 0.25 → 0.5 to keep
the total stiffness reduction comparable, matching v2's approach).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from reduced_model_semirigid import BuildingGeometry  # noqa: E402
sys.path.insert(0, str(_REPO))
from damage_scenarios import _pristine_geom            # noqa: E402

# Re-use v1's SampleParams and supporting functions so the HDF5 schema
# is identical to v1 (avoids the SampleParamsV2 list-field bridge).
from ml_pipeline.variation import (   # noqa: E402
    SampleParams, sample_params,
    hole_ratio, crack_ratio, bolt_jsr_ratio,
    JITTER_E, JITTER_RHO, JITTER_JSR, JITTER_DAMPING,
    JITTER_PLATE_DIM, JITTER_COL_DIM,
    JITTER_PLATE_MASS_KG, JITTER_BASE_MASS_KG,
)

from ml_pipeline.case_design import (   # noqa: E402
    CaseLocation,
    TYPE_PRISTINE, TYPE_BOLT, TYPE_CRACK, TYPE_HOLE, TYPE_MASS,
    END_BD, END_AD, SEVERITY_BOUNDS,
)


def geometry_from_params(p: SampleParams) -> BuildingGeometry:
    """Apply v1 jitter + v2-style asymmetric Crack/Hole damage."""
    g = _pristine_geom()

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

    if p.type_code == TYPE_BOLT:
        r = bolt_jsr_ratio(p.severity)
        end_idx = 1 if p.end == END_AD else 0
        g.joint_stiffness_per_end[p.storey, :, end_idx] *= r
    elif p.type_code == TYPE_CRACK:
        r = crack_ratio(p.severity)
        # V2-style asymmetric: only 2 corners (paired by end), exponent ^0.5
        corners = (0, 1) if p.end != END_AD else (2, 3)
        g.column_factor[p.storey, list(corners)] *= r ** 0.5
    elif p.type_code == TYPE_HOLE:
        r = hole_ratio(p.severity)
        corners = (0, 1) if p.end != END_AD else (2, 3)
        g.column_factor[p.storey, list(corners)] *= r ** 0.5
    elif p.type_code == TYPE_MASS:
        plate = int(p.end)
        if 0 <= plate <= 3:
            g.plate_extra_mass[plate] += p.severity

    return g
