"""Physical-parameter sampling for the synthetic damage dataset.

Every sample draws material, geometric and damage parameters from
bounded distributions.  Variation is **physical** (manufacturing
tolerances, real material spread, calibrated joint scatter) — there is
no measurement noise.

All randomness flows from a single ``numpy.random.Generator`` so the
dataset is reproducible from a seed.
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
# After importing reduced_model_semirigid, sys.path[0] is pymodal's
# los_alamos_3story example dir (it self-inserts).  Put the PhD_LANL
# repo back at the head so the local damage_scenarios.py wins.
sys.path.insert(0, str(_REPO))
from damage_scenarios import _pristine_geom            # noqa: E402

from ml_pipeline.case_design import (   # noqa: E402
    CaseLocation,
    TYPE_PRISTINE, TYPE_BOLT, TYPE_CRACK, TYPE_HOLE, TYPE_MASS,
    END_BD, END_AD, SEVERITY_BOUNDS,
)


# Multiplicative jitter ranges (uniform).  Centred on 1.0 unless stated.
JITTER_E         = (0.98, 1.02)   # Young's modulus +/-2 %
JITTER_RHO       = (0.99, 1.01)   # density        +/-1 %
JITTER_JSR       = (0.95, 1.05)   # joint stiffness +/-5 %
JITTER_DAMPING   = (0.80, 1.20)   # modal damping  +/-20 %
JITTER_PLATE_DIM = (0.995, 1.005) # plate dimensions +/-0.5 %
JITTER_COL_DIM   = (0.995, 1.005) # column dimensions +/-0.5 %
JITTER_PLATE_MASS_KG = (-0.05, 0.05)   # +/-50 g per plate (absolute)
JITTER_BASE_MASS_KG  = (-0.10, 0.10)   # base-extra-mass +/-100 g


@dataclass
class SampleParams:
    """All scalars needed to reproduce one sample."""

    sample_id:      int
    type_code:      int
    storey:         int    # -1 if N/A
    end:            int    # -1 if N/A (mass: plate index 0..3)
    severity:       float  # interpretation depends on type
    severity_units: str

    young_factor:    float
    density_factor:  float
    jsr_factor:      float
    damping_factor:  float
    plate_lx_factor: float
    plate_ly_factor: float
    plate_lz_factor: float
    col_lx_factor:   float
    col_ly_factor:   float
    base_extra_mass_dkg: float
    plate_extra_mass_dkg_fl1: float
    plate_extra_mass_dkg_fl2: float
    plate_extra_mass_dkg_fl3: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# ── physical effects of damage parameters ────────────────────────────────────
def hole_ratio(diameter_mm: float) -> float:
    """Storey-stiffness ratio for a drilled hole of given diameter.

    Linear interpolation of the calibrated discrete table
    (4 mm -> 0.98, 6 mm -> 0.97) extended to the [1, 6] mm range.
    """
    return float(max(0.80, 1.0 - 0.005 * diameter_mm))


def crack_ratio(size_mm: float) -> float:
    """Storey-stiffness ratio for a through-thickness crack.

    Linear interpolation of the calibrated discrete table
    (5 mm -> 0.96, 8 mm -> 0.94) extended to [1, 8] mm.
    """
    return float(max(0.70, 1.0 - 0.008 * size_mm))


def bolt_jsr_ratio(percent: float) -> float:
    """Per-end JSR multiplier for a bolt loosened to ``percent`` severity.

    Smooth interpolation through the calibrated points
    (11→0.85, 20→0.70, 50→0.55, 85→0.39) extrapolated linearly out to
    [5, 95] % severity.
    """
    # Piecewise-linear interp on the calibrated anchors.
    pcts   = np.array([5.0, 11.0, 20.0, 50.0, 85.0, 95.0])
    ratios = np.array([0.94, 0.85, 0.70, 0.55, 0.39, 0.30])
    return float(np.clip(np.interp(percent, pcts, ratios), 0.05, 0.99))


# ── per-sample sampler ───────────────────────────────────────────────────────
def _uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def sample_params(sample_id: int, loc: CaseLocation,
                   rng: np.random.Generator) -> SampleParams:
    """Draw a SampleParams record for one (sample_id, location) pair."""
    tcode = loc.type_code
    lo, hi = SEVERITY_BOUNDS[tcode]
    severity = _uniform(rng, lo, hi) if tcode != TYPE_PRISTINE else 0.0

    return SampleParams(
        sample_id=sample_id,
        type_code=tcode,
        storey=loc.storey,
        end=loc.end,
        severity=severity,
        severity_units={0: "", 1: "percent", 2: "mm", 3: "mm", 4: "kg"}[tcode],
        young_factor=    _uniform(rng, *JITTER_E),
        density_factor=  _uniform(rng, *JITTER_RHO),
        jsr_factor=      _uniform(rng, *JITTER_JSR),
        damping_factor=  _uniform(rng, *JITTER_DAMPING),
        plate_lx_factor= _uniform(rng, *JITTER_PLATE_DIM),
        plate_ly_factor= _uniform(rng, *JITTER_PLATE_DIM),
        plate_lz_factor= _uniform(rng, *JITTER_PLATE_DIM),
        col_lx_factor=   _uniform(rng, *JITTER_COL_DIM),
        col_ly_factor=   _uniform(rng, *JITTER_COL_DIM),
        base_extra_mass_dkg=    _uniform(rng, *JITTER_BASE_MASS_KG),
        plate_extra_mass_dkg_fl1=_uniform(rng, *JITTER_PLATE_MASS_KG),
        plate_extra_mass_dkg_fl2=_uniform(rng, *JITTER_PLATE_MASS_KG),
        plate_extra_mass_dkg_fl3=_uniform(rng, *JITTER_PLATE_MASS_KG),
    )


# ── geometry construction from params ────────────────────────────────────────
def _columns_all() -> list[int]:
    return [0, 1, 2, 3]


def geometry_from_params(p: SampleParams) -> BuildingGeometry:
    """Build the calibrated geometry and apply this sample's perturbations."""
    g = _pristine_geom()

    # Material / geometry jitter.
    g.young            *= p.young_factor
    g.density          *= p.density_factor
    g.damping          *= p.damping_factor
    g.plate_lx         *= p.plate_lx_factor
    g.plate_ly         *= p.plate_ly_factor
    g.plate_lz         *= p.plate_lz_factor
    g.col_lx           *= p.col_lx_factor
    g.col_ly           *= p.col_ly_factor
    g.base_extra_mass  += p.base_extra_mass_dkg

    # Apply JSR jitter to the per-end JSR table (or scalar fallback).
    if g.joint_stiffness_per_end is None:
        g.joint_stiffness_per_end = np.full((g.n_stories, 4, 2),
                                              float(g.joint_stiffness_ratio),
                                              dtype=float)
    g.joint_stiffness_per_end *= p.jsr_factor

    # Plate-extra-mass jitter (per upper floor).
    g.plate_extra_mass[1] += p.plate_extra_mass_dkg_fl1
    g.plate_extra_mass[2] += p.plate_extra_mass_dkg_fl2
    g.plate_extra_mass[3] += p.plate_extra_mass_dkg_fl3

    # Damage on top of the jittered baseline.
    if p.type_code == TYPE_BOLT:
        r = bolt_jsr_ratio(p.severity)
        end_idx = 1 if p.end == END_AD else 0
        g.joint_stiffness_per_end[p.storey, :, end_idx] *= r
    elif p.type_code == TYPE_CRACK:
        r = crack_ratio(p.severity)
        g.column_factor[p.storey, _columns_all()] *= r ** 0.25
    elif p.type_code == TYPE_HOLE:
        r = hole_ratio(p.severity)
        g.column_factor[p.storey, _columns_all()] *= r ** 0.25
    elif p.type_code == TYPE_MASS:
        plate = int(p.end)        # 0..3
        if 0 <= plate <= 3:
            g.plate_extra_mass[plate] += p.severity
    # Pristine: nothing extra.

    return g
