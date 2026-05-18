"""P1.2: widened domain-randomization ranges for the synth generator.

This is a *sibling* of variation.py that defines the broadened jitter
distribution called for in the sim-to-real plan.  It is NOT wired
into the active generation pipeline; P2.1 will promote it to
variation.py and re-run generate_dataset.py.

Why each range was widened
--------------------------
* `JITTER_JSR`: log-uniform [0.3, 3.0] per joint, not scalar +/- 5 %.
  case_overrides.py already documents per-case JSR multipliers
  spanning 0.3-3.0x for individual corners/ends, so the existing
  per-case calibration data say the bolted-joint scatter on the IQS
  rig spans ~10x.  The +/-5 % jitter on top of the calibrated nominal
  is roughly 60x too narrow to cover real bolted-joint behaviour.

* `JITTER_DAMPING`: log-uniform [0.5, 3.0] PER MODE, not scalar
  +/- 20 %.  Real bolted-joint damping varies 2-4x sample-to-sample,
  and the calibrated per-mode damping in calibration_result.npz has
  modes with ratios ranging from 0.6 % to 2.6 % -- a 4.3x spread that
  the +/-20 % scalar jitter cannot reproduce.

* `JITTER_E`, `JITTER_RHO`: widened modestly.  +/-2 % E is on the
  tight side for an aluminium structure; +/-5 % is closer to a real
  manufacturing tolerance band, especially given the floor plates are
  thin and have visible roller-marks on the IQS specimen.

* New `JITTER_SENSOR_GAIN`: +/-10 % per channel.  The 9
  accelerometers in the IQS test have their own calibration drift;
  the only sensor compensation currently in the codebase is the S2
  polarity flip.

* New `JITTER_SENSOR_PHASE_DEG`: +/-2 deg per channel.  Phase mis-
  match between channels biases cross-channel features (CFDAC, FRF
  imaginary part).

* New `JITTER_INPUT_GAIN`: 0.7 - 1.4 multiplicative on the chirp
  amplitude.  The IQS shaker is driven open-loop and the effective
  force at the structure depends on shaker compliance.  Per-sample
  varying input gain teaches the model that absolute FRF magnitude
  is *not* a reliable damage indicator on its own.

* New `INPUT_SHELF_FREQ_HZ`, `INPUT_SHELF_DB`: random first-order
  low-shelf filter at 30 Hz, +/-3 dB gain.  Models the dominant
  frequency-coloring effect of a real shaker / load cell stack.

The sanity unit-test at the bottom confirms `geometry_from_params`
produces well-conditioned mass and stiffness matrices at the extreme
ends of the new ranges.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from reduced_model_semirigid import BuildingGeometry  # noqa: E402
sys.path.insert(0, str(_REPO))
from damage_scenarios import _pristine_geom            # noqa: E402

from ml_pipeline.case_design import (   # noqa: E402
    CaseLocation,
    TYPE_PRISTINE, TYPE_BOLT, TYPE_CRACK, TYPE_HOLE, TYPE_MASS,
    END_BD, END_AD, SEVERITY_BOUNDS,
)


# Material / geometric jitter (widened from variation.py).
JITTER_E         = (0.95, 1.05)   # Young's modulus +/- 5 %  (was +/- 2 %)
JITTER_RHO       = (0.97, 1.03)   # density        +/- 3 %  (was +/- 1 %)
JITTER_PLATE_DIM = (0.99,  1.01)  # plate dims +/-1 %  (was +/- 0.5 %)
JITTER_COL_DIM   = (0.99,  1.01)  # column dims +/-1 % (was +/- 0.5 %)
JITTER_PLATE_MASS_KG = (-0.05, 0.05)   # +/-50 g per plate (unchanged)
JITTER_BASE_MASS_KG  = (-0.10, 0.10)   # base-extra-mass +/-100 g (unchanged)

# Heavy-tailed / per-mode / per-sensor jitter (NEW or widened).
JITTER_JSR_LOG_LO,    JITTER_JSR_LOG_HI    = 0.3, 3.0   # log-uniform, was +/- 5 % scalar
JITTER_DAMPING_LOG_LO,JITTER_DAMPING_LOG_HI= 0.5, 3.0   # log-uniform, per mode

JITTER_SENSOR_GAIN      = (0.90, 1.10)   # per-channel multiplicative (NEW)
JITTER_SENSOR_PHASE_DEG = (-2.0, 2.0)    # per-channel small phase rotation (NEW)
JITTER_INPUT_GAIN       = (0.70, 1.40)   # chirp amplitude scaling (NEW)
INPUT_SHELF_FREQ_HZ     = 30.0           # low-shelf knee (constant)
INPUT_SHELF_DB          = (-3.0, 3.0)    # shelf gain +/-3 dB (NEW)


@dataclass
class SampleParamsV2:
    """All scalars needed to reproduce one sample (P1.2 widened set)."""

    # Identical to v1 -------------------------------------------------
    sample_id:      int
    type_code:      int
    storey:         int
    end:            int
    severity:       float
    severity_units: str

    young_factor:    float
    density_factor:  float
    plate_lx_factor: float
    plate_ly_factor: float
    plate_lz_factor: float
    col_lx_factor:   float
    col_ly_factor:   float
    base_extra_mass_dkg: float
    plate_extra_mass_dkg_fl1: float
    plate_extra_mass_dkg_fl2: float
    plate_extra_mass_dkg_fl3: float

    # P1.2 -- per-end JSR factor table (n_stories x 4 columns x 2 ends).
    # Stored as a flat list of length 24 in the same order as
    # BuildingGeometry.joint_stiffness_per_end.flatten().  This lets us
    # serialise it through the SampleParams record without changing
    # the HDF5 schema beyond a new params/<key> column per slot.
    jsr_factor_per_end: list[float]

    # P1.2 -- per-mode damping multiplier (length = number of modes).
    damping_factor_per_mode: list[float]

    # P1.2 -- per-channel sensor gain (9 elements).
    sensor_gain: list[float]

    # P1.2 -- per-channel sensor phase rotation, in radians (9 elements).
    sensor_phase_rad: list[float]

    # P1.2 -- input chirp gain and low-shelf gain in dB.
    input_gain: float
    input_shelf_db: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# ── physical effects of damage parameters (unchanged from v1) ────────────────
def hole_ratio(diameter_mm: float) -> float:
    return float(max(0.80, 1.0 - 0.005 * diameter_mm))


def crack_ratio(size_mm: float) -> float:
    return float(max(0.70, 1.0 - 0.008 * size_mm))


def bolt_jsr_ratio(percent: float) -> float:
    pcts   = np.array([5.0, 11.0, 20.0, 50.0, 85.0, 95.0])
    ratios = np.array([0.94, 0.85, 0.70, 0.55, 0.39, 0.30])
    return float(np.clip(np.interp(percent, pcts, ratios), 0.05, 0.99))


# ── per-sample sampler ───────────────────────────────────────────────────────
def _uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _probe_n_modes(geom: BuildingGeometry) -> int:
    """Return the per-sample mode count expected by the damping model."""
    damping = getattr(geom, 'damping_modes', None)
    if damping is not None and np.size(damping) > 0:
        return int(np.size(damping))
    # Fall back to whatever the reduced-order model uses by default.
    return 16


def sample_params(sample_id: int, loc: CaseLocation,
                   rng: np.random.Generator,
                   n_modes: Optional[int] = None,
                   ) -> SampleParamsV2:
    """Draw a SampleParamsV2 record for one (sample_id, location) pair.

    ``n_modes`` controls the length of ``damping_factor_per_mode``;
    callers that don't yet know it can pass ``None`` and we will
    probe the pristine geometry once.
    """
    tcode = loc.type_code
    lo, hi = SEVERITY_BOUNDS[tcode]
    severity = _uniform(rng, lo, hi) if tcode != TYPE_PRISTINE else 0.0

    if n_modes is None:
        n_modes = _probe_n_modes(_pristine_geom())

    # Per-end JSR factors: n_stories=3, 4 columns, 2 ends = 24 entries
    jsr_per_end = [_log_uniform(rng, JITTER_JSR_LOG_LO, JITTER_JSR_LOG_HI)
                       for _ in range(3 * 4 * 2)]

    damping_per_mode = [_log_uniform(rng, JITTER_DAMPING_LOG_LO,
                                            JITTER_DAMPING_LOG_HI)
                            for _ in range(n_modes)]

    sensor_gain = [_uniform(rng, *JITTER_SENSOR_GAIN) for _ in range(9)]
    sensor_phase = [_uniform(rng, *JITTER_SENSOR_PHASE_DEG) * np.pi / 180.0
                       for _ in range(9)]

    return SampleParamsV2(
        sample_id=sample_id,
        type_code=tcode,
        storey=loc.storey,
        end=loc.end,
        severity=severity,
        severity_units={0: "", 1: "percent", 2: "mm", 3: "mm", 4: "kg"}[tcode],
        young_factor=    _uniform(rng, *JITTER_E),
        density_factor=  _uniform(rng, *JITTER_RHO),
        plate_lx_factor= _uniform(rng, *JITTER_PLATE_DIM),
        plate_ly_factor= _uniform(rng, *JITTER_PLATE_DIM),
        plate_lz_factor= _uniform(rng, *JITTER_PLATE_DIM),
        col_lx_factor=   _uniform(rng, *JITTER_COL_DIM),
        col_ly_factor=   _uniform(rng, *JITTER_COL_DIM),
        base_extra_mass_dkg=    _uniform(rng, *JITTER_BASE_MASS_KG),
        plate_extra_mass_dkg_fl1=_uniform(rng, *JITTER_PLATE_MASS_KG),
        plate_extra_mass_dkg_fl2=_uniform(rng, *JITTER_PLATE_MASS_KG),
        plate_extra_mass_dkg_fl3=_uniform(rng, *JITTER_PLATE_MASS_KG),
        jsr_factor_per_end=jsr_per_end,
        damping_factor_per_mode=damping_per_mode,
        sensor_gain=sensor_gain,
        sensor_phase_rad=sensor_phase,
        input_gain=_uniform(rng, *JITTER_INPUT_GAIN),
        input_shelf_db=_uniform(rng, *INPUT_SHELF_DB),
    )


# ── geometry construction from params ────────────────────────────────────────
def _columns_all() -> list[int]:
    return [0, 1, 2, 3]


def geometry_from_params(p: SampleParamsV2) -> BuildingGeometry:
    """Build the calibrated geometry and apply this sample's perturbations.

    P1.2 differences from variation.py:
    * Per-end JSR jitter table (24 independent factors) instead of one
      scalar applied uniformly.
    * Per-mode damping factor list applied to ``damping_modes`` when
      that attribute exists; falls back to a scalar otherwise.
    """
    g = _pristine_geom()

    g.young            *= p.young_factor
    g.density          *= p.density_factor
    g.plate_lx         *= p.plate_lx_factor
    g.plate_ly         *= p.plate_ly_factor
    g.plate_lz         *= p.plate_lz_factor
    g.col_lx           *= p.col_lx_factor
    g.col_ly           *= p.col_ly_factor
    g.base_extra_mass  += p.base_extra_mass_dkg

    # Per-mode damping if available.
    damping_modes = getattr(g, 'damping_modes', None)
    if damping_modes is not None and np.size(damping_modes) > 0:
        scale = np.asarray(p.damping_factor_per_mode[:np.size(damping_modes)])
        g.damping_modes = (np.asarray(damping_modes).flatten() *
                              scale).reshape(np.shape(damping_modes))
    else:
        # Fall back to applying the geometric-mean factor as a scalar.
        g.damping *= float(np.exp(np.mean(np.log(p.damping_factor_per_mode))))

    # Per-end JSR jitter.
    if g.joint_stiffness_per_end is None:
        g.joint_stiffness_per_end = np.full((g.n_stories, 4, 2),
                                              float(g.joint_stiffness_ratio),
                                              dtype=float)
    jsr_arr = np.asarray(p.jsr_factor_per_end).reshape((3, 4, 2))
    g.joint_stiffness_per_end = g.joint_stiffness_per_end * jsr_arr

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
        # NOTE: P2.2 will replace this with the per-corner asymmetric
        # variant (BD -> corners 0,1; AD -> corners 2,3) that lifts
        # the ROM ceiling on col_location.  Keeping symmetric for now
        # so P1.2 is purely about jitter widening.
        g.column_factor[p.storey, _columns_all()] *= r ** 0.25
    elif p.type_code == TYPE_HOLE:
        r = hole_ratio(p.severity)
        g.column_factor[p.storey, _columns_all()] *= r ** 0.25
    elif p.type_code == TYPE_MASS:
        plate = int(p.end)
        if 0 <= plate <= 3:
            g.plate_extra_mass[plate] += p.severity

    return g


# ── post-processing helpers (used at FRF synthesis time in P2.1) ─────────────
def apply_sensor_calibration(H: np.ndarray, sensor_gain: list[float],
                                sensor_phase_rad: list[float]) -> np.ndarray:
    """Apply per-channel multiplicative gain and additive phase rotation
    to a complex FRF tensor of shape ``(n_freq, 9)``.

    P1.2 is a *definitions* step; the actual call site lands in P2.1.
    """
    g = np.asarray(sensor_gain, dtype=np.complex64)
    ph = np.exp(1j * np.asarray(sensor_phase_rad, dtype=np.float64))
    return H.astype(np.complex64) * (g * ph)[None, :].astype(np.complex64)


def apply_input_coloring(freqs_hz: np.ndarray, F_fft: np.ndarray,
                            input_gain: float, shelf_db: float,
                            knee_hz: float = INPUT_SHELF_FREQ_HZ) -> np.ndarray:
    """Apply a 1st-order low-shelf coloring filter and overall gain to
    the input force spectrum ``F_fft = rfft(chirp)``.

    H_shelf(f) = 1 + (10**(shelf_db/20) - 1) / (1 + (f/knee)**2)

    A negative ``shelf_db`` rolls off the low frequencies; positive
    boosts them.  At ``f = knee_hz`` the shelf is halfway between
    unity and 10**(shelf_db/20).
    """
    boost = 10.0 ** (shelf_db / 20.0)
    shelf = 1.0 + (boost - 1.0) / (1.0 + (np.asarray(freqs_hz) / knee_hz) ** 2)
    return F_fft * (input_gain * shelf).astype(F_fft.dtype)


# ── self-test ────────────────────────────────────────────────────────────────
def _self_test() -> None:
    """Smoke test that the new ranges produce well-conditioned matrices."""
    rng = np.random.default_rng(20260518)
    print("variation_v2 self-test")
    for tc in (TYPE_PRISTINE, TYPE_BOLT, TYPE_CRACK, TYPE_HOLE, TYPE_MASS):
        loc = CaseLocation(type_code=tc, storey=0, end=0,
                              label=f"selftest_{tc}")
        for trial in range(50):
            p = sample_params(trial, loc, rng)
            g = geometry_from_params(p)
            # The per-end JSR table after jitter + damage must stay
            # positive everywhere.
            assert (g.joint_stiffness_per_end > 0).all(), \
                f"JSR went non-positive at trial {trial}/type {tc}"
            assert (g.column_factor > 0).all(), \
                f"column_factor went non-positive at trial {trial}/type {tc}"
        print(f"  type {tc}: 50 trials OK")


if __name__ == "__main__":
    _self_test()
