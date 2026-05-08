"""Reduced-order model of the LANL 3SBB benchmark, tuned for the IQS dataset.

Semi-rigid joint model: JSR = k_r * L / EI  (tuned to match experiment)
Effective column stiffness: k_eff = k_ff * JSR/(JSR+6)

Sensor layout (9 accelerometers, Y-direction, HDF5 channel order)
------------------------------------------------------------------
Channel  Signal  Location           Position
  ch0     S2     Base plate         (xl, +Y face, base z)   [*INVERTED polarity*]
  ch1     S5     Floor 3 (top)      (xl, +Y face, z3)
  ch2     S6     Floor 2 (middle)   (xl, +Y face, z2)
  ch3     S7     Floor 1 (bottom)   (xl, +Y face, z1)
  ch4     S8     Base plate         (xh, +Y face, base z)
  ch5     S11    Floor 3 (top)      (xh, +Y face, z3)   [same as S5, no torsion excitation]
  ch6     S12    Floor 2 (middle)   (xh, +Y face, z2)   [same as S6]
  ch7     S13    Floor 1 (bottom)   (xh, +Y face, z1)   [same as S7]
  ch8     S14    Base plate         (cx, -Y face, base z)  [shaker reference]

Note: S2 (ch0) has inverted mounting polarity in the experiment; the model
outputs the physically correct sign.  When comparing model to experiment,
negate the experimental S2 channel (or the synthetic ch0).

Identification method: empirical mode-shape analysis of Pristine dataset.
At 3 Hz (quasi-static): S2 shows ratio -1.0 to S14 -> inverted mounting.
Mode-shape stability test across resonances identifies S2, S8, S14 as base
plate sensors (constant ratio ~1.0).  Floor pairs identified by amplitude
ordering at first resonance (~17 Hz above the base):
  S5/S11 -> Floor 3  (|H|/|H_S14| ~ 1.94, largest amplification)
  S6/S12 -> Floor 2  (|H|/|H_S14| ~ 0.93)
  S7/S13 -> Floor 1  (|H|/|H_S14| ~ 0.16, lowest amplification)

Shaker input
------------
Mid-face of the base plate's -Y side, Y-direction: (cx, 0, z_base)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE))
sys.path.insert(1, str(_HERE.parent / "pymodal" / "examples" / "los_alamos_3story"))

from reduced_model_semirigid import BuildingGeometry, modes, compute_frf_matrix
import params as P

JSR_TUNED    = 7.9009
N_FREQ       = 1601
FREQ_MAX     = 100.0
FREQ_ARRAY   = np.linspace(0.0, FREQ_MAX, N_FREQ)
SIGNAL_NAMES = [2, 5, 6, 7, 8, 11, 12, 13, 14]   # HDF5 column order
N_OUTPUTS    = 9


def _sensor_positions(geom):
    """Nine Y-direction sensors matching the HDF5 channel order.

    Layout derived from empirical mode-shape analysis of the Pristine dataset.
    Since the shaker is at the plate centre and drives only translational Y
    modes, xl-face and xh-face sensors on the same floor return identical
    FRFs.  xl is used for S2/S5/S6/S7; xh for S8/S11/S12/S13.

    Returns list of ((x, y, z), direction) tuples in HDF5 channel order.
    """
    xl = geom.col_lx / 2.0          # low-x  column centre  (~0.0127 m)
    xh = geom.plate_lx - xl         # high-x column centre  (~0.2923 m)
    cx = geom.plate_lx / 2.0        # plate centre x        (~0.1525 m)
    PL = geom.plate_ly              # +Y face of plate      (0.305 m)
    z0 = geom.plate_z_centre(0)    # base plate centre z
    z1 = geom.plate_z_centre(1)    # floor 1 centre z
    z2 = geom.plate_z_centre(2)    # floor 2 centre z
    z3 = geom.plate_z_centre(3)    # floor 3 centre z

    return [
        ((xl,  PL, z0), 'Y'),   # ch0  S2   base plate, +Y face, xl  [INVERTED in exp.]
        ((xl,  PL, z3), 'Y'),   # ch1  S5   floor 3,    +Y face, xl
        ((xl,  PL, z2), 'Y'),   # ch2  S6   floor 2,    +Y face, xl
        ((xl,  PL, z1), 'Y'),   # ch3  S7   floor 1,    +Y face, xl
        ((xh,  PL, z0), 'Y'),   # ch4  S8   base plate, +Y face, xh
        ((xh,  PL, z3), 'Y'),   # ch5  S11  floor 3,    +Y face, xh
        ((xh,  PL, z2), 'Y'),   # ch6  S12  floor 2,    +Y face, xh
        ((xh,  PL, z1), 'Y'),   # ch7  S13  floor 1,    +Y face, xh
        ((cx,  0., z0), 'Y'),   # ch8  S14  base plate, -Y face, cx  [shaker ref]
    ]


def _input_position(geom):
    """Shaker at the mid-face of the base plate's -Y side, Y-direction."""
    cx = geom.plate_lx / 2.0
    z0 = geom.plate_z_centre(0)
    return [((cx, 0.0, z0), 'Y')]


def pristine_geometry(screw_mass_per_joint=0.0):
    """Return calibrated pristine BuildingGeometry.

    Parameters are loaded from ``calibration_result.npz`` when available;
    falls back to nominal JSR=7.9009, cf=1, damping=0.005 otherwise.
    """
    _CAL = _HERE / 'calibration_result.npz'
    if _CAL.exists():
        cal = np.load(_CAL)
        g = BuildingGeometry(
            joint_stiffness_ratio = float(cal['jsr']),
            damping               = float(cal['damping']),
            base_extra_mass       = float(cal['base_extra_mass']),
            screw_mass_per_joint  = screw_mass_per_joint,
        )
        g.column_factor = np.array([[float(cal['cf_s1'])] * 4,
                                    [float(cal['cf_s2'])] * 4,
                                    [float(cal['cf_s3'])] * 4], dtype=float)
        return g
    # Fallback: nominal uncalibrated geometry
    return BuildingGeometry(column_factor=np.ones((3, 4)),
                            joint_stiffness_ratio=JSR_TUNED,
                            damping=0.005,
                            screw_mass_per_joint=screw_mass_per_joint)


def compute_frf_3sbb(geom, freq_array=None):
    if freq_array is None:
        freq_array = FREQ_ARRAY
    return compute_frf_matrix(freq_array, _input_position(geom),
                              _sensor_positions(geom), geom)


def natural_frequencies(geom):
    return modes(geom)
