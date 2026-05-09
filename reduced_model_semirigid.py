"""Reduced-order shear-building model for the LANL 3SBB benchmark with
semi-rigid bolted-connection correction and screw-mass lumped-mass addition.

Physical background — joint flexibility
----------------------------------------
Original model: column ends fixed → k_ff = 12EI/L³
Real structure: bolted connection modelled as rotational spring k_r.
Slope-deflection for symmetric end springs in a sway frame:
    k_eff = k_ff * JSR / (JSR + 6)    where JSR = k_r * L / EI
Limits:
  * JSR → ∞  →  k_eff = k_ff  (fixed-fixed)
  * JSR  = 0  →  k_eff = 0    (pinned-pinned)
  * JSR  = 6  →  k_eff = k_ff / 2

Physical background — screw mass
---------------------------------
Each column–plate junction has 2 screws (steel, ~2.9 g each by default).
The 3SBB has 4 columns per storey × 3 storeys = 12 columns, with 2 joints
per column (top + bottom).  Counting joints per plate:

  Base plate:      4 column bases  → 4 joints                →  8 screws
  Floor 1 or 2:    4 col tops from below + 4 col bases above  → 16 screws
  Floor 3 (roof):  4 column tops                              →  8 screws

``screw_mass_per_joint`` is the total mass of the 2 screws at one junction
(default 0 g to keep backward compatibility).  The added mass is treated as
uniformly distributed over the plate for the translational DOF, and the extra
rotational inertia about the plate centroid is computed from the joint
x-positions.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import eigh

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES = os.path.join(os.path.dirname(_HERE), 'pymodal', 'examples', 'los_alamos_3story')
if _EXAMPLES not in sys.path:
    sys.path.insert(0, _EXAMPLES)
import params as P  # noqa: E402

Point = Tuple[float, float, float]
Direction = str  # 'X', 'Y' or 'Z'


# ---------------------------------------------------------------------------
# Geometry container
# ---------------------------------------------------------------------------
@dataclass
class BuildingGeometry:
    """All scalar parameters for one realisation of the building.

    Defaults pull from :mod:`params` so the unperturbed geometry matches
    the nominal LANL 3-storey benchmark.

    ``column_factor`` is a ``(n_stories, 4)`` array of section-scale
    factors.  Both column dimensions are multiplied by the corresponding
    factor, so the lateral stiffness of that column scales as
    ``factor**4``.

    ``joint_stiffness_ratio`` (JSR = k_r * L / EI) encodes bolted-connection
    flexibility.  Use ``float('inf')`` for the original fixed-fixed model.

    ``screw_mass_per_joint`` is the total mass (kg) of the 2 screws at each
    column–plate junction.  Default 0.0 (backward compatible).
    """

    plate_lx: float = P.PLATE_LX
    plate_ly: float = P.PLATE_LY
    plate_lz: float = P.PLATE_LZ
    col_lx:   float = P.COL_LX
    col_ly:   float = P.COL_LY
    inter_storey_gap: float = P.INTER_STOREY_GAP
    column_gap:       float = P.COLUMN_GAP
    n_stories: int = P.N_STORIES
    young:    float = P.ALU_E
    poisson:  float = P.ALU_NU
    density:  float = P.ALU_RHO
    damping:  float = P.MODAL_DAMPING
    rail_direction: str = P.RAIL_DIRECTION
    column_factor: np.ndarray = field(default=None)
    joint_stiffness_ratio: float = float('inf')  # JSR; inf = fixed-fixed
    # Per-end JSR override.  Shape (n_stories, 4, 2): last axis is
    # [bottom_end, top_end].  Where it is finite, replaces the scalar
    # ``joint_stiffness_ratio`` for that specific (storey, corner, end).
    # The lateral stiffness then uses the *asymmetric* semi-rigid formula
    #     k_eff = k_ff · (J_t·J_b + J_t + J_b) / (J_t·J_b + 4(J_t+J_b) + 12)
    # which reduces to k_ff · J/(J+6) when J_t = J_b = J.  This lets the
    # damage scenarios distinguish AD (top-end) from BD (bottom-end) bolt
    # loosening — without it both reduce the same column lateral
    # stiffness identically, which fails on cases like ``D(85%) 2BD`` and
    # ``D(85%) 1AD + D(85%) 1BD``.
    joint_stiffness_per_end: np.ndarray = field(default=None)
    screw_mass_per_joint: float = 0.0            # kg per joint (2 screws)
    base_extra_mass: float = 0.0                 # kg — shaker + attachment on base plate
    # plate_extra_mass is a length-(n_stories+1) array of extra translational
    # mass added to each plate (plate 0 = base, plate n = roof).  Used to
    # model the IQS "Mass <Floor>" damage scenarios that strap an extra block
    # on top of a specific plate.  Stored independently of base_extra_mass
    # so that calibrated shaker mass and added test mass remain separable.
    plate_extra_mass: np.ndarray = field(default=None)
    # Out-of-plane plate flexural mode (one extra DOF per upper plate,
    # tuned-attachment style: a hidden mass `plate_flex_mass` connected to
    # the plate's Y-translation DOF by a spring of stiffness
    # `(2π · plate_flex_freq_hz)² · plate_flex_mass`.  When freq or mass is
    # zero the flex DOFs are inert and the model reduces to the previous
    # rigid-plate version.  Captures the experimental rise in floor-3 FRFs
    # near 95 Hz that the rigid-plate Y-chain cannot reach.
    plate_flex_freq_hz: float = 0.0              # natural frequency of the
                                                 #   1st plate flexural mode
    plate_flex_mass:    float = 0.0              # default modal mass for
                                                 #   1st flex mode (kg)
    # Optional per-plate override (length-n_stories array, ordered fl1..fl_n).
    # Where it is > 0 it replaces ``plate_flex_mass`` on that plate.  Lets
    # the calibrator turn the flex DOF on for one plate (typically floor 3,
    # which is where the experimental rise above 75 Hz lives) without
    # disturbing the other plates' SCI.
    plate_flex_mass_per_floor: np.ndarray = field(default=None)
    # Second flex mode (one tuned attachment per plate, second resonance
    # higher than the first).  Captures the broad-band high-frequency rise
    # that extends past 95 Hz on floor-3 sensors.
    plate_flex2_freq_hz: float = 0.0
    plate_flex2_mass_per_floor: np.ndarray = field(default=None)
    # When True, output sensors at the plate's z are wired to the flex
    # DOF instead of the main plate-Y DOF for plates that have an active
    # flex set.  Models the physical sensor location: S5/S11 sit on top
    # of the plate, so they "see" the upper half of a discretised plate
    # rather than the lower half (where the columns attach).  Removes
    # the H_yy anti-resonance from the rise band (H_qy has no zero
    # below its resonance).
    sensors_on_flex: bool = False
    # Grounded oscillators: each is a hidden mass that lives "outside" the
    # building, connected to ground by its own spring, and coupled to a
    # specific plate's Y DOF only through a dashpot (no spring coupling).
    # Solved by the direct-inversion FRF path because the resulting damping
    # matrix is non-proportional (the modal-superposition path can't see the
    # cross-mode dissipation).  Each entry is a dict:
    #   {'plate':         int (1..n_stories),
    #    'mass':          float (kg),
    #    'freq_hz':       float (Hz)  → grounded spring k = (2π f)² m,
    #    'damp_coupling': float (N·s/m, optional, defaults to 0)}
    grounded_oscillators: list = field(default_factory=list)
    # Free-form dashpot couplings: list of (dof_a, dof_b, c_val) tuples added
    # directly to the C matrix (with the standard
    # `[[+c, -c], [-c, +c]]` sub-block).  Used by the direct-inversion FRF
    # path when modelling non-proportional damping that doesn't fit the
    # `grounded_oscillators` shape.
    dashpot_couplings: list = field(default_factory=list)

    def __post_init__(self):
        if self.column_factor is None:
            self.column_factor = np.ones((self.n_stories, 4), dtype=float)
        else:
            self.column_factor = np.asarray(self.column_factor, dtype=float)
        if self.joint_stiffness_per_end is not None:
            arr = np.asarray(self.joint_stiffness_per_end, dtype=float)
            if arr.shape != (self.n_stories, 4, 2):
                raise ValueError(
                    'joint_stiffness_per_end must have shape '
                    f'({self.n_stories}, 4, 2); got {arr.shape}'
                )
            self.joint_stiffness_per_end = arr
        if self.plate_extra_mass is None:
            self.plate_extra_mass = np.zeros(self.n_stories + 1, dtype=float)
        else:
            self.plate_extra_mass = np.asarray(self.plate_extra_mass, dtype=float)
            if self.plate_extra_mass.size != self.n_stories + 1:
                raise ValueError(
                    f'plate_extra_mass must have length {self.n_stories + 1}'
                )

    # ── coordinate helpers ------------------------------------------------
    @property
    def storey_height(self) -> float:
        return self.plate_lz + self.inter_storey_gap

    @property
    def plate_centroid(self) -> Tuple[float, float]:
        return self.plate_lx / 2.0, self.plate_ly / 2.0

    def plate_z_centre(self, k: int) -> float:
        return k * self.storey_height + self.plate_lz / 2.0

    def column_box_centres(self) -> np.ndarray:
        x_lo = self.col_lx / 2.0
        x_hi = self.plate_lx - self.col_lx / 2.0
        y_neg = -self.col_ly / 2.0 - self.column_gap
        y_pos = self.plate_ly + self.col_ly / 2.0 + self.column_gap
        return np.array([(x_lo, y_neg), (x_hi, y_neg),
                         (x_lo, y_pos), (x_hi, y_pos)])

    def column_attachment_points(self) -> np.ndarray:
        x_lo = self.col_lx / 2.0
        x_hi = self.plate_lx - self.col_lx / 2.0
        return np.array([(x_lo, 0.0), (x_hi, 0.0),
                         (x_lo, self.plate_ly), (x_hi, self.plate_ly)])

    def _flex_set_mass_per_floor(self, which: int) -> np.ndarray:
        """Per-plate mass for flex set 1 (`which=1`) or set 2 (`which=2`)."""
        if which == 1:
            arr = self.plate_flex_mass_per_floor
            scalar = self.plate_flex_mass
        elif which == 2:
            arr = self.plate_flex2_mass_per_floor
            scalar = 0.0
        else:
            raise ValueError(f'flex set must be 1 or 2, got {which}')
        if arr is not None:
            arr = np.asarray(arr, dtype=float)
            if arr.size != self.n_stories:
                raise ValueError(
                    f'plate_flex{which}_mass_per_floor must have length '
                    f'{self.n_stories}'
                )
            return arr
        return np.full(self.n_stories, float(scalar))

    @property
    def _flex_mass_per_floor(self) -> np.ndarray:
        """Backwards-compat: per-plate mass of the *first* flex mode."""
        return self._flex_set_mass_per_floor(1)

    def _flex_freq_hz(self, which: int) -> float:
        return float(self.plate_flex_freq_hz if which == 1
                     else self.plate_flex2_freq_hz)

    def _flex_set_active(self, which: int) -> bool:
        return (self._flex_freq_hz(which) > 0.0
                and bool(np.any(self._flex_set_mass_per_floor(which) > 0.0)))

    @property
    def has_plate_flex(self) -> bool:
        return self._flex_set_active(1) or self._flex_set_active(2)

    @property
    def n_flex(self) -> int:
        n = 0
        if self._flex_set_active(1):
            n += int((self._flex_set_mass_per_floor(1) > 0.0).sum())
        if self._flex_set_active(2):
            n += int((self._flex_set_mass_per_floor(2) > 0.0).sum())
        return n

    def _active_count_set1(self) -> int:
        if not self._flex_set_active(1):
            return 0
        return int((self._flex_set_mass_per_floor(1) > 0.0).sum())

    def _flex_index_for_plate(self, plate_index: int, which: int = 1):
        """Position of plate *plate_index* in the active flex list of
        ``which`` (1 or 2), or None if its mass is zero.
        """
        if not self._flex_set_active(which):
            return None
        masses = self._flex_set_mass_per_floor(which)
        if masses[plate_index - 1] <= 0.0:
            return None
        active_before = int((masses[:plate_index - 1] > 0.0).sum())
        # Set-2 entries live after all set-1 entries
        offset = self._active_count_set1() if which == 2 else 0
        return offset + active_before

    @property
    def n_dof_base(self) -> int:
        """DOF count without flex DOFs (base Y + (x, y, θ) per upper plate)."""
        return 1 + 3 * self.n_stories

    @property
    def n_grounded(self) -> int:
        """Number of active grounded oscillators."""
        return sum(1 for go in (self.grounded_oscillators or [])
                   if float(go.get('mass', 0)) > 0
                   and float(go.get('freq_hz', 0)) > 0)

    def grounded_dof(self, idx: int) -> int:
        """Index in the global state vector of the *idx*-th grounded
        oscillator (0-based, in the order they appear in the list)."""
        return self.n_dof_base + self.n_flex + idx

    @property
    def n_dof(self) -> int:
        return self.n_dof_base + self.n_flex + self.n_grounded

    def upper_dof_slice(self, plate_index: int) -> Tuple[int, int, int]:
        if not (1 <= plate_index <= self.n_stories):
            raise ValueError("plate_index must be in 1..n_stories")
        s = plate_index
        return 3 * s - 2, 3 * s - 1, 3 * s

    def flex_dof(self, plate_index: int, which: int = 1):
        """Index of the *which*-th flexural DOF for plate *plate_index*
        (1..n_stories), or None when this plate has no flex mass attached
        in that set.
        """
        if not (1 <= plate_index <= self.n_stories):
            raise ValueError('plate_index must be in 1..n_stories')
        slot = self._flex_index_for_plate(plate_index, which)
        if slot is None:
            return None
        return self.n_dof_base + slot


# ---------------------------------------------------------------------------
# Stiffness and mass matrices
# ---------------------------------------------------------------------------
def _semirigid_factor(j_t: float, j_b: float) -> float:
    """Asymmetric semi-rigid joint correction.

        k_eff = k_ff · (J_t·J_b + J_t + J_b) / (J_t·J_b + 4(J_t+J_b) + 12)

    Reduces to ``J/(J+6)`` when J_t = J_b = J.  ``+inf`` is treated as
    fully rigid (no reduction); a zero or negative value as pinned (k=0).
    """
    if not np.isfinite(j_t) and not np.isfinite(j_b):
        return 1.0                       # both rigid → no correction
    if j_t <= 0.0 or j_b <= 0.0:
        return 0.0                       # at least one pinned → no shear
    if not np.isfinite(j_t):             # top rigid, bottom finite
        # Limit J_t → ∞ of the asymmetric formula:
        # ratio → (J_b + 1) / (J_b + 4)
        return (j_b + 1.0) / (j_b + 4.0)
    if not np.isfinite(j_b):
        return (j_t + 1.0) / (j_t + 4.0)
    num = j_t * j_b + j_t + j_b
    den = j_t * j_b + 4.0 * (j_t + j_b) + 12.0
    return num / den


def _column_base_stiffnesses(geom: BuildingGeometry) -> Tuple[float, float]:
    """Bare fixed-fixed lateral stiffness of one column (no JSR correction).

    Returns ``(k_ff_x, k_ff_y)`` — the per-storey shear stiffness for X
    and Y direction translations of a column with both ends fully rigid
    and the section dimensions at ``column_factor = 1``.  The semi-rigid
    correction is applied per-column inside ``stiffness_matrix`` because
    each column may have different per-end JSR values.
    """
    L  = geom.storey_height
    I_yy = geom.col_ly * geom.col_lx ** 3 / 12.0   # deflection in X
    I_xx = geom.col_lx * geom.col_ly ** 3 / 12.0   # deflection in Y
    return (12.0 * geom.young * I_yy / L ** 3,
            12.0 * geom.young * I_xx / L ** 3)


def _column_lateral_stiffnesses(geom: BuildingGeometry) -> Tuple[float, float]:
    """Backwards-compat scalar entry point for callers that still want
    one (kx, ky) for a nominal column.  Uses the global scalar
    ``joint_stiffness_ratio`` symmetrically; per-end overrides are
    only honoured by ``stiffness_matrix``.
    """
    kx, ky = _column_base_stiffnesses(geom)
    jsr = geom.joint_stiffness_ratio
    cf = _semirigid_factor(jsr, jsr) if np.isfinite(jsr) and jsr > 0.0 else 1.0
    return kx * cf, ky * cf


def _column_jsr_pair(geom: BuildingGeometry, storey: int, corner: int) -> Tuple[float, float]:
    """Return (J_top, J_bottom) for the column at ``(storey, corner)``."""
    if geom.joint_stiffness_per_end is not None:
        j_b = float(geom.joint_stiffness_per_end[storey, corner, 0])
        j_t = float(geom.joint_stiffness_per_end[storey, corner, 1])
        return j_t, j_b
    j = float(geom.joint_stiffness_ratio)
    return j, j


def _column_local_K(kx_eff: float, ky_eff: float) -> np.ndarray:
    K = np.zeros((4, 4))
    K[0, 0] =  kx_eff;  K[0, 2] = -kx_eff
    K[1, 1] =  ky_eff;  K[1, 3] = -ky_eff
    K[2, 0] = -kx_eff;  K[2, 2] =  kx_eff
    K[3, 1] = -ky_eff;  K[3, 3] =  ky_eff
    return K


def _T_top(xc: float, yc: float) -> np.ndarray:
    return np.array([[1.0, 0.0, -yc],
                     [0.0, 1.0,  xc]])


def _T_base(xc: float, yc: float, geom: BuildingGeometry) -> np.ndarray:
    rail = geom.rail_direction.upper()
    T = np.zeros((2, 1))
    if rail == 'Y':
        T[1, 0] = 1.0
    elif rail == 'X':
        T[0, 0] = 1.0
    else:
        raise ValueError("RAIL_DIRECTION must be 'X' or 'Y'")
    return T


def stiffness_matrix(geom: BuildingGeometry) -> np.ndarray:
    n     = geom.n_stories
    n_dof = geom.n_dof
    n_dof_base = geom.n_dof_base
    K     = np.zeros((n_dof, n_dof))

    # ── Plate flexural DOFs (tuned attachments) ─────────────────────────
    # For every upper plate we attach a hidden mass on a spring of stiffness
    # k_flex = (2π f)² m_flex to the plate's Y DOF.  Adds a 2×2 sub-matrix
    # `[[k, -k], [-k, k]]` in (y_plate, q_flex) at the appropriate indices.
    if geom.has_plate_flex:
        for which in (1, 2):
            if not geom._flex_set_active(which):
                continue
            omega_f = 2.0 * np.pi * geom._flex_freq_hz(which)
            masses  = geom._flex_set_mass_per_floor(which)
            for s in range(1, n + 1):
                iq = geom.flex_dof(s, which)
                if iq is None:
                    continue
                m_s = float(masses[s - 1])
                k_flex = omega_f * omega_f * m_s
                _, iy, _ = geom.upper_dof_slice(s)
                K[iy, iy] += k_flex
                K[iq, iq] += k_flex
                K[iy, iq] -= k_flex
                K[iq, iy] -= k_flex

    # ── Grounded oscillators ────────────────────────────────────────────
    # Diagonal entry only — each grounded oscillator is connected to
    # ground (not to the plate) through its own spring.  The plate
    # coupling is dissipative and lives in the C matrix instead.
    for idx, go in enumerate(geom.grounded_oscillators or []):
        m_g = float(go.get('mass', 0.0))
        f_g = float(go.get('freq_hz', 0.0))
        if m_g <= 0.0 or f_g <= 0.0:
            continue
        ig = geom.grounded_dof(idx)
        K[ig, ig] += (2.0 * np.pi * f_g) ** 2 * m_g
    cx, cy = geom.plate_centroid
    centres = geom.column_attachment_points()
    kx_ff, ky_ff = _column_base_stiffnesses(geom)

    for s in range(n):
        for c, (xc_abs, yc_abs) in enumerate(centres):
            factor = float(geom.column_factor[s, c])
            if factor <= 0.0:
                continue
            j_t, j_b = _column_jsr_pair(geom, s, c)
            cf_jsr   = _semirigid_factor(j_t, j_b)
            scale    = factor ** 4
            kx_eff   = kx_ff * scale * cf_jsr
            ky_eff   = ky_ff * scale * cf_jsr
            xc       = xc_abs - cx
            yc       = yc_abs - cy
            K_local  = _column_local_K(kx_eff, ky_eff)
            T_top   = _T_top(xc, yc)
            if s == 0:
                T_bot = _T_base(xc, yc, geom)
                T = np.zeros((4, 4))
                T[:2, 1:4] = T_top
                T[2:, 0:1] = T_bot
                idx = [0, 1, 2, 3]
            else:
                T_bot = T_top
                T = np.zeros((4, 6))
                T[:2, 0:3] = T_top
                T[2:, 3:6] = T_bot
                ix_top = list(geom.upper_dof_slice(s + 1))
                ix_bot = list(geom.upper_dof_slice(s))
                idx = ix_top + ix_bot
            K_block = T.T @ K_local @ T
            for i_loc, i_glob in enumerate(idx):
                for j_loc, j_glob in enumerate(idx):
                    K[i_glob, j_glob] += K_block[i_loc, j_loc]
    return K


def _screw_mass_per_plate(geom: BuildingGeometry, plate_k: int) -> float:
    """Total added screw mass for plate k (0=base, 1..n=floors).

    Each column–plate junction carries ``screw_mass_per_joint`` kg.
    - Base plate (k=0):   4 column bases attach below → 4 joints
    - Intermediate (1..n-1): 4 column tops below + 4 bases above → 8 joints
    - Top plate (k=n):    4 column tops attach above → 4 joints
    """
    if geom.screw_mass_per_joint == 0.0:
        return 0.0
    n = geom.n_stories
    if plate_k == 0 or plate_k == n:
        n_joints = 4
    else:
        n_joints = 8
    return n_joints * geom.screw_mass_per_joint


def _screw_J_per_plate(geom: BuildingGeometry, plate_k: int) -> float:
    """Rotational inertia contribution from screw mass at plate k.

    Screws are at the 4 column attachment x-positions (xl_i, yl_i).
    ΔJ = Σ_i (m_i_screw / n_joints_i) × r_i² where r_i = distance from
    plate centroid.  Since n_joints per column = 1 at each plate face and
    the mass per screw-point is screw_mass_per_joint / 4 (4 column positions):
    """
    if geom.screw_mass_per_joint == 0.0:
        return 0.0
    m_total = _screw_mass_per_plate(geom, plate_k)
    # Mass per attachment point
    m_pt = m_total / 4.0
    cx, cy = geom.plate_centroid
    pts = geom.column_attachment_points()   # shape (4,2): (x,y) per column
    J = 0.0
    for (xi, yi) in pts:
        J += m_pt * ((xi - cx) ** 2 + (yi - cy) ** 2)
    return J


def mass_matrix(geom: BuildingGeometry) -> np.ndarray:
    n     = geom.n_stories
    n_dof = geom.n_dof
    M     = np.zeros((n_dof, n_dof))

    # Plate mass (aluminium only)
    m_plate = geom.density * geom.plate_lx * geom.plate_ly * geom.plate_lz
    J_plate = m_plate * (geom.plate_lx ** 2 + geom.plate_ly ** 2) / 12.0

    # Base plate (DOF 0 = Y-translation); add shaker / attachment mass
    # plus any user-supplied per-plate extra mass (e.g. a "Mass Base" weight)
    m_base = (m_plate + _screw_mass_per_plate(geom, 0)
              + geom.base_extra_mass + float(geom.plate_extra_mass[0]))
    M[0, 0] = m_base

    # Upper floor plates
    for s in range(1, n + 1):
        ix, iy, it = geom.upper_dof_slice(s)
        m_extra_s = float(geom.plate_extra_mass[s])
        m_s = m_plate + _screw_mass_per_plate(geom, s) + m_extra_s
        J_s = J_plate + _screw_J_per_plate(geom, s)
        M[ix, ix] = m_s
        M[iy, iy] = m_s
        M[it, it] = J_s

    # Plate flexural DOFs (extra masses per upper plate, only where active)
    if geom.has_plate_flex:
        for which in (1, 2):
            if not geom._flex_set_active(which):
                continue
            masses = geom._flex_set_mass_per_floor(which)
            for s in range(1, n + 1):
                iq = geom.flex_dof(s, which)
                if iq is None:
                    continue
                M[iq, iq] = float(masses[s - 1])

    # Grounded oscillators
    for idx, go in enumerate(geom.grounded_oscillators or []):
        m_g = float(go.get('mass', 0.0))
        if m_g <= 0.0 or float(go.get('freq_hz', 0.0)) <= 0.0:
            continue
        ig = geom.grounded_dof(idx)
        M[ig, ig] = m_g

    return M


def modes(geom: BuildingGeometry):
    """Solve generalised eigenvalue problem.  Returns (freqs_hz, V, M)."""
    K   = stiffness_matrix(geom)
    M   = mass_matrix(geom)
    w2, V = eigh(K, M)
    w2  = np.clip(w2, 0.0, None)
    freqs = np.sqrt(w2) / (2.0 * np.pi)
    return freqs, V, M


# ---------------------------------------------------------------------------
# Point-on-mesh → generalised DOF vector
# ---------------------------------------------------------------------------
def point_to_dof_vector(point: Point,
                         direction: Direction,
                         geom: BuildingGeometry) -> np.ndarray:
    direction = direction.upper().strip()
    if direction not in ('X', 'Y', 'Z'):
        raise ValueError("direction must be 'X', 'Y' or 'Z'")
    x, y, z = point
    cx, cy = geom.plate_centroid
    rx = x - cx
    ry = y - cy

    plate_centres = np.array([geom.plate_z_centre(k)
                               for k in range(geom.n_stories + 1)])
    plate_idx = int(np.argmin(np.abs(plate_centres - z)))

    b = np.zeros(geom.n_dof)
    if plate_idx == 0:
        if direction == geom.rail_direction.upper():
            b[0] = 1.0
        return b

    s = plate_idx
    ix, iy, it = geom.upper_dof_slice(s)
    # If the plate has an active flex set and ``sensors_on_flex`` is on,
    # redirect Y outputs to the flex (upper-half) DOF.  H from base to q
    # has no anti-resonance below the new mode (the q-row of the inverse
    # impedance is constant in the numerator), so the rise band fills
    # cleanly while the lower-frequency Y modes still come through the
    # plate-Y → q coupling spring.
    if (direction == 'Y'
            and getattr(geom, 'sensors_on_flex', False)
            and geom._flex_set_active(1)):
        iq = geom.flex_dof(s, 1)
        if iq is not None:
            b[iq] = 1.0
            return b
    if direction == 'X':
        b[ix] = 1.0
        b[it] = -ry
    elif direction == 'Y':
        b[iy] = 1.0
        b[it] = rx
    return b


# ---------------------------------------------------------------------------
# FRF computation (modal superposition, accelerance in mm/s²/N)
# ---------------------------------------------------------------------------

def damping_matrix(geom: BuildingGeometry, damping=None) -> np.ndarray:
    """Construct a full (n_dof, n_dof) viscous damping matrix.

    Two contributions are summed:

    1. **Modal (proportional) damping** built from the per-mode ratios
       ``damping`` (scalar or array, same convention as
       ``compute_frf_matrix``).  Equivalent to the modal-superposition
       kernel when no other damping source is present, since
       ``C = M Φ diag(2ζ_r ω_r) Φᵀ M`` with mass-normalised modes Φ
       reproduces the same per-mode dissipation.
    2. **Explicit dashpot couplings** from ``geom.dashpot_couplings`` —
       a list of ``(dof_a, dof_b, c_val)`` triples added as
       ``[[+c, -c], [-c, +c]]`` sub-blocks — *plus* the dashpot side of
       any active ``geom.grounded_oscillators`` (each contributes an
       off-diagonal coupling `c_yg` between its plate's Y DOF and the
       grounded mass's DOF).

    Building the C matrix lets the direct-inversion FRF path handle
    non-proportional damping, which is what the grounded-oscillator
    formulation needs (the modal-superposition path can't see
    cross-mode dissipation).
    """
    if damping is None:
        damping = geom.damping
    K = stiffness_matrix(geom)
    M = mass_matrix(geom)
    omega2, V = eigh(K, M)
    omega = np.sqrt(np.clip(omega2, 0.0, None))
    n_dof = K.shape[0]

    damp_arr = np.atleast_1d(np.asarray(damping, dtype=float)).ravel()
    if damp_arr.size == 1:
        zeta_full = np.full(n_dof, float(damp_arr[0]))
    else:
        zeta_full = np.full(n_dof, float(damp_arr[-1]))
        m_fill = min(int(damp_arr.size), n_dof)
        zeta_full[:m_fill] = damp_arr[:m_fill]

    # zeta_full is indexed by elastic mode order, but eigh returns rigid
    # modes first.  Skip rigid (omega < 1e-3) and re-align.
    is_rigid = omega < 1e-3
    n_rigid = int(is_rigid.sum())
    diag = np.zeros(n_dof)
    n_el = n_dof - n_rigid
    if n_el > 0:
        zeta_el = np.full(n_el, zeta_full[-1] if zeta_full.size else 0.0)
        m_fill = min(zeta_full.size, n_el)
        zeta_el[:m_fill] = zeta_full[:m_fill]
        diag[~is_rigid] = 2.0 * zeta_el * omega[~is_rigid]

    # C = M V diag V^T M  (mass-normalised modal coordinates)
    C = M @ V @ np.diag(diag) @ V.T @ M

    # Explicit dashpot couplings
    for entry in (geom.dashpot_couplings or []):
        a, b, c_val = entry
        if c_val == 0.0:
            continue
        C[a, a] += c_val
        C[b, b] += c_val
        C[a, b] -= c_val
        C[b, a] -= c_val

    # Grounded-oscillator dashpot side: c_yg between plate Y DOF and the
    # grounded oscillator's DOF.  (Spring side already lives in K.)
    for idx, go in enumerate(geom.grounded_oscillators or []):
        c_val = float(go.get('damp_coupling', 0.0))
        if c_val <= 0.0:
            continue
        plate = int(go.get('plate', 0))
        if not (1 <= plate <= geom.n_stories):
            continue
        _, iy, _ = geom.upper_dof_slice(plate)
        ig = geom.grounded_dof(idx)
        C[iy, iy] += c_val
        C[ig, ig] += c_val
        C[iy, ig] -= c_val
        C[ig, iy] -= c_val

    return C


def compute_frf_direct(freq_array: np.ndarray,
                        inputs:  Sequence,
                        outputs: Sequence,
                        geom: BuildingGeometry,
                        damping=None) -> np.ndarray:
    """Direct frequency-domain accelerance FRF.

    ``H_a(ω) = -ω² · Cᵀ · (-ω²M + jωC + K)⁻¹ · B``

    Handles non-proportional damping (grounded oscillators with dashpot
    coupling, free-form dashpot couplings) that the modal-superposition
    path cannot represent.  Singular K at the rigid-body mode is
    regularised by adding a small grounding stiffness ``ε·M`` so the
    impedance matrix is invertible at every ω; this puts the rigid mode
    at ~1/(2π)·√ε ≈ 0.16 Hz when ε = 1, well below the 5 Hz analysis
    band.
    """
    K = stiffness_matrix(geom)
    M = mass_matrix(geom)
    Cmat = damping_matrix(geom, damping)
    n_dof = K.shape[0]

    # Regularise the rigid mode so Z(ω) is invertible at every frequency.
    omega2_eig, V_eig = eigh(K, M)
    EPS_GROUND = 1.0     # N·s²·m⁻¹ — gives a ~0.16 Hz rigid-mode pole
    for r in np.where(omega2_eig < 1e-6)[0]:
        phi = V_eig[:, r]
        K = K + EPS_GROUND * np.outer(phi, phi)

    B = np.stack([point_to_dof_vector(p, d, geom) for p, d in inputs],  axis=1)
    Co = np.stack([point_to_dof_vector(p, d, geom) for p, d in outputs], axis=1)

    omega = 2.0 * np.pi * np.asarray(freq_array, dtype=float).ravel()
    n_freq = len(omega)
    H = np.zeros((n_freq, Co.shape[1], B.shape[1]), dtype=complex)
    for i, w in enumerate(omega):
        Z = (-w * w) * M + (1j * w) * Cmat + K
        X = np.linalg.solve(Z, B)
        H[i] = (-w * w) * (Co.T @ X)
    return H


def compute_frf_matrix(freq_array: np.ndarray,
                        inputs:  Sequence,
                        outputs: Sequence,
                        geom: BuildingGeometry,
                        damping=None) -> np.ndarray:
    """Vectorised (n_freq, n_outputs, n_inputs) accelerance FRF matrix.

    Dispatches to either modal superposition (fast, default) or direct
    frequency-domain inversion (when non-proportional damping is
    present — grounded oscillators with dashpot coupling, or any
    explicit ``geom.dashpot_couplings``).

    ``damping`` can be:
      * ``None``  — use ``geom.damping`` (scalar, uniform)
      * ``float`` — uniform modal damping ratio
      * array-like (length >= 1) — per-mode damping ratios ordered by
        increasing frequency; the last value is repeated for any
        remaining modes.  Rigid-body modes are skipped.
    """
    if damping is None:
        damping = geom.damping

    # Direct inversion path: needed whenever damping is non-proportional.
    needs_direct = bool(geom.dashpot_couplings) or any(
        float(go.get('damp_coupling', 0.0)) > 0.0
        and float(go.get('mass', 0.0)) > 0.0
        and float(go.get('freq_hz', 0.0)) > 0.0
        for go in (geom.grounded_oscillators or [])
    )
    if needs_direct:
        return compute_frf_direct(freq_array, inputs, outputs, geom, damping)

    freqs, V, _ = modes(geom)
    omega   = 2.0 * np.pi * np.asarray(freq_array, dtype=float).ravel()
    omega_n = 2.0 * np.pi * freqs

    B = np.stack([point_to_dof_vector(p, d, geom) for p, d in inputs],  axis=1)
    C = np.stack([point_to_dof_vector(p, d, geom) for p, d in outputs], axis=1)
    bphi = V.T @ B   # (n_modes, n_inputs)
    cphi = V.T @ C   # (n_modes, n_outputs)

    is_rigid   = omega_n < 1e-3
    is_elastic = ~is_rigid

    n_freq = len(omega)
    H = np.zeros((n_freq, C.shape[1], B.shape[1]), dtype=complex)

    # ── Rigid-body contribution to accelerance ────────────────────────────
    # The 3SBB base plate is free to translate in Y on a rail, so the
    # eigenvalue problem returns one (or more) zero-frequency mode.  In
    # accelerance form `H_a = -ω² · receptance`, that mode collapses to a
    # *constant* `(c·φ_R)(b·φ_R)` ≈ 1/M_total — the "rigid-body floor"
    # that the experiment shows at low frequencies and that the elastic-only
    # modal sum was dropping (giving |H| → 0 as ω → 0).
    if is_rigid.any():
        b_r = bphi[is_rigid]                 # (n_r, n_inputs)
        c_r = cphi[is_rigid]                 # (n_r, n_outputs)
        H_rigid = np.einsum('ro,ri->oi', c_r, b_r)   # (n_outputs, n_inputs)
        H += H_rigid[None, :, :]

    if is_elastic.any():
        wn  = omega_n[is_elastic]   # (n_el,)
        b_e = bphi[is_elastic]      # (n_el, n_inputs)
        c_e = cphi[is_elastic]      # (n_el, n_outputs)

        # Build per-mode zeta vector, shape (n_el,)
        n_el     = int(wn.shape[0])
        damp_arr = np.asarray(damping, dtype=float).ravel()
        if damp_arr.size == 1:
            zeta = np.full(n_el, float(damp_arr[0]))
        else:
            # pad with last value; fill up to min(given, n_el) elements
            zeta    = np.full(n_el, float(damp_arr[-1]))
            m       = min(int(damp_arr.size), n_el)
            zeta[:m] = damp_arr[:m]

        # Modal kernel: H_r(w) = -w^2 / (wn_r^2 - w^2 + 2i*z_r*wn_r*w)
        # Shapes: wn (n_el,), zeta (n_el,), omega (n_freq,)
        wn2    = (wn ** 2).reshape(n_el, 1)        # (n_el, 1)
        zwn    = (zeta * wn).reshape(n_el, 1)       # (n_el, 1)
        om2    = (omega ** 2).reshape(1, n_freq)    # (1, n_freq)
        om1    = omega.reshape(1, n_freq)            # (1, n_freq)

        den    = (wn2 - om2) + 2j * zwn * om1       # (n_el, n_freq)
        kernel = -om2 / den                          # (n_el, n_freq)

        # H[freq, out, in] = sum_r  c_e[r,out] * b_e[r,in] * kernel[r,freq]
        elastic_H = np.einsum('ro,ri,rf->foi', c_e, b_e, kernel)
        H += elastic_H

    return H
