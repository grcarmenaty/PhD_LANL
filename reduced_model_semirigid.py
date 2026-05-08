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
                                                 #   plate flexural attachment
    plate_flex_mass:    float = 0.0              # default modal mass (kg)
    # Optional per-plate override (length-n_stories array, ordered fl1..fl_n).
    # Where it is > 0 it replaces ``plate_flex_mass`` on that plate.  Lets
    # the calibrator turn the flex DOF on for one plate (typically floor 3,
    # which is where the experimental rise above 75 Hz lives) without
    # disturbing the other plates' SCI.
    plate_flex_mass_per_floor: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.column_factor is None:
            self.column_factor = np.ones((self.n_stories, 4), dtype=float)
        else:
            self.column_factor = np.asarray(self.column_factor, dtype=float)
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

    @property
    def _flex_mass_per_floor(self) -> np.ndarray:
        """Length-n_stories vector of flex-attachment mass per upper plate."""
        if self.plate_flex_mass_per_floor is not None:
            arr = np.asarray(self.plate_flex_mass_per_floor, dtype=float)
            if arr.size != self.n_stories:
                raise ValueError(
                    f'plate_flex_mass_per_floor must have length {self.n_stories}'
                )
            return arr
        return np.full(self.n_stories, float(self.plate_flex_mass))

    @property
    def has_plate_flex(self) -> bool:
        return (self.plate_flex_freq_hz > 0.0
                and bool(np.any(self._flex_mass_per_floor > 0.0)))

    @property
    def n_flex(self) -> int:
        return int((self._flex_mass_per_floor > 0.0).sum())

    def _flex_index_for_plate(self, plate_index: int):
        """Internal: position of plate *plate_index* in the active flex list,
        or None if its flex mass is zero.
        """
        masses = self._flex_mass_per_floor
        if masses[plate_index - 1] <= 0.0:
            return None
        active_before = int((masses[:plate_index - 1] > 0.0).sum())
        return active_before

    @property
    def n_dof_base(self) -> int:
        """DOF count without flex DOFs (base Y + (x, y, θ) per upper plate)."""
        return 1 + 3 * self.n_stories

    @property
    def n_dof(self) -> int:
        return self.n_dof_base + self.n_flex

    def upper_dof_slice(self, plate_index: int) -> Tuple[int, int, int]:
        if not (1 <= plate_index <= self.n_stories):
            raise ValueError("plate_index must be in 1..n_stories")
        s = plate_index
        return 3 * s - 2, 3 * s - 1, 3 * s

    def flex_dof(self, plate_index: int):
        """Index of the flexural DOF for plate *plate_index* (1..n_stories),
        or None when this plate has no flex mass attached.
        """
        if not (1 <= plate_index <= self.n_stories):
            raise ValueError('plate_index must be in 1..n_stories')
        slot = self._flex_index_for_plate(plate_index)
        if slot is None:
            return None
        return self.n_dof_base + slot


# ---------------------------------------------------------------------------
# Stiffness and mass matrices
# ---------------------------------------------------------------------------
def _column_lateral_stiffnesses(geom: BuildingGeometry) -> Tuple[float, float]:
    """Translational stiffness of one nominal (factor=1) column.

    Applies the semi-rigid correction k_eff = k_ff * JSR/(JSR+6) when
    ``geom.joint_stiffness_ratio`` is finite.
    """
    L  = geom.storey_height
    I_yy = geom.col_ly * geom.col_lx ** 3 / 12.0   # deflection in X
    I_xx = geom.col_lx * geom.col_ly ** 3 / 12.0   # deflection in Y
    kx = 12.0 * geom.young * I_yy / L ** 3
    ky = 12.0 * geom.young * I_xx / L ** 3

    jsr = geom.joint_stiffness_ratio
    if not np.isinf(jsr) and jsr > 0.0:
        cf = jsr / (jsr + 6.0)
        kx *= cf
        ky *= cf

    return kx, ky


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
        omega_f = 2.0 * np.pi * geom.plate_flex_freq_hz
        masses  = geom._flex_mass_per_floor
        for s in range(1, n + 1):
            iq = geom.flex_dof(s)
            if iq is None:
                continue
            m_s = float(masses[s - 1])
            k_flex = omega_f * omega_f * m_s
            _, iy, _ = geom.upper_dof_slice(s)
            K[iy, iy] += k_flex
            K[iq, iq] += k_flex
            K[iy, iq] -= k_flex
            K[iq, iy] -= k_flex
    cx, cy = geom.plate_centroid
    centres = geom.column_attachment_points()
    kx, ky  = _column_lateral_stiffnesses(geom)

    for s in range(n):
        for c, (xc_abs, yc_abs) in enumerate(centres):
            factor = float(geom.column_factor[s, c])
            if factor <= 0.0:
                continue
            scale   = factor ** 4
            xc      = xc_abs - cx
            yc      = yc_abs - cy
            K_local = _column_local_K(kx * scale, ky * scale)
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

    # Plate flexural DOFs (one extra mass per upper plate, only where active)
    if geom.has_plate_flex:
        masses = geom._flex_mass_per_floor
        for s in range(1, n + 1):
            iq = geom.flex_dof(s)
            if iq is None:
                continue
            M[iq, iq] = float(masses[s - 1])

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

def compute_frf_matrix(freq_array: np.ndarray,
                        inputs:  Sequence,
                        outputs: Sequence,
                        geom: BuildingGeometry,
                        damping=None) -> np.ndarray:
    """Vectorised (n_freq, n_outputs, n_inputs) accelerance FRF matrix.

    ``damping`` can be:
      * ``None``  — use ``geom.damping`` (scalar, uniform)
      * ``float`` — uniform modal damping ratio
      * array-like (length >= 1) — per-mode damping ratios ordered by
        increasing frequency; the last value is repeated for any
        remaining modes.  Rigid-body modes are skipped.
    """
    if damping is None:
        damping = geom.damping

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
