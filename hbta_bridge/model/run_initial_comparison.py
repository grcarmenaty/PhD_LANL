"""HBTA bow-string truss — pure-Python 3D beam-network FEM.

Builds the truss geometry from the sensor positions in
data_100Hz.h5, assembles K and M via 3D Euler-Bernoulli frame
elements (6 DOF/node), solves the modal problem, synthesises the
input-output FRF at the 12 selected sensor channels for a Y-direction
shaker at MVS position P1, and compares against the experimental
median H1 FRF for the UDS class.

Outputs:
  figures/modes_table.txt        - first 12 modal frequencies
  figures/sci_scoreboard.txt     - SCI per scenario
  figures/frf_magnitude.png      - per-channel |H_exp| vs |H_model|
  figures/cfdac_uds.png          - CFDAC matrices (exp self vs model)
  best_params_anchor.json        - first-pass anchor parameters
"""
from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.ndimage as ndi

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)
LOG_DIR = ROOT.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

log = logging.getLogger("hbta_model")
log.setLevel(logging.INFO)
log.handlers.clear()
_fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                          datefmt="%H:%M:%S")
_fh = logging.FileHandler(LOG_DIR / "model_initial.log", mode="w")
_fh.setFormatter(_fmt); log.addHandler(_fh)
_sh = logging.StreamHandler(sys.stdout); _sh.setFormatter(_fmt); log.addHandler(_sh)


# ──────────────────────────────────────────────────────────────────────
# 1. Geometry
# ──────────────────────────────────────────────────────────────────────

# Panel-point X coordinates (from AG sensor positions)
PANEL_X = np.array([-14.0, -10.5, -7.0, -3.5, 0.0, 3.5, 7.0, 10.5, 14.0])
N_PANELS = len(PANEL_X) - 1     # 8 panels of 3.5 m each
TRUSS_HALF_WIDTH = 2.25         # y = ±2.25 m
DECK_Z = 0.6                    # bottom-chord level

# Top-chord (arch) z-coordinate at each panel point (combined N+S sensors)
#   Springers: z = 3.4 at x = ±14 (from AG10/AG18)
#   Quarter:   z = 4.1 at x = ±10.5 (from AG02/AG08)
#   Mid-q:     z = 4.6 at x = ±7   (from AG12/AG16)
#   3/8:       z = 4.9 at x = ±3.5 (from AG04/AG06)
#   Crown:     z = 5.0 at x = 0    (from AG14)
ARCH_Z = np.array([3.4, 4.1, 4.6, 4.9, 5.0, 4.9, 4.6, 4.1, 3.4])

# Sensor selection mirrored from build_hbta_pymodal.py:53
SENSOR_NAMES = ["AG02", "AG04", "AG05", "AG06", "AG08",
                "AG11", "AG13", "AG14", "AG15", "AG17",
                "AL10", "AL26"]


@dataclass
class Joint:
    name: str
    x: float
    y: float
    z: float


@dataclass
class Member:
    j1: int       # joint index
    j2: int
    group: str    # ARCH | BOTCHORD | VERTICAL | DIAGONAL | CROSS_GIRDER
    # section properties filled from material/section table
    A: float = 0.0
    Iy: float = 0.0
    Iz: float = 0.0
    J:  float = 0.0


def build_geometry():
    """Construct joint list and member list for the bow-string truss.

    Layout per panel point (every 3.5 m along X):
      - 4 truss nodes:  bot_N/S (y=±2.25, z=0.6), top_N/S (y=±2.25, z=ARCH_Z)
      - 1 deck midline node: mid (y=0, z=0.6) -- splits the cross-girder
                               and hosts the shaker
    Stringers run longitudinally at y=±0.55 at deck level (z=0.19) but
    are connected only via short cross-girder taps to keep the joint
    count down; instead we add longitudinal beams at y=0 (the midline
    'spine') along the deck.
    """
    joints = []
    j_idx = {}

    for chord, z_fun in [("bot", lambda i: DECK_Z),
                          ("top", lambda i: ARCH_Z[i])]:
        for side, y in [("N", +TRUSS_HALF_WIDTH), ("S", -TRUSS_HALF_WIDTH)]:
            for i, x in enumerate(PANEL_X):
                name = f"{chord}_{side}_{i}"
                j_idx[(chord, side, i)] = len(joints)
                joints.append(Joint(name, float(x), float(y), float(z_fun(i))))

    # Cross-girder midpoint nodes at y=0 (one per panel point)
    for i, x in enumerate(PANEL_X):
        j_idx[("mid", i)] = len(joints)
        joints.append(Joint(f"mid_{i}", float(x), 0.0, DECK_Z))

    members = []

    # Bottom chord beams: 2 sides × 8 segments
    for side in ("N", "S"):
        for i in range(N_PANELS):
            members.append(Member(j_idx[("bot", side, i)],
                                    j_idx[("bot", side, i + 1)],
                                    "BOTCHORD"))

    # Top arch chord beams: 2 sides × 8 segments
    for side in ("N", "S"):
        for i in range(N_PANELS):
            members.append(Member(j_idx[("top", side, i)],
                                    j_idx[("top", side, i + 1)],
                                    "ARCH"))

    # Verticals at every panel point: 2 sides × 9 posts
    for side in ("N", "S"):
        for i in range(len(PANEL_X)):
            members.append(Member(j_idx[("bot", side, i)],
                                    j_idx[("top", side, i)],
                                    "VERTICAL"))

    # X-diagonals per panel: 2 sides × 8 panels × 2 diagonals
    for side in ("N", "S"):
        for i in range(N_PANELS):
            members.append(Member(j_idx[("bot", side, i)],
                                    j_idx[("top", side, i + 1)],
                                    "DIAGONAL"))
            members.append(Member(j_idx[("top", side, i)],
                                    j_idx[("bot", side, i + 1)],
                                    "DIAGONAL"))

    # Cross-girders: split into north-half and south-half at midpoint
    for i in range(len(PANEL_X)):
        members.append(Member(j_idx[("bot", "N", i)],
                                j_idx[("mid", i)],
                                "CROSS_GIRDER"))
        members.append(Member(j_idx[("mid", i)],
                                j_idx[("bot", "S", i)],
                                "CROSS_GIRDER"))

    # Longitudinal "spine" beams at midline deck level (connect mid nodes)
    for i in range(N_PANELS):
        members.append(Member(j_idx[("mid", i)],
                                j_idx[("mid", i + 1)],
                                "STRINGER"))

    # Lateral X-bracing under the deck (in horizontal X-Y plane at z=0.6):
    # one X-pair per panel, between bot_N_i, bot_S_i+1 and bot_S_i, bot_N_i+1
    for i in range(N_PANELS):
        members.append(Member(j_idx[("bot", "N", i)],
                                j_idx[("bot", "S", i + 1)],
                                "LAT_BRACE"))
        members.append(Member(j_idx[("bot", "S", i)],
                                j_idx[("bot", "N", i + 1)],
                                "LAT_BRACE"))

    # Pinned BCs at the 4 corner bottom-chord nodes
    bc_nodes = [j_idx[("bot", "N", 0)],
                j_idx[("bot", "N", N_PANELS)],
                j_idx[("bot", "S", 0)],
                j_idx[("bot", "S", N_PANELS)]]

    log.info(f"geometry: {len(joints)} joints, {len(members)} members, "
             f"{len(bc_nodes)} fixed nodes")
    return joints, members, bc_nodes, j_idx


# ──────────────────────────────────────────────────────────────────────
# 2. Section properties (initial estimates, model.md §4)
# ──────────────────────────────────────────────────────────────────────

SECTION_PROPS = {
    # group:        (A m²,    Iy m⁴,    Iz m⁴,    J m⁴)
    "ARCH":         (0.012,   2.5e-4,   2.5e-4,   5.0e-4),
    "BOTCHORD":     (0.010,   1.5e-4,   4.0e-4,   5.5e-4),
    "VERTICAL":     (0.0045,  5.0e-5,   1.5e-4,   2.0e-4),
    "DIAGONAL":     (0.0045,  5.0e-5,   1.5e-4,   2.0e-4),
    "CROSS_GIRDER": (0.013,   1.0e-4,   6.0e-4,   7.0e-4),
    "STRINGER":     (0.006,   4.0e-5,   1.5e-4,   2.0e-4),
    "LAT_BRACE":    (0.002,   1.5e-6,   1.5e-6,   3.0e-6),
}

E_STEEL = 210.0e9
NU_STEEL = 0.30
G_STEEL = E_STEEL / (2.0 * (1.0 + NU_STEEL))
RHO_STEEL = 7850.0
DECK_LUMPED_KG_PER_M = 200.0    # deck timber + rails distributed at panel pts


def assign_sections(members):
    for m in members:
        A, Iy, Iz, J = SECTION_PROPS[m.group]
        m.A, m.Iy, m.Iz, m.J = A, Iy, Iz, J


# ──────────────────────────────────────────────────────────────────────
# 3. 3D Euler-Bernoulli frame element (6 DOF/node)
# ──────────────────────────────────────────────────────────────────────

def element_kr_local(E, G, A, Iy, Iz, J, L):
    """12x12 stiffness matrix in local coordinates."""
    k = np.zeros((12, 12))
    # axial
    k[0, 0] = k[6, 6] =  E * A / L
    k[0, 6]           = -E * A / L
    # torsion
    k[3, 3] = k[9, 9] =  G * J / L
    k[3, 9]           = -G * J / L
    # bending about z (in local x-y plane) – uses Iz, v=local-y, theta_z
    a = E * Iz / L**3
    k[1, 1] =  12 * a;    k[1, 5]  =  6*L * a;   k[1, 7]  = -12 * a;    k[1, 11] =  6*L * a
    k[5, 5] =  4*L*L * a; k[5, 7]  = -6*L * a;   k[5, 11] =  2*L*L * a
    k[7, 7] =  12 * a;    k[7, 11] = -6*L * a
    k[11, 11] = 4*L*L * a
    # bending about y (in local x-z plane) – uses Iy, w=local-z, theta_y
    b = E * Iy / L**3
    k[2, 2] =  12 * b;    k[2, 4]  = -6*L * b;   k[2, 8]  = -12 * b;    k[2, 10] = -6*L * b
    k[4, 4] =  4*L*L * b; k[4, 8]  =  6*L * b;   k[4, 10] =  2*L*L * b
    k[8, 8] =  12 * b;    k[8, 10] =  6*L * b
    k[10, 10] = 4*L*L * b
    # symmetric
    return k + k.T - np.diag(np.diag(k))


def element_mass_local(rho, A, J, L):
    """12x12 consistent mass matrix in local coordinates."""
    m = np.zeros((12, 12))
    fac = rho * A * L / 420.0
    # axial
    m[0, 0] = m[6, 6] = 140 * fac
    m[0, 6]           =  70 * fac
    # torsion (rotational inertia = J / A · ρA)
    rt = rho * J * L / 6.0
    m[3, 3] = m[9, 9] =  2 * rt
    m[3, 9]           =      rt
    # bending in x-y plane (translational v and rotational theta_z)
    m[1, 1]  = 156 * fac;   m[1, 5]   =  22*L * fac
    m[1, 7]  =  54 * fac;   m[1, 11]  = -13*L * fac
    m[5, 5]  = 4*L*L * fac; m[5, 7]   =  13*L * fac
    m[5, 11] = -3*L*L * fac
    m[7, 7]  = 156 * fac;   m[7, 11]  = -22*L * fac
    m[11, 11] = 4*L*L * fac
    # bending in x-z plane (translational w and rotational theta_y, signs flipped)
    m[2, 2]  = 156 * fac;   m[2, 4]   = -22*L * fac
    m[2, 8]  =  54 * fac;   m[2, 10]  =  13*L * fac
    m[4, 4]  = 4*L*L * fac; m[4, 8]   = -13*L * fac
    m[4, 10] = -3*L*L * fac
    m[8, 8]  = 156 * fac;   m[8, 10]  =  22*L * fac
    m[10, 10] = 4*L*L * fac
    return m + m.T - np.diag(np.diag(m))


def transformation_matrix(p1, p2):
    """3x3 direction cosines from global to local. Local x' along p1→p2."""
    dx = p2 - p1
    L = float(np.linalg.norm(dx))
    if L < 1e-12:
        raise ValueError("zero-length member")
    e1 = dx / L
    # Choose reference vector for e3: world z if member nearly horizontal,
    # else world y. (Avoids near-zero cross-product.)
    if abs(e1[2]) < 0.95:
        ref = np.array([0.0, 0.0, 1.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])
    e2 = np.cross(ref, e1)
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(e1, e2)
    R = np.vstack([e1, e2, e3])
    return R, L


def expand_T(R):
    """Build the 12x12 block-diagonal transformation from 3x3 R."""
    T = np.zeros((12, 12))
    for i in range(4):
        T[3*i:3*i+3, 3*i:3*i+3] = R
    return T


# ──────────────────────────────────────────────────────────────────────
# 4. Global assembly
# ──────────────────────────────────────────────────────────────────────

def assemble(joints, members):
    n_dof = 6 * len(joints)
    K = np.zeros((n_dof, n_dof))
    M = np.zeros((n_dof, n_dof))

    for m in members:
        p1 = np.array([joints[m.j1].x, joints[m.j1].y, joints[m.j1].z])
        p2 = np.array([joints[m.j2].x, joints[m.j2].y, joints[m.j2].z])
        R, L = transformation_matrix(p1, p2)
        T = expand_T(R)
        ke_loc = element_kr_local(E_STEEL, G_STEEL, m.A, m.Iy, m.Iz, m.J, L)
        me_loc = element_mass_local(RHO_STEEL, m.A, m.J, L)
        ke = T.T @ ke_loc @ T
        me = T.T @ me_loc @ T
        # Scatter to global DOFs
        dofs = list(range(6*m.j1, 6*m.j1+6)) + list(range(6*m.j2, 6*m.j2+6))
        for i, gi in enumerate(dofs):
            for j, gj in enumerate(dofs):
                K[gi, gj] += ke[i, j]
                M[gi, gj] += me[i, j]

    # Add lumped deck mass at every bottom-chord node (linear distribution)
    panel_len = PANEL_X[1] - PANEL_X[0]
    lumped = DECK_LUMPED_KG_PER_M * panel_len   # half each on N and S
    for j in joints:
        if "bot_" in j.name:
            i_node = joints.index(j)
            for k in range(3):       # translational only
                M[6*i_node + k, 6*i_node + k] += lumped / 2.0

    return K, M


def apply_pinned_bc(K, M, bc_nodes):
    """Remove translation DOFs for the 4 corner nodes (rotations free).
    Returns reduced K, M and the DOF mask used."""
    n_dof = K.shape[0]
    fixed = []
    for n in bc_nodes:
        fixed.extend([6*n, 6*n + 1, 6*n + 2])
    free = np.setdiff1d(np.arange(n_dof), fixed)
    Kf = K[np.ix_(free, free)]
    Mf = M[np.ix_(free, free)]
    return Kf, Mf, free


def solve_modes(Kf, Mf, n_modes=40):
    """Generalized eigenvalue problem K v = λ M v, return freqs in Hz."""
    # eigh with B handles SPD generalized problem
    eigvals, eigvecs = la.eigh(Kf, Mf, subset_by_index=[0, n_modes - 1])
    # Guard against tiny-negative numerical eigenvalues
    eigvals = np.maximum(eigvals, 0.0)
    freqs = np.sqrt(eigvals) / (2.0 * math.pi)
    return freqs, eigvecs


# ──────────────────────────────────────────────────────────────────────
# 5. Sensor & shaker mapping to mesh
# ──────────────────────────────────────────────────────────────────────

# Experimental sensor 3D positions (extracted from data_100Hz.h5 attrs)
SENSOR_POS = {
    "AG02": (-10.5, +2.25, 4.1),
    "AG04": ( -3.5, +2.25, 4.9),
    "AG05": (  0.0, +2.25, 0.6),
    "AG06": ( +3.5, +2.25, 4.9),
    "AG08": (+10.5, +2.25, 4.1),
    "AG11": (-10.5, -2.25, 0.6),
    "AG13": ( -3.5, -2.25, 0.6),
    "AG14": (  0.0, -2.25, 5.0),
    "AG15": ( +3.5, -2.25, 0.6),
    "AG17": (+10.5, -2.25, 0.6),
    "AL10": ( -8.75, -0.55, 0.19),
    "AL26": ( +5.25, -0.55, 0.19),
}

# Shaker positions (from record attr mvs_position_coordinates)
MVS_POS = {
    "P1": (7.5, 0.0, 0.6),
    "P2": (-7.5, 0.0, 0.6),       # assumption — confirm from a P2 record attr
}


def map_pos_to_joint(joints, p):
    """Return joint index nearest to position p."""
    arr = np.array([[j.x, j.y, j.z] for j in joints])
    d = np.linalg.norm(arr - np.array(p), axis=1)
    return int(np.argmin(d)), float(d.min())


def sensor_dofs(joints, direction):
    """Map each sensor to (joint, dof_offset).  direction in {'y','z'}.
    Returns ndarray of length 12 with the global DOF index per sensor."""
    dof_off = {"x": 0, "y": 1, "z": 2}[direction]
    out = []
    for name in SENSOR_NAMES:
        idx, dist = map_pos_to_joint(joints, SENSOR_POS[name])
        out.append(6*idx + dof_off)
        if dist > 0.5:
            log.warning(f"sensor {name} → joint {joints[idx].name} "
                        f"distance={dist:.2f} m (poor match)")
    return np.array(out, dtype=np.int64)


def shaker_dof(joints, position_key, direction):
    dof_off = {"x": 0, "y": 1, "z": 2}[direction]
    idx, dist = map_pos_to_joint(joints, MVS_POS[position_key])
    log.info(f"shaker {position_key} ({direction}) → joint "
             f"{joints[idx].name} distance={dist:.2f} m")
    return 6*idx + dof_off


# ──────────────────────────────────────────────────────────────────────
# 6. Modal-superposition FRF synthesis
# ──────────────────────────────────────────────────────────────────────

def piecewise_zeta(freqs, scale=1.0):
    """Per-mode damping ratios. `scale` multiplies the baseline."""
    z = np.full_like(freqs, 0.01)
    z[freqs < 5.0]  = 0.005
    z[(freqs >= 5.0) & (freqs < 20.0)] = 0.010
    z[freqs >= 20.0] = 0.020
    return z * scale


def modal_frf(eigvecs, freqs_modes, zeta, sens_dofs, input_dof,
              freq_grid, free_dof_index):
    """Compute H(ω) at sensors due to a unit force at input_dof.

    Returns complex array (N_F, N_CH) = accelerance ω² · receptance.
    """
    # Map sens/input DOFs from full to reduced index space
    full2red = -np.ones(eigvecs.shape[0] + max(sens_dofs.max(), input_dof) + 1,
                          dtype=int)
    # Build mapping from full-DOF to reduced index using the free_dof_index
    full_to_red = -np.ones(free_dof_index.max() + 1, dtype=int)
    full_to_red[free_dof_index] = np.arange(len(free_dof_index))

    if input_dof not in free_dof_index:
        raise SystemExit(f"input_dof {input_dof} is constrained")
    in_red = int(full_to_red[input_dof])

    sens_red = []
    for d in sens_dofs:
        if d not in free_dof_index:
            log.warning(f"sensor dof {d} is constrained — using 0 row")
            sens_red.append(-1)
        else:
            sens_red.append(int(full_to_red[d]))
    sens_red = np.array(sens_red)

    phi_sens = np.zeros((len(sens_red), eigvecs.shape[1]))
    valid = sens_red >= 0
    phi_sens[valid] = eigvecs[sens_red[valid], :]
    phi_in = eigvecs[in_red, :]                # (N_modes,)

    omega   = 2 * np.pi * freq_grid            # (N_F,)
    omega_r = 2 * np.pi * freqs_modes          # (N_modes,)
    # Receptance modal denominator: (ω_r² − ω² + 2 j ζ_r ω_r ω)
    den = (omega_r[None, :]**2 - omega[:, None]**2
            + 2j * zeta[None, :] * omega_r[None, :] * omega[:, None])
    # u_i = Σ_r (φ_{i,r} · φ_{in,r}) / den_r       (N_F, N_CH)
    num = phi_sens * phi_in[None, :]           # (N_CH, N_modes)
    H_disp = (num[None, :, :] / den[:, None, :]).sum(axis=2)
    # Accelerance: multiply by -ω² (sign convention for a/F)
    H_acc = -(omega[:, None]**2) * H_disp
    return H_acc      # (N_F, N_CH)  complex


# ──────────────────────────────────────────────────────────────────────
# 7. Experimental data loading & comparison
# ──────────────────────────────────────────────────────────────────────

def load_uds_y(chunks_dir: Path):
    """Load all UDS class windows, return median |FRF_H1| over windows
    for the Y-direction records only. Returns (freqs, |H_med|) where
    |H_med| has shape (N_F, N_CH)."""
    frfs, srcs = [], []
    for p in sorted(chunks_dir.glob("chunk_*.h5")):
        with h5py.File(p, "r") as f:
            lab = f["labels/class_code"][:]
            src = f["labels/source_record"][:]
            uds = (lab == 0)
            if not uds.any():
                continue
            frfs.append(f["frf_H1"][uds])
            srcs.extend([s.decode() for s in src[uds]])
    frfs = np.concatenate(frfs)                   # (n_w, N_F, N_CH)
    srcs = np.array(srcs)
    y_mask = np.array(["_Y_" in s for s in srcs])
    log.info(f"experimental UDS windows: total={len(srcs)}  "
             f"Y-direction={int(y_mask.sum())}")
    H_y = frfs[y_mask]
    H_med = np.median(np.abs(H_y), axis=0)         # (N_F, N_CH)
    return H_med, H_y


def cfdac(H):
    """Multi-channel CFDAC matrix from (N_F, N_CH) FRF.

      CFDAC_ij = |<H_i, H_j>|^2 / (||H_i||^2 * ||H_j||^2)

    where <a,b> = sum_k a_k^* · b_k across channels.
    """
    inner = H.conj() @ H.T                          # (N_F, N_F) complex
    norms = np.sqrt(np.einsum("fk,fk->f", H.conj(), H).real)  # (N_F,)
    den = np.outer(norms, norms)
    with np.errstate(divide="ignore", invalid="ignore"):
        C = (np.abs(inner) ** 2) / np.maximum(den ** 2, 1e-30)
    return np.clip(C.real, 0.0, 1.0)


def sci(C1, C2):
    a = C1 - C1.mean(); b = C2 - C2.mean()
    num = (a * b).sum()
    den = math.sqrt((a * a).sum() * (b * b).sum())
    return float(num / den) if den > 0 else 0.0


def smoothed_log_cfdac(H, sigma=4.0):
    """CFDAC built on Gaussian-smoothed log|H| per channel."""
    H_log = np.log10(np.maximum(np.abs(H), 1e-30))
    for k in range(H.shape[1]):
        H_log[:, k] = ndi.gaussian_filter1d(H_log[:, k], sigma=sigma)
    H_smooth = 10.0 ** H_log    # back to magnitude
    return cfdac(H_smooth.astype(np.complex64))


# ──────────────────────────────────────────────────────────────────────
# 8. Main runner
# ──────────────────────────────────────────────────────────────────────

def main(e_factor=1.0, zeta_scale=1.0):
    log.info("== HBTA initial FEM run ==")
    log.info(f"E_steel × {e_factor:.3f}   damping × {zeta_scale:.2f}")

    # Adjust global modulus through the section table (cheap hack:
    # scale all sections' EI uniformly via Young's modulus).
    global E_STEEL, G_STEEL
    base_E = 210.0e9
    E_STEEL = base_E * e_factor
    G_STEEL = E_STEEL / (2.0 * (1.0 + NU_STEEL))

    joints, members, bc_nodes, j_idx = build_geometry()
    assign_sections(members)
    K, M = assemble(joints, members)
    log.info(f"global K shape {K.shape}  density {np.count_nonzero(K)/K.size:.3f}")

    Kf, Mf, free = apply_pinned_bc(K, M, bc_nodes)
    freqs_modes, eigvecs = solve_modes(Kf, Mf, n_modes=40)
    log.info(f"first 10 modes [Hz]: " + ", ".join(f"{f:.2f}" for f in freqs_modes[:10]))

    # Save modal table
    txt = "Mode  Freq[Hz]\n" + "\n".join(
        f"{i+1:3d}    {f:7.3f}" for i, f in enumerate(freqs_modes[:20]))
    (FIG_DIR / "modes_table.txt").write_text(txt)

    # --- FRF synthesis for Y-direction shaker at P1 ---
    direction = "y"
    sens = sensor_dofs(joints, direction)
    in_dof = shaker_dof(joints, "P1", direction)

    fs = 100.0
    N_T = 1024
    N_F = N_T // 2 + 1
    freq_grid = np.fft.rfftfreq(N_T, d=1/fs)  # (513,)
    zeta = piecewise_zeta(freqs_modes, scale=zeta_scale)

    H_model = modal_frf(eigvecs, freqs_modes, zeta, sens, in_dof, freq_grid, free)
    log.info(f"model H shape {H_model.shape}  band 0.5-25 Hz = "
             f"bins {int(0.5/fs*N_T)}..{int(25.0/fs*N_T)}")

    # --- Experimental UDS comparison ---
    chunks_dir = ROOT.parent / "output" / "chunks"
    H_exp_med, H_exp_all = load_uds_y(chunks_dir)
    log.info(f"experimental H_med shape {H_exp_med.shape}")

    # ---------- Magnitude scale fit (one global gain, in-band fit) ----
    # Use 4-25 Hz to avoid the low-freq sweep-onset artifact peaks in H1
    # (the H1 estimator amplifies noise where input spectrum is small).
    band = (freq_grid >= 4.0) & (freq_grid <= 25.0)
    ch_keep = np.arange(10)        # arch sensors only (skip AL deck channels)
    num = np.sum(np.abs(H_exp_med[np.ix_(band, ch_keep)]))
    den = np.sum(np.abs(H_model [np.ix_(band, ch_keep)])) + 1e-30
    gain = float(num / den)
    H_model_scaled = H_model * gain
    log.info(f"global gain fit (band 4-25 Hz, arch ch) = {gain:.3e}")

    # ---------- Scores ----------
    C_exp = cfdac(H_exp_med[band])
    C_mod = cfdac(H_model_scaled[band])
    raw_sci   = sci(C_exp, C_mod)
    C_exp_sm = smoothed_log_cfdac(H_exp_med[band], sigma=4.0)
    C_mod_sm = smoothed_log_cfdac(H_model_scaled[band], sigma=4.0)
    sm_sci    = sci(C_exp_sm, C_mod_sm)
    log.info(f"raw CFDAC SCI = {raw_sci:.3f}   smoothed = {sm_sci:.3f}")

    (FIG_DIR / "sci_scoreboard.txt").write_text(
        f"raw  CFDAC SCI  = {raw_sci:.4f}\n"
        f"smooth log-SCI  = {sm_sci:.4f}\n"
        f"gain            = {gain:.3e}\n"
        f"E_factor        = {e_factor}\n"
        f"first 5 modes   = " + ", ".join(f"{f:.3f}" for f in freqs_modes[:5]) + "\n")

    # ---------- Figures ----------
    fig, ax = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    plot_chs = [0, 2, 3, 4, 7, 9]            # 6 representative arch sensors
    for k, ch in enumerate(plot_chs):
        r, c = k // 2, k % 2
        ax[r, c].semilogy(freq_grid[1:], np.abs(H_exp_med[1:, ch]),
                            "r-", lw=1.0, label="experiment")
        ax[r, c].semilogy(freq_grid[1:], np.abs(H_model_scaled[1:, ch]),
                            "b-", lw=1.0, label="model (anchored)")
        ax[r, c].set_title(f"{SENSOR_NAMES[ch]}  (ch{ch})", fontsize=9)
        ax[r, c].set_xlim(0.5, 25)
        ax[r, c].set_ylim(1e-3, 1e3)
        ax[r, c].grid(True, alpha=0.4)
        if k == 0:
            ax[r, c].legend(fontsize=8)
    for r in range(3):
        ax[r, 0].set_ylabel("|H1|  [m/s² / N]")
    for c in range(2):
        ax[-1, c].set_xlabel("frequency [Hz]")
    fig.suptitle(f"HBTA UDS Y-sweep — experimental vs FEM (E factor {e_factor:.2f}, "
                  f"raw-SCI={raw_sci:.3f}, smooth-SCI={sm_sci:.3f})", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "frf_magnitude.png", dpi=130)
    plt.close(fig)
    log.info(f"wrote {FIG_DIR/'frf_magnitude.png'}")

    # CFDAC matrices
    fig, ax = plt.subplots(2, 2, figsize=(9, 8))
    extent = [freq_grid[band].min(), freq_grid[band].max(),
              freq_grid[band].max(), freq_grid[band].min()]
    ax[0, 0].imshow(C_exp,    extent=extent, cmap="viridis", vmin=0, vmax=1)
    ax[0, 0].set_title("CFDAC raw — experiment", fontsize=9)
    ax[0, 1].imshow(C_mod,    extent=extent, cmap="viridis", vmin=0, vmax=1)
    ax[0, 1].set_title("CFDAC raw — model", fontsize=9)
    ax[1, 0].imshow(C_exp_sm, extent=extent, cmap="viridis", vmin=0, vmax=1)
    ax[1, 0].set_title("CFDAC smooth-log — experiment", fontsize=9)
    ax[1, 1].imshow(C_mod_sm, extent=extent, cmap="viridis", vmin=0, vmax=1)
    ax[1, 1].set_title("CFDAC smooth-log — model", fontsize=9)
    for a in ax.flat:
        a.set_xlabel("f₁ [Hz]"); a.set_ylabel("f₂ [Hz]")
    fig.suptitle(f"CFDAC matrices (UDS Y-sweep) — raw SCI {raw_sci:.3f},  smooth SCI {sm_sci:.3f}",
                  fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cfdac_uds.png", dpi=130)
    plt.close(fig)
    log.info(f"wrote {FIG_DIR/'cfdac_uds.png'}")

    # ---------- Per-damage-class FRF overlay (single sensor) -------
    log.info("loading per-class experimental medians for cross-class plot…")
    chunks_dir2 = ROOT.parent / "output" / "chunks"
    by_class = {}
    for p in sorted(chunks_dir2.glob("chunk_*.h5")):
        with h5py.File(p, "r") as f:
            lab = f["labels/class_code"][:]
            src = f["labels/source_record"][:]
            frf = f["frf_H1"][:]
            for c in range(9):
                m = (lab == c) & np.array([b"_Y_" in s for s in src])
                if not m.any(): continue
                by_class.setdefault(c, []).append(frf[m])
    medians = {c: np.median(np.abs(np.concatenate(by_class[c])), axis=0)
                for c in by_class}

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, 9))
    ch = 2     # AG05 — midspan north wall
    for c in sorted(medians):
        nm = "UDS" if c == 0 else f"DS{c}"
        ax.semilogy(freq_grid[1:], medians[c][1:, ch],
                     color=colors[c], lw=1.0, label=nm)
    ax.semilogy(freq_grid[1:], np.abs(H_model_scaled[1:, ch]),
                 "k--", lw=1.5, label="FEM (anchored)")
    ax.set_xlim(0.5, 25); ax.set_ylim(1e-3, 1e2)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("|H1|  AG05 (m/s² / m/s²)")
    ax.set_title(f"HBTA per-class median |H1| vs FEM anchored — "
                 f"sensor AG05 (Y-sweep)", fontsize=10)
    ax.legend(ncol=2, fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "per_class_vs_model.png", dpi=130)
    plt.close(fig)
    log.info(f"wrote {FIG_DIR/'per_class_vs_model.png'}")

    # ---------- Save anchor params ----------
    (ROOT / "best_params_anchor.json").write_text(json.dumps({
        "E_factor": e_factor,
        "gain": gain,
        "modes_Hz_first_5": [float(f) for f in freqs_modes[:5]],
        "raw_SCI": raw_sci,
        "smooth_SCI": sm_sci,
        "n_modes_total": int(len(freqs_modes)),
    }, indent=2))
    log.info("done.")

    return dict(freqs_modes=freqs_modes, H_model_scaled=H_model_scaled,
                H_exp_med=H_exp_med, freq_grid=freq_grid, gain=gain,
                raw_sci=raw_sci, sm_sci=sm_sci)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--e-factor",  type=float, default=1.0)
    ap.add_argument("--zeta-scale", type=float, default=1.0)
    args = ap.parse_args()
    main(e_factor=args.e_factor, zeta_scale=args.zeta_scale)
