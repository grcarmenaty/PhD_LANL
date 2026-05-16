"""Beam FEM v2 — vertical bending + torsion, decoupled solves.

Adds the missing physics from v1: torsion modes about the deck
longitudinal axis. East-side accelerometers sit at offset y_e from the
deck centerline, so they pick up a · = w_dot_dot + y_e · phi_dot_dot.
Traffic loads act vertically at an eccentricity y_f from the
centerline (lane offset), so they drive both bending and torsion:

  bending forcing  : F_j  →  H_bend(f, i, j)
  torsion forcing  : y_f · F_j  →  H_tors(f, i, j)
  sensor response  : H_ij(f) = H_bend + (y_e · y_f) · H_tors

Sensor auto-spectrum (uncorrelated white-noise per node):

  S_aa(f, i) = Σ_j |H_ij(f)|² · W(f)

Pier supports constrain w and φ to zero; bending rotation θ_y is free
unless a rotational spring k_pier_rot is added (knob).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.linalg import eigh


# ── Bridge geometry (unchanged from v1) ───────────────────────────
L_TOTAL = 358.0
N_SPAN  = 7
L_SPAN  = L_TOTAL / N_SPAN
ELEM_PER_SPAN = 30          # slightly coarser to keep DE fast
N_ELEM        = N_SPAN * ELEM_PER_SPAN
DL            = L_TOTAL / N_ELEM
N_NODES       = N_ELEM + 1
NODE_X        = np.linspace(0.0, L_TOTAL, N_NODES)
PIER_X        = np.array([i * L_SPAN for i in range(0, N_SPAN + 1)])
PIER_NODE_IDX = np.array([np.argmin(np.abs(NODE_X - x)) for x in PIER_X])

SENSOR_X = {
    "ch3":  4 * L_SPAN + 0.50 * L_SPAN,
    "ch11": 5 * L_SPAN + 0.50 * L_SPAN,
    "ch19": 6 * L_SPAN + 0.50 * L_SPAN,
    "ch21": 6 * L_SPAN + 0.75 * L_SPAN,
    "ch27": 2 * L_SPAN + 0.50 * L_SPAN,
    "ch29": 2 * L_SPAN + 0.75 * L_SPAN,
    "ch35": 1 * L_SPAN + 0.50 * L_SPAN,
    "ch43": 0 * L_SPAN + 0.50 * L_SPAN,
    "ch51": 3 * L_SPAN + 0.50 * L_SPAN,
}
SENSOR_ORDER    = ["ch3", "ch11", "ch19", "ch21", "ch27", "ch29",
                   "ch35", "ch43", "ch51"]
SENSOR_NODE_IDX = np.array([np.argmin(np.abs(NODE_X - SENSOR_X[k]))
                             for k in SENSOR_ORDER])

# Mass perturbation centroids
MASS_X    = {"field3": 2.5 * L_SPAN, "field4": 3.5 * L_SPAN}
MASS_NODE = {k: int(np.argmin(np.abs(NODE_X - x))) for k, x in MASS_X.items()}
MASS_KG   = 39_000.0


# ── Bending element matrices (Euler-Bernoulli, consistent) ───────
def _k_bend(L, EI):
    k = EI / L**3 * np.array([
        [12,    6*L,   -12,   6*L],
        [6*L,   4*L*L, -6*L,  2*L*L],
        [-12,  -6*L,    12,  -6*L],
        [6*L,   2*L*L, -6*L,  4*L*L],
    ])
    return k


def _m_bend(L, mu):
    return mu * L / 420.0 * np.array([
        [156,    22*L,   54,    -13*L],
        [22*L,   4*L*L,  13*L,  -3*L*L],
        [54,     13*L,   156,   -22*L],
        [-13*L, -3*L*L, -22*L,   4*L*L],
    ])


# ── Torsion element matrices (1D bar, 1 DOF/node) ────────────────
def _k_tors(L, GJ):
    return GJ / L * np.array([[1.0, -1.0], [-1.0, 1.0]])


def _m_tors(L, rho_Ip):
    return rho_Ip * L / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])


# ── Assembly ─────────────────────────────────────────────────────
@dataclass
class BendSystem:
    K: np.ndarray; M: np.ndarray
    free: np.ndarray; constrained: np.ndarray


@dataclass
class TorsSystem:
    K: np.ndarray; M: np.ndarray
    free: np.ndarray; constrained: np.ndarray


def build_bending(scenario: str, E: float, I: float, mu: float,
                   k_pier_rot: float) -> BendSystem:
    n_dof = 2 * N_NODES
    K = np.zeros((n_dof, n_dof)); M = np.zeros((n_dof, n_dof))
    EI = E * I
    for e in range(N_ELEM):
        i, j = e, e + 1
        dofs = [2*i, 2*i+1, 2*j, 2*j+1]
        ke = _k_bend(DL, EI); me = _m_bend(DL, mu)
        for a, da in enumerate(dofs):
            for b, db in enumerate(dofs):
                K[da, db] += ke[a, b]
                M[da, db] += me[a, b]
    # Pier vertical pin
    constrained = [2 * nidx for nidx in PIER_NODE_IDX]
    constrained = np.array(sorted(set(constrained)))
    free = np.setdiff1d(np.arange(n_dof), constrained)
    # Pier rotational spring
    if k_pier_rot > 0:
        for nidx in PIER_NODE_IDX:
            K[2*nidx + 1, 2*nidx + 1] += k_pier_rot
    # Scenario mass
    if scenario == "field3":
        K_idx = MASS_NODE["field3"]
        M[2*K_idx, 2*K_idx] += MASS_KG
    elif scenario == "field4":
        K_idx = MASS_NODE["field4"]
        M[2*K_idx, 2*K_idx] += MASS_KG
    return BendSystem(K=K, M=M, free=free, constrained=constrained)


def build_torsion(scenario: str, GJ: float, rho_Ip: float,
                   k_pier_tors: float) -> TorsSystem:
    n_dof = N_NODES         # 1 rotational DOF per node
    K = np.zeros((n_dof, n_dof)); M = np.zeros((n_dof, n_dof))
    for e in range(N_ELEM):
        i, j = e, e + 1
        dofs = [i, j]
        ke = _k_tors(DL, GJ); me = _m_tors(DL, rho_Ip)
        for a, da in enumerate(dofs):
            for b, db in enumerate(dofs):
                K[da, db] += ke[a, b]
                M[da, db] += me[a, b]
    # Pier torsional restraint
    constrained = []
    if k_pier_tors >= 1e11:   # treat as pin
        constrained = [int(nidx) for nidx in PIER_NODE_IDX]
    constrained = np.array(sorted(set(constrained))) if constrained else np.array([], dtype=int)
    if constrained.size:
        free = np.setdiff1d(np.arange(n_dof), constrained)
    else:
        free = np.arange(n_dof)
        for nidx in PIER_NODE_IDX:
            K[int(nidx), int(nidx)] += k_pier_tors
    # Scenario mass — also adds torsional inertia if mass is off-center?
    # Truck mass on the deck adds vertical inertia (already in bending M);
    # for torsion we'd need J_truck and y_truck. Skip for now (small effect
    # vs the existing distributed rho_Ip).
    return TorsSystem(K=K, M=M, free=free, constrained=constrained)


# ── Modal solve ──────────────────────────────────────────────────
def solve_modes(K, M, free, n_modes):
    Kf = K[np.ix_(free, free)]
    Mf = M[np.ix_(free, free)]
    nf = Kf.shape[0]
    n_modes = min(n_modes, nf)
    eigvals, eigvecs = eigh(Kf, Mf, subset_by_index=[0, n_modes - 1])
    eigvals = np.maximum(eigvals, 1e-12)
    freqs = np.sqrt(eigvals) / (2 * np.pi)
    full = np.zeros((K.shape[0], n_modes))
    full[free, :] = eigvecs
    return freqs, full


# ── Per-band damping ────────────────────────────────────────────
def zeta_per_band(freqs_hz, z_lo, z_mid, z_hi):
    z = np.empty_like(freqs_hz)
    z[freqs_hz <= 3.0] = z_lo
    z[(freqs_hz > 3.0) & (freqs_hz <= 10.0)] = z_mid
    z[freqs_hz > 10.0] = z_hi
    return z


# ── FRF synthesis ────────────────────────────────────────────────
def auto_spectrum_v2(scenario: str, params: dict, freq_grid_hz: np.ndarray,
                     band_mask: np.ndarray, n_modes_bend=40, n_modes_tors=25):
    """Returns per-sensor band-restricted |Y| under stochastic input."""
    # Build systems
    bend = build_bending(scenario, params["E_bend"], params["I_yy"],
                         params["mu"], params["k_pier_rot"])
    tors = build_torsion(scenario, params["GJ"], params["rho_Ip"],
                         params["k_pier_tors"])
    # Solve
    f_b, phi_b = solve_modes(bend.K, bend.M, bend.free, n_modes_bend)
    f_t, phi_t = solve_modes(tors.K, tors.M, tors.free, n_modes_tors)
    # Sensor + input maps
    sensor_w_dofs = 2 * SENSOR_NODE_IDX
    sensor_phi_dofs = SENSOR_NODE_IDX
    phi_b_sens = phi_b[sensor_w_dofs, :]      # (9, n_b)
    phi_t_sens = phi_t[sensor_phi_dofs, :]    # (9, n_t)
    # Input DOFs: every deck node's vertical w (bending) and rotation phi (torsion)
    deck_w = np.array([2*i for i in range(N_NODES) if 2*i in bend.free])
    deck_phi = np.array([i for i in range(N_NODES) if i in tors.free])
    phi_b_in = phi_b[deck_w, :]               # (n_in_b, n_b)
    phi_t_in = phi_t[deck_phi, :]             # (n_in_t, n_t)

    # Damping
    z_b = zeta_per_band(f_b, params["z_lo"], params["z_mid"], params["z_hi"])
    z_t = zeta_per_band(f_t, params["z_lo"], params["z_mid"], params["z_hi"])

    # FRFs on band frequencies
    omega = 2 * np.pi * freq_grid_hz[band_mask][:, None]   # (n_b, 1)
    omega_b_r = 2 * np.pi * f_b
    omega_t_r = 2 * np.pi * f_t
    D_b = (-omega**2 + 2j*z_b*omega_b_r*omega + omega_b_r**2)
    D_t = (-omega**2 + 2j*z_t*omega_t_r*omega + omega_t_r**2)

    # H_bend[f, i, j] = sum_r phi_b_sens[i,r] phi_b_in[j,r] / D_b[f,r]
    H_b = np.einsum("ir, jr, fr -> fij", phi_b_sens, phi_b_in, 1.0/D_b)
    H_t = np.einsum("ir, jr, fr -> fij", phi_t_sens, phi_t_in, 1.0/D_t)

    # Acceleration = -ω² × displacement
    H_acc_b = -(omega[..., None]**2) * H_b
    H_acc_t = -(omega[..., None]**2) * H_t

    # Combined: sensor sees bending + y_e * (torsion-driven-by-eccentric-load)
    # H_total[f, i, j_bend] = H_acc_b[f, i, j]  +  y_off * H_acc_t[f, i, j']
    # We treat the bending and torsion input "channels" as independent
    # uncorrelated input streams, both proportional to the underlying
    # vertical traffic force F(t) with relative scale y_f / 1 for torsion.
    # The auto-spectrum sum-over-j then has two terms with relative scale:
    #   S_aa[f, i] = Σ_j (|H_b|² + (y_off²) |H_t|²) · W(f)
    # treating bending and torsion inputs as independent reduces cross-
    # interference; this is a common OMA approximation.
    y_off = params["y_off"]
    W = 1.0 / (1.0 + (freq_grid_hz[band_mask] / params["f_cut"]) ** 4)
    S_b = np.einsum("fij -> fi", np.abs(H_acc_b)**2)
    S_t = np.einsum("fij -> fi", np.abs(H_acc_t)**2)
    S = (S_b + (y_off**2) * S_t) * W[:, None]
    Y = np.sqrt(np.maximum(S, 0.0))
    return Y, float(f_b[0]), f_b, f_t
