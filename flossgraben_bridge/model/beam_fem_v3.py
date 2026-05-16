"""Beam FEM v3 — adds finite pier compliance and a global sensor-x offset.

Changes from v2:
  - Pier supports are no longer rigid pins. Each pier provides a vertical
    spring k_pier_v and a rotational spring k_pier_rot at the deck node
    instead of zero-displacement constraint. Inverse-Helmholtz physics:
    finite k_pier_v lets the deck "bounce" at the pier, injecting low-
    frequency modes that aren't present in the rigid-pin model.
  - Pier torsional spring k_pier_tors becomes a proper knob (was 1e12 in v2).
  - All 9 sensor x-positions are shifted by a single global Δx_sensor [m].
    Captures systematic error in the assumed quarter-span layout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.linalg import eigh


# Geometry constants — identical to v2
L_TOTAL = 358.0
N_SPAN  = 7
L_SPAN  = L_TOTAL / N_SPAN
ELEM_PER_SPAN = 30
N_ELEM        = N_SPAN * ELEM_PER_SPAN
DL            = L_TOTAL / N_ELEM
N_NODES       = N_ELEM + 1
NODE_X        = np.linspace(0.0, L_TOTAL, N_NODES)
PIER_X        = np.array([i * L_SPAN for i in range(0, N_SPAN + 1)])
PIER_NODE_IDX = np.array([np.argmin(np.abs(NODE_X - x)) for x in PIER_X])

_SENSOR_X_BASE = {
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
SENSOR_ORDER = ["ch3", "ch11", "ch19", "ch21", "ch27", "ch29",
                "ch35", "ch43", "ch51"]


def sensor_node_indices(dx_sensor: float = 0.0) -> np.ndarray:
    """Map sensors to mesh nodes after applying a global Δx offset."""
    return np.array([
        np.argmin(np.abs(NODE_X - (_SENSOR_X_BASE[k] + dx_sensor)))
        for k in SENSOR_ORDER
    ])


MASS_X    = {"field3": 2.5 * L_SPAN, "field4": 3.5 * L_SPAN}
MASS_NODE = {k: int(np.argmin(np.abs(NODE_X - x))) for k, x in MASS_X.items()}
MASS_KG   = 39_000.0


def _k_bend(L, EI):
    return EI / L**3 * np.array([
        [12,    6*L,   -12,   6*L],
        [6*L,   4*L*L, -6*L,  2*L*L],
        [-12,  -6*L,    12,  -6*L],
        [6*L,   2*L*L, -6*L,  4*L*L],
    ])


def _m_bend(L, mu):
    return mu * L / 420.0 * np.array([
        [156,    22*L,   54,    -13*L],
        [22*L,   4*L*L,  13*L,  -3*L*L],
        [54,     13*L,   156,   -22*L],
        [-13*L, -3*L*L, -22*L,   4*L*L],
    ])


def _k_tors(L, GJ):
    return GJ / L * np.array([[1.0, -1.0], [-1.0, 1.0]])


def _m_tors(L, rho_Ip):
    return rho_Ip * L / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])


@dataclass
class BendSystem:
    K: np.ndarray; M: np.ndarray
    free: np.ndarray; constrained: np.ndarray


@dataclass
class TorsSystem:
    K: np.ndarray; M: np.ndarray
    free: np.ndarray; constrained: np.ndarray


def build_bending(scenario, E, I, mu, k_pier_v, k_pier_rot):
    """Build bending system with FINITE pier compliance.

    Vertical DOF at piers is unconstrained but gets a ground spring k_pier_v.
    Rotational DOF at piers gets a ground spring k_pier_rot.
    Only abutments (x=0 and x=L_TOTAL) remain hard pin (w=0).
    """
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
    # Hard pin only at abutments (first and last node)
    constrained = [0, 2 * (N_NODES - 1)]
    constrained = np.array(sorted(set(constrained)))
    free = np.setdiff1d(np.arange(n_dof), constrained)
    # Pier springs — all interior piers
    for nidx in PIER_NODE_IDX:
        if nidx == 0 or nidx == N_NODES - 1:
            continue
        K[2*nidx,     2*nidx]     += k_pier_v
        K[2*nidx + 1, 2*nidx + 1] += k_pier_rot
    # Scenario mass
    if scenario == "field3":
        nm = MASS_NODE["field3"]
        M[2*nm, 2*nm] += MASS_KG
    elif scenario == "field4":
        nm = MASS_NODE["field4"]
        M[2*nm, 2*nm] += MASS_KG
    return BendSystem(K=K, M=M, free=free, constrained=constrained)


def build_torsion(scenario, GJ, rho_Ip, k_pier_tors):
    n_dof = N_NODES
    K = np.zeros((n_dof, n_dof)); M = np.zeros((n_dof, n_dof))
    for e in range(N_ELEM):
        i, j = e, e + 1
        ke = _k_tors(DL, GJ); me = _m_tors(DL, rho_Ip)
        for a, da in enumerate([i, j]):
            for b, db in enumerate([i, j]):
                K[da, db] += ke[a, b]
                M[da, db] += me[a, b]
    # No hard constraints; just ground springs at every pier
    free = np.arange(n_dof)
    for nidx in PIER_NODE_IDX:
        K[int(nidx), int(nidx)] += k_pier_tors
    constrained = np.array([], dtype=int)
    return TorsSystem(K=K, M=M, free=free, constrained=constrained)


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


def zeta_per_band(freqs_hz, z_lo, z_mid, z_hi):
    z = np.empty_like(freqs_hz)
    z[freqs_hz <= 3.0] = z_lo
    z[(freqs_hz > 3.0) & (freqs_hz <= 10.0)] = z_mid
    z[freqs_hz > 10.0] = z_hi
    return z


def auto_spectrum_v3(scenario, params, freq_grid_hz, band_mask,
                      n_modes_bend=50, n_modes_tors=30):
    bend = build_bending(scenario, params["E_bend"], params["I_yy"],
                         params["mu"], params["k_pier_v"],
                         params["k_pier_rot"])
    tors = build_torsion(scenario, params["GJ"], params["rho_Ip"],
                         params["k_pier_tors"])
    f_b, phi_b = solve_modes(bend.K, bend.M, bend.free, n_modes_bend)
    f_t, phi_t = solve_modes(tors.K, tors.M, tors.free, n_modes_tors)
    sens_nodes = sensor_node_indices(params["dx_sensor"])
    sensor_w_dofs = 2 * sens_nodes
    sensor_phi_dofs = sens_nodes
    phi_b_sens = phi_b[sensor_w_dofs, :]
    phi_t_sens = phi_t[sensor_phi_dofs, :]
    deck_w = np.array([2*i for i in range(N_NODES) if 2*i in bend.free])
    deck_phi = np.array([i for i in range(N_NODES) if i in tors.free])
    phi_b_in = phi_b[deck_w, :]
    phi_t_in = phi_t[deck_phi, :]
    z_b = zeta_per_band(f_b, params["z_lo"], params["z_mid"], params["z_hi"])
    z_t = zeta_per_band(f_t, params["z_lo"], params["z_mid"], params["z_hi"])
    omega = 2 * np.pi * freq_grid_hz[band_mask][:, None]
    omega_b_r = 2 * np.pi * f_b
    omega_t_r = 2 * np.pi * f_t
    D_b = (-omega**2 + 2j*z_b*omega_b_r*omega + omega_b_r**2)
    D_t = (-omega**2 + 2j*z_t*omega_t_r*omega + omega_t_r**2)
    H_b = np.einsum("ir, jr, fr -> fij", phi_b_sens, phi_b_in, 1.0/D_b)
    H_t = np.einsum("ir, jr, fr -> fij", phi_t_sens, phi_t_in, 1.0/D_t)
    H_acc_b = -(omega[..., None]**2) * H_b
    H_acc_t = -(omega[..., None]**2) * H_t
    y_off = params["y_off"]
    W = 1.0 / (1.0 + (freq_grid_hz[band_mask] / params["f_cut"]) ** 4)
    S_b = np.einsum("fij -> fi", np.abs(H_acc_b)**2)
    S_t = np.einsum("fij -> fi", np.abs(H_acc_t)**2)
    S = (S_b + (y_off**2) * S_t) * W[:, None]
    Y = np.sqrt(np.maximum(S, 0.0))
    return Y, float(f_b[0]), f_b, f_t
