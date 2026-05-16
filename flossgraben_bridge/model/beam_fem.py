"""Continuum-beam FEM for the Flossgraben bridge.

Pure-Python equivalent of the Salome / Code_Aster stack described in
`flossgraben_bridge/model.md`. Same physics (Euler-Bernoulli beam,
consistent mass matrix, eigh modal decomposition, modal-superposition
response synthesis); Python loop replaces the GEOM/SMESH/CALC_MODES
chain because Salome is not available in this remote environment.

Outputs (per scenario):
  - eigenfrequencies and mode shapes
  - acceleration FRF magnitude at the 9 sensor positions, 0–25 Hz
  - synthetic time-domain windows (1024 samples @ 256 Hz) matching the
    experimental dataset schema
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from scipy.linalg import eigh


# ── Bridge parameters (from model.md §2 and §4) ──────────────────────
L_TOTAL  = 358.0
N_SPAN   = 7
L_SPAN   = L_TOTAL / N_SPAN

E_DECK   = 16.0e9        # Pa — effective modulus (cracked concrete + prestress);
                          #      gross-section value 34 GPa, anchored to experimental
                          #      reference f_1 ≈ 1.75 Hz (see model.md §15.1)
RHO_C    = 2500.0        # kg/m³
A_DECK   = 9.4           # m²
I_YY     = 12.6          # m⁴
MU       = RHO_C * A_DECK  # kg/m (linear mass density)

# Element grid: 40 elements per span ≈ 1.28 m
ELEM_PER_SPAN = 40
N_ELEM        = N_SPAN * ELEM_PER_SPAN
DL            = L_TOTAL / N_ELEM
NODE_X        = np.linspace(0.0, L_TOTAL, N_ELEM + 1)

# Pier and abutment locations (vertical pin supports)
PIER_X        = np.array([i * L_SPAN for i in range(0, N_SPAN + 1)])
PIER_NODE_IDX = np.array([np.argmin(np.abs(NODE_X - x)) for x in PIER_X])

# Sensor locations (model.md §10.1) — 9 East-side accelerometers
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
SENSOR_ORDER     = ["ch3", "ch11", "ch19", "ch21", "ch27", "ch29",
                    "ch35", "ch43", "ch51"]
SENSOR_NODE_IDX  = np.array([np.argmin(np.abs(NODE_X - SENSOR_X[k]))
                              for k in SENSOR_ORDER])

# 39 t mass perturbation centroids
MASS_X = {"field3": 2.5 * L_SPAN, "field4": 3.5 * L_SPAN}
MASS_NODE = {k: int(np.argmin(np.abs(NODE_X - x))) for k, x in MASS_X.items()}
MASS_KG   = 39_000.0

# Rayleigh damping (model.md §4)
ALPHA_RAYL = 0.05
BETA_RAYL  = 1.0e-4


# ── Element matrices (Euler-Bernoulli, consistent) ───────────────────
def _k_elem(L: float, EI: float) -> np.ndarray:
    """4×4 EB beam stiffness in DOFs (w_i, θ_i, w_j, θ_j)."""
    k = EI / L ** 3 * np.array([
        [12,    6*L,   -12,   6*L],
        [6*L,   4*L*L, -6*L,  2*L*L],
        [-12,  -6*L,    12,  -6*L],
        [6*L,   2*L*L, -6*L,  4*L*L],
    ])
    return k


def _m_elem(L: float, mu: float) -> np.ndarray:
    """4×4 EB consistent mass matrix."""
    m = mu * L / 420.0 * np.array([
        [156,    22*L,   54,    -13*L],
        [22*L,   4*L*L,  13*L,  -3*L*L],
        [54,     13*L,   156,   -22*L],
        [-13*L, -3*L*L, -22*L,   4*L*L],
    ])
    return m


# ── Assembly ─────────────────────────────────────────────────────────
@dataclass
class BeamModel:
    n_nodes: int
    dof_per_node: int = 2
    elements: list[tuple[int, int]] = field(default_factory=list)
    K: np.ndarray = field(default=None)
    M: np.ndarray = field(default=None)
    free_dofs: np.ndarray = field(default=None)
    constrained_dofs: np.ndarray = field(default=None)
    extra_mass: dict[int, float] = field(default_factory=dict)


def build(scenario: str) -> BeamModel:
    n_nodes = N_ELEM + 1
    n_dof   = n_nodes * 2     # w, θ per node
    K = np.zeros((n_dof, n_dof))
    M = np.zeros((n_dof, n_dof))
    EI = E_DECK * I_YY

    elements = []
    for e in range(N_ELEM):
        i, j = e, e + 1
        dofs = [2*i, 2*i + 1, 2*j, 2*j + 1]
        ke = _k_elem(DL, EI)
        me = _m_elem(DL, MU)
        for a, da in enumerate(dofs):
            for b, db in enumerate(dofs):
                K[da, db] += ke[a, b]
                M[da, db] += me[a, b]
        elements.append((i, j))

    # Pier / abutment supports: pin vertical displacement only
    constrained = []
    for nidx in PIER_NODE_IDX:
        constrained.append(2 * nidx)      # w = 0
    constrained = np.array(sorted(set(constrained)))
    free = np.setdiff1d(np.arange(n_dof), constrained)

    # Scenario-dependent lumped mass
    extra = {}
    if scenario == "field3":
        extra[MASS_NODE["field3"]] = MASS_KG
    elif scenario == "field4":
        extra[MASS_NODE["field4"]] = MASS_KG
    for node, mass in extra.items():
        M[2*node, 2*node] += mass

    return BeamModel(
        n_nodes=n_nodes, elements=elements, K=K, M=M,
        free_dofs=free, constrained_dofs=constrained, extra_mass=extra)


# ── Modal analysis ───────────────────────────────────────────────────
def solve_modes(model: BeamModel, n_modes: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Return (freqs_Hz, mode_shapes) for the first n_modes vertical modes."""
    Kf = model.K[np.ix_(model.free_dofs, model.free_dofs)]
    Mf = model.M[np.ix_(model.free_dofs, model.free_dofs)]
    # Regularise tiny negatives from numerical roundoff
    eigvals, eigvecs = eigh(Kf, Mf, subset_by_index=[0, n_modes - 1])
    eigvals = np.maximum(eigvals, 1e-12)
    freqs = np.sqrt(eigvals) / (2 * np.pi)
    # Expand to full DOF vector (constrained DOFs = 0)
    full = np.zeros((model.K.shape[0], n_modes))
    full[model.free_dofs, :] = eigvecs
    return freqs, full


def vertical_shape_at_sensors(modes_full: np.ndarray) -> np.ndarray:
    """Extract vertical (w) displacement at each sensor node from full DOFs."""
    sensor_dofs = 2 * SENSOR_NODE_IDX     # w-dofs
    return modes_full[sensor_dofs, :]     # (9, n_modes)


# ── Response synthesis ───────────────────────────────────────────────
def rayleigh_zeta(freqs_hz: np.ndarray, alpha: float = ALPHA_RAYL,
                  beta: float = BETA_RAYL) -> np.ndarray:
    omega = 2 * np.pi * freqs_hz
    omega = np.maximum(omega, 1e-3)
    return 0.5 * (alpha / omega + beta * omega)


def acc_frf_at_sensors(model: BeamModel, freq_grid_hz: np.ndarray,
                       n_modes: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Acceleration FRF |H| at every sensor for each input deck w-DOF.

    Returns
    -------
    freqs_modes : (n_modes,)  modal frequencies in Hz
    phi_sens    : (9, n_modes) mode shapes at sensor w-DOFs
    H_acc       : (n_f, 9, n_input_nodes) complex accelerance,
                  with n_input_nodes = deck nodes (free DOFs that are w)
    """
    freqs_modes, phi_full = solve_modes(model, n_modes=n_modes)
    phi_sens = vertical_shape_at_sensors(phi_full)  # (9, n_modes)
    # Input DOFs: vertical (w) of every deck node that is free
    deck_w_dofs = np.array([2 * i for i in range(model.n_nodes)
                             if 2 * i in model.free_dofs])
    phi_in = phi_full[deck_w_dofs, :]   # (n_in, n_modes)
    n_in = phi_in.shape[0]

    zeta = rayleigh_zeta(freqs_modes)
    omega_r = 2 * np.pi * freqs_modes
    omega = 2 * np.pi * freq_grid_hz[:, None]   # (n_f, 1)

    # H_disp_ij(ω) = Σ_r φ_sens[i,r] φ_in[j,r] / (-ω² + 2jζω_rω + ω_r²)
    denom = (-omega**2 + 2j * zeta * omega_r * omega + omega_r**2)  # (n_f, n_modes)
    H_disp = np.einsum("ir, jr, fr -> fij",
                       phi_sens, phi_in, 1.0 / denom)
    H_acc = -(omega[..., None]**2) * H_disp        # broadcast (n_f,1,1)
    return freqs_modes, phi_sens, H_acc


# ── Auto-spectrum under stochastic-traffic input ────────────────────
def auto_spectrum(model: BeamModel, freq_grid_hz: np.ndarray,
                  f_cut_hz: float = 8.0, n_modes: int = 60) -> np.ndarray:
    """Per-sensor response auto-spectrum S_aa(f) under uniform white-noise
    forcing along the deck, low-pass shaped at f_cut_hz."""
    _, _, H_acc = acc_frf_at_sensors(model, freq_grid_hz, n_modes=n_modes)
    # Input PSD W(ω) = 1 / (1 + (ω/ω_c)^4)
    W = 1.0 / (1.0 + (freq_grid_hz / f_cut_hz) ** 4)
    # S_aa[f, i] = Σ_j |H_acc[f, i, j]|² · W[f]
    S = np.einsum("fij, f -> fi", np.abs(H_acc) ** 2, W)
    return S


# ── Time-domain windows ──────────────────────────────────────────────
def synthesise_windows(S_aa: np.ndarray, n_windows: int, n_t: int = 1024,
                       fs_hz: float = 256.0, rng_seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """From a per-sensor auto-spectrum, generate `n_windows` realisations
    of (n_t, n_ch) time-domain windows and the matching complex64
    rfft spectrum tensor of shape (n_windows, n_t/2+1, n_ch, 1)."""
    rng = np.random.default_rng(rng_seed)
    n_f, n_ch = S_aa.shape
    assert n_f == n_t // 2 + 1
    df = fs_hz / n_t

    # Spectrum scale: |Y(f)| = sqrt(S * df) so that |Y|² ≈ S·df, matching
    # the experimental rfft-of-window convention up to a sqrt(2/N) factor.
    mag = np.sqrt(S_aa * df).astype(np.float32)

    signals = np.zeros((n_windows, n_t, n_ch), dtype=np.float32)
    specs   = np.zeros((n_windows, n_f, n_ch, 1), dtype=np.complex64)
    for w in range(n_windows):
        phases = rng.uniform(-np.pi, np.pi, size=mag.shape)
        # Pin DC and Nyquist phases to real for real-valued ifft
        phases[0, :] = 0.0
        phases[-1, :] = 0.0
        Y = mag * np.exp(1j * phases)
        sig = np.fft.irfft(Y, n=n_t, axis=0).astype(np.float32)
        specs[w, :, :, 0] = Y
        signals[w, :, :] = sig
    return signals, specs


# ── CLI entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    grid = np.arange(513) * (256.0 / 1024.0)
    for sc in ("reference", "field3", "field4"):
        model = build(sc)
        freqs, _ = solve_modes(model, n_modes=20)
        print(f"{sc}: first 8 modes [Hz] =",
              np.array2string(freqs[:8], precision=3, separator=", "))
