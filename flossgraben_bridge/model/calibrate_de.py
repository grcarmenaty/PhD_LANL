"""Calibration loop: SciPy differential_evolution maximizing mean SCI
across the 3 Flossgraben scenarios.

Knobs (model.md §4):
  - E_c effective deck modulus  [GPa]
  - α Rayleigh mass-proportional damping
  - β Rayleigh stiffness-proportional damping
  - f_cut traffic-input PSD low-pass corner [Hz]
  - k_pier_rot rotational stiffness at pier supports [N·m/rad]  (0 = pin)

Loss = -mean SCI across (reference, field3, field4), with a small
soft penalty on f_1 deviation from the experimental peak at 1.75 Hz
to keep the optimiser from drifting modes.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import differential_evolution

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import beam_fem as bf

EXP_H5 = HERE.parent / "output" / "flossgraben_collection.h5"

FS = 256.0
N_T = 1024
N_F = 513
FREQ_GRID = np.arange(N_F) * (FS / N_T)
BAND_LO, BAND_HI = 0.5, 25.0
BAND = (FREQ_GRID >= BAND_LO) & (FREQ_GRID <= BAND_HI)

# Experimental anchor peak for soft-penalty term
F1_EXP = 1.75      # Hz
W_FREQ = 4.0       # weight on (f1_model − f1_exp)²

# Number of modes per eval (kept high enough to span 0.5–25 Hz)
N_MODES = 40
# Sub-sample windows per class for cached experimental CFDAC
N_PER_CLASS = 500


# ── CFDAC / SCI (3SBB-identical, generate_docs_images.py:196-205) ─
def cfdac(H):
    inner = H.conj() @ H.T
    d = np.real(np.diag(inner)).copy(); d[d < 1e-30] = 1e-30
    return (np.abs(inner) ** 2) / np.outer(d, d)


def sci(C1, C2):
    a = C1.ravel() - C1.mean(); b = C2.ravel() - C2.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float((a @ b) ** 2 / d ** 2) if d > 0 else 0.0


# ── Experimental cache ────────────────────────────────────────────
def load_experimental_cfdac():
    name_by_class = {0: "reference", 1: "field3", 2: "field4"}
    out_spec = {v: [] for v in name_by_class.values()}
    with h5py.File(EXP_H5, "r") as f:
        meas = f["measurements"]
        for name in meas:
            if name == "_axes":
                continue
            label = int(round(float(meas[name]["label"][()])))
            sc = name_by_class.get(label)
            if sc is None or len(out_spec[sc]) >= N_PER_CLASS:
                continue
            out_spec[sc].append(np.asarray(meas[name]["data"]))
            if all(len(v) >= N_PER_CLASS for v in out_spec.values()):
                break
    out = {}
    for sc, lst in out_spec.items():
        arr = np.stack(lst)                                 # (n, 513, 9, 1)
        med = np.median(np.abs(arr[..., 0]).astype(np.float64), axis=0)
        H_band = med[BAND].astype(np.complex128)            # (n_b, 9)
        out[sc] = dict(H=H_band, C=cfdac(H_band), peak1=med)
    return out


# ── Custom-stiffness build, supporting pier rotational springs ───
def build_with_pier_rot_spring(scenario: str, e_deck: float, k_rot: float) -> bf.BeamModel:
    """Override beam_fem.build: keep w-constraint at piers but add a
    rotational spring k_rot to ground at the pier rotational DOF."""
    model = bf.build(scenario)        # builds at module-level E_DECK
    # Recompute K with the requested E (linear scaling of bending stiffness)
    scale = e_deck / bf.E_DECK
    model.K = model.K * scale         # bending K scales linearly with E
    # Add rotational ground springs at every pier node
    if k_rot > 0:
        for nidx in bf.PIER_NODE_IDX:
            dof_theta = 2 * nidx + 1
            model.K[dof_theta, dof_theta] += k_rot
    return model


def acc_frf_fast(model: bf.BeamModel, alpha: float, beta: float,
                  n_modes: int = N_MODES) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Modal-superposition accelerance evaluated only on the SCI band."""
    Kf = model.K[np.ix_(model.free_dofs, model.free_dofs)]
    Mf = model.M[np.ix_(model.free_dofs, model.free_dofs)]
    eigvals, eigvecs = eigh(Kf, Mf, subset_by_index=[0, n_modes - 1])
    eigvals = np.maximum(eigvals, 1e-12)
    freqs = np.sqrt(eigvals) / (2 * np.pi)
    full = np.zeros((model.K.shape[0], n_modes))
    full[model.free_dofs, :] = eigvecs
    sensor_dofs = 2 * bf.SENSOR_NODE_IDX
    phi_sens = full[sensor_dofs, :]      # (9, n_modes)
    deck_w = np.array([2 * i for i in range(model.n_nodes)
                        if 2 * i in model.free_dofs])
    phi_in = full[deck_w, :]             # (n_in, n_modes)
    omega_r = 2 * np.pi * freqs
    omega = 2 * np.pi * FREQ_GRID[BAND][:, None]
    zeta = 0.5 * (alpha / np.maximum(omega_r, 1e-3)
                  + beta * omega_r)
    denom = (-omega**2 + 2j * zeta * omega_r * omega + omega_r**2)
    # |H_acc[f, i]|² = sum_j |sum_r phi_sens[i,r] phi_in[j,r] / denom[f,r]|² · ω⁴
    #    auto-spec already integrates over j so we can split:
    H_disp = np.einsum("ir, jr, fr -> fij",
                       phi_sens, phi_in, 1.0 / denom)
    H_acc = -(omega[..., None]**2) * H_disp
    return freqs, phi_sens, H_acc


def model_band_spec(model: bf.BeamModel, alpha: float, beta: float,
                     f_cut: float) -> tuple[np.ndarray, float]:
    """Return synthetic band-restricted |Y| at 9 sensors, plus f_1."""
    freqs, _, H_acc = acc_frf_fast(model, alpha, beta)
    W = 1.0 / (1.0 + (FREQ_GRID[BAND] / f_cut) ** 4)
    S = np.einsum("fij, f -> fi", np.abs(H_acc) ** 2, W)
    Y = np.sqrt(np.maximum(S, 0.0))
    return Y, float(freqs[0])


# ── Loss ─────────────────────────────────────────────────────────
def make_loss(exp_cache):
    counter = {"n": 0}
    best = {"score": -np.inf, "x": None}
    t0 = time.time()
    def loss(x):
        e_gpa, alpha, beta, f_cut, log_krot = x
        e_deck = e_gpa * 1e9
        k_rot = 10.0 ** log_krot
        sci_list = []
        f1_dev = 0.0
        for sc in ("reference", "field3", "field4"):
            model = build_with_pier_rot_spring(sc, e_deck, k_rot)
            try:
                Y, f1 = model_band_spec(model, alpha, beta, f_cut)
            except Exception:
                return 10.0
            if sc == "reference":
                f1_dev = ((f1 - F1_EXP) / F1_EXP) ** 2
            H_band = Y.astype(np.complex128)
            C = cfdac(H_band)
            sci_list.append(sci(exp_cache[sc]["C"], C))
        mean_sci = float(np.mean(sci_list))
        score = mean_sci - W_FREQ * f1_dev
        counter["n"] += 1
        if score > best["score"]:
            best["score"] = score
            best["x"] = x.copy()
            print(f"  [eval {counter['n']:4d} t={time.time()-t0:6.1f}s] "
                  f"SCI={mean_sci:.4f} (ref={sci_list[0]:.3f} "
                  f"f3={sci_list[1]:.3f} f4={sci_list[2]:.3f}) "
                  f"f1={f1:.3f}Hz "
                  f"E={e_gpa:.2f}GPa α={alpha:.3g} β={beta:.3g} "
                  f"fc={f_cut:.2f}Hz krot=1e{log_krot:.2f}")
        return -score
    return loss, best


# ── Driver ───────────────────────────────────────────────────────
def main():
    print("Loading experimental CFDAC cache…")
    exp_cache = load_experimental_cfdac()
    print(f"  ref / field3 / field4: H_band shape = {exp_cache['reference']['H'].shape}")

    loss, best = make_loss(exp_cache)

    # Bounds: [E_GPa, alpha, beta, f_cut_Hz, log10(k_rot)]
    bounds = [
        (5.0,   45.0),        # E_c [GPa] — effective concrete modulus
        (0.001,  5.0),        # Rayleigh α
        (1e-6, 1e-2),         # Rayleigh β
        (1.0,  25.0),         # input PSD low-pass corner [Hz]
        (0.0,  11.0),         # log10(k_rot) — pier rotational spring
    ]

    print("Starting differential_evolution…")
    res = differential_evolution(
        loss, bounds, seed=20260516, maxiter=40, popsize=15,
        tol=1e-4, workers=1, polish=True, disp=False)
    print(f"\nDE done. Best score = {-res.fun:.4f}")
    print(f"Best params: E={res.x[0]:.3f} GPa  α={res.x[1]:.4g}  "
          f"β={res.x[2]:.4g}  f_cut={res.x[3]:.3f}  k_rot=1e{res.x[4]:.3f}")

    # Save best params
    out = {
        "best_score": float(-res.fun),
        "best_params": {
            "E_GPa": float(res.x[0]),
            "alpha_rayl": float(res.x[1]),
            "beta_rayl": float(res.x[2]),
            "f_cut_traffic_Hz": float(res.x[3]),
            "k_pier_rot_log10": float(res.x[4]),
        },
        "iterations": int(res.nit),
        "n_evals": int(res.nfev),
    }
    (HERE / "best_params.json").write_text(json.dumps(out, indent=2))
    print(f"\nWritten {HERE / 'best_params.json'}")


if __name__ == "__main__":
    main()
