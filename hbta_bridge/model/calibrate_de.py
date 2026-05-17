"""Differential-evolution calibration of HBTA stage-1 model.

Optimises 6 knobs against the UDS Y-sweep median |frf_H1|:
    E_factor          0.3 -- 2.0    Young's modulus scale
    A_arch_scale      0.5 -- 3.0    cross-section area of ARCH chord
    A_chord_scale     0.5 -- 3.0    area of bottom chord
    I_arch_scale      0.3 -- 3.0    bending inertia of ARCH chord
    zeta_scale        1   -- 30     uniform damping multiplier
    deck_mass_factor  0.3 -- 3.0    deck distributed mass

Loss = -smooth_log_SCI on the 2.5-30 Hz band (negative because DE
minimises).  Runs sequentially, ~0.3-0.5 s per eval.  ~1000 evals.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import scipy.optimize as so

import os, sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
# Force single-thread BLAS so eigh stays predictable
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import run_initial_comparison as R           # noqa: E402

ROOT = Path(__file__).resolve().parent

# Load experimental UDS Y data once (slow part)
print("Loading experimental UDS Y-sweep data…")
H_exp_med, _ = R.load_uds_y(ROOT.parent / "output" / "chunks")
fs = 100.0
N_T = 1024
freq_grid = np.fft.rfftfreq(N_T, d=1/fs)
band = (freq_grid >= 2.5) & (freq_grid <= 30.0)
ch_keep = np.arange(10)
C_exp_sm = R.smoothed_log_cfdac(H_exp_med[band], sigma=4.0)
C_exp_rw = R.cfdac(H_exp_med[band])

# Cached geometry
_joints, _members, _bc_nodes, _ = R.build_geometry()
print(f"  geometry: {len(_joints)} joints, {len(_members)} members")

# Original section-property baseline
_BASE = dict(R.SECTION_PROPS)


def evaluate(params):
    e_factor, a_arch, a_chord, i_arch, zeta_scale, deck_m = params
    # Patch global parameters
    R.E_STEEL = 210.0e9 * e_factor
    R.G_STEEL = R.E_STEEL / (2.0 * (1.0 + R.NU_STEEL))
    R.DECK_LUMPED_KG_PER_M = 200.0 * deck_m
    # Section overrides per group
    R.SECTION_PROPS["ARCH"]     = (_BASE["ARCH"][0]     * a_arch,
                                     _BASE["ARCH"][1]     * i_arch,
                                     _BASE["ARCH"][2]     * i_arch,
                                     _BASE["ARCH"][3])
    R.SECTION_PROPS["BOTCHORD"] = (_BASE["BOTCHORD"][0] * a_chord,
                                     _BASE["BOTCHORD"][1],
                                     _BASE["BOTCHORD"][2],
                                     _BASE["BOTCHORD"][3])
    # Re-build members with new sections
    members = [R.Member(m.j1, m.j2, m.group) for m in _members]
    R.assign_sections(members)
    K, M = R.assemble(_joints, members)
    Kf, Mf, free = R.apply_pinned_bc(K, M, _bc_nodes)
    try:
        freqs_modes, eigvecs = R.solve_modes(Kf, Mf, n_modes=40)
    except Exception:
        return 1.0     # bad config
    sens = R.sensor_dofs(_joints, "y")
    in_dof = R.shaker_dof(_joints, "P1", "y")
    zeta = R.piecewise_zeta(freqs_modes, scale=zeta_scale)
    H = R.modal_frf(eigvecs, freqs_modes, zeta, sens, in_dof, freq_grid, free)
    # Fit gain
    num = np.sum(np.abs(H_exp_med[np.ix_(band, ch_keep)]))
    den = np.sum(np.abs(H        [np.ix_(band, ch_keep)])) + 1e-30
    H_s = H * float(num / den)
    C_mod_sm = R.smoothed_log_cfdac(H_s[band], sigma=4.0)
    C_mod_rw = R.cfdac(H_s[band])
    s_sm = R.sci(C_exp_sm, C_mod_sm)
    s_rw = R.sci(C_exp_rw, C_mod_rw)
    # Composite loss: weight smooth + raw equally
    loss = -0.6 * s_sm - 0.4 * s_rw
    return loss


bounds = [(0.3, 2.0),   # E_factor
          (0.5, 3.0),   # A_arch
          (0.5, 3.0),   # A_chord
          (0.3, 3.0),   # I_arch
          (1.0, 30.0),  # zeta_scale
          (0.3, 3.0)]   # deck_m


_call = {"n": 0, "best": 0.0, "best_x": None}
def cb(xk, convergence):
    print(f"  gen {_call['n']:3d}   conv={convergence:.3f}   "
          f"best loss = {_call['best']:.4f}  → "
          f"E={_call['best_x'][0]:.2f}  A_arch={_call['best_x'][1]:.2f}  "
          f"A_ch={_call['best_x'][2]:.2f}  I_arch={_call['best_x'][3]:.2f}  "
          f"ζ×{_call['best_x'][4]:.1f}  m_deck×{_call['best_x'][5]:.2f}",
          flush=True)
    _call["n"] += 1


def loss_track(params):
    v = evaluate(params)
    if v < _call["best"]:
        _call["best"] = v
        _call["best_x"] = params.copy()
    return v


print("Starting differential_evolution…")
t0 = time.time()
res = so.differential_evolution(
    loss_track, bounds=bounds,
    maxiter=80, popsize=15, tol=1e-3,
    mutation=(0.4, 1.4), recombination=0.7,
    seed=42, polish=True, workers=1,
    init="sobol", callback=cb,
    updating="deferred",
)
elapsed = time.time() - t0
print(f"\nDE done in {elapsed:.1f}s, evals={res.nfev}")
print(f"best loss = {res.fun:.4f}")
print(f"best x:")
names = ["E_factor","A_arch","A_chord","I_arch","zeta_scale","deck_m"]
for n, v in zip(names, res.x):
    print(f"  {n:12s} = {v:.3f}")

# Final eval to get the actual SCI values
final = evaluate(res.x)
# Re-evaluate to capture both metrics
e_factor, a_arch, a_chord, i_arch, zeta_scale, deck_m = res.x
R.E_STEEL = 210.0e9 * e_factor
R.G_STEEL = R.E_STEEL / (2.0 * (1.0 + R.NU_STEEL))
R.DECK_LUMPED_KG_PER_M = 200.0 * deck_m
R.SECTION_PROPS["ARCH"]     = (_BASE["ARCH"][0]     * a_arch,
                                 _BASE["ARCH"][1]     * i_arch,
                                 _BASE["ARCH"][2]     * i_arch,
                                 _BASE["ARCH"][3])
R.SECTION_PROPS["BOTCHORD"] = (_BASE["BOTCHORD"][0] * a_chord,
                                 _BASE["BOTCHORD"][1],
                                 _BASE["BOTCHORD"][2],
                                 _BASE["BOTCHORD"][3])
members = [R.Member(m.j1, m.j2, m.group) for m in _members]
R.assign_sections(members)
K, M = R.assemble(_joints, members)
Kf, Mf, free = R.apply_pinned_bc(K, M, _bc_nodes)
freqs_modes, eigvecs = R.solve_modes(Kf, Mf, n_modes=40)
sens = R.sensor_dofs(_joints, "y")
in_dof = R.shaker_dof(_joints, "P1", "y")
zeta = R.piecewise_zeta(freqs_modes, scale=zeta_scale)
H = R.modal_frf(eigvecs, freqs_modes, zeta, sens, in_dof, freq_grid, free)
num = np.sum(np.abs(H_exp_med[np.ix_(band, ch_keep)]))
den = np.sum(np.abs(H        [np.ix_(band, ch_keep)])) + 1e-30
gain = float(num / den)
H_s = H * gain
s_sm = R.sci(C_exp_sm, R.smoothed_log_cfdac(H_s[band], sigma=4.0))
s_rw = R.sci(C_exp_rw, R.cfdac(H_s[band]))
print(f"\nFinal: smooth-SCI = {s_sm:.4f}   raw-SCI = {s_rw:.4f}")
print(f"first 5 modes: {freqs_modes[:5]}")

(ROOT / "best_params_de.json").write_text(json.dumps({
    "E_factor": float(res.x[0]),
    "A_arch_scale": float(res.x[1]),
    "A_chord_scale": float(res.x[2]),
    "I_arch_scale": float(res.x[3]),
    "zeta_scale": float(res.x[4]),
    "deck_mass_factor": float(res.x[5]),
    "smooth_SCI": float(s_sm),
    "raw_SCI": float(s_rw),
    "gain": gain,
    "n_evals": int(res.nfev),
    "first_5_modes_Hz": [float(f) for f in freqs_modes[:5]],
}, indent=2))
print(f"\nwrote {ROOT/'best_params_de.json'}")
