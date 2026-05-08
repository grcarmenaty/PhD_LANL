"""calibrate_3sbb_v4.py — high-correlation calibration of the 3SBB reduced model.

Improvements over v3
--------------------
1.  **Richer parameter set** (13 free vars vs 5):

      * 3 column factors  (cf_s1, cf_s2, cf_s3)
      * 3 per-storey JSR  (jsr_s1, jsr_s2, jsr_s3)
      * 3 floor extra masses  (m_floor1, m_floor2, m_floor3)
      * 1 base extra mass     (m_base)
      * 3 modal damping ratios for the Y-modes (zeta_1, zeta_2, zeta_3)

    The previous v3 had a single JSR and a single base mass, which meant the
    free-base 4-DOF Y-chain was constrained to nearly-uniform parameters and
    the Y-mode frequency ratios could not approach the experimental targets
    [1, 2.385, 3.257].  Allowing per-storey JSR + heavier base mass + per-floor
    extra mass removes that bottleneck.

2.  **FRF *correlation* objective**, not just log-magnitude MSE.  We minimise
    a weighted sum of:

      * 1 - Pearson(log|H_model|, log|H_exp|)           (per Y-floor channel)
      * 1 - cosine_similarity(H_model_complex, H_exp_complex)
      * mean squared (f_model - f_exp)/f_exp            (3 Y-modes)
      * 1 - MAC(phi_model, phi_exp)                     (3 modes × 6 sensors)

    Pearson + complex cosine similarity directly maximise the correlation
    coefficient that downstream CFDAC/MAC analysis cares about.

3.  **All 6 floor sensors** (S5/S6/S7 + S11/S12/S13) participate in the FRF
    fit, not just 3.  Mode-shape MAC is computed across these 6 floor sensors
    to disambiguate ambiguous shapes.

Outputs
-------
``calibration_result.npz`` — superset of v3 fields with backwards-compatible
``jsr`` (mean of per-storey values) and added ``jsr_per_storey`` and
``floor_extra_mass`` fields.  ``damage_scenarios.py`` and ``model_3sbb.py``
read the richer fields when present and fall back to the scalar otherwise.
"""

from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.signal import find_peaks
import h5py

_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE))
from reduced_model_semirigid import (BuildingGeometry, modes,
                                      compute_frf_matrix, point_to_dof_vector)
import params as P

# ─── Experimental data ───────────────────────────────────────────────────────
# Loader auto-detects the slim per-case "median" file (preferred) or the
# original full per-measurement archive.
HDF5_MEDIAN = _HERE / "experimental_frfs_median.h5"
HDF5_FULL   = _HERE / "experimental_frfs.h5"
PRISTINE_LABEL = 59
FLOOR_Y_CHS = [1, 2, 3, 5, 6, 7]   # S5, S6, S7, S11, S12, S13  (all upper floors)
F_LO, F_HI  = 10.0, 80.0


def _load_pristine_summary():
    """Return (freq, |H|_summary, complex_summary) for the Pristine case.

    Prefers ``experimental_frfs_median.h5`` (slim, committed) so the script
    runs without the 305 MB raw archive.  Falls back to the full file when
    the median has not been built yet.
    """
    if HDF5_MEDIAN.exists():
        with h5py.File(HDF5_MEDIAN, "r") as f:
            freq = np.array(f["freq"])
            cases = [c.decode() if isinstance(c, bytes) else str(c)
                     for c in np.array(f["case_names"])]
            ci = cases.index("Pristine")
            mag_med  = np.array(f["mag_median"][ci])
            cplx_med = np.array(f["complex_median"][ci])
            n_meas   = int(f["case_counts"][ci])
        print(f"  Median dataset, Pristine: {n_meas} measurements aggregated "
              f"(loaded from {HDF5_MEDIAN.name})")
        return freq, mag_med, cplx_med

    if not HDF5_FULL.exists():
        raise FileNotFoundError(
            f"Neither {HDF5_MEDIAN} nor {HDF5_FULL} exist. Run "
            f"build_median_dataset.py to produce the slim file.")
    with h5py.File(HDF5_FULL, "r") as f:
        freq    = np.array(f["freq"])
        data    = np.array(f["data"])
        labels  = np.array(f["labels"])
    idx_p   = np.where(labels == PRISTINE_LABEL)[0]
    H_raw   = data[idx_p]
    print(f"  Full dataset, Pristine: {len(idx_p)} measurements")
    return freq, np.mean(np.abs(H_raw), axis=0), np.mean(H_raw, axis=0)


print("Loading experimental FRFs …", flush=True)
freq_exp, H_mean, H_cmean = _load_pristine_summary()

# ─── Identify Y-mode resonances and half-power damping (S7 = ch3) ────────────
def half_power_zeta(H_mag, freq, pk_idx):
    half  = H_mag[pk_idx] / np.sqrt(2.0)
    left  = np.where((freq < freq[pk_idx]) & (H_mag < half))[0]
    right = np.where((freq > freq[pk_idx]) & (H_mag < half))[0]
    if len(left) and len(right):
        return (freq[right[0]] - freq[left[-1]]) / (2.0 * freq[pk_idx])
    return 0.01

mask_band = (freq_exp >= F_LO) & (freq_exp <= F_HI)
ch3_band  = H_mean[mask_band, 3]
freq_band = freq_exp[mask_band]
peaks, props = find_peaks(ch3_band, distance=40, prominence=ch3_band.max() * 0.05)
top3 = peaks[np.argsort(props['prominences'])[-3:]]
top3 = top3[np.argsort(freq_band[top3])]
F_EXP = freq_band[top3]
ZETA3 = np.array([half_power_zeta(ch3_band, freq_band, pk) for pk in top3])
print(f"  Y-mode targets:  {F_EXP} Hz")
print(f"  Half-power zeta: {ZETA3 * 100} %")

# ─── Experimental mode shapes from complex average at each resonance ─────────
# Use the 6 floor sensors (S5,S6,S7,S11,S12,S13) so MAC has 6 components.
phi_exp = np.zeros((3, 6))
for j, fr in enumerate(F_EXP):
    fi = np.argmin(np.abs(freq_exp - fr))
    H_at = H_cmean[fi, FLOOR_Y_CHS]
    ref  = H_at[0] if abs(H_at[0]) > 1e-30 else 1.0     # normalise to S5
    v    = H_at / ref
    phi_exp[j] = np.real(v) * (np.abs(v) / (np.abs(np.real(v)) + 1e-30))

print("Experimental mode shapes (S5,S6,S7,S11,S12,S13 normalised to S5):")
for j in range(3):
    print(f"  Mode {j+1}: " + " ".join(f"{x:+.3f}" for x in phi_exp[j]))

# ─── Fit-band reference ──────────────────────────────────────────────────────
freq_fit  = freq_exp[mask_band]
H_exp_y   = H_mean[mask_band][:, FLOOR_Y_CHS]                  # (Nf, 6)  magnitude
logH_exp  = np.log10(H_exp_y + 1e-12)
H_cmean_y = H_cmean[mask_band][:, FLOOR_Y_CHS]                 # (Nf, 6)  complex

# Pre-centre log-magnitude for Pearson correlation (per channel)
logH_exp_c    = logH_exp - logH_exp.mean(axis=0, keepdims=True)
logH_exp_norm = np.linalg.norm(logH_exp_c, axis=0) + 1e-30

# Pre-normalise complex experimental for cosine similarity (per channel)
H_exp_norm_c = H_cmean_y / (np.linalg.norm(H_cmean_y, axis=0, keepdims=True) + 1e-30)

# ─── Sensor / shaker positions (matches model_3sbb.py) ───────────────────────
geom_nom = BuildingGeometry()
cx = geom_nom.plate_lx / 2.0
PL = geom_nom.plate_ly
xl = geom_nom.col_lx / 2.0
xh = geom_nom.plate_lx - xl
z0 = geom_nom.plate_z_centre(0)
z1 = geom_nom.plate_z_centre(1)
z2 = geom_nom.plate_z_centre(2)
z3 = geom_nom.plate_z_centre(3)

INPUT_PT   = ((cx, 0.0, z0), 'Y')
OUTPUT_PTS = [
    ((xl, PL, z3), 'Y'),   # ch1  S5  fl3 xl
    ((xl, PL, z2), 'Y'),   # ch2  S6  fl2 xl
    ((xl, PL, z1), 'Y'),   # ch3  S7  fl1 xl
    ((xh, PL, z3), 'Y'),   # ch5  S11 fl3 xh
    ((xh, PL, z2), 'Y'),   # ch6  S12 fl2 xh
    ((xh, PL, z1), 'Y'),   # ch7  S13 fl1 xh
]


def geom_from_x(x):
    """13-vector unpack:
       0..2  cf_s1..3       (column factor)
       3..5  log10 jsr_s1..3
       6..8  m_floor_1..3   (kg)
       9     m_base         (kg)
       10..12  zeta_1..3
    """
    cf  = x[0:3]
    jsr = 10.0 ** np.asarray(x[3:6])
    mf  = x[6:9]
    mb  = x[9]
    g = BuildingGeometry(
        joint_stiffness_ratio              = float(np.exp(np.log(jsr).mean())),
        joint_stiffness_ratio_per_storey   = jsr.tolist(),
        base_extra_mass                    = float(mb),
        floor_extra_mass                   = mf.tolist(),
    )
    g.column_factor = np.array([[cf[0]] * 4, [cf[1]] * 4, [cf[2]] * 4], dtype=float)
    return g


def y_modes_and_shape(geom):
    """Return (f_y[3], phi_y[3,6]) for the 3 Y-modes ranked by base-Y participation."""
    freqs_all, V, _ = modes(geom)
    b = point_to_dof_vector((cx, 0.0, z0), 'Y', geom)
    bphi = np.abs(V.T @ b)
    ie = np.where(freqs_all > 0.5)[0]
    if len(ie) < 3:
        return None, None
    order = np.argsort(-bphi[ie])
    y_idx = ie[order[:3]]
    y_idx = y_idx[np.argsort(freqs_all[y_idx])]    # frequency order
    f_y = freqs_all[y_idx]

    # Sensor mode-shape vector at the 6 floor sensors
    C = np.stack([point_to_dof_vector(p, d, geom) for p, d in OUTPUT_PTS], axis=1)
    cphi = V.T @ C                                  # (n_modes, 6)
    phi_y = cphi[y_idx]                             # (3, 6)
    # normalise to first column (S5) so MAC is sign-invariant after this
    for j in range(3):
        ref = phi_y[j, 0] if abs(phi_y[j, 0]) > 1e-30 else 1.0
        phi_y[j] = phi_y[j] / ref
    return f_y, phi_y


# ─── Objective: FRF correlation + frequency residual + MAC ──────────────────
W_PEARSON = 1.0     # 1 - Pearson(log|H_mod|, log|H_exp|)
W_COS     = 1.0     # 1 - cosine_similarity(H_mod_complex, H_exp_complex)
W_FREQ    = 8.0     # frequency residuals (3 Y-modes)
W_MAC     = 2.0     # 1 - MAC for each of 3 modes


def _pearson_per_channel(A, B, B_centred=None, B_norm=None):
    """Mean (1 - Pearson) across columns of A vs B; B normalised inputs cached."""
    Ac = A - A.mean(axis=0, keepdims=True)
    if B_centred is None:
        B_centred = B - B.mean(axis=0, keepdims=True)
    if B_norm is None:
        B_norm = np.linalg.norm(B_centred, axis=0) + 1e-30
    A_norm = np.linalg.norm(Ac, axis=0) + 1e-30
    cov    = np.sum(Ac * B_centred, axis=0)
    rho    = cov / (A_norm * B_norm)                # length n_channels
    return float(np.mean(1.0 - rho))


def _cosine_complex(H_mod, H_exp_normed):
    H_mod_n = H_mod / (np.linalg.norm(H_mod, axis=0, keepdims=True) + 1e-30)
    cos     = np.real(np.sum(H_mod_n * H_exp_normed.conj(), axis=0))
    return float(np.mean(1.0 - cos))


def _mac(phi_a, phi_b):
    num = np.dot(phi_a, phi_b) ** 2
    den = np.dot(phi_a, phi_a) * np.dot(phi_b, phi_b) + 1e-30
    return float(num / den)


def objective(x, return_breakdown=False):
    # Hard guards
    if (x[6] < 0 or x[7] < 0 or x[8] < 0 or x[9] < 0 or
            any(z < 1e-4 for z in x[10:13]) or
            any(c < 0.3 for c in x[0:3])):
        return 1e9

    geom = geom_from_x(x)
    zeta3 = np.asarray(x[10:13], dtype=float)

    # Build damping array: Y-modes get zeta3 by frequency rank, others get mean.
    try:
        f_all, V, _ = modes(geom)
    except Exception:
        return 1e9
    ie = np.where(f_all > 0.5)[0]
    if len(ie) < 3:
        return 1e9
    b = point_to_dof_vector((cx, 0.0, z0), 'Y', geom)
    bphi = np.abs(V.T @ b)
    order   = np.argsort(-bphi[ie])
    y_idx   = ie[order[:3]]
    y_idx   = y_idx[np.argsort(f_all[y_idx])]
    zeta_full = np.full(len(ie), float(zeta3.mean()))
    for rank, idx in enumerate(y_idx):
        local_pos = int(np.where(ie == idx)[0][0])
        zeta_full[local_pos] = zeta3[rank]

    try:
        H_mod = compute_frf_matrix(
            freq_fit, inputs=[INPUT_PT], outputs=OUTPUT_PTS,
            geom=geom, damping=zeta_full,
        )                          # (Nf, 6, 1) complex
    except Exception:
        return 1e9

    H_mod_c = H_mod[:, :, 0]                       # (Nf, 6) complex
    log_mod = np.log10(np.abs(H_mod_c) + 1e-12)

    L_pear = _pearson_per_channel(log_mod, logH_exp,
                                  B_centred=logH_exp_c,
                                  B_norm=logH_exp_norm)
    L_cos  = _cosine_complex(H_mod_c, H_exp_norm_c)

    f_y    = f_all[y_idx]
    rel    = (f_y - F_EXP) / F_EXP
    L_freq = float(np.sum(rel ** 2))

    # MAC at each model resonance using sensor mode shapes
    C = np.stack([point_to_dof_vector(p, d, geom) for p, d in OUTPUT_PTS], axis=1)
    cphi = V.T @ C
    phi_y = cphi[y_idx]
    L_mac = 0.0
    for j in range(3):
        ref = phi_y[j, 0] if abs(phi_y[j, 0]) > 1e-30 else 1.0
        phi_norm = phi_y[j] / ref
        L_mac += 1.0 - _mac(phi_exp[j], phi_norm)
    L_mac /= 3.0

    L = (W_PEARSON * L_pear + W_COS * L_cos
         + W_FREQ * L_freq + W_MAC * L_mac)
    if return_breakdown:
        return L, dict(pearson=L_pear, cos=L_cos, freq=L_freq, mac=L_mac,
                       freqs=f_y.tolist())
    return float(L)


# ─── Bounds & initial guess ──────────────────────────────────────────────────
bounds = [
    (0.5, 2.0),          # cf_s1
    (0.5, 2.0),          # cf_s2
    (0.5, 2.0),          # cf_s3
    (np.log10(2.0), np.log10(80.0)),  # log10 jsr_s1
    (np.log10(2.0), np.log10(80.0)),  # log10 jsr_s2
    (np.log10(2.0), np.log10(80.0)),  # log10 jsr_s3
    (0.0, 5.0),          # m_floor1 (kg)
    (0.0, 5.0),          # m_floor2
    (0.0, 5.0),          # m_floor3
    (0.0, 60.0),         # m_base   (shaker + base attachments, can be heavy)
    (0.001, 0.05),       # zeta_1
    (0.001, 0.05),       # zeta_2
    (0.001, 0.05),       # zeta_3
]

# Seed near v3 solution but with heavier base + per-storey JSR fan-out
x0 = np.array([
    1.10, 1.05, 1.00,                       # cf
    np.log10(15.0), np.log10(20.0), np.log10(25.0),  # log10 jsr
    0.2, 0.2, 0.2,                          # m_floor
    20.0,                                   # m_base
    ZETA3[0], ZETA3[1], ZETA3[2],
])
print(f"\nInitial cost: {objective(x0):.6f}")
fv0, brk0 = objective(x0, return_breakdown=True)
print(f"  breakdown: pearson={brk0['pearson']:.4f}  cos={brk0['cos']:.4f} "
      f"freq={brk0['freq']:.5f}  mac={brk0['mac']:.4f}  f_y={brk0['freqs']}")

# ─── Stage 1: differential evolution ─────────────────────────────────────────
print("\n── Stage 1: differential evolution ──────────────────────────────")
t0 = time.time()
res_de = differential_evolution(
    objective, bounds, seed=42,
    maxiter=800, popsize=24, tol=1e-9,
    mutation=(0.5, 1.5), recombination=0.9,
    workers=-1, disp=True, polish=False,
    updating='deferred', x0=x0, init='sobol',
)
print(f"\nDE cost = {res_de.fun:.6f}  in {time.time()-t0:.1f}s")

# ─── Stage 2: Nelder-Mead polish ─────────────────────────────────────────────
print("\n── Stage 2: Nelder-Mead polish ──────────────────────────────────")
res_nm = minimize(objective, res_de.x, method='Nelder-Mead',
                  options={'maxiter': 50000, 'xatol': 1e-10, 'fatol': 1e-10,
                           'adaptive': True, 'disp': True})
x_opt = res_nm.x
print(f"NM cost = {res_nm.fun:.6f}")

# ─── Report ──────────────────────────────────────────────────────────────────
geom_opt = geom_from_x(x_opt)
f_y_opt, phi_y_opt = y_modes_and_shape(geom_opt)
fv, brk  = objective(x_opt, return_breakdown=True)

cf1, cf2, cf3   = x_opt[0:3]
jsr1, jsr2, jsr3 = (10.0 ** np.asarray(x_opt[3:6])).tolist()
mf1, mf2, mf3   = x_opt[6:9]
mb              = x_opt[9]
z1, z2, z3      = x_opt[10:13]

print("\n" + "=" * 64)
print("OPTIMAL PARAMETERS")
print("=" * 64)
print(f"  column factor  : [{cf1:.4f}, {cf2:.4f}, {cf3:.4f}]")
print(f"  per-storey JSR : [{jsr1:.3f}, {jsr2:.3f}, {jsr3:.3f}]")
print(f"  floor extra m  : [{mf1:.3f}, {mf2:.3f}, {mf3:.3f}] kg")
print(f"  base extra m   : {mb:.3f} kg")
print(f"  Y-mode zeta    : [{z1*100:.3f}, {z2*100:.3f}, {z3*100:.3f}] %")

print("\nFREQUENCY COMPARISON (Y-modes):")
for j in range(3):
    err = (f_y_opt[j] - F_EXP[j]) / F_EXP[j] * 100.0
    mac = _mac(phi_exp[j], phi_y_opt[j])
    print(f"  Mode {j+1}:  exp={F_EXP[j]:7.3f} Hz  mod={f_y_opt[j]:7.3f} Hz  "
          f"err={err:+6.2f}%   MAC={mac:.4f}")

print(f"\nObjective breakdown:")
for k, v in brk.items():
    if k != 'freqs':
        print(f"  {k:10s} = {v:.6f}")

# ─── Save (backwards-compatible superset of v3) ──────────────────────────────
out = _HERE / "calibration_result.npz"
zeta_full = np.array([z1, z2, z3, np.mean([z1, z2, z3])])    # 4 used by FRF code
np.savez(out,
         # legacy scalar fields
         jsr             = float(np.mean([jsr1, jsr2, jsr3])),
         base_extra_mass = mb,
         cf_s1           = cf1, cf_s2 = cf2, cf_s3 = cf3,
         damping         = float(z1),
         damping_modes   = zeta_full,
         f_cal           = f_y_opt, f_exp = F_EXP,
         phi_cal         = phi_y_opt, phi_exp = phi_exp,
         # v4 additions
         jsr_per_storey   = np.array([jsr1, jsr2, jsr3]),
         floor_extra_mass = np.array([mf1, mf2, mf3]),
         zeta_y_modes     = np.array([z1, z2, z3]),
         param_names      = np.array([
             'cf_s1', 'cf_s2', 'cf_s3',
             'jsr_s1', 'jsr_s2', 'jsr_s3',
             'm_floor1', 'm_floor2', 'm_floor3',
             'base_extra_mass_kg',
             'zeta_1', 'zeta_2', 'zeta_3']),
         k_story1 = 0.0, k_story2 = 0.0, k_story3 = 0.0,
)
print(f"\nSaved → {out}")
print("\n[calibrate_3sbb_v4.py] Done.")
