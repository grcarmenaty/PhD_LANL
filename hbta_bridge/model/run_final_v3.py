"""Regenerate figures + scores for the DE-round-3 (channel-normalised) best."""
from __future__ import annotations

import json, os, sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

import run_initial_comparison as R

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures_v3"
FIG.mkdir(exist_ok=True)

BEST = dict(
    E_factor=0.37,
    A_arch_scale=3.13,
    A_chord_scale=3.56,
    I_arch_scale=0.56,
    zeta_scale=19.3,
    deck_mass_factor=0.32,
)
print("== DE round-3 best (channel-normalised loss) ==")
for k, v in BEST.items():
    print(f"  {k:18s} = {v}")

R.E_STEEL = 210.0e9 * BEST["E_factor"]
R.G_STEEL = R.E_STEEL / (2.0 * (1.0 + R.NU_STEEL))
R.DECK_LUMPED_KG_PER_M = 200.0 * BEST["deck_mass_factor"]
_B = dict(R.SECTION_PROPS)
R.SECTION_PROPS["ARCH"]     = (_B["ARCH"][0]*BEST["A_arch_scale"],
                                  _B["ARCH"][1]*BEST["I_arch_scale"],
                                  _B["ARCH"][2]*BEST["I_arch_scale"],
                                  _B["ARCH"][3])
R.SECTION_PROPS["BOTCHORD"] = (_B["BOTCHORD"][0]*BEST["A_chord_scale"],
                                  _B["BOTCHORD"][1], _B["BOTCHORD"][2],
                                  _B["BOTCHORD"][3])

joints, members, bc_nodes, _ = R.build_geometry()
R.assign_sections(members)
K, M = R.assemble(joints, members)
Kf, Mf, free = R.apply_pinned_bc(K, M, bc_nodes)
freqs_modes, eigvecs = R.solve_modes(Kf, Mf, n_modes=40)
print(f"first 12 modes: {', '.join(f'{f:.2f}' for f in freqs_modes[:12])}")

sens = R.sensor_dofs(joints, "y")
in_dof = R.shaker_dof(joints, "P1", "y")
zeta = R.piecewise_zeta(freqs_modes, scale=BEST["zeta_scale"])

fs = 100.0
N_T = 1024
freq_grid = np.fft.rfftfreq(N_T, d=1/fs)
H = R.modal_frf(eigvecs, freqs_modes, zeta, sens, in_dof, freq_grid, free)
H_exp_med, H_exp_all = R.load_uds_y(ROOT.parent / "output" / "chunks")

band = (freq_grid >= 2.5) & (freq_grid <= 30.0)

def chan_norm(X):
    p = np.max(np.abs(X), axis=0, keepdims=True)
    return X / np.maximum(p, 1e-30)

# Channel-normalised metric (primary)
H_exp_n = chan_norm(H_exp_med[band])
H_mod_n = chan_norm(H[band])
C_exp_sm_n = R.smoothed_log_cfdac(H_exp_n, sigma=4.0)
C_mod_sm_n = R.smoothed_log_cfdac(H_mod_n, sigma=4.0)
C_exp_rw_n = R.cfdac(H_exp_n)
C_mod_rw_n = R.cfdac(H_mod_n)
sm_n = R.sci(C_exp_sm_n, C_mod_sm_n)
rw_n = R.sci(C_exp_rw_n, C_mod_rw_n)

# Global-gain metric (legacy, for v2-comparison)
ch_keep = np.arange(10)
gain = float(np.sum(np.abs(H_exp_med[np.ix_(band, ch_keep)])) /
              (np.sum(np.abs(H[np.ix_(band, ch_keep)])) + 1e-30))
H_s = H * gain
sm_g = R.sci(R.smoothed_log_cfdac(H_exp_med[band]), R.smoothed_log_cfdac(H_s[band]))
rw_g = R.sci(R.cfdac(H_exp_med[band]), R.cfdac(H_s[band]))

print(f"\nchan-norm  : raw_SCI={rw_n:.4f}  smooth_SCI={sm_n:.4f}")
print(f"global-gain: raw_SCI={rw_g:.4f}  smooth_SCI={sm_g:.4f}")

# Modes table
(FIG / "modes_table.txt").write_text("Mode  Freq[Hz]\n" + "\n".join(
    f"{i+1:3d}    {f:7.3f}" for i, f in enumerate(freqs_modes[:20])))

(FIG / "sci_scoreboard.txt").write_text(
    "DE round-3 (channel-normalised loss):\n"
    f"  chan-norm   raw_SCI = {rw_n:.4f}   smooth_SCI = {sm_n:.4f}\n"
    f"  global-gain raw_SCI = {rw_g:.4f}   smooth_SCI = {sm_g:.4f}\n"
    f"  gain                = {gain:.3e}\n"
    + "\n".join(f"  {k:18s} = {v}" for k, v in BEST.items()) + "\n"
    f"  first_5_modes_Hz    = {', '.join(f'{f:.3f}' for f in freqs_modes[:5])}\n")

# Smoothed log|H| envelopes (12 channels)
sig = 4.0
H_exp_l = np.log10(np.maximum(np.abs(H_exp_med), 1e-30))
H_mod_l = np.log10(np.maximum(np.abs(H_s),       1e-30))
for k in range(12):
    H_exp_l[:, k] = ndi.gaussian_filter1d(H_exp_l[:, k], sig)
    H_mod_l[:, k] = ndi.gaussian_filter1d(H_mod_l[:, k], sig)

fig, axes = plt.subplots(4, 3, figsize=(12, 11), sharex=True, sharey=True)
for k, ax in enumerate(axes.flat):
    ax.plot(freq_grid, H_exp_l[:, k], "r-", lw=1.0, label="experiment")
    ax.plot(freq_grid, H_mod_l[:, k], "b-", lw=1.0, label="model (DE v3)")
    ax.set_xlim(2.5, 25); ax.grid(True, alpha=0.4)
    ax.set_title(f"{R.SENSOR_NAMES[k]}", fontsize=9)
    if k == 0: ax.legend(fontsize=8, loc="lower left")
for ax in axes[-1, :]: ax.set_xlabel("freq [Hz]")
for ax in axes[:, 0]: ax.set_ylabel("log10 |H|")
fig.suptitle(f"HBTA UDS Y-sweep — smoothed log|H| per channel "
              f"(DE-v3, chan-norm smooth-SCI={sm_n:.3f})", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "smooth_envelopes_v3.png", dpi=130)
plt.close(fig)
print(f"wrote {FIG/'smooth_envelopes_v3.png'}")

# Per-channel FRF magnitude (6 sensors)
fig, ax = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
plot_chs = [0, 2, 3, 4, 7, 9]
for k, ch in enumerate(plot_chs):
    r, c = k // 2, k % 2
    ax[r, c].semilogy(freq_grid[1:], np.abs(H_exp_med[1:, ch]),
                        "r-", lw=1.0, label="experiment")
    ax[r, c].semilogy(freq_grid[1:], np.abs(H_s[1:, ch]),
                        "b-", lw=1.0, label="model (DE v3)")
    ax[r, c].set_title(f"{R.SENSOR_NAMES[ch]}  (ch{ch})", fontsize=9)
    ax[r, c].set_xlim(0.5, 25)
    ax[r, c].set_ylim(1e-3, 1e3)
    ax[r, c].grid(True, alpha=0.4)
    if k == 0: ax[r, c].legend(fontsize=8)
for r in range(3): ax[r, 0].set_ylabel("|H1|")
for c in range(2): ax[-1, c].set_xlabel("frequency [Hz]")
fig.suptitle(f"HBTA UDS Y-sweep — experiment vs FEM (DE-v3, "
              f"chan-norm smooth-SCI={sm_n:.3f})", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "frf_magnitude_v3.png", dpi=130)
plt.close(fig)
print(f"wrote {FIG/'frf_magnitude_v3.png'}")

# CFDAC (smooth-log, channel-normalised)
fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
ext = [freq_grid[band].min(), freq_grid[band].max(),
        freq_grid[band].max(), freq_grid[band].min()]
ax[0].imshow(C_exp_sm_n, extent=ext, cmap="viridis", vmin=0, vmax=1)
ax[0].set_title("smooth-log CFDAC — experiment (channel-norm)", fontsize=9)
ax[1].imshow(C_mod_sm_n, extent=ext, cmap="viridis", vmin=0, vmax=1)
ax[1].set_title("smooth-log CFDAC — model (channel-norm)", fontsize=9)
for a in ax:
    a.set_xlabel("f₁ [Hz]"); a.set_ylabel("f₂ [Hz]")
fig.suptitle(f"channel-normalised CFDAC (UDS Y-sweep) — smooth SCI {sm_n:.3f}",
              fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "cfdac_uds_v3.png", dpi=130)
plt.close(fig)
print(f"wrote {FIG/'cfdac_uds_v3.png'}")

# Per-class median FRF
chunks = ROOT.parent / "output" / "chunks"
by_class = {}
for p in sorted(chunks.glob("chunk_*.h5")):
    with h5py.File(p, "r") as f:
        lab = f["labels/class_code"][:]
        src = f["labels/source_record"][:]
        frf = f["frf_H1"][:]
        for c in range(9):
            m = (lab == c) & np.array([b"_Y_" in s for s in src])
            if m.any():
                by_class.setdefault(c, []).append(frf[m])
medians = {c: np.median(np.abs(np.concatenate(by_class[c])), axis=0)
            for c in by_class}

fig, ax = plt.subplots(1, 1, figsize=(9, 5))
colors = plt.cm.viridis(np.linspace(0, 1, 9))
ch = 2
for c in sorted(medians):
    nm = "UDS" if c == 0 else f"DS{c}"
    ax.semilogy(freq_grid[1:], medians[c][1:, ch],
                 color=colors[c], lw=1.0, label=nm)
ax.semilogy(freq_grid[1:], np.abs(H_s[1:, ch]),
             "k--", lw=1.8, label="FEM (DE v3)")
ax.set_xlim(0.5, 25); ax.set_ylim(1e-3, 1e2)
ax.set_xlabel("frequency [Hz]")
ax.set_ylabel("|H1|  AG05")
ax.set_title("HBTA per-class median |H1| vs FEM DE-v3 — AG05 (Y-sweep)",
              fontsize=10)
ax.legend(ncol=2, fontsize=8, loc="lower left")
ax.grid(True, alpha=0.4)
fig.tight_layout()
fig.savefig(FIG / "per_class_vs_model_v3.png", dpi=130)
plt.close(fig)
print(f"wrote {FIG/'per_class_vs_model_v3.png'}")

(ROOT / "best_params_de_v3.json").write_text(json.dumps({
    **BEST,
    "smooth_SCI_chnorm": float(sm_n),
    "raw_SCI_chnorm": float(rw_n),
    "smooth_SCI_globgain": float(sm_g),
    "raw_SCI_globgain": float(rw_g),
    "gain": gain,
    "first_5_modes_Hz": [float(f) for f in freqs_modes[:5]],
    "n_joints": len(joints),
    "n_members": len(members),
    "metric_note": "DE optimised on per-channel-normalised CFDAC; "
                    "smooth_SCI_chnorm is the primary metric.",
}, indent=2))
print(f"wrote {ROOT/'best_params_de_v3.json'}")
print("done.")
