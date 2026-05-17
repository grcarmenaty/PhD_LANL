"""Regenerate all figures + score tables for the DE-round-2 best params."""
from __future__ import annotations

import json
import os
import sys
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
FIG_DIR = ROOT / "figures_v2"
FIG_DIR.mkdir(exist_ok=True)

# Best DE round-2 parameters
BEST = dict(
    E_factor=0.65,
    A_arch_scale=1.75,
    A_chord_scale=3.68,
    I_arch_scale=5.61,
    zeta_scale=14.5,
    deck_mass_factor=6.12,
)
print("=== DE round-2 best params ===")
for k, v in BEST.items():
    print(f"  {k:18s} = {v}")

R.E_STEEL = 210.0e9 * BEST["E_factor"]
R.G_STEEL = R.E_STEEL / (2.0 * (1.0 + R.NU_STEEL))
R.DECK_LUMPED_KG_PER_M = 200.0 * BEST["deck_mass_factor"]
_BASE = dict(R.SECTION_PROPS)
R.SECTION_PROPS["ARCH"]     = (_BASE["ARCH"][0]     * BEST["A_arch_scale"],
                                 _BASE["ARCH"][1]     * BEST["I_arch_scale"],
                                 _BASE["ARCH"][2]     * BEST["I_arch_scale"],
                                 _BASE["ARCH"][3])
R.SECTION_PROPS["BOTCHORD"] = (_BASE["BOTCHORD"][0] * BEST["A_chord_scale"],
                                 _BASE["BOTCHORD"][1],
                                 _BASE["BOTCHORD"][2],
                                 _BASE["BOTCHORD"][3])

joints, members, bc_nodes, _ = R.build_geometry()
R.assign_sections(members)
K, M = R.assemble(joints, members)
Kf, Mf, free = R.apply_pinned_bc(K, M, bc_nodes)
freqs_modes, eigvecs = R.solve_modes(Kf, Mf, n_modes=40)
print(f"first 10 modes [Hz]: {', '.join(f'{f:.2f}' for f in freqs_modes[:10])}")

sens = R.sensor_dofs(joints, "y")
in_dof = R.shaker_dof(joints, "P1", "y")
zeta = R.piecewise_zeta(freqs_modes, scale=BEST["zeta_scale"])

fs = 100.0
N_T = 1024
freq_grid = np.fft.rfftfreq(N_T, d=1/fs)
H = R.modal_frf(eigvecs, freqs_modes, zeta, sens, in_dof, freq_grid, free)

# Load experimental
print("loading experimental UDS Y data…")
H_exp_med, H_exp_all = R.load_uds_y(ROOT.parent / "output" / "chunks")

band = (freq_grid >= 2.5) & (freq_grid <= 30.0)
ch_keep = np.arange(10)
gain = float(np.sum(np.abs(H_exp_med[np.ix_(band, ch_keep)])) /
              (np.sum(np.abs(H[np.ix_(band, ch_keep)])) + 1e-30))
H_s = H * gain
print(f"global gain (band 2.5-30 Hz, arch ch) = {gain:.3e}")

C_exp_rw = R.cfdac(H_exp_med[band])
C_mod_rw = R.cfdac(H_s[band])
C_exp_sm = R.smoothed_log_cfdac(H_exp_med[band], sigma=4.0)
C_mod_sm = R.smoothed_log_cfdac(H_s[band], sigma=4.0)
raw_sci = R.sci(C_exp_rw, C_mod_rw)
sm_sci  = R.sci(C_exp_sm, C_mod_sm)
print(f"raw  SCI = {raw_sci:.4f}")
print(f"smooth SCI = {sm_sci:.4f}")

# ─── Modes table ───────────────────────────────────────────────────
txt = "Mode  Freq[Hz]\n" + "\n".join(
    f"{i+1:3d}    {f:7.3f}" for i, f in enumerate(freqs_modes[:20]))
(FIG_DIR / "modes_table.txt").write_text(txt)

# ─── SCI scoreboard ───────────────────────────────────────────────
(FIG_DIR / "sci_scoreboard.txt").write_text(
    f"raw  CFDAC SCI  = {raw_sci:.4f}\n"
    f"smooth log-SCI  = {sm_sci:.4f}\n"
    f"gain            = {gain:.3e}\n"
    f"E_factor        = {BEST['E_factor']}\n"
    f"A_arch_scale    = {BEST['A_arch_scale']}\n"
    f"A_chord_scale   = {BEST['A_chord_scale']}\n"
    f"I_arch_scale    = {BEST['I_arch_scale']}\n"
    f"zeta_scale      = {BEST['zeta_scale']}\n"
    f"deck_mass_factor= {BEST['deck_mass_factor']}\n"
    f"first 5 modes   = " + ", ".join(f"{f:.3f}" for f in freqs_modes[:5]) + "\n")

# ─── Per-channel FRF overlay ──────────────────────────────────────
fig, ax = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
plot_chs = [0, 2, 3, 4, 7, 9]
for k, ch in enumerate(plot_chs):
    r, c = k // 2, k % 2
    ax[r, c].semilogy(freq_grid[1:], np.abs(H_exp_med[1:, ch]),
                        "r-", lw=1.0, label="experiment")
    ax[r, c].semilogy(freq_grid[1:], np.abs(H_s[1:, ch]),
                        "b-", lw=1.0, label="model (DE v2)")
    ax[r, c].set_title(f"{R.SENSOR_NAMES[ch]}  (ch{ch})", fontsize=9)
    ax[r, c].set_xlim(0.5, 25)
    ax[r, c].set_ylim(1e-3, 1e3)
    ax[r, c].grid(True, alpha=0.4)
    if k == 0:
        ax[r, c].legend(fontsize=8)
for r in range(3): ax[r, 0].set_ylabel("|H1|")
for c in range(2): ax[-1, c].set_xlabel("frequency [Hz]")
fig.suptitle(f"HBTA UDS Y-sweep — experiment vs FEM (DE-v2, "
              f"raw-SCI={raw_sci:.3f}, smooth-SCI={sm_sci:.3f})", fontsize=10)
fig.tight_layout()
fig.savefig(FIG_DIR / "frf_magnitude_v2.png", dpi=130)
plt.close(fig)
print(f"wrote {FIG_DIR/'frf_magnitude_v2.png'}")

# ─── Smoothed log-|H| envelopes (12 channels, 4x3 grid) ─────────────
sig = 4.0
H_exp_log = np.log10(np.maximum(np.abs(H_exp_med), 1e-30))
H_mod_log = np.log10(np.maximum(np.abs(H_s), 1e-30))
for k in range(12):
    H_exp_log[:, k] = ndi.gaussian_filter1d(H_exp_log[:, k], sig)
    H_mod_log[:, k] = ndi.gaussian_filter1d(H_mod_log[:, k], sig)
fig, axes = plt.subplots(4, 3, figsize=(11, 10), sharex=True, sharey=True)
for k, ax in enumerate(axes.flat):
    ax.plot(freq_grid, H_exp_log[:, k], "r-", lw=1.0, label="exp")
    ax.plot(freq_grid, H_mod_log[:, k], "b-", lw=1.0, label="model")
    ax.set_xlim(2.5, 25); ax.grid(True, alpha=0.4)
    ax.set_title(f"{R.SENSOR_NAMES[k]}", fontsize=9)
    if k == 0: ax.legend(fontsize=8)
for ax in axes[-1, :]: ax.set_xlabel("freq [Hz]")
for ax in axes[:, 0]: ax.set_ylabel("log10 |H|")
fig.suptitle(f"HBTA UDS Y-sweep — smoothed log|H| envelope per channel  "
             f"(DE-v2, smooth-SCI={sm_sci:.3f})", fontsize=10)
fig.tight_layout()
fig.savefig(FIG_DIR / "smooth_envelopes_v2.png", dpi=130)
plt.close(fig)
print(f"wrote {FIG_DIR/'smooth_envelopes_v2.png'}")

# ─── CFDAC matrices (raw + smoothed) ─────────────────────────────
fig, ax = plt.subplots(2, 2, figsize=(9, 8))
extent = [freq_grid[band].min(), freq_grid[band].max(),
          freq_grid[band].max(), freq_grid[band].min()]
ax[0, 0].imshow(C_exp_rw, extent=extent, cmap="viridis", vmin=0, vmax=1)
ax[0, 0].set_title("CFDAC raw — experiment", fontsize=9)
ax[0, 1].imshow(C_mod_rw, extent=extent, cmap="viridis", vmin=0, vmax=1)
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
fig.savefig(FIG_DIR / "cfdac_uds_v2.png", dpi=130)
plt.close(fig)
print(f"wrote {FIG_DIR/'cfdac_uds_v2.png'}")

# ─── Per-class median FRF vs model ───────────────────────────────
chunks = ROOT.parent / "output" / "chunks"
by_class = {}
for p in sorted(chunks.glob("chunk_*.h5")):
    with h5py.File(p, "r") as f:
        lab = f["labels/class_code"][:]
        src = f["labels/source_record"][:]
        frf = f["frf_H1"][:]
        for c in range(9):
            mask = (lab == c) & np.array([b"_Y_" in s for s in src])
            if not mask.any(): continue
            by_class.setdefault(c, []).append(frf[mask])
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
             "k--", lw=1.5, label="FEM (DE v2)")
ax.set_xlim(0.5, 25); ax.set_ylim(1e-3, 1e2)
ax.set_xlabel("frequency [Hz]")
ax.set_ylabel(f"|H1|  AG05")
ax.set_title("HBTA per-class median |H1| vs FEM DE-v2 — AG05 (Y-sweep)",
              fontsize=10)
ax.legend(ncol=2, fontsize=8, loc="lower left")
ax.grid(True, alpha=0.4)
fig.tight_layout()
fig.savefig(FIG_DIR / "per_class_vs_model_v2.png", dpi=130)
plt.close(fig)
print(f"wrote {FIG_DIR/'per_class_vs_model_v2.png'}")

# Save params
(ROOT / "best_params_de_v2.json").write_text(json.dumps({
    **BEST,
    "smooth_SCI": float(sm_sci),
    "raw_SCI": float(raw_sci),
    "gain": gain,
    "first_5_modes_Hz": [float(f) for f in freqs_modes[:5]],
    "n_joints": len(joints),
    "n_members": len(members),
}, indent=2))
print(f"wrote {ROOT/'best_params_de_v2.json'}")
print("done.")
