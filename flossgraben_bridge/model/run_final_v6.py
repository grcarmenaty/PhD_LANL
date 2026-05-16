"""Generate final figures using v6 best parameters.

Produces all of model.md §15's figures (time waveform, FRF magnitude,
CFDAC matrices, modes table, SCI scoreboard) under the v6-best
calibration. Both smoothed-SCI (matched to OMA data character) and
raw-SCI are reported.
"""
from __future__ import annotations
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import beam_fem_v3 as bf3

FIG = HERE / "figures_v6"
FIG.mkdir(exist_ok=True)
EXP_H5 = HERE.parent / "output" / "flossgraben_collection.h5"

FS, N_T, N_F = 256.0, 1024, 513
FREQ_GRID = np.arange(N_F) * (FS / N_T)
BAND_LO, BAND_HI = 0.5, 25.0
BAND = (FREQ_GRID >= BAND_LO) & (FREQ_GRID <= BAND_HI)
SIGMA_BINS = 5.0


def smooth(spec_mag):
    return gaussian_filter1d(np.log10(spec_mag + 1e-30), sigma=SIGMA_BINS, axis=0)


def cfdac(H):
    inner = H.conj() @ H.T
    d = np.real(np.diag(inner)).copy(); d[d < 1e-30] = 1e-30
    return (np.abs(inner) ** 2) / np.outer(d, d)


def sci_metric(C1, C2):
    a = C1.ravel() - C1.mean(); b = C2.ravel() - C2.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float((a @ b) ** 2 / d ** 2) if d > 0 else 0.0


def load_experimental():
    name_by_class = {0: "reference", 1: "field3", 2: "field4"}
    spec_lists = {v: [] for v in name_by_class.values()}
    with h5py.File(EXP_H5, "r") as f:
        meas = f["measurements"]
        for name in meas:
            if name == "_axes": continue
            label = int(round(float(meas[name]["label"][()])))
            sc = name_by_class.get(label)
            if sc is None or len(spec_lists[sc]) >= 500: continue
            spec_lists[sc].append(np.asarray(meas[name]["data"]))
            if all(len(v) >= 500 for v in spec_lists.values()): break
    out = {}
    for sc, lst in spec_lists.items():
        arr = np.stack(lst)
        out[sc] = dict(specs=arr,
                       median_mag=np.median(np.abs(arr[..., 0]).astype(np.float64), axis=0))
    return out


def load_experimental_signals(scenario, n=1):
    chunks_dir = HERE.parent / "output" / "chunks"
    if not chunks_dir.exists():
        return None
    cls_code = {"reference": 0, "field3": 4, "field4": 4}[scenario]
    cls_end  = {"reference": -1, "field3": 3, "field4": 4}[scenario]
    out = []
    for p in sorted(chunks_dir.glob("chunk_*.h5")):
        with h5py.File(p, "r") as f:
            tc = f["labels/type_code"][:]
            end = f["labels/end"][:]
            mask = (tc == cls_code) & (end == cls_end)
            if mask.any():
                idxs = np.flatnonzero(mask)[: max(0, n - len(out))]
                out.extend(np.asarray(f["signals"][idxs]))
        if len(out) >= n: break
    return np.stack(out[:n]) if out else None


def main():
    print("Loading v6 best params and experimental data…")
    v6 = json.loads((HERE / "best_params_v6.json").read_text())
    bp = v6["best_params"]
    params = dict(
        E_bend=bp["E_GPa"] * 1e9,
        I_yy=bp["I_yy"], mu=23500.0,
        GJ=10.0 ** bp["log_GJ"], rho_Ip=10.0 ** bp["log_rho_Ip"],
        z_lo=bp["z_lo"], z_mid=bp["z_mid"], z_hi=bp["z_hi"],
        y_off=bp["y_off"], f_cut=bp["f_cut"],
        k_pier_v=10.0 ** bp["log_k_pier_v"],
        k_pier_rot=10.0 ** bp["log_k_pier_rot"],
        k_pier_tors=10.0 ** bp["log_k_pier_tors"],
        dx_sensor=bp["dx_sensor"],
    )
    print(f"params: E={bp['E_GPa']:.2f} GPa  I={bp['I_yy']:.2f} m^4")
    print(f"        log10(GJ)={bp['log_GJ']:.2f}  log10(ρIp)={bp['log_rho_Ip']:.2f}")
    print(f"        ζ_band=[{bp['z_lo']:.3f}, {bp['z_mid']:.3f}, {bp['z_hi']:.3f}]")
    print(f"        y_off={bp['y_off']:.2f} m²  f_cut={bp['f_cut']:.2f} Hz")
    print(f"        pier-v=1e{bp['log_k_pier_v']:.2f}  pier-rot=1e{bp['log_k_pier_rot']:.2f}  "
          f"pier-tors=1e{bp['log_k_pier_tors']:.2f}")
    print(f"        Δx_sensor={bp['dx_sensor']:.2f} m")

    exp = load_experimental()
    plt.rcParams.update({"font.size": 9, "figure.dpi": 110})

    # ── Build model spectra (3 scenarios) ──────────────────────────
    results = {}
    for sc in ("reference", "field3", "field4"):
        Y, f1, fb, ft = bf3.auto_spectrum_v3(sc, params, FREQ_GRID, BAND)
        results[sc] = dict(Y=Y, f_bend=fb, f_tors=ft, f1=f1)
    print(f"\nModel f_1: ref={results['reference']['f1']:.3f}  "
          f"f3={results['field3']['f1']:.3f}  f4={results['field4']['f1']:.3f}")

    # ── Modes table ────────────────────────────────────────────────
    with open(FIG / "modes_table.txt", "w") as out:
        out.write("First 12 bending modes [Hz]\n" + "-"*42 + "\n")
        out.write(f"{'mode':>4}  {'reference':>10}  {'field3':>10}  {'field4':>10}\n")
        for r in range(12):
            out.write(f"{r+1:>4}  "
                      f"{results['reference']['f_bend'][r]:>10.3f}  "
                      f"{results['field3']['f_bend'][r]:>10.3f}  "
                      f"{results['field4']['f_bend'][r]:>10.3f}\n")
        out.write("\nFirst 8 torsion modes [Hz]\n" + "-"*42 + "\n")
        out.write(f"{'mode':>4}  {'reference':>10}  {'field3':>10}  {'field4':>10}\n")
        for r in range(8):
            out.write(f"{r+1:>4}  "
                      f"{results['reference']['f_tors'][r]:>10.3f}  "
                      f"{results['field3']['f_tors'][r]:>10.3f}  "
                      f"{results['field4']['f_tors'][r]:>10.3f}\n")

    # ── Time waveform ──────────────────────────────────────────────
    rng = np.random.default_rng(42)
    sensor_pick = ["ch43", "ch27", "ch51"]
    sensor_idx = [bf3.SENSOR_ORDER.index(c) for c in sensor_pick]
    t = np.arange(N_T) / FS
    fig, axes = plt.subplots(3, 3, figsize=(14, 7.5), sharex=True)
    for col, sc in enumerate(("reference", "field3", "field4")):
        # Build full-spectrum from band-restricted Y by padding (used for irfft)
        Y_band = results[sc]["Y"]
        n_bins = BAND.sum()
        # Random-phase realization
        phases = rng.uniform(-np.pi, np.pi, size=(n_bins, 9))
        phases[0, :] = 0.0; phases[-1, :] = 0.0
        Y_complex = Y_band * np.exp(1j * phases)
        full = np.zeros((N_F, 9), dtype=np.complex128)
        full[BAND] = Y_complex
        sig = np.fft.irfft(full, n=N_T, axis=0)
        exp_sig = load_experimental_signals(sc, n=1)
        for row, (cname, ch) in enumerate(zip(sensor_pick, sensor_idx)):
            ax = axes[row, col]
            syn = sig[:, ch]
            syn_n = syn / (syn.std() + 1e-30)
            ax.plot(t, syn_n, lw=0.7, color="C0", label="model")
            if exp_sig is not None:
                en = exp_sig[0, :, ch] / (exp_sig[0, :, ch].std() + 1e-30)
                ax.plot(t, en, lw=0.5, color="C3", alpha=0.7, label="experiment")
            if row == 0: ax.set_title(sc, fontweight="bold")
            if col == 0: ax.set_ylabel(f"{cname}\n(σ-normalised a)")
            if row == 2: ax.set_xlabel("time [s]")
            ax.grid(alpha=0.3)
            if row == 0 and col == 0: ax.legend(loc="upper right", fontsize=7)
    fig.suptitle("Time-waveform comparison — v6 calibration (σ-normalised)",
                  fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "time_waveform_v6.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── Auto-spectrum magnitude ────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(14, 8), sharex=True, sharey="row")
    for col, sc in enumerate(("reference", "field3", "field4")):
        exp_med = exp[sc]["median_mag"]
        syn = np.zeros((N_F, 9))
        syn[BAND] = results[sc]["Y"]
        peak_band = (FREQ_GRID >= 1.0) & (FREQ_GRID <= 4.0)
        gain = exp_med[peak_band].max() / (syn[peak_band].max() + 1e-30)
        syn_scaled = syn * gain
        for row, (cname, ch) in enumerate(zip(sensor_pick, sensor_idx)):
            ax = axes[row, col]
            ax.semilogy(FREQ_GRID, exp_med[:, ch], color="C3", lw=0.9, label="experiment")
            ax.semilogy(FREQ_GRID, syn_scaled[:, ch], color="C0", lw=1.0, label="model (v6)")
            ax.set_xlim(0, 25)
            if row == 0: ax.set_title(sc, fontweight="bold")
            if col == 0: ax.set_ylabel(f"{cname}\n|spectrum|")
            if row == 2: ax.set_xlabel("frequency [Hz]")
            ax.grid(alpha=0.3, which="both")
            if row == 0 and col == 0: ax.legend(loc="upper right", fontsize=7)
            for fm in results[sc]["f_bend"][:12]:
                if fm <= 25:
                    ax.axvline(fm, color="0.7", lw=0.4, alpha=0.5, zorder=0)
    fig.suptitle("Auto-spectrum magnitude (v6) — model vs experiment, "
                  "grey ticks = model bending modes",
                  fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "frf_magnitude_v6.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── CFDAC (raw and smoothed) ───────────────────────────────────
    sci_results = {}
    for sc in ("reference", "field3", "field4"):
        Y = results[sc]["Y"]
        exp_band = exp[sc]["median_mag"][BAND]
        # Raw
        C_exp_raw = cfdac(exp_band.astype(np.complex128))
        C_syn_raw = cfdac(Y.astype(np.complex128))
        s_raw = sci_metric(C_exp_raw, C_syn_raw)
        # Smoothed
        E_s = smooth(exp_band); E_s -= E_s.mean(axis=0, keepdims=True)
        Y_s = smooth(Y); Y_s -= Y_s.mean(axis=0, keepdims=True)
        C_exp_sm = cfdac(E_s.astype(np.complex128))
        C_syn_sm = cfdac(Y_s.astype(np.complex128))
        s_sm = sci_metric(C_exp_sm, C_syn_sm)
        sci_results[sc] = dict(raw=s_raw, smooth=s_sm)

        # Save figure: 2 rows × 2 cols (raw exp/model + smoothed exp/model)
        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        extent = [BAND_LO, BAND_HI, BAND_HI, BAND_LO]
        for (axR, axC, mat, ttl) in [
            (axes[0, 0], "raw",    C_exp_raw, f"experiment (raw |H|)"),
            (axes[0, 1], "raw",    C_syn_raw, f"model (raw |H|)"),
            (axes[1, 0], "smooth", C_exp_sm,  f"experiment (smoothed log|H|, σ={SIGMA_BINS:.0f})"),
            (axes[1, 1], "smooth", C_syn_sm,  f"model (smoothed log|H|, σ={SIGMA_BINS:.0f})"),
        ]:
            im = axR.imshow(mat, extent=extent, cmap="viridis", vmin=0, vmax=1,
                             aspect="auto", interpolation="nearest")
            axR.set_xlabel("frequency [Hz]"); axR.set_ylabel("frequency [Hz]")
            axR.set_title(ttl)
        fig.suptitle(
            f"CFDAC — scenario: {sc}  |  raw-SCI = {s_raw:.3f}   "
            f"smooth-SCI = {s_sm:.3f}",
            fontweight="bold")
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="CFDAC")
        fig.savefig(FIG / f"cfdac_{sc}_v6.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    # ── Scoreboard ─────────────────────────────────────────────────
    with open(FIG / "sci_scoreboard_v6.txt", "w") as out:
        out.write(f"v6 calibration — σ={SIGMA_BINS} bins ({SIGMA_BINS*0.25:.2f} Hz) smoothing\n")
        out.write("-"*68 + "\n")
        out.write(f"{'scenario':>12}  {'raw-SCI':>10}  {'smooth-SCI':>12}\n")
        out.write("-"*68 + "\n")
        for sc, s in sci_results.items():
            out.write(f"{sc:>12s}  {s['raw']:>10.4f}  {s['smooth']:>12.4f}\n")
        out.write("-"*68 + "\n")
        m_raw = np.mean([s["raw"] for s in sci_results.values()])
        m_sm  = np.mean([s["smooth"] for s in sci_results.values()])
        mn_sm = np.min([s["smooth"] for s in sci_results.values()])
        out.write(f"{'mean':>12s}  {m_raw:>10.4f}  {m_sm:>12.4f}\n")
        out.write(f"{'min':>12s}  {'-':>10s}  {mn_sm:>12.4f}\n")

    print("\nFinal scoreboard:")
    for sc, s in sci_results.items():
        print(f"  {sc:>10s}  raw={s['raw']:.4f}  smooth={s['smooth']:.4f}")
    print(f"  {'mean':>10s}  raw={np.mean([s['raw'] for s in sci_results.values()]):.4f}  "
          f"smooth={np.mean([s['smooth'] for s in sci_results.values()]):.4f}")
    print(f"\nWrote figures to {FIG}/")


if __name__ == "__main__":
    main()
