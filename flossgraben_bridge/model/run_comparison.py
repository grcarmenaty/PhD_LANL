"""Run the beam FEM for the 3 Flossgraben scenarios, compare against
the experimental dataset, and save figures + SCI scores.

Outputs to flossgraben_bridge/model/figures/:
  - modes_table.txt                — first 8 modes per scenario
  - time_waveform_<ch>.png         — synthetic vs experimental waveform
  - frf_magnitude.png              — per-sensor magnitude (3 scenarios)
  - cfdac_<scenario>.png           — CFDAC matrices (exp vs synth)
  - sci_scoreboard.txt             — SCI per scenario
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import beam_fem as bf


HERE = Path(__file__).resolve().parent
FIG  = HERE / "figures"
FIG.mkdir(exist_ok=True)
EXP_H5 = HERE.parent / "output" / "flossgraben_collection.h5"

FS = 256.0
N_T = 1024
N_F = 513
FREQ_GRID = np.arange(N_F) * (FS / N_T)         # 0–128 Hz, Δf=0.25 Hz
BAND_LO, BAND_HI = 0.5, 25.0                     # SCI / CFDAC band (model.md §11.2)
BAND = (FREQ_GRID >= BAND_LO) & (FREQ_GRID <= BAND_HI)


# ── Experimental data loading ────────────────────────────────────────
def load_experimental():
    """Return dict {scenario: spectra_arr} where spectra_arr shape is
    (n_per_class, 513, 9, 1) complex64, sampled from the pymodal
    collection.

    Layout: /measurements/<name>/data and /<name>/label, with `data`
    shape (513, 9, 1) complex64. Labels: 0=reference, 1=field3, 2=field4.
    """
    n_per_class = 500
    name_by_class = {0: "reference", 1: "field3", 2: "field4"}
    out = {v: [] for v in name_by_class.values()}
    with h5py.File(EXP_H5, "r") as f:
        meas = f["measurements"]
        for name in meas:
            if name == "_axes":
                continue
            label = int(round(float(meas[name]["label"][()])))
            sc = name_by_class.get(label)
            if sc is None or len(out[sc]) >= n_per_class:
                continue
            out[sc].append(np.asarray(meas[name]["data"]))
            if all(len(v) >= n_per_class for v in out.values()):
                break
    return {k: np.stack(v) for k, v in out.items()}


def load_experimental_signals(scenario: str, n: int = 4) -> np.ndarray:
    """Load a few time-domain experimental windows from the chunked HDF5
    dataset (companion to the pymodal frf file). Returns (n, 1024, 9)."""
    chunks_dir = HERE.parent / "output" / "chunks"
    if not chunks_dir.exists():
        return None
    chunk_paths = sorted(chunks_dir.glob("chunk_*.h5"))
    cls_code = {"reference": 0, "field3": 4, "field4": 4}[scenario]
    cls_end  = {"reference": -1, "field3": 3, "field4": 4}[scenario]
    out = []
    for p in chunk_paths:
        with h5py.File(p, "r") as f:
            tc = f["labels/type_code"][:]
            end = f["labels/end"][:]
            mask = (tc == cls_code) & (end == cls_end)
            if mask.any():
                idxs = np.flatnonzero(mask)[: max(0, n - len(out))]
                out.extend(np.asarray(f["signals"][idxs]))
        if len(out) >= n:
            break
    return np.stack(out[:n]) if out else None


# ── CFDAC + SCI (from generate_docs_images.py:196-205) ───────────────
def cfdac(H_band: np.ndarray) -> np.ndarray:
    """H_band shape (n_freq, n_ch_total_flat) → (n_freq, n_freq)."""
    inner = H_band.conj() @ H_band.T
    d = np.real(np.diag(inner)).copy()
    d[d < 1e-30] = 1e-30
    return (np.abs(inner) ** 2) / np.outer(d, d)


def sci(C1: np.ndarray, C2: np.ndarray) -> float:
    a = C1.ravel() - C1.mean()
    b = C2.ravel() - C2.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float((a @ b) ** 2 / d ** 2) if d > 0 else 0.0


# ── Driver ───────────────────────────────────────────────────────────
def main():
    print("Loading experimental spectra…")
    exp_specs = load_experimental()
    print({k: v.shape for k, v in exp_specs.items()})

    print("Building beam FEM and running 3 scenarios…")
    results = {}
    for sc in ("reference", "field3", "field4"):
        model = bf.build(sc)
        freqs_m, phi_full = bf.solve_modes(model, n_modes=60)
        S_aa = bf.auto_spectrum(model, FREQ_GRID, f_cut_hz=8.0, n_modes=60)
        signals_syn, specs_syn = bf.synthesise_windows(
            S_aa, n_windows=500, n_t=N_T, fs_hz=FS, rng_seed=42)
        results[sc] = dict(modes=freqs_m, S_aa=S_aa,
                           signals=signals_syn, specs=specs_syn)
        print(f"  {sc}: first 8 modes = "
              f"{np.array2string(freqs_m[:8], precision=3, separator=', ')}")

    # ── Modes table ────────────────────────────────────────────────
    with open(FIG / "modes_table.txt", "w") as out:
        out.write("First 12 modal frequencies [Hz]\n")
        out.write("---------------------------------\n")
        out.write(f"{'mode':>4}  {'reference':>10}  {'field3':>10}  {'field4':>10}\n")
        for r in range(12):
            out.write(f"{r+1:>4}  "
                      f"{results['reference']['modes'][r]:>10.3f}  "
                      f"{results['field3']['modes'][r]:>10.3f}  "
                      f"{results['field4']['modes'][r]:>10.3f}\n")

    # ── Time-waveform comparison ───────────────────────────────────
    plt.rcParams.update({"font.size": 9, "figure.dpi": 110})
    fig, axes = plt.subplots(3, 3, figsize=(14, 7.5), sharex=True)
    sensor_pick = ["ch43", "ch27", "ch51"]   # span 1, span 3, span 4
    sensor_idx  = [bf.SENSOR_ORDER.index(c) for c in sensor_pick]
    t = np.arange(N_T) / FS
    for col, sc in enumerate(("reference", "field3", "field4")):
        exp_sig = load_experimental_signals(sc, n=1)
        for row, (cname, ch) in enumerate(zip(sensor_pick, sensor_idx)):
            ax = axes[row, col]
            syn = results[sc]["signals"][0, :, ch]
            # Normalise both to unit-std for visual comparison (scale is
            # arbitrary in this uncalibrated run — we have no force)
            syn_n = syn / (syn.std() + 1e-30)
            ax.plot(t, syn_n, lw=0.7, color="C0", label="model")
            if exp_sig is not None:
                exp_n = exp_sig[0, :, ch] / (exp_sig[0, :, ch].std() + 1e-30)
                ax.plot(t, exp_n, lw=0.5, color="C3", alpha=0.7, label="experiment")
            if row == 0:
                ax.set_title(sc, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{cname}\n(σ-normalised a)")
            if row == 2:
                ax.set_xlabel("time [s]")
            ax.grid(alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(loc="upper right", fontsize=7)
    fig.suptitle("Time-waveform comparison (one window, σ-normalised)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "time_waveform.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── FRF magnitude comparison (auto-spectrum, log scale) ───────
    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(3, 3, figsize=(14, 8), sharex=True, sharey="row")
    for col, sc in enumerate(("reference", "field3", "field4")):
        # Median experimental |spec| across windows
        exp_mag = np.abs(exp_specs[sc][:, :, :, 0]).astype(np.float64)
        exp_med = np.median(exp_mag, axis=0)            # (513, 9)
        syn_med = np.sqrt(results[sc]["S_aa"]).astype(np.float64)  # (513, 9)
        # Scale model to match experimental magnitude in [1, 4] Hz peak
        peak_band = (FREQ_GRID >= 1.0) & (FREQ_GRID <= 4.0)
        gain = exp_med[peak_band].max() / (syn_med[peak_band].max() + 1e-30)
        syn_scaled = syn_med * gain
        for row, (cname, ch) in enumerate(zip(sensor_pick, sensor_idx)):
            ax = axes[row, col]
            ax.semilogy(FREQ_GRID, exp_med[:, ch], color="C3", lw=0.9,
                        label="experiment (median)")
            ax.semilogy(FREQ_GRID, syn_scaled[:, ch], color="C0", lw=1.0,
                        label="model")
            ax.set_xlim(0, 25)
            if row == 0:
                ax.set_title(sc, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{cname}\n|spectrum|")
            if row == 2:
                ax.set_xlabel("frequency [Hz]")
            ax.grid(alpha=0.3, which="both")
            if row == 0 and col == 0:
                ax.legend(loc="upper right", fontsize=7)
            # Overlay modal-frequency vertical lines
            for fm in results[sc]["modes"][:12]:
                if fm <= 25:
                    ax.axvline(fm, color="0.7", lw=0.5, alpha=0.5, zorder=0)
    fig.suptitle("Auto-spectrum magnitude (0–25 Hz). Grey ticks = model modes.",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "frf_magnitude.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ── CFDAC + SCI ───────────────────────────────────────────────
    sci_scores = {}
    for sc in ("reference", "field3", "field4"):
        exp_med_spec = np.median(np.abs(exp_specs[sc][:, :, :, 0]).astype(np.float64),
                                  axis=0)         # (513, 9)
        syn_med_spec = np.sqrt(results[sc]["S_aa"]).astype(np.float64)  # (513, 9)
        # Restrict to band
        H_exp = exp_med_spec[BAND]                                    # (n_b, 9)
        H_syn = syn_med_spec[BAND]                                    # (n_b, 9)
        # Treat as complex by reinjecting unit phase from experiment so CFDAC
        # operates on the same |H|² structure (phase is arbitrary in OMA).
        C_exp = cfdac(H_exp.astype(np.complex128))
        C_syn = cfdac(H_syn.astype(np.complex128))
        s = sci(C_exp, C_syn)
        sci_scores[sc] = s

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
        n_b = H_exp.shape[0]
        extent = [BAND_LO, BAND_HI, BAND_HI, BAND_LO]
        for ax, C, ttl in [(axes[0], C_exp, "experiment"),
                           (axes[1], C_syn, "model")]:
            im = ax.imshow(C, extent=extent, cmap="viridis", vmin=0, vmax=1,
                            aspect="auto", interpolation="nearest")
            ax.set_xlabel("frequency [Hz]")
            ax.set_ylabel("frequency [Hz]")
            ax.set_title(ttl)
        fig.suptitle(f"CFDAC — scenario: {sc}  |  SCI = {s:.3f}",
                     fontweight="bold")
        fig.colorbar(im, ax=axes, shrink=0.85, label="CFDAC")
        fig.savefig(FIG / f"cfdac_{sc}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    with open(FIG / "sci_scoreboard.txt", "w") as out:
        out.write(f"SCI band: {BAND_LO}-{BAND_HI} Hz\n")
        out.write("-" * 32 + "\n")
        for sc, s in sci_scores.items():
            out.write(f"{sc:>10s} : SCI = {s:.4f}\n")
        out.write(f"{'mean':>10s} : SCI = {np.mean(list(sci_scores.values())):.4f}\n")
    print("\nSCI scores:")
    for sc, s in sci_scores.items():
        print(f"  {sc:>10s} : {s:.4f}")

    # Dump structured summary for embedding
    summary = {
        "modes_first8": {k: results[k]["modes"][:8].tolist()
                          for k in results},
        "sci": sci_scores,
        "band": [BAND_LO, BAND_HI],
        "model_params": {
            "L_total": bf.L_TOTAL, "N_span": bf.N_SPAN,
            "E_deck": bf.E_DECK, "I_yy": bf.I_YY, "A_deck": bf.A_DECK,
            "rho_c": bf.RHO_C, "alpha": bf.ALPHA_RAYL,
            "beta": bf.BETA_RAYL, "f_cut_traffic": 8.0,
        },
    }
    (FIG / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nFigures and summary written to {FIG}")


if __name__ == "__main__":
    main()
