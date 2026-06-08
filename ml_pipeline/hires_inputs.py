"""Visualise EVERY input representation the zoo consumes, on real samples.

For a synthetic and an experimental example (pristine + a high-severity damage
case) it renders: the 81-d modal vector, the 22 named damage indicators, the
frf_mag / frf_realimag / timeseries sequences, and all CFDAC channels (real /
imag / mag / phase) — exactly as the models receive them (same normalisation).
A separate panel shows the CFDAC-magnitude fingerprint of each damage class.

Writes figures to results/figures/hires/inputs_*.png and a small
results_hires/inputs.json (which sample indices were shown).

Run: python ml_pipeline/hires_inputs.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
from ml_pipeline.hires_zoo import cfdac_torch, CFDAC_FEATURES
from ml_pipeline.hires_tab import _timeseries_from_frf
from ml_pipeline.features import modal_features, indicator_features, INDICATOR_NAMES

FIG = _REPO/"results"/"figures"/"hires"; FIG.mkdir(parents=True, exist_ok=True)
SYN = _REPO/"dataset"/"features_hires.h5"
EXP = _REPO/"dataset"/"experimental_features_hires.h5"
TYPE_NAMES = ["pristine", "bolt", "crack", "hole", "mass"]


def pick(h5):
    """Return dict: freqs, ref(complex N,9), and for pristine + each damage a sample idx
    plus its complex FRF. Damaged pick = highest-severity of that type."""
    import h5py
    with h5py.File(h5, "r") as f:
        tc = f["type_code"][:].astype(int); sev = f["severity"][:].astype(float)
        freqs = f["freqs"][:].astype(np.float32)
        ref = f["reference/frf_complex"][:].astype(np.complex64)        # (N,9)
        idx = {}
        idx[0] = int(np.where(tc == 0)[0][0])
        for c in range(1, 5):
            w = np.where(tc == c)[0]
            idx[c] = int(w[np.argmax(sev[w])]) if len(w) else None
        H = {}
        for c, i in idx.items():
            if i is None: continue
            H[c] = (f["frf_real"][i] + 1j * f["frf_imag"][i]).astype(np.complex64)   # (N,9)
        sevv = {c: (float(sev[idx[c]]) if idx[c] is not None else None) for c in idx}
    return {"freqs": freqs, "ref": ref, "idx": idx, "H": H, "sev": sevv}


def cfdac_channels(ref, H, channels=("real", "imag", "mag", "phase")):
    """ref:(N,9) complex, H:(N,9) complex -> dict ch-> (N,N) normalised, as the model sees it."""
    r = torch.from_numpy(ref); h = torch.from_numpy(H)[None]
    out = cfdac_torch(r, h, channels, normalize=True)[0]    # (C,N,N)
    return {ch: out[i].numpy() for i, ch in enumerate(channels)}


def main():
    S = pick(SYN); E = pick(EXP)
    fr = S["freqs"]

    # ---------- (1) tabular features: modal (81) + indicators (22) ----------
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.6))
    # modal for synth pristine vs synth bolt
    import h5py
    with h5py.File(SYN, "r") as f:
        mag_p = np.abs(f["frf_real"][S["idx"][0]] + 1j*f["frf_imag"][S["idx"][0]]).astype(np.float32)
        mag_b = np.abs(f["frf_real"][S["idx"][1]] + 1j*f["frf_imag"][S["idx"][1]]).astype(np.float32)
    mv_p = modal_features(mag_p, fr); mv_b = modal_features(mag_b, fr)
    x = np.arange(81)
    ax[0].plot(x, mv_p, ".-", ms=3, lw=.7, label="pristine", color="#7f7f7f")
    ax[0].plot(x, mv_b, ".-", ms=3, lw=.7, label=f"bolt (sev {S['sev'][1]:.0f})", color="#1f77b4")
    for k in range(1, 9):
        ax[0].axvline(9*k-0.5, color="k", alpha=.12, lw=.7)
    ax[0].set_title("(a) `modal` — 81-d vector (9 features × 9 channels)\n"
                    "[pk1_f,pk1_a, pk2_f,pk2_a, pk3_f,pk3_a, mean_logA, std_logA, band_E] per channel",
                    fontweight="bold", fontsize=9)
    ax[0].set_xlabel("feature index (vertical lines = channel boundaries)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    # indicators (synth bolt vs synth crack) — computed vs the pristine ref
    iv_b = indicator_features(S["H"][1], S["ref"]); iv_c = indicator_features(S["H"][2], S["ref"])
    xi = np.arange(22); w = 0.4
    ax[1].bar(xi - w/2, iv_b, w, label=f"bolt (sev {S['sev'][1]:.0f})", color="#1f77b4")
    ax[1].bar(xi + w/2, iv_c, w, label=f"crack (sev {S['sev'][2]:.0f})", color="#d62728")
    ax[1].set_xticks(xi); ax[1].set_xticklabels(INDICATOR_NAMES, rotation=90, fontsize=6)
    ax[1].set_title("(b) `indicators` — 22 pymodal damage indicators vs the pristine reference",
                    fontweight="bold", fontsize=9)
    ax[1].legend(fontsize=8); ax[1].grid(axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig(FIG/"inputs_tabular.png", dpi=130); plt.close(fig)

    # ---------- (2) sequence features: frf_mag / frf_realimag / timeseries ----------
    def znorm(a):  # per-channel z over axis 0 (freq)
        return (a - a.mean(0)) / (a.std(0) + 1e-6)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    ch = 0  # show drive-point channel
    for tag, src, col in [("synth pristine", S["H"][0], "#7f7f7f"), ("synth bolt", S["H"][1], "#1f77b4"),
                          ("exp bolt", E["H"].get(1), "#d62728")]:
        if src is None: continue
        lm = np.log10(np.clip(np.abs(src[:, ch]), 1e-12, None)); lm = (lm-lm.mean())/(lm.std()+1e-6)
        ax[0].plot(fr, lm, lw=1, label=tag, color=col, alpha=.85)
    ax[0].set_title("(a) `frf_mag` — log|H(f)|, z-normed (drive-point ch.)\nfed as 9 channels × 1601",
                    fontweight="bold", fontsize=9)
    ax[0].set_xlabel("frequency (Hz)"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3); ax[0].set_xlim(0, 100)
    # realimag
    H = S["H"][1][:, ch]
    re = (H.real-H.real.mean())/(H.real.std()+1e-6); im = (H.imag-H.imag.mean())/(H.imag.std()+1e-6)
    ax[1].plot(fr, re, lw=1, label="Re H", color="#1f77b4"); ax[1].plot(fr, im, lw=1, label="Im H", color="#ff7f0e")
    ax[1].set_title("(b) `frf_realimag` — Re/Im of H(f), z-normed\nfed as 18 channels × 1601 (synth bolt)",
                    fontweight="bold", fontsize=9)
    ax[1].set_xlabel("frequency (Hz)"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3); ax[1].set_xlim(0, 100)
    # timeseries (reconstructed)
    ts_p = _timeseries_from_frf(S["H"][0][None].real, S["H"][0][None].imag, fr)[0, ch]
    ts_b = _timeseries_from_frf(S["H"][1][None].real, S["H"][1][None].imag, fr)[0, ch]
    t = np.arange(len(ts_b))/256.0
    ax[2].plot(t, ts_p, lw=.6, label="pristine", color="#7f7f7f", alpha=.8)
    ax[2].plot(t, ts_b, lw=.6, label="bolt", color="#1f77b4", alpha=.8)
    ax[2].set_title("(c) `timeseries` — band-limited response\nirfft(H·chirp), z-normed; 9 × 4096 (synth)",
                    fontweight="bold", fontsize=9)
    ax[2].set_xlabel("time (s)"); ax[2].legend(fontsize=7); ax[2].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(FIG/"inputs_sequences.png", dpi=130); plt.close(fig)

    # ---------- (3) CFDAC channel variants for one damaged sample ----------
    cc = cfdac_channels(S["ref"], S["H"][1])     # synth bolt
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.4))
    cmaps = {"real": "RdBu_r", "imag": "RdBu_r", "mag": "viridis", "phase": "twilight"}
    for a, chn in zip(ax, ["real", "imag", "mag", "phase"]):
        M = cc[chn]; v = np.percentile(np.abs(M), 99)
        im = a.imshow(M, cmap=cmaps[chn], aspect="auto",
                      vmin=(-v if chn in ("real", "imag", "mag") else -1),
                      vmax=(v if chn in ("real", "imag", "mag") else 1),
                      extent=[0, 100, 100, 0])
        a.set_title(f"cfdac_{chn}", fontweight="bold", fontsize=10)
        a.set_xlabel("freq j (Hz)"); a.set_ylabel("freq i (Hz)")
        plt.colorbar(im, ax=a, fraction=.046)
    fig.suptitle("CFDAC channels (synth bolt vs pristine reference) — the 4 base maps; "
                 "`cfdac_realimag`=Re+Im, `cfdac_magphase`=mag+phase, `cfdac_all`=all four stacked",
                 fontweight="bold", fontsize=11, y=1.03)
    plt.tight_layout(); plt.savefig(FIG/"inputs_cfdac_variants.png", dpi=130); plt.close(fig)

    # ---------- (4) CFDAC-mag fingerprint per damage class (synth & exp) ----------
    fig, axs = plt.subplots(2, 5, figsize=(18, 7.2))
    for row, (D, dom) in enumerate([(S, "synth"), (E, "exp")]):
        for c in range(5):
            a = axs[row, c]
            if D["idx"].get(c) is None or c not in D["H"]:
                a.axis("off"); continue
            M = cfdac_channels(D["ref"], D["H"][c], channels=("mag",))["mag"]
            v = np.percentile(np.abs(M), 99)
            a.imshow(M, cmap="viridis", aspect="auto", vmin=-v, vmax=v, extent=[0, 100, 100, 0])
            sv = D["sev"].get(c)
            a.set_title(f"{dom} · {TYPE_NAMES[c]}" + (f" (sev {sv:.0f})" if sv else ""), fontsize=9, fontweight="bold")
            if c == 0: a.set_ylabel("freq i (Hz)", fontsize=8)
            if row == 1: a.set_xlabel("freq j (Hz)", fontsize=8)
    fig.suptitle("CFDAC magnitude fingerprint per damage class — top: synthetic, bottom: experimental\n"
                 "(pristine is near-diagonal; damage spreads energy off-diagonal — and the synth/exp "
                 "patterns differ, which is the domain gap the image models must cross)",
                 fontweight="bold", fontsize=12, y=1.02)
    plt.tight_layout(); plt.savefig(FIG/"inputs_cfdac_classes.png", dpi=130); plt.close(fig)

    (_REPO/"results_hires"/"inputs.json").write_text(json.dumps(
        {"synth_idx": S["idx"], "synth_sev": S["sev"], "exp_idx": E["idx"], "exp_sev": E["sev"]}, indent=1))
    print("wrote inputs_{tabular,sequences,cfdac_variants,cfdac_classes}.png + results_hires/inputs.json")
    print("synth severities:", S["sev"], "\nexp severities:", E["sev"])


if __name__ == "__main__":
    main()
