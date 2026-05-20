"""Generate side-by-side synth vs real feature-example plots, one per
feature type, used to ground the per-task documentation.

Output: ``results/figures/feature_examples/<feature>.png``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml_pipeline.case_design import TYPE_NAMES  # noqa: E402
from ml_pipeline.evaluate import build_experimental_features, primary_op  # noqa: E402
from ml_pipeline.train import load_labels, load_feature  # noqa: E402
from ml_pipeline.features import INDICATOR_NAMES  # noqa: E402


CLASSES = [0, 1, 2, 3, 4]
COLORS  = ["#5a5a5a", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def _pick_synth(labels, type_code):
    rows = np.where(labels["type_code"] == type_code)[0]
    return int(rows[0]) if len(rows) else None


def _pick_real(exp_labels, type_code):
    rows = np.where(exp_labels["type_code"] == type_code)[0]
    return int(rows[0]) if len(rows) else None


def plot_modal(features_path, exp, out_path):
    labels = load_labels(features_path)
    syn = {k: _pick_synth(labels, k) for k in CLASSES}
    real = {k: _pick_real(exp["labels"], k) for k in CLASSES}
    with h5py.File(features_path, "r") as f:
        syn_modal = f["modal"][:]
    real_modal = exp["modal"]

    fig, axes = plt.subplots(5, 2, figsize=(12, 11), sharex=True)
    for row, k in enumerate(CLASSES):
        for col, (mat, idx, tag) in enumerate(
            (("synthetic", syn[k], "synth"), ("experimental", real[k], "real"))
        ):
            ax = axes[row, col]
            if idx is None:
                ax.set_visible(False); continue
            data = (syn_modal if col == 0 else real_modal)[idx]
            ax.bar(np.arange(len(data)), data, color=COLORS[k])
            ax.set_title(f"{TYPE_NAMES[k]} — {mat} sample {idx}", fontsize=10)
            ax.set_xlabel("modal feature index (81-d)")
            ax.grid(axis="y", linestyle=":", alpha=0.5)
            ax.set_xlim(-0.5, 81)
            for ch in range(9):
                ax.axvline(ch * 9 - 0.5, color="k", lw=0.3, alpha=0.4)
    fig.suptitle("Modal-peak features — synth (left) vs experimental (right)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_indicators(features_path, exp, out_path):
    labels = load_labels(features_path)
    syn = {k: _pick_synth(labels, k) for k in CLASSES}
    real = {k: _pick_real(exp["labels"], k) for k in CLASSES}
    with h5py.File(features_path, "r") as f:
        syn_ind = f["indicators"][:]
    real_ind = exp["indicators"]

    fig, axes = plt.subplots(5, 2, figsize=(13, 12), sharex=True)
    for row, k in enumerate(CLASSES):
        for col, (mat, idx) in enumerate(
            (("synthetic", syn[k]), ("experimental", real[k]))
        ):
            ax = axes[row, col]
            if idx is None:
                ax.set_visible(False); continue
            data = (syn_ind if col == 0 else real_ind)[idx]
            ax.bar(range(len(data)), data, color=COLORS[k])
            ax.set_title(f"{TYPE_NAMES[k]} — {mat} sample {idx}", fontsize=10)
            ax.set_xticks(range(len(INDICATOR_NAMES)))
            ax.set_xticklabels(INDICATOR_NAMES, rotation=70, fontsize=7)
            ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.suptitle("pymodal damage indicators (22-d) — synth vs experimental")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_frf_mag(features_path, exp, out_path):
    labels = load_labels(features_path)
    syn = {k: _pick_synth(labels, k) for k in CLASSES}
    real = {k: _pick_real(exp["labels"], k) for k in CLASSES}
    with h5py.File(features_path, "r") as f:
        freqs = f["freqs"][:]
        syn_mag = f["frf_mag"]
    real_mag = exp["frf_mag"]

    fig, axes = plt.subplots(5, 2, figsize=(13, 14), sharex=True)
    with h5py.File(features_path, "r") as f:
        for row, k in enumerate(CLASSES):
            for col, (mat, idx, source) in enumerate(
                (("synthetic", syn[k], f["frf_mag"]),
                 ("experimental", real[k], real_mag))
            ):
                ax = axes[row, col]
                if idx is None:
                    ax.set_visible(False); continue
                data = source[idx]
                for ch in range(data.shape[1]):
                    ax.semilogy(freqs, data[:, ch], lw=0.6, alpha=0.7)
                ax.set_title(f"{TYPE_NAMES[k]} — {mat} sample {idx}", fontsize=10)
                ax.grid(linestyle=":", alpha=0.5)
                if row == 4: ax.set_xlabel("frequency [Hz]")
                if col == 0: ax.set_ylabel("|H(f)|")
    fig.suptitle("FRF magnitudes |H(f)| — synth vs experimental")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_cfdac(features_path, exp, out_path):
    labels = load_labels(features_path)
    syn = {k: _pick_synth(labels, k) for k in CLASSES}
    real = {k: _pick_real(exp["labels"], k) for k in CLASSES}
    real_cfdac = exp.get("cfdac")

    fig, axes = plt.subplots(5, 2, figsize=(10, 17))
    with h5py.File(features_path, "r") as f:
        syn_re = f["cfdac_real"]; syn_im = f["cfdac_imag"]
        for row, k in enumerate(CLASSES):
            for col, (mat, idx) in enumerate(
                (("synthetic", syn[k]), ("experimental", real[k]))
            ):
                ax = axes[row, col]
                if idx is None:
                    ax.set_visible(False); continue
                if col == 0:
                    m = np.sqrt(syn_re[idx] ** 2 + syn_im[idx] ** 2)
                else:
                    if real_cfdac is None:
                        ax.set_visible(False); continue
                    m = np.sqrt(real_cfdac[idx, 0] ** 2 + real_cfdac[idx, 1] ** 2)
                im = ax.imshow(m, cmap="viridis", origin="lower", vmin=0, vmax=1)
                ax.set_title(f"{TYPE_NAMES[k]} — {mat} sample {idx}", fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("|CFDAC| 128 × 128 — synth vs experimental")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_timeseries(features_path, exp, out_path):
    labels = load_labels(features_path)
    syn = {k: _pick_synth(labels, k) for k in CLASSES}
    real = {k: _pick_real(exp["labels"], k) for k in CLASSES}
    with h5py.File(features_path, "r") as f:
        t = np.arange(1024) / 256.0
        syn_ts = f["timeseries"]
    real_ts = exp["timeseries"]

    fig, axes = plt.subplots(5, 2, figsize=(13, 13), sharex=True)
    with h5py.File(features_path, "r") as f:
        for row, k in enumerate(CLASSES):
            for col, (mat, idx, source) in enumerate(
                (("synthetic", syn[k], f["timeseries"]),
                 ("experimental", real[k], real_ts))
            ):
                ax = axes[row, col]
                if idx is None:
                    ax.set_visible(False); continue
                data = source[idx]
                for ch in range(data.shape[1]):
                    ax.plot(t, data[:, ch], lw=0.5, alpha=0.6)
                ax.set_title(f"{TYPE_NAMES[k]} — {mat} sample {idx}", fontsize=10)
                ax.grid(linestyle=":", alpha=0.5)
                if row == 4: ax.set_xlabel("time [s]")
                if col == 0: ax.set_ylabel("acceleration [m/s²]")
    fig.suptitle("9-channel acceleration time series — synth vs experimental")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out_dir = _REPO / "results" / "figures" / "feature_examples"
    features_path = _REPO / "dataset" / "features.h5"
    print("building experimental features …")
    exp = build_experimental_features(_REPO / "median_frfs.h5", features_path)

    print("modal …");      plot_modal(features_path, exp, out_dir / "modal.png")
    print("indicators …"); plot_indicators(features_path, exp, out_dir / "indicators.png")
    print("frf_mag …");    plot_frf_mag(features_path, exp, out_dir / "frf_mag.png")
    print("timeseries …"); plot_timeseries(features_path, exp, out_dir / "timeseries.png")
    print("cfdac …");      plot_cfdac(features_path, exp, out_dir / "cfdac.png")
    print(f"done → {out_dir}")


if __name__ == "__main__":
    main()
