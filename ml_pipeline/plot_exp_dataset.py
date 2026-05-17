"""Experimental-dataset exploration figures.

Renders five PNGs into ``results/figures/dataset_exp/`` describing the
680-case ``experimental_features_balanced.h5`` set:

  class_counts.png    4-panel bar chart - binary, type, col_location,
                       mass_location class counts.
  severity_hist.png   Severity distribution per damage type.
  signals_ts.png      One real time series per damage type (all 9
                       sensors overlaid).
  signals_frf.png     One real FRF magnitude per damage type.
  signals_cfdac.png   One real |CFDAC| matrix per damage type.

These figures are inherited by both REPORT.md and REPORT_noisy_mixed.md
via the new section 2.7 inserted by ``integrate_report.py``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml_pipeline.case_design import TYPE_NAMES  # noqa: E402
from ml_pipeline.tasks import build_targets  # noqa: E402
from ml_pipeline.train import load_labels  # noqa: E402
from ml_pipeline.plots_advanced import (  # noqa: E402
    _class_labels,
    _safe_savefig,
)


def _picks(type_code: np.ndarray) -> dict[int, int]:
    rng = np.random.default_rng(0)
    out: dict[int, int] = {}
    for k in range(5):
        idx = np.where(type_code == k)[0]
        if len(idx):
            out[k] = int(rng.choice(idx))
    return out


def plot_class_counts(exp_path: Path, out_dir: Path) -> None:
    L = load_labels(exp_path)
    tasks = build_targets(L["type_code"], L["storey"], L["end"], L["severity"])
    cls_tasks = [(tn, tn_data) for tn, tn_data in tasks.items()
                       if tn_data[2] == "cls"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (tn, (mask, y, _)) in zip(axes.flat, cls_tasks):
        labels = _class_labels(tn)
        counts = np.bincount(y.astype(int), minlength=len(labels))
        bars = ax.bar(labels, counts, color="#1f77b4")
        for b, c in zip(bars, counts):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                       f"{int(c)}", ha="center", va="bottom", fontsize=9)
        ax.set_title(f"{tn}  (n_pool={int(mask.sum())})")
        ax.tick_params(axis="x", rotation=20)
        ax.set_ylabel("samples")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.suptitle("Experimental balanced set - per-task class counts (680 cases total)")
    fig.tight_layout()
    _safe_savefig(fig, out_dir / "class_counts.png")


def plot_severity_hist(exp_path: Path, out_dir: Path) -> None:
    L = load_labels(exp_path)
    tc = L["type_code"]
    sev = L["severity"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for tc_val in (1, 2, 3, 4):
        mask = tc == tc_val
        if mask.sum() == 0:
            continue
        ax.hist(sev[mask], bins=25, alpha=0.55,
                  label=f"{TYPE_NAMES[tc_val]}  (n={int(mask.sum())})")
    ax.set_xlabel("severity (raw, units depend on type)")
    ax.set_ylabel("count")
    ax.set_title("Experimental severity distribution per damage type")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    _safe_savefig(fig, out_dir / "severity_hist.png")


def plot_signal_examples(exp_path: Path, out_dir: Path) -> None:
    with h5py.File(exp_path, "r") as f:
        tc = f["type_code"][:]
        picks = _picks(tc)
        freqs = f["freqs"][:]
        time_axis = np.arange(1024) / 256.0

        fig_ts, axts = plt.subplots(5, 1, figsize=(9, 11), sharex=True)
        fig_fr, axfr = plt.subplots(5, 1, figsize=(9, 11), sharex=True)
        fig_cf, axcf = plt.subplots(1, 5, figsize=(20, 4))
        for row, (k, i) in enumerate(picks.items()):
            ts = f["timeseries"][i]
            for ch in range(ts.shape[1]):
                axts[row].plot(time_axis, ts[:, ch], lw=0.5, alpha=0.7)
            axts[row].set_title(f"EXP {TYPE_NAMES[k]} - sample id {i}")
            axts[row].grid(linestyle=":", alpha=0.5)

            mag = f["frf_mag"][i]
            for ch in range(mag.shape[1]):
                axfr[row].semilogy(freqs, mag[:, ch], lw=0.7, alpha=0.7)
            axfr[row].set_title(f"EXP |H(f)| - {TYPE_NAMES[k]}  (id {i})")
            axfr[row].grid(linestyle=":", alpha=0.5)

            if "cfdac_real" in f:
                cre = f["cfdac_real"][i]; cim = f["cfdac_imag"][i]
                cmag = np.sqrt(cre ** 2 + cim ** 2)
                axcf[row].imshow(cmag, origin="lower", cmap="viridis",
                                       vmin=0, vmax=1)
                axcf[row].set_title(f"EXP |CFDAC| - {TYPE_NAMES[k]}")
                axcf[row].set_xticks([]); axcf[row].set_yticks([])
        axts[-1].set_xlabel("time [s]")
        axfr[-1].set_xlabel("frequency [Hz]")
        fig_ts.suptitle("Experimental acceleration time series  "
                              "(1 sample / class, 9 sensors overlaid)")
        fig_fr.suptitle("Experimental FRF magnitude  "
                              "(1 sample / class, 9 sensors overlaid)")
        fig_cf.suptitle("Experimental |CFDAC| vs the pristine reference "
                              "(1 sample / class)")
        _safe_savefig(fig_ts, out_dir / "signals_ts.png")
        _safe_savefig(fig_fr, out_dir / "signals_frf.png")
        _safe_savefig(fig_cf, out_dir / "signals_cfdac.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp", type=Path,
                      default=_REPO / "dataset"
                              / "experimental_features_balanced.h5")
    p.add_argument("--out", type=Path,
                      default=_REPO / "results" / "figures" / "dataset_exp")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    print("class counts...", flush=True)
    plot_class_counts(args.exp, args.out)
    print("severity hist...", flush=True)
    plot_severity_hist(args.exp, args.out)
    print("signal examples...", flush=True)
    plot_signal_examples(args.exp, args.out)
    n = sum(1 for _ in args.out.rglob("*.png"))
    print(f"wrote {n} PNGs to {args.out}", flush=True)


if __name__ == "__main__":
    main()
