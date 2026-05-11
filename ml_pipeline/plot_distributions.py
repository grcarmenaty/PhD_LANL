"""Generate the two distribution plots the report is missing:

1. Per-indicator value histograms across all damage cases — 22 panels.
2. Per-(type, location) sample-count bar chart for both synth and exp.
"""
from pathlib import Path
import sys
sys.path.insert(0, '/home/user/PhD_LANL')

import h5py
import numpy as np
import matplotlib.pyplot as plt

from ml_pipeline.case_design import (
    TYPE_PRISTINE, TYPE_BOLT, TYPE_CRACK, TYPE_HOLE, TYPE_MASS,
)
from ml_pipeline.features import INDICATOR_NAMES
from ml_pipeline.train import load_labels


TYPE_NAMES = {TYPE_PRISTINE: 'Pristine', TYPE_BOLT: 'Bolt',
              TYPE_CRACK: 'Crack', TYPE_HOLE: 'Hole', TYPE_MASS: 'Mass'}
OUT = Path("/home/user/PhD_LANL/results/figures/dataset")
OUT.mkdir(parents=True, exist_ok=True)


def plot_indicator_distributions():
    with h5py.File("/home/user/PhD_LANL/dataset/features.h5", "r") as f:
        ind_syn = f["indicators"][:]
        tc_syn = f["labels/type_code"][:]
    with h5py.File("/home/user/PhD_LANL/dataset/experimental_features.h5", "r") as f:
        ind_exp = f["indicators"][:]
        tc_exp = f["type_code"][:]
    # damage subsets
    syn_d = ind_syn[tc_syn != TYPE_PRISTINE]
    exp_d = ind_exp[tc_exp != TYPE_PRISTINE]

    nrows, ncols = 6, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 16))
    axes = axes.ravel()
    for k, name in enumerate(INDICATOR_NAMES):
        ax = axes[k]
        s = syn_d[:, k]
        e = exp_d[:, k]
        # Robust clipping: 1-99 percentile of both
        lo = np.nanpercentile(np.concatenate([s, e]), 1)
        hi = np.nanpercentile(np.concatenate([s, e]), 99)
        bins = np.linspace(lo, hi, 50)
        ax.hist(s, bins=bins, density=True, alpha=0.55, color='C0',
                label=f'synth (n={len(s)})')
        ax.hist(e, bins=bins, density=True, alpha=0.55, color='C3',
                label=f'exp (n={len(e)})')
        ax.set_title(name, fontsize=10)
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.legend(fontsize=7, loc='best')
    for k in range(len(INDICATOR_NAMES), len(axes)):
        axes[k].axis('off')
    fig.suptitle("Per-indicator value distribution across damage cases — "
                 "synthetic vs experimental",
                 fontsize=12, y=0.99)
    fig.tight_layout()
    fig.savefig(OUT / "indicator_distributions.png", dpi=110,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/'indicator_distributions.png'}")


def plot_location_distributions():
    def per_cell(path: Path, label_group: bool):
        with h5py.File(path, "r") as f:
            g = f["labels"] if label_group else f
            tc = g["type_code"][:]
            sto = g["storey"][:]
            end = g["end"][:]
        rows = []
        for t in (TYPE_PRISTINE, TYPE_BOLT, TYPE_CRACK, TYPE_HOLE, TYPE_MASS):
            m = tc == t
            n_total = int(m.sum())
            if t == TYPE_PRISTINE:
                rows.append((f"Pristine", n_total))
                continue
            if t == TYPE_MASS:
                for plate, lbl in enumerate(("Base", "F1", "F2", "F3")):
                    rows.append((f"Mass-{lbl}", int((end[m] == plate).sum())))
                continue
            for s in range(3):
                for e in range(2):
                    rows.append((f"{TYPE_NAMES[t]}-S{s+1}{'BD' if e == 0 else 'AD'}",
                                 int(((sto[m] == s) & (end[m] == e)).sum())))
        return rows

    syn = per_cell(Path("/home/user/PhD_LANL/dataset/features.h5"), True)
    exp = per_cell(Path("/home/user/PhD_LANL/dataset/experimental_features.h5"), False)
    labels = [r[0] for r in syn]
    counts_syn = [r[1] for r in syn]
    counts_exp = [r[1] for r in exp]

    x = np.arange(len(labels))
    width = 0.4
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width/2, counts_syn, width=width, color='C0',
           label="synthetic (10 000 total)")
    ax.bar(x + width/2, counts_exp, width=width, color='C3',
           label="experimental (2 638 total)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha='right', fontsize=8)
    ax.set_ylabel("# samples")
    ax.set_title("Per-(type, location) sample count — synthetic vs experimental")
    ax.grid(axis='y', linestyle=':', alpha=0.5)
    ax.legend()
    # Annotate counts above bars
    for xi, v_s in zip(x, counts_syn):
        ax.text(xi - width/2, v_s + 8, str(v_s), ha='center', fontsize=6.5, color='C0')
    for xi, v_e in zip(x, counts_exp):
        ax.text(xi + width/2, v_e + 8, str(v_e), ha='center', fontsize=6.5, color='C3')
    fig.tight_layout()
    fig.savefig(OUT / "location_distribution.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/'location_distribution.png'}")


if __name__ == "__main__":
    plot_indicator_distributions()
    plot_location_distributions()
