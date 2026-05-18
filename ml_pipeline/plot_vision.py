"""Diagnostic plots for the vision-model sweep (synth-only training,
zero-shot cross-domain eval).

Reads results/vision_eval.json and the cnn2d baseline rows from
results/p1_1/experimental_full_evaluation.json, and produces:

  results/figures/vision/vision_vs_cnn2d_bar.png
  results/figures/vision/synth_vs_exp_scatter.png
  results/figures/vision/per_feature_grid.png          (if multiple features)
  results/figures/vision/runtime_vs_accuracy.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
OUT_DIR = _REPO / "results" / "figures" / "vision"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BACKBONE_PARAMS_M = {
    "resnet50":        25.6,
    "efficientnet_b0":  5.3,
    "convnext_tiny":   28.6,
    "swin_t":          28.3,
    "vit_b_16":        86.6,
}
BACKBONE_ORDER = list(BACKBONE_PARAMS_M.keys())


def _load_vision():
    return json.loads((_REPO / "results" / "vision_eval.json").read_text())


def _cnn2d_baseline(task: str, feature: str | None = None):
    """Pull the best cnn2d zero-shot baseline from p1_1 for this task.

    If ``feature`` is None, return the best cnn2d cell on this task
    across every CFDAC variant (the right comparison point for a new
    backbone trained on a single variant).  If ``feature`` is given,
    return that specific cell (None if it wasn't trained).
    """
    p = _REPO / "results" / "p1_1" / "experimental_full_evaluation.json"
    if not p.exists():
        return None, None
    rows = json.loads(p.read_text())
    best = None
    for r in rows:
        if r["task"] != task or r["model"] != "cnn2d":
            continue
        if feature is not None and r["feature"] != feature:
            continue
        if best is None or r["value"] > best[0]:
            best = (r["value"], r["feature"])
    return best if best is not None else (None, None)


def plot_vision_vs_cnn2d_bar():
    rows = _load_vision()
    # Group by (task, feature) → list of (backbone, exp_value, synth_value)
    by_cell = {}
    for r in rows:
        key = (r["task"], r["feature"])
        by_cell.setdefault(key, []).append(r)
    n_cells = len(by_cell)
    fig, axes = plt.subplots(1, n_cells, figsize=(5 * n_cells, 5),
                                  sharey=True)
    if n_cells == 1:
        axes = [axes]
    for ax, (key, cell_rows) in zip(axes, sorted(by_cell.items())):
        task, feature = key
        # Order by parameter count
        cell_rows_sorted = sorted(
            cell_rows, key=lambda r: BACKBONE_PARAMS_M.get(r["backbone"], 0))
        labels = [r["backbone"].replace("_", "\n") for r in cell_rows_sorted]
        exp_vals = [r["exp_value"] for r in cell_rows_sorted]
        synth_vals = [r["synth_test"] for r in cell_rows_sorted]
        x = np.arange(len(labels))
        w = 0.36
        ax.bar(x - w/2, synth_vals, w, label="synth test",
                  color="#aaaaaa", edgecolor="black", linewidth=0.4)
        ax.bar(x + w/2, exp_vals, w, label="exp zero-shot",
                  color="#d9534f", edgecolor="black", linewidth=0.4)
        # Reference 1: cnn2d on the SAME feature (if that cell exists)
        cnn2d_same, _ = _cnn2d_baseline(task, feature)
        if cnn2d_same is not None:
            ax.axhline(cnn2d_same, color="#4a90d9", linestyle="--",
                          linewidth=2,
                          label=f"cnn2d/{feature} ({cnn2d_same:.2f})")
        # Reference 2: best cnn2d on any CFDAC variant
        cnn2d_best_val, cnn2d_best_feat = _cnn2d_baseline(task)
        if cnn2d_best_val is not None:
            ax.axhline(cnn2d_best_val, color="#1f77b4", linestyle=":",
                          linewidth=2,
                          label=f"best cnn2d/{cnn2d_best_feat} ({cnn2d_best_val:.2f})")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(f"{task} / {feature}")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.axhline(0, color="black", linewidth=0.5)
        for xi, (sv, ev) in enumerate(zip(synth_vals, exp_vals)):
            ax.text(xi - w/2, sv + 0.015, f"{sv:.2f}", ha="center",
                      fontsize=7)
            ax.text(xi + w/2, ev + 0.015, f"{ev:.2f}", ha="center",
                      fontsize=7)
        if ax is axes[0]:
            ax.set_ylabel("accuracy / R²")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("ImageNet-pretrained vision backbones on CFDAC "
                  "(synth-only training, zero-shot cross-domain)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT_DIR / "vision_vs_cnn2d_bar.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def plot_synth_vs_exp_scatter():
    rows = _load_vision()
    fig, ax = plt.subplots(figsize=(7, 6.5))
    colors = plt.cm.tab10.colors
    feats = sorted({r["feature"] for r in rows})
    markers = ["o", "s", "^", "D", "v"]
    feat2marker = {f: markers[i % len(markers)] for i, f in enumerate(feats)}
    for i, name in enumerate(BACKBONE_ORDER):
        for r in rows:
            if r["backbone"] != name: continue
            ax.scatter(r["synth_test"], r["exp_value"],
                          color=colors[i], marker=feat2marker[r["feature"]],
                          s=140, alpha=0.85, edgecolor="black", linewidth=0.6,
                          label=f"{name} ({BACKBONE_PARAMS_M[name]:.0f}M, "
                                  f"{r['feature']})")
            ax.annotate(f"{name[:4]}/{r['feature'][6:]}",
                              (r["synth_test"], r["exp_value"]),
                              fontsize=7, alpha=0.8,
                              xytext=(5, 5), textcoords="offset points")
    lims = [0, 1]
    ax.plot(lims, lims, "--", color="gray", linewidth=1, label="y=x (perfect transfer)")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("synth test metric")
    ax.set_ylabel("exp zero-shot metric")
    ax.set_title("Synth → exp transfer for each (backbone, feature)\n"
                  "above diagonal = transfers ≥ synth; below = generalisation gap")
    ax.grid(linestyle=":", alpha=0.4)
    # De-duplicate the legend
    handles, labels = ax.get_legend_handles_labels()
    seen = set(); uh, ul = [], []
    for h, l in zip(handles, labels):
        bb = l.split()[0]
        if bb in seen: continue
        seen.add(bb); uh.append(h); ul.append(l.split(" (")[0])
    ax.legend(uh, ul, fontsize=8, loc="upper left")
    fig.tight_layout()
    out = OUT_DIR / "synth_vs_exp_scatter.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def plot_runtime_vs_accuracy():
    rows = _load_vision()
    fig, ax = plt.subplots(figsize=(7, 5))
    feats = sorted({r["feature"] for r in rows})
    markers = ["o", "s", "^", "D"]
    feat2marker = {f: markers[i % len(markers)] for i, f in enumerate(feats)}
    colors = plt.cm.tab10.colors
    for i, name in enumerate(BACKBONE_ORDER):
        for r in rows:
            if r["backbone"] != name: continue
            ax.scatter(r["runtime_s"], r["exp_value"],
                          color=colors[i], marker=feat2marker[r["feature"]],
                          s=160, alpha=0.85, edgecolor="black", linewidth=0.6,
                          label=name)
            ax.annotate(f"{name[:4]}", (r["runtime_s"], r["exp_value"]),
                              fontsize=8, alpha=0.8,
                              xytext=(5, 0), textcoords="offset points",
                              va="center")
    ax.set_xlabel("training runtime per cell (sec)")
    ax.set_ylabel("exp zero-shot metric")
    ax.set_title("Pareto: training cost vs cross-domain accuracy")
    ax.set_xscale("log")
    ax.grid(linestyle=":", alpha=0.4)
    # Dedup legend
    handles, labels = ax.get_legend_handles_labels()
    seen = set(); uh, ul = [], []
    for h, l in zip(handles, labels):
        if l in seen: continue
        seen.add(l); uh.append(h); ul.append(l)
    ax.legend(uh, ul, fontsize=8, loc="lower right")
    fig.tight_layout()
    out = OUT_DIR / "runtime_vs_accuracy.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    plot_vision_vs_cnn2d_bar()
    plot_synth_vs_exp_scatter()
    plot_runtime_vs_accuracy()
    print(f"\nfigures in {OUT_DIR}")


if __name__ == "__main__":
    main()
