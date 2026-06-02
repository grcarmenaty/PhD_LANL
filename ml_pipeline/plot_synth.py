"""Plots for the synthetic-domain (pre-transfer) report.

Outputs → results/figures/synth/:
  fig1_bespoke_by_task.png   per-task bars of bespoke synth test (model/feature)
  fig2_val_vs_test.png       synth val vs synth test (generalisation/overfit)
  fig3_runtime_vs_test.png   training runtime vs synth test
  fig4_gap.png               synth test vs experimental (the sim-to-real gap)
  fig5_vision_synth.png      vision synth test by task × backbone (as available)
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parent.parent
OUT = _REPO / "results" / "figures" / "synth"
OUT.mkdir(parents=True, exist_ok=True)

TM = json.load(open(_REPO / "results" / "training_metrics.json"))
CELLS = json.load(open(_REPO / "results" / "cells_v1_v2_v2a.json"))["cells"]

TASK_ORDER = ["binary", "type", "col_location", "mass_location", "severity"]
RANDOM = {"binary": 0.5, "type": 0.20, "col_location": 1/9,
          "mass_location": 1/3, "severity": 0.0}


def _label(r):
    return f"{r['model']}/{r['feature']}"


# ---------------------------------------------------------------- fig1
def fig1_bespoke_by_task():
    by_task = defaultdict(list)
    for r in TM:
        by_task[r["task"]].append(r)
    tasks = [t for t in TASK_ORDER if t in by_task]
    fig, axes = plt.subplots(len(tasks), 1,
                              figsize=(9, 2.6 * len(tasks)))
    if len(tasks) == 1:
        axes = [axes]
    for ax, task in zip(axes, tasks):
        rows = sorted(by_task[task], key=lambda r: r["metric_test"])
        labels = [_label(r) for r in rows]
        vals = [r["metric_test"] for r in rows]
        y = np.arange(len(rows))
        colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(rows)))
        ax.barh(y, vals, color=colors, edgecolor="black", lw=0.4)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
        mname = rows[0]["metric_name"]
        ax.axvline(RANDOM[task], ls=":", color="red", alpha=0.7,
                    label=f"chance={RANDOM[task]:.2f}")
        ax.set_title(f"`{task}` — synthetic test {mname}", fontweight="bold",
                      fontsize=10)
        ax.set_xlabel(mname)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(axis="x", alpha=0.3)
        if mname == "accuracy":
            ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(OUT / "fig1_bespoke_by_task.png", dpi=130)
    plt.close(fig)
    print("wrote fig1_bespoke_by_task.png")


# ---------------------------------------------------------------- fig2
def fig2_val_vs_test():
    fig, ax = plt.subplots(figsize=(7, 6))
    tasks = sorted({r["task"] for r in TM})
    cmap = {t: c for t, c in zip(tasks, plt.cm.tab10(np.linspace(0, 1, len(tasks))))}
    for r in TM:
        # only compare like-for-like (accuracy tasks together; severity R2 separate)
        ax.scatter(r["metric_val"], r["metric_test"], color=cmap[r["task"]],
                    s=40, edgecolor="black", lw=0.4, alpha=0.85)
    lims = [-0.1, 1.05]
    ax.plot(lims, lims, ls="--", color="grey", alpha=0.7, label="val = test")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("synthetic validation score")
    ax.set_ylabel("synthetic test score")
    ax.set_title("Generalisation within the synthetic domain\n"
                  "(points on the diagonal = no val→test overfit)",
                  fontweight="bold", fontsize=11)
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=cmap[t],
                           label=t, markeredgecolor="black") for t in tasks]
    handles.append(plt.Line2D([0], [0], ls="--", color="grey", label="val=test"))
    ax.legend(handles=handles, fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "fig2_val_vs_test.png", dpi=130)
    plt.close(fig)
    print("wrote fig2_val_vs_test.png")


# ---------------------------------------------------------------- fig3
def fig3_runtime_vs_test():
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    tasks = sorted({r["task"] for r in TM})
    cmap = {t: c for t, c in zip(tasks, plt.cm.tab10(np.linspace(0, 1, len(tasks))))}
    for r in TM:
        ax.scatter(max(r["runtime_s"], 0.5), r["metric_test"],
                    color=cmap[r["task"]], s=40, edgecolor="black",
                    lw=0.4, alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlabel("training runtime (s, log scale)")
    ax.set_ylabel("synthetic test score")
    ax.set_title("Cost vs in-domain accuracy (bespoke models)",
                  fontweight="bold", fontsize=11)
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=cmap[t],
                           label=t, markeredgecolor="black") for t in tasks]
    ax.legend(handles=handles, fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(OUT / "fig3_runtime_vs_test.png", dpi=130)
    plt.close(fig)
    print("wrote fig3_runtime_vs_test.png")


# ---------------------------------------------------------------- fig4
def fig4_gap():
    """Synth test vs experimental (same metric) for joinable cells."""
    pairs = []  # (label, task, synth, exp, metric)
    for r in TM:
        key = f"{r['task']}/{r['model']}/{r['feature']}"
        if key not in CELLS:
            continue
        ev = CELLS[key]["metrics"].get("v1")
        if not ev:
            continue
        em = ev["mean"]
        if r["task"] == "severity":
            synth = r["metric_test"]; exp = em.get("r2"); metric = "R²"
        else:
            synth = r["metric_test"]; exp = em.get("accuracy"); metric = "acc"
        if exp is None:
            continue
        pairs.append((f"{r['task']}: {r['model']}/{r['feature']}",
                      r["task"], synth, exp, metric))
    if not pairs:
        print("fig4: no joinable cells"); return
    pairs.sort(key=lambda p: p[2] - p[3], reverse=True)
    labels = [p[0] for p in pairs]
    synth = np.array([p[2] for p in pairs])
    exp = np.array([p[3] for p in pairs])
    y = np.arange(len(pairs))
    fig, ax = plt.subplots(figsize=(10, max(5, 0.32 * len(pairs) + 1)))
    ax.barh(y - 0.2, synth, 0.4, color="#2ca02c", edgecolor="black",
            lw=0.4, label="synthetic test")
    ax.barh(y + 0.2, exp, 0.4, color="#d62728", edgecolor="black",
            lw=0.4, label="experimental (zero-shot)")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("score (accuracy for classification; R² for severity)")
    ax.set_title("The sim-to-real gap, cell by cell (v1)\n"
                  "green = learned on synthetic · red = survives to real data",
                  fontweight="bold", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "fig4_gap.png", dpi=130)
    plt.close(fig)
    print(f"wrote fig4_gap.png ({len(pairs)} joinable cells)")


# ---------------------------------------------------------------- fig5
def fig5_vision_synth():
    vis_root = _REPO / "results_vision"
    rows = []
    if vis_root.exists():
        for d in vis_root.iterdir():
            if not d.is_dir() or "_seed" not in d.name:
                continue
            variant = d.name.split("_seed")[0]
            pc = d / "per_case_vision"
            if not pc.exists():
                continue
            for jf in pc.glob("*.json"):
                try:
                    meta = json.load(open(jf)).get("meta", {})
                except Exception:
                    continue
                if meta.get("synth_test") is None:
                    continue
                rows.append((meta["task"], meta["backbone"], variant,
                             meta["synth_test"]))
    if not rows:
        print("fig5: no vision cells yet — skipping"); return
    # average over seeds+features per (task, backbone, variant)
    agg = defaultdict(list)
    for task, bk, variant, st in rows:
        agg[(task, bk, variant)].append(st)
    tasks = sorted({k[0] for k in agg})
    backbones = ["convnext_tiny", "resnet50", "vit_b_16"]
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(tasks) + 2), 5))
    x = np.arange(len(tasks))
    width = 0.25
    bcol = {"convnext_tiny": "#1f77b4", "resnet50": "#ff7f0e",
            "vit_b_16": "#9467bd"}
    for i, bk in enumerate(backbones):
        vals = []
        for t in tasks:
            allv = [v for var in ("v1", "v2", "v2a")
                    for v in agg.get((t, bk, var), [])]
            vals.append(np.mean(allv) if allv else np.nan)
        ax.bar(x + (i - 1) * width, vals, width, color=bcol[bk],
               edgecolor="black", lw=0.4, label=bk)
    ax.set_xticks(x); ax.set_xticklabels(tasks, rotation=30, ha="right",
                                          fontsize=8)
    ax.set_ylabel("synthetic test score (acc / R²)")
    ax.set_title("Vision backbones — synthetic test (avg over variants+seeds "
                  "done so far)", fontweight="bold", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "fig5_vision_synth.png", dpi=130)
    plt.close(fig)
    print(f"wrote fig5_vision_synth.png ({len(rows)} vision cells)")


if __name__ == "__main__":
    fig1_bespoke_by_task()
    fig2_val_vs_test()
    fig3_runtime_vs_test()
    fig4_gap()
    fig5_vision_synth()
    print(f"\nfigures in {OUT}")
