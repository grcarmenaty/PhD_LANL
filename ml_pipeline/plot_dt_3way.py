"""Generate plots for the DT-stratified 3-way verdict report.

Outputs to results/figures/dt_3way/:
  fig1_severity_distribution.png — per-damage-type severity histogram +
                                    stiffness-reduction overlay (why DT
                                    works for bolt but not crack/hole)
  fig2_dt_curves_grid.png         — macroF1 vs DT stiffness reduction,
                                    grid of 6 panels (one per binary task),
                                    3 lines per panel (v1 / v2 / v2a)
  fig3_tier_bars.png              — per-task tier (all / med+ / severe)
                                    macroF1 bars, 3 variants side-by-side,
                                    flags the floor cells from preregistered
                                    v2a criteria
  fig4_feature_axis.png           — per-damage-type axis sweep: is_bolt vs
                                    %loose, is_hole vs Ø, is_crack vs depth,
                                    3 lines (v1 / v2 / v2a)
  fig5_severity_mae.png           — severity regression MAE vs bolt
                                    threshold, 3 lines
  fig6_vision_vs_bespoke.png      — type task, vision backbones vs bespoke
                                    cells under per-tier filter (v1 only)
  fig7_failure_modes.png          — binary + is_pristine flat/dropping
                                    curves showing the synth-to-real
                                    detection bottleneck
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml_pipeline.case_design import (
    TYPE_PRISTINE, TYPE_BOLT, TYPE_CRACK, TYPE_HOLE, TYPE_MASS)
from ml_pipeline.dt_sweep import stiffness_reduction, _load_exp

OUT = _REPO / "results" / "figures" / "dt_3way"
OUT.mkdir(parents=True, exist_ok=True)

VARIANT_COLORS = {"v1": "#1f77b4", "v2": "#d62728", "v2a": "#2ca02c"}
TYPE_NAMES = {TYPE_PRISTINE: "pristine", TYPE_BOLT: "bolt",
              TYPE_CRACK: "crack", TYPE_HOLE: "hole", TYPE_MASS: "mass"}


def load_compare():
    return json.load(open(_REPO / "results" / "dt_compare_v1_v2_v2a.json"))


def load_feature():
    return json.load(open(_REPO / "results" / "dt_feature_sweep.json"))


# ─── Fig 1 ──────────────────────────────────────────────────────────────
def fig1_severity_distribution():
    tc, sev, sto, end = _load_exp(_REPO / "dataset" / "experimental_features.h5")
    sr = stiffness_reduction(tc, sev)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    # Left: raw severity per type (in native units)
    ax = axes[0]
    types = [TYPE_BOLT, TYPE_CRACK, TYPE_HOLE, TYPE_MASS]
    units = {TYPE_BOLT: "% loose", TYPE_CRACK: "mm",
             TYPE_HOLE: "mm", TYPE_MASS: "kg"}
    colors = ["#e07b00", "#7f3f98", "#0099a8", "#888888"]
    for i, (t, c) in enumerate(zip(types, colors)):
        m = (tc == t)
        if not m.any(): continue
        s = sev[m]
        ax.hist(s, bins=np.arange(s.min(), s.max()+1.5, 1),
                alpha=0.7, color=c, edgecolor="black",
                label=f"{TYPE_NAMES[t]} (n={m.sum()})  units: {units[t]}")
    ax.set_xlabel("severity (per-type native units)")
    ax.set_ylabel("count")
    ax.set_title("(a) Raw damage severity by type\n(experimental test set, n=2638)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    # Right: stiffness reduction (DT axis)
    ax = axes[1]
    for t, c in zip(types[:3], colors[:3]):  # bolt/crack/hole only
        m = (tc == t)
        if not m.any(): continue
        s = sr[m]
        ax.hist(s, bins=np.linspace(0, 0.7, 40),
                alpha=0.7, color=c, edgecolor="black",
                label=f"{TYPE_NAMES[t]} (n={m.sum()})")
    ax.set_xlabel("fractional stiffness reduction")
    ax.set_ylabel("count")
    ax.set_title("(b) Stiffness reduction (the DT axis)\nmass excluded — inertia change, not stiffness")
    ax.set_xlim(0, 0.7)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    # Annotations
    for dt in (0.05, 0.20, 0.45):
        ax.axvline(dt, ls="--", color="gray", alpha=0.5, lw=0.8)
        ax.text(dt, ax.get_ylim()[1]*0.92, f"DT={dt:.2f}",
                rotation=90, fontsize=7, ha="right", va="top", color="gray")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_severity_distribution.png", dpi=130)
    plt.close(fig)
    print("wrote fig1_severity_distribution.png")


# ─── Fig 2 ──────────────────────────────────────────────────────────────
def fig2_dt_curves_grid():
    cmp = load_compare()
    bp = cmp["best_per_task"]
    dt_grid = cmp["dt_grid"]
    binary_tasks = ["binary", "is_bolt", "is_pristine", "is_crack", "is_hole", "is_mass"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharey=True)
    for ax, task in zip(axes.flat, binary_tasks):
        for v in ("v1", "v2", "v2a"):
            xs, ys, sds = [], [], []
            for dt in dt_grid:
                r = bp.get(task, {}).get(v, {}).get(f"{dt:.2f}")
                if r is None: continue
                if r.get("macro_f1_mean") is None: continue
                xs.append(dt); ys.append(r["macro_f1_mean"])
                sds.append(r.get("macro_f1_sd") or 0)
            if xs:
                xs, ys, sds = map(np.array, (xs, ys, sds))
                ax.plot(xs, ys, "o-", color=VARIANT_COLORS[v], label=v, lw=2, ms=5)
                ax.fill_between(xs, ys - sds, ys + sds,
                                 color=VARIANT_COLORS[v], alpha=0.15)
        ax.axhline(0.5, ls=":", color="black", alpha=0.4, lw=0.8)
        ax.set_title(task, fontsize=11, fontweight="bold")
        ax.set_xlabel("DT (min stiffness reduction)")
        ax.set_xlim(-0.02, 0.62)
        ax.set_ylim(0.3, 0.9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right" if task == "is_bolt" else "upper right")
    axes[0,0].set_ylabel("macro-F1 (best cell, 3-seed mean)")
    axes[1,0].set_ylabel("macro-F1 (best cell, 3-seed mean)")
    fig.suptitle("Best-cell-per-task macro-F1 vs DT stiffness reduction — v1 / v2 / v2a (3-seed mean ±sd)",
                  fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_dt_curves_grid.png", dpi=130)
    plt.close(fig)
    print("wrote fig2_dt_curves_grid.png")


# ─── Fig 3 ──────────────────────────────────────────────────────────────
def fig3_tier_bars():
    feat = load_feature()
    best = feat["best"]
    tasks = ["binary", "is_pristine", "type", "col_location",
             "is_crack", "is_hole", "is_mass"]
    # tier-axis tasks only (binary/is_pristine/type/col_location use tier;
    # is_X bolt/crack/hole/mass use per_type — we render those in fig4)
    tier_tasks = ["binary", "is_pristine", "type", "col_location", "mass_location"]
    tiers = ["all", "med+", "severe"]
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(tier_tasks))
    width = 0.25
    for i, v in enumerate(("v1", "v2", "v2a")):
        means = []
        sds = []
        for t in tier_tasks:
            row = best.get(t, {}).get(v, {})
            # Prefer tier_pos for binary/type/col_location/mass_location,
            # tier_neg for is_pristine
            sub = (row.get("tier_neg") if t == "is_pristine"
                   else row.get("tier_pos"))
            if not sub:
                means.append(np.nan); sds.append(0); continue
            r = sub.get("severe") or sub.get("all")
            means.append(r["macro_f1"] if r else np.nan)
            sds.append(0)
        ax.bar(x + i*width, means, width, yerr=sds,
                label=v, color=VARIANT_COLORS[v], edgecolor="black", lw=0.5)
    ax.set_xticks(x + width)
    ax.set_xticklabels(tier_tasks, rotation=15)
    ax.set_ylabel("macro-F1 (best cell @ severe tier)")
    ax.set_title("Per-task macroF1 under most-restrictive tier filter (severe)\n"
                  "Compares the three physics variants on their best-cell-per-task")
    ax.axhline(0.5, ls=":", color="black", alpha=0.4, lw=0.8,
                label="0.5 random (binary)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_tier_bars.png", dpi=130)
    plt.close(fig)
    print("wrote fig3_tier_bars.png")


# ─── Fig 4 ──────────────────────────────────────────────────────────────
def fig4_feature_axis():
    feat = load_feature()
    best = feat["best"]
    panels = [
        ("is_bolt",  "bolt",  "bolt % looseness",       [0, 11, 50, 85]),
        ("is_crack", "crack", "crack depth (mm)",       [0, 5, 8]),
        ("is_hole",  "hole",  "hole diameter (mm)",     [0, 4, 6]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (task, dname, xlabel, grid) in zip(axes, panels):
        for v in ("v1", "v2", "v2a"):
            sub = best.get(task, {}).get(v, {}).get("per_type", {}).get(dname, {})
            xs, ys = [], []
            for thr in grid:
                r = sub.get(str(thr))
                if r and r.get("macro_f1") is not None:
                    xs.append(thr); ys.append(r["macro_f1"])
            if xs:
                ax.plot(xs, ys, "o-", color=VARIANT_COLORS[v], lw=2,
                        ms=8, label=v)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("macro-F1 (best cell)")
        ax.set_title(task, fontweight="bold")
        ax.axhline(0.5, ls=":", color="black", alpha=0.4)
        ax.set_ylim(0.35, 0.9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("Per-damage-type feature-axis sweep — macroF1 climbs with severity",
                  fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_feature_axis.png", dpi=130)
    plt.close(fig)
    print("wrote fig4_feature_axis.png")


# ─── Fig 5 ──────────────────────────────────────────────────────────────
def fig5_severity_mae():
    feat = load_feature()
    best = feat["best"].get("severity", {})
    fig, ax = plt.subplots(figsize=(8, 5))
    grid = [0, 11, 50, 85]
    for v in ("v1", "v2", "v2a"):
        sub = best.get(v, {}).get("per_type", {}).get("bolt", {})
        xs, ys = [], []
        for thr in grid:
            r = sub.get(str(thr))
            if r and r.get("mae") is not None:
                xs.append(thr); ys.append(r["mae"])
        if xs:
            ax.plot(xs, ys, "o-", color=VARIANT_COLORS[v], lw=2,
                    ms=8, label=v)
    ax.set_xlabel("bolt % looseness (positives filter)")
    ax.set_ylabel("MAE (best cell, lower is better)")
    ax.set_title("Severity regression — MAE drops at high bolt severity\n(bolt-restricted positives only)")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_severity_mae.png", dpi=130)
    plt.close(fig)
    print("wrote fig5_severity_mae.png")


# ─── Fig 6 ──────────────────────────────────────────────────────────────
def fig6_vision_vs_bespoke():
    vc = json.load(open(_REPO / "results" / "dt_vision_check_v1_type.json"))
    tiers = ["all", "med+", "severe"]
    # Best vision and best bespoke per tier
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(tiers))
    width = 0.35
    bv_means, bv_lbl, bb_means, bb_sd, bb_lbl = [], [], [], [], []
    for tier in tiers:
        bv = max(((k, v[tier]) for k, v in vc.items() if k.startswith("vision/") and v.get(tier)),
                  key=lambda x: x[1]["macro_f1_mean"], default=(None, None))
        bb = max(((k, v[tier]) for k, v in vc.items() if k.startswith("bespoke/") and v.get(tier)),
                  key=lambda x: x[1]["macro_f1_mean"], default=(None, None))
        bv_means.append(bv[1]["macro_f1_mean"] if bv[1] else np.nan)
        bb_means.append(bb[1]["macro_f1_mean"] if bb[1] else np.nan)
        bb_sd.append(bb[1]["macro_f1_sd"] if bb[1] else 0)
        bv_lbl.append("/".join(bv[0].split("/")[1:]) if bv[0] else "")
        bb_lbl.append("/".join(bb[0].split("/")[1:]) if bb[0] else "")
    ax.bar(x - width/2, bv_means, width, color="#9467bd",
            edgecolor="black", label="best vision (1 seed)")
    ax.bar(x + width/2, bb_means, width, yerr=bb_sd, color="#1f77b4",
            edgecolor="black", label="best bespoke (3-seed)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\nvision: {bv_lbl[i][:18]}\nbespoke: {bb_lbl[i][:18]}"
                        for i, t in enumerate(tiers)],
                        fontsize=8)
    ax.set_ylabel("macro-F1 — type task")
    ax.set_title("Vision-backbone vs bespoke models on `type` (v1)\n"
                  "Bespoke wins at every tier, gap narrows under DT-restriction")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for i, (bv, bb) in enumerate(zip(bv_means, bb_means)):
        if not np.isnan(bv): ax.text(i - width/2, bv + 0.005, f"{bv:.3f}",
                                       ha="center", fontsize=8)
        if not np.isnan(bb): ax.text(i + width/2, bb + 0.005, f"{bb:.3f}",
                                       ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_vision_vs_bespoke.png", dpi=130)
    plt.close(fig)
    print("wrote fig6_vision_vs_bespoke.png")


# ─── Fig 7 ──────────────────────────────────────────────────────────────
def fig7_failure_modes():
    feat = load_feature()
    best = feat["best"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    tiers = ["all", "med+", "severe"]
    for ax, task, axis_key in zip(axes,
                                    ("binary", "is_pristine"),
                                    ("tier_pos", "tier_neg")):
        x = np.arange(len(tiers))
        width = 0.25
        for i, v in enumerate(("v1", "v2", "v2a")):
            sub = best.get(task, {}).get(v, {}).get(axis_key, {})
            means = [sub.get(t, {}).get("macro_f1", np.nan) for t in tiers]
            ax.bar(x + i*width, means, width, color=VARIANT_COLORS[v],
                    edgecolor="black", lw=0.5, label=v)
        ax.set_xticks(x + width)
        ax.set_xticklabels(tiers)
        ax.axhline(0.5, ls=":", color="black", alpha=0.4)
        ax.set_title(task + ("\n(tier on positives)" if task == "binary"
                              else "\n(tier on negatives)"),
                      fontweight="bold")
        ax.set_xlabel("severity tier of filtered samples")
        ax.set_ylabel("macro-F1 (best cell)")
        ax.set_ylim(0.35, 0.55)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Detection failure mode — restriction to severe damage does NOT help\n"
                  "models cannot reliably answer \"is this damaged at all?\"",
                  fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "fig7_failure_modes.png", dpi=130)
    plt.close(fig)
    print("wrote fig7_failure_modes.png")


if __name__ == "__main__":
    fig1_severity_distribution()
    fig2_dt_curves_grid()
    fig3_tier_bars()
    fig4_feature_axis()
    fig5_severity_mae()
    fig6_vision_vs_bespoke()
    fig7_failure_modes()
    print(f"\nall plots in {OUT}")
