"""Plots and detail tables for §10 (transfer-learning) and §11
(resolution sweep).

Reads:
    results/transfer_learning.json
    results/resolution_sweep.json

Writes under results/figures/sweeps/:
    transfer/per_task_curves.png        line plot of best cell per task vs k
    transfer/delta_vs_zeroshot.png       bar chart at k=50 % vs zero-shot
    transfer/heatmap_<task>.png          (model,feature,unfreeze) × k heatmap
    transfer/unfreeze_compare.png        head vs head_proj scatter
    resolution/per_task_curves.png       best cell per task vs ratio
    resolution/heatmap_<task>.png        (model, feature) × ratio heatmap
    resolution/robustness.png            per-cell drop r=1→r=0.5
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parent.parent

TASKS = ("binary", "type", "severity", "col_location", "mass_location")
FRACTIONS = (0.10, 0.20, 0.30, 0.40, 0.50)
RATIOS = (1.000, 0.875, 0.750, 0.625, 0.500)

ZERO_SHOT = {
    "binary":         (0.941, "MLP/modal"),
    "type":           (0.403, "2-D CNN/cfdac_mag"),
    "severity":       (0.022, "XGB/modal"),
    "col_location":   (0.287, "1-D CNN/frf_mag"),
    "mass_location":  (0.338, "2-D CNN/cfdac_all"),
}


def _save(fig, path: Path, dpi: int = 110) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ── Transfer-learning plots ─────────────────────────────────────────────────
def plot_transfer(rows: list[dict], out: Path) -> None:
    if not rows:
        return
    out = out / "transfer"

    # 1. Per-task line plot: best cell per (task, k) across all unfreeze depths
    fig, ax = plt.subplots(figsize=(9, 5))
    for task in TASKS:
        ys = []
        for k in FRACTIONS:
            rk = [r for r in rows
                       if r["task"] == task and r["fraction"] == k]
            ys.append(max(r["value"] for r in rk) if rk else np.nan)
        ax.plot([100 * k for k in FRACTIONS], ys, "-o", label=task)
        # Annotate zero-shot baseline as a dashed horizontal line
        zs, _ = ZERO_SHOT[task]
        ax.axhline(zs, linestyle=":", alpha=0.4,
                       color=ax.get_lines()[-1].get_color())
    ax.set_xlabel("fine-tune fraction k (%)")
    ax.set_ylabel("best held-out metric (accuracy / R²)")
    ax.set_title("Transfer-learning lift per task vs experimental fraction "
                     "(dashed = zero-shot best, § 9)")
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    _save(fig, out / "per_task_curves.png")

    # 2. Δ vs zero-shot at k=50 %
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels, vals, anns = [], [], []
    for task in TASKS:
        rk = [r for r in rows
                   if r["task"] == task and r["fraction"] == 0.5]
        if not rk:
            continue
        best = max(rk, key=lambda r: r["value"])
        zs, _ = ZERO_SHOT[task]
        labels.append(task); vals.append(best["value"] - zs)
        anns.append(f"{best['model']}/{best['feature']}/{best['unfreeze']}")
    colors = ["tab:green" if v >= 0 else "tab:red" for v in vals]
    bars = ax.barh(labels, vals, color=colors)
    for i, (b, a) in enumerate(zip(bars, anns)):
        ax.text(b.get_width(), b.get_y() + b.get_height() / 2,
                    f"  {a}", va="center", fontsize=8)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Δ metric at k=50 % vs zero-shot")
    ax.set_title("Sim-to-real lift from fine-tuning (k = 50 %)")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    _save(fig, out / "delta_vs_zeroshot.png")

    # 3. Per-task heatmap (cell × k)
    for task in TASKS:
        cells = sorted({(r["model"], r["feature"], r["unfreeze"])
                          for r in rows if r["task"] == task})
        if not cells:
            continue
        mat = np.full((len(cells), len(FRACTIONS)), np.nan)
        for i, (m, f, u) in enumerate(cells):
            for j, k in enumerate(FRACTIONS):
                v = [r["value"] for r in rows
                       if r["task"] == task and r["model"] == m
                       and r["feature"] == f and r["unfreeze"] == u
                       and r["fraction"] == k]
                if v:
                    mat[i, j] = v[0]
        fig, ax = plt.subplots(figsize=(7, 0.25 * len(cells) + 1.5))
        im = ax.imshow(mat, cmap="viridis", aspect="auto",
                          vmin=np.nanmin(mat), vmax=np.nanmax(mat))
        ax.set_xticks(range(len(FRACTIONS)))
        ax.set_xticklabels([f"{int(k*100)}%" for k in FRACTIONS])
        ax.set_yticks(range(len(cells)))
        ax.set_yticklabels([f"{m}/{f}/{u}" for (m, f, u) in cells],
                                fontsize=7)
        ax.set_xlabel("fine-tune fraction")
        ax.set_title(f"{task} — held-out metric  (rows: model/feature/unfreeze)")
        fig.colorbar(im, fraction=0.04, pad=0.04)
        _save(fig, out / f"heatmap_{task}.png")

    # 4. head vs head_proj scatter (at k=50 %)
    fig, ax = plt.subplots(figsize=(6, 6))
    pairs = defaultdict(dict)
    for r in rows:
        if r["fraction"] != 0.5:
            continue
        pairs[(r["task"], r["model"], r["feature"])][r["unfreeze"]] = r["value"]
    xs, ys, cs, lbls = [], [], [], []
    color_by_task = {"binary": "tab:blue", "type": "tab:orange",
                          "severity": "tab:green",
                          "col_location": "tab:red",
                          "mass_location": "tab:purple"}
    for (task, m, f), v in pairs.items():
        if "head" in v and "head_proj" in v:
            xs.append(v["head"]); ys.append(v["head_proj"])
            cs.append(color_by_task[task])
            lbls.append(task)
    ax.scatter(xs, ys, s=18, c=cs, alpha=0.7)
    lo = min(min(xs), min(ys)); hi = max(max(xs), max(ys))
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.7)
    ax.set_xlabel("head only (k=50 %)")
    ax.set_ylabel("head + projection (k=50 %)")
    ax.set_title("Unfreeze depth comparison")
    handles = [plt.Line2D([], [], marker="o", color=color_by_task[t],
                                  linestyle="", label=t)
                  for t in TASKS]
    ax.legend(handles=handles, fontsize=9, loc="lower right")
    ax.grid(linestyle=":", alpha=0.5)
    _save(fig, out / "unfreeze_compare.png")


# ── Resolution-sweep plots ───────────────────────────────────────────────────
def plot_resolution(rows: list[dict], out: Path) -> None:
    if not rows:
        return
    out = out / "resolution"

    # 1. Per-task best cell vs ratio
    fig, ax = plt.subplots(figsize=(9, 5))
    for task in TASKS:
        ys = []
        for r_ratio in RATIOS:
            rk = [r for r in rows
                       if r["task"] == task and abs(r["ratio"] - r_ratio) < 1e-3]
            ys.append(max(r["value"] for r in rk) if rk else np.nan)
        ax.plot(RATIOS, ys, "-o", label=task)
    ax.set_xlabel("feature resolution ratio")
    ax.set_ylabel("synth test metric (best cell per ratio)")
    ax.set_title("Best synth-test metric per task vs feature resolution")
    ax.invert_xaxis()  # decimation goes 1.0 → 0.5 left-to-right
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(loc="best", fontsize=9)
    _save(fig, out / "per_task_curves.png")

    # 2. Per-task heatmap (cell × ratio)
    for task in TASKS:
        cells = sorted({(r["model"], r["feature"])
                            for r in rows if r["task"] == task})
        if not cells:
            continue
        mat = np.full((len(cells), len(RATIOS)), np.nan)
        for i, (m, f) in enumerate(cells):
            for j, ratio in enumerate(RATIOS):
                v = [r["value"] for r in rows
                       if r["task"] == task and r["model"] == m
                       and r["feature"] == f
                       and abs(r["ratio"] - ratio) < 1e-3]
                if v:
                    mat[i, j] = v[0]
        fig, ax = plt.subplots(figsize=(7, 0.20 * len(cells) + 1.5))
        im = ax.imshow(mat, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(RATIOS)))
        ax.set_xticklabels([f"{r:.3f}" for r in RATIOS])
        ax.set_yticks(range(len(cells)))
        ax.set_yticklabels([f"{m}/{f}" for (m, f) in cells], fontsize=7)
        ax.set_xlabel("resolution ratio")
        ax.set_title(f"{task} — synth test metric  (rows: model/feature)")
        fig.colorbar(im, fraction=0.04, pad=0.04)
        _save(fig, out / f"heatmap_{task}.png")

    # 3. Robustness scatter: r=1 metric vs (metric at r=0.5).  Diagonal =
    # perfect robustness; below diagonal = degraded by resolution drop.
    fig, ax = plt.subplots(figsize=(7, 7))
    color_by_task = {"binary": "tab:blue", "type": "tab:orange",
                          "severity": "tab:green",
                          "col_location": "tab:red",
                          "mass_location": "tab:purple"}
    cells_at_full: dict[tuple, float] = {}
    cells_at_half: dict[tuple, float] = {}
    for r in rows:
        if abs(r["ratio"] - 1.000) < 1e-3:
            cells_at_full[(r["task"], r["model"], r["feature"])] = r["value"]
        if abs(r["ratio"] - 0.500) < 1e-3:
            cells_at_half[(r["task"], r["model"], r["feature"])] = r["value"]
    xs, ys, cs = [], [], []
    for key, full_val in cells_at_full.items():
        if key in cells_at_half:
            xs.append(full_val); ys.append(cells_at_half[key])
            cs.append(color_by_task[key[0]])
    ax.scatter(xs, ys, s=18, c=cs, alpha=0.7)
    lo = min(xs + ys); hi = max(xs + ys)
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.7,
                label="perfect robustness")
    ax.set_xlabel("metric at full resolution (r = 1.000)")
    ax.set_ylabel("metric at half resolution (r = 0.500)")
    ax.set_title("Resolution robustness — closer to diagonal = better")
    handles = [plt.Line2D([], [], marker="o", color=color_by_task[t],
                                  linestyle="", label=t)
                  for t in TASKS]
    ax.legend(handles=handles, fontsize=9, loc="upper left")
    ax.grid(linestyle=":", alpha=0.5)
    _save(fig, out / "robustness.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=_REPO / "results")
    args = p.parse_args()
    out = args.results / "figures" / "sweeps"
    tl = args.results / "transfer_learning.json"
    rs = args.results / "resolution_sweep.json"
    if tl.exists():
        plot_transfer(json.loads(tl.read_text()), out)
        print(f"wrote transfer-learning figures to {out / 'transfer'}")
    if rs.exists():
        plot_resolution(json.loads(rs.read_text()), out)
        print(f"wrote resolution figures to {out / 'resolution'}")


if __name__ == "__main__":
    main()
