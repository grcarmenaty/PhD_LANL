"""Per-task cell-zoo plots: each cell on its own row, three coloured bars
(v1 / v2 / v2a) with error bars; sorted by best variant's mean.

Outputs: results/figures/cell_zoo/<task>.png  (10 files)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parent.parent
OUT = _REPO / "results" / "figures" / "cell_zoo"
OUT.mkdir(parents=True, exist_ok=True)
COLORS = {"v1": "#1f77b4", "v2": "#d62728", "v2a": "#2ca02c"}

J = json.load(open(_REPO / "results" / "cells_v1_v2_v2a.json"))
cells = J["cells"]
tasks = sorted({c["task"] for c in cells.values()})


def render_task(task: str):
    task_cells = sorted(
        (k, v) for k, v in cells.items() if v["task"] == task
    )
    if not task_cells: return
    kind = task_cells[0][1]["kind"]
    score_key = "mae" if kind == "regression" else "macro_f1"
    score_label = "MAE (lower = better)" if kind == "regression" else "macro-F1"
    # Sort by best score across variants
    def sort_key(item):
        _, rec = item
        vals = [rec["metrics"][v]["mean"][score_key]
                for v in ("v1", "v2", "v2a")
                if rec["metrics"].get(v) and rec["metrics"][v]["mean"].get(score_key) is not None]
        if not vals: return 0
        return (min if kind == "regression" else max)(vals)
    task_cells = sorted(task_cells, key=sort_key, reverse=(kind != "regression"))

    n = len(task_cells)
    fig_h = max(4.5, 0.3 * n + 1.5)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    y = np.arange(n)
    width = 0.27
    for i, v in enumerate(("v1", "v2", "v2a")):
        means, sds = [], []
        for _, rec in task_cells:
            m = rec["metrics"].get(v)
            if not m or m["mean"].get(score_key) is None:
                means.append(np.nan); sds.append(0); continue
            means.append(m["mean"][score_key])
            sds.append(m["sd"][score_key])
        ax.barh(y + (i - 1) * width, means, width, xerr=sds, color=COLORS[v],
                edgecolor="black", lw=0.4, label=v, capsize=2)
    labels = [f"{rec['model']}/{rec['feature']}" for _, rec in task_cells]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(score_label)
    # Reference lines
    if kind == "cls_binary":
        prior = task_cells[0][1]["metrics"]["v1"]["mean"].get("class_prior", 0.5)
        ax.axvline(0.5, ls=":", color="black", alpha=0.5,
                    label=f"random macroF1=0.5")
    elif kind == "cls_multi":
        rec0 = task_cells[0][1]["metrics"]["v1"]["mean"]
        n_cls = task_cells[0][1]["metrics"]["v1"]["seeds"]["42"].get("n_classes")
        if n_cls:
            ax.axvline(1.0 / n_cls, ls=":", color="black", alpha=0.5,
                        label=f"random macroF1=1/{n_cls}={1/n_cls:.2f}")
    ax.set_title(f"Task: {task}  —  all {n} cells (pooled, 3-seed mean ±sd)",
                  fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(axis="x", alpha=0.3)
    if kind != "regression":
        ax.set_xlim(0, max(0.9, ax.get_xlim()[1]))
    plt.tight_layout()
    out_path = OUT / f"{task}.png"
    plt.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


for t in tasks:
    render_task(t)
print(f"\n{len(tasks)} plots in {OUT}")
