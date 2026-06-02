"""Per-task DT-stratified curve plots — one PNG per task, larger and clearer
than the 6-panel grid. Includes:
  - macroF1 vs DT for v1/v2/v2a (best cell at each DT)
  - cell-of-best annotations
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parent.parent
OUT = _REPO / "results" / "figures" / "dt_per_task"
OUT.mkdir(parents=True, exist_ok=True)
COLORS = {"v1": "#1f77b4", "v2": "#d62728", "v2a": "#2ca02c"}

J = json.load(open(_REPO / "results" / "dt_compare_v1_v2_v2a.json"))
bp = J["best_per_task"]
dt_grid = J["dt_grid"]
tasks = sorted(bp)

for task in tasks:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for v in ("v1", "v2", "v2a"):
        xs, ys, sds, cells = [], [], [], []
        for dt in dt_grid:
            r = bp.get(task, {}).get(v, {}).get(f"{dt:.2f}")
            if r is None: continue
            score = r.get("macro_f1_mean") if r.get("macro_f1_mean") is not None else r.get("mae_mean")
            if score is None: continue
            xs.append(dt); ys.append(score)
            sds.append(r.get("macro_f1_sd") or r.get("mae_sd") or 0)
            cells.append(r.get("cell", ""))
        if xs:
            xs, ys, sds = map(np.array, (xs, ys, sds))
            ax.plot(xs, ys, "o-", color=COLORS[v], label=v, lw=2.2, ms=7)
            ax.fill_between(xs, ys - sds, ys + sds,
                             color=COLORS[v], alpha=0.15)
            # Annotate best cell at the highest DT for each variant
            ax.annotate(cells[-1],
                         xy=(xs[-1], ys[-1]),
                         xytext=(8, 4 + 12 * ("v1 v2 v2a".split().index(v))),
                         textcoords="offset points", fontsize=7,
                         color=COLORS[v], alpha=0.85)
    score_kind = "MAE" if task == "severity" else "macro-F1"
    ax.set_ylabel(f"best-cell {score_kind} (3-seed mean ±sd)")
    ax.set_xlabel("DT — minimum stiffness reduction kept in positives")
    if task != "severity":
        ax.axhline(0.5, ls=":", color="black", alpha=0.4, lw=0.8)
    ax.set_xlim(-0.02, 0.62)
    if task != "severity":
        ax.set_ylim(0.3, 0.9)
    ax.set_title(f"DT sweep — {task}  (v1 / v2 / v2a, 3-seed)", fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / f"{task}.png", dpi=130)
    plt.close(fig)
    print(f"wrote {task}.png")

print(f"\n{len(tasks)} plots in {OUT}")
