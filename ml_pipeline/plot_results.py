"""Render summary plots for the trained-model and experimental-eval results.

Produces:

  results/figures/train_metrics_by_task.png
  results/figures/experimental_metrics_by_task.png

Both as grouped bar charts (model x feature, one panel per task).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_REPO = Path(__file__).resolve().parent.parent


def _bar_plot(rows, out_path: Path, title: str) -> None:
    tasks = sorted({r["task"] for r in rows})
    feats = sorted({r["feature"] for r in rows})
    models = sorted({r["model"] for r in rows})

    fig, axes = plt.subplots(1, len(tasks), figsize=(3.6 * len(tasks), 4), sharey=False)
    if len(tasks) == 1:
        axes = [axes]
    for ax, task in zip(axes, tasks):
        rows_t = [r for r in rows if r["task"] == task]
        if not rows_t:
            ax.set_title(task); continue
        # Build matrix model x feature.
        M = np.full((len(models), len(feats)), np.nan)
        for r in rows_t:
            i = models.index(r["model"])
            j = feats.index(r["feature"])
            val = r.get("metric_test", r.get("value"))
            M[i, j] = val
        x = np.arange(len(feats))
        width = 0.8 / max(len(models), 1)
        for k, model in enumerate(models):
            ys = M[k]
            offsets = x + (k - len(models)/2 + 0.5) * width
            ax.bar(offsets, np.where(np.isnan(ys), 0, ys),
                    width=width, label=model)
        ax.set_xticks(x); ax.set_xticklabels(feats, rotation=30, ha="right")
        ax.set_title(task)
        ax.set_ylim(0, max(1.05, np.nanmax(M) * 1.05) if np.isfinite(M).any() else 1)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        if ax is axes[0]:
            ax.set_ylabel("test metric")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=_REPO / "results")
    args = p.parse_args()
    figs = args.results / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    # Synth test bars — built from every results/hpo/<task>__<model>__<feature>.json.
    hpo_dir = args.results / "hpo"
    train_rows = []
    for path in sorted(hpo_dir.glob("*.json")):
        blob = json.loads(path.read_text())
        if blob.get("best_metric_test") is None:
            continue
        train_rows.append({"task": blob["task"], "model": blob["model"],
                              "feature": blob["feature"],
                              "metric_test": blob["best_metric_test"]})
    if train_rows:
        _bar_plot(train_rows, figs / "train_metrics_by_task.png",
                    "Synthetic test-set metric per (model, feature, task)")

    # Experimental bars — prefer the full 2 638-case eval if present.
    full_exp_path = args.results / "experimental_full_evaluation.json"
    if full_exp_path.exists():
        rows = json.loads(full_exp_path.read_text())
        for r in rows:
            r["metric_test"] = r.get("value")
        _bar_plot(rows, figs / "experimental_metrics_by_task.png",
                    "Full 2 638-case experimental metric per (model, feature, task)")
    bal_exp_path = args.results / "balanced" / "experimental_full_evaluation.json"
    if bal_exp_path.exists():
        rows = json.loads(bal_exp_path.read_text())
        for r in rows:
            r["metric_test"] = r.get("value")
        _bar_plot(rows, figs / "experimental_balanced_metrics_by_task.png",
                    "Balanced 680-case experimental metric per (model, feature, task)")
    print(f"Plots written to {figs}")


if __name__ == "__main__":
    main()
