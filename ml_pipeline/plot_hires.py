"""Plots for the hi-res 1601 CFDAC sweep.

  results/figures/hires/hires_synth_vs_exp.png  — per-cell synth (in-domain)
        macro-F1/R2 vs experimental zero-shot macro-F1, with the chance line.
  results/figures/hires/hires_vs_baseline.png   — per-task experimental
        balanced-accuracy of the 1601 cell vs the best 128 baseline cell,
        with the chance line, showing full-res does not help.

Reads results_hires/hires_summary.json (written by hires_summary.py).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parent.parent
OUT = _REPO / "results" / "figures" / "hires"
OUT.mkdir(parents=True, exist_ok=True)
S = json.loads((_REPO / "results_hires" / "hires_summary.json").read_text())

cls = {k: v for k, v in S.items() if v["kind"] == "cls"}
order = sorted(cls, key=lambda k: cls[k]["task"])
tasks = [cls[k]["task"] for k in order]
synth = [(cls[k]["synth"]["test_macro_f1"] or 0.0) for k in order]
exp_mf1 = [cls[k]["exp"]["macro_f1"] for k in order]
exp_bal = [cls[k]["exp"]["balanced_acc"] for k in order]
chance = [1.0 / cls[k]["n_out"] for k in order]
base = [((cls[k]["baseline_128_task_best_v1"] or {}).get("macro_f1") or np.nan) for k in order]

# ---- Fig 1: synth (in-domain) vs experimental (zero-shot) macro-F1 ----
fig, ax = plt.subplots(figsize=(12, 5.5))
x = np.arange(len(order)); w = 0.38
ax.bar(x - w/2, synth, w, label="synth test (in-domain) macro-F1", color="#2ca02c", edgecolor="black", lw=0.4)
ax.bar(x + w/2, exp_mf1, w, label="experimental (zero-shot) macro-F1", color="#d62728", edgecolor="black", lw=0.4)
ax.plot(x, chance, "k_", ms=18, mew=2, label="random macro-F1 (1/n_classes)")
for i, k in enumerate(order):
    if cls[k]["exp"]["collapse"]:
        ax.annotate("collapse", (x[i]+w/2, exp_mf1[i]), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=7, color="#d62728")
ax.set_xticks(x); ax.set_xticklabels(tasks, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("macro-F1"); ax.set_ylim(0, 1)
ax.set_title("Hi-res 1601² CFDAC — in-domain learning vs zero-shot transfer\n"
             "(models learn on synth; nearly all collapse to chance on experimental)",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(OUT / "hires_synth_vs_exp.png", dpi=130); plt.close(fig)
print("wrote", OUT / "hires_synth_vs_exp.png")

# ---- Fig 2: experimental balanced-acc 1601 vs best 128 baseline macro-F1 ----
fig, ax = plt.subplots(figsize=(12, 5.5))
ax.bar(x - w/2, exp_bal, w, label="1601² cell — exp balanced-acc", color="#1f77b4", edgecolor="black", lw=0.4)
ax.bar(x + w/2, base, w, label="best 128² baseline (v1) — exp macro-F1", color="#ff7f0e", edgecolor="black", lw=0.4)
ax.plot(x, chance, "k_", ms=18, mew=2, label="chance")
ax.set_xticks(x); ax.set_xticklabels(tasks, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("score"); ax.set_ylim(0, 0.9)
ax.set_title("Does full resolution help? Hi-res 1601² (balanced-acc) vs best 128² baseline (macro-F1)\n"
             "Full resolution is uniformly NOT better — most 1601² cells sit at chance",
             fontweight="bold")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(OUT / "hires_vs_baseline.png", dpi=130); plt.close(fig)
print("wrote", OUT / "hires_vs_baseline.png")
