"""Diagnostic plots for the binary-trenchcoat decomposition of `type`.

Reads results/trenchcoat_eval.json and produces:

  results/figures/trenchcoat/aggregator_compare.png
  results/figures/trenchcoat/confusion.png
  results/figures/trenchcoat/per_binary_auc.png
  results/figures/trenchcoat/proba_distribution.png
  results/figures/trenchcoat/uncertainty_hist.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, roc_curve, auc,
)

_REPO = Path(__file__).resolve().parent.parent
OUT_DIR = _REPO / "results" / "figures" / "trenchcoat"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TYPE_NAMES = ["Pristine", "Bolt", "Crack", "Hole", "Mass"]


def _load():
    return json.loads((_REPO / "results" / "trenchcoat_eval.json").read_text())


def plot_aggregator_compare():
    d = _load()
    aggs = d.get("aggregators", {})
    if not aggs:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = list(aggs.keys())
    accs = [aggs[n]["accuracy"] for n in names]
    f1s = [aggs[n]["macro_f1"] for n in names]
    x = np.arange(len(names))
    w = 0.36
    ax.bar(x - w/2, accs, w, label="accuracy",
              color="#aaaaaa", edgecolor="black", linewidth=0.4)
    ax.bar(x + w/2, f1s, w, label="macro-F1",
              color="#d9534f", edgecolor="black", linewidth=0.4)
    # Class-prior accuracy floor on unbalanced 2638-case exp:
    ax.axhline(0.507, color="#4a90d9", linestyle="--", linewidth=2,
                  label="predict-Bolt class-prior floor (0.51)")
    # Multi-class baseline from REPORT_vision: convnext_tiny/cfdac_all
    ax.axhline(0.331, color="#7eb854", linestyle=":", linewidth=2,
                  label="multi-class convnext_tiny acc (0.33)")
    ax.axhline(0.253, color="#7eb854", linestyle="-.", linewidth=2,
                  label="multi-class convnext_tiny macro-F1 (0.25)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("metric on full 2638-case exp")
    ax.set_title("Trenchcoat aggregators — accuracy vs macro-F1\n"
                  f"backbone={d['backbone']} / feature={d['feature']}")
    ax.set_ylim(0, 0.6)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(fontsize=8, loc="upper right")
    for xi, (a, f1) in enumerate(zip(accs, f1s)):
        ax.text(xi - w/2, a + 0.01, f"{a:.2f}", ha="center",
                  va="bottom", fontsize=8)
        ax.text(xi + w/2, f1 + 0.01, f"{f1:.2f}", ha="center",
                  va="bottom", fontsize=8)
    fig.tight_layout()
    out = OUT_DIR / "aggregator_compare.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def plot_confusion():
    d = _load()
    cm = np.array(d["confusion_matrix"])
    cm_norm = np.array(d["confusion_matrix_normalised"])
    f1_per = d["aggregator_per_class_f1"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, M, ttl, fmt in zip(axes, [cm, cm_norm],
                                          [f"counts (n={cm.sum()})",
                                            "row-normalised recall"],
                                          ["{:d}", "{:.2f}"]):
        im = ax.imshow(M, cmap="Blues", vmin=0,
                          vmax=cm.max() if M is cm else 1)
        ax.set_xticks(range(5)); ax.set_xticklabels(TYPE_NAMES, rotation=30,
                                                                ha="right", fontsize=9)
        ax.set_yticks(range(5)); ax.set_yticklabels(TYPE_NAMES, fontsize=9)
        ax.set_xlabel("predicted (argmax of 5 binaries)")
        ax.set_ylabel("true")
        for i in range(5):
            for j in range(5):
                v = cm[i, j] if M is cm else cm_norm[i, j]
                txt = fmt.format(v)
                ax.text(j, i, txt, ha="center", va="center",
                          fontsize=8,
                          color="white" if M[i, j] > 0.5*M.max() else "black")
        ax.set_title(ttl)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    fig.suptitle(f"Trenchcoat aggregator confusion — best={d.get('best_aggregator', 'naive')}\n"
                  f"acc={d['aggregator_accuracy']:.3f}, macro-F1={d['aggregator_macro_f1']:.3f}",
                  fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = OUT_DIR / "confusion.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def plot_per_binary_auc():
    d = _load()
    per_case = d["per_case"]
    proba = np.array([r["proba"] for r in per_case])
    tc = np.array([r["type_code"] for r in per_case])

    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = ["#888888", "#d9534f", "#e6a23c", "#7eb854", "#4a90d9"]
    aucs = []
    for k, (name, c) in enumerate(zip(TYPE_NAMES, colors)):
        y_bin = (tc == k).astype(int)
        if y_bin.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin, proba[:, k])
        a = auc(fpr, tpr)
        aucs.append(a)
        ax.plot(fpr, tpr, color=c, linewidth=2,
                  label=f"is_{name} (AUC={a:.2f}, n_pos={int(y_bin.sum())})")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"Per-binary ROC — trenchcoat ({d['backbone']}/{d['feature']})\n"
                  f"macro-AUC = {np.mean(aucs):.3f}")
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    out = OUT_DIR / "per_binary_auc.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def plot_proba_distribution():
    """Per-binary probability distributions broken down by true class.

    Reveals whether each binary actually discriminates positives from
    negatives or just outputs a near-constant.
    """
    d = _load()
    per_case = d["per_case"]
    proba = np.array([r["proba"] for r in per_case])
    tc = np.array([r["type_code"] for r in per_case])

    fig, axes = plt.subplots(1, 5, figsize=(16, 4), sharey=True)
    for k_bin, (ax, name) in enumerate(zip(axes, TYPE_NAMES)):
        ax.hist(proba[tc != k_bin, k_bin], bins=40, range=(0, 1),
                  alpha=0.55, color="#888888",
                  label=f"true ≠ {name} (n={int((tc != k_bin).sum())})",
                  density=True)
        ax.hist(proba[tc == k_bin, k_bin], bins=40, range=(0, 1),
                  alpha=0.7, color="#d9534f",
                  label=f"true = {name} (n={int((tc == k_bin).sum())})",
                  density=True)
        ax.set_title(f"is_{name}")
        ax.set_xlabel("P(positive)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(linestyle=":", alpha=0.4)
        ax.axvline(proba[tc == k_bin, k_bin].mean(), color="red",
                      linestyle="-", linewidth=1)
        ax.axvline(proba[tc != k_bin, k_bin].mean(), color="black",
                      linestyle="-", linewidth=1)
    axes[0].set_ylabel("density")
    fig.suptitle("Per-binary probability distributions by true class\n"
                  "(vertical lines: per-class mean prob. — should be "
                  "well-separated if the binary discriminates)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = OUT_DIR / "proba_distribution.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def plot_uncertainty_hist():
    d = _load()
    per_case = d["per_case"]
    unc = np.array([r["uncertainty"] for r in per_case])
    yes_count = np.array([r["binary_yes_count"] for r in per_case])
    tc = np.array([r["type_code"] for r in per_case])
    pred = np.array([r["pred"] for r in per_case])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    correct = pred == tc
    axes[0].hist(unc[correct],   bins=30, alpha=0.6, color="#7eb854",
                    label=f"correct (n={int(correct.sum())})", density=True)
    axes[0].hist(unc[~correct],  bins=30, alpha=0.6, color="#d9534f",
                    label=f"incorrect (n={int((~correct).sum())})", density=True)
    axes[0].set_xlabel("per-sample uncertainty (1 − max prob)")
    axes[0].set_ylabel("density")
    axes[0].set_title("Are confident predictions also correct?")
    axes[0].legend(fontsize=9); axes[0].grid(linestyle=":", alpha=0.4)

    bins_yes = np.bincount(yes_count, minlength=6)
    axes[1].bar(range(len(bins_yes)), bins_yes, color="#4a90d9",
                    edgecolor="black", linewidth=0.4)
    axes[1].set_xlabel("number of binaries with P(pos) > 0.5")
    axes[1].set_ylabel("samples")
    axes[1].set_title("How often do the 5 binaries agree / disagree?")
    for xi, v in enumerate(bins_yes):
        axes[1].text(xi, v + max(bins_yes)*0.01, str(v),
                          ha="center", fontsize=8)
    axes[1].grid(axis="y", linestyle=":", alpha=0.4)

    fig.suptitle("Trenchcoat — per-sample uncertainty + binary-agreement diagnostics",
                  fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = OUT_DIR / "uncertainty_hist.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    plot_aggregator_compare()
    plot_confusion()
    plot_per_binary_auc()
    plot_proba_distribution()
    plot_uncertainty_hist()
    print(f"\nfigures in {OUT_DIR}")


if __name__ == "__main__":
    main()
