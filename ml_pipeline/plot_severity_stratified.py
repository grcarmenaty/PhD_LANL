"""Accuracy/macro-F1 as a function of severity threshold.

The question: do models do better when restricted to the most extreme
damage cases?  Answers via a curve of metric vs severity threshold,
computed for several synth-only models on the type task using the full
2638-case experimental set.

Severity is normalised per damage type so that severity=0 means
"least extreme" and severity=1 means "most extreme" within that
type's physical range:

  Bolt:   5  → 95   percent loosening
  Crack:  1  → 8    mm  through-thickness
  Hole:   1  → 6    mm  diameter
  Mass:   0.1 → 2.5 kg  added

Pristine has severity 0 and is excluded from this analysis (it is
not "more extreme" of anything).

Inputs:
  results/baseline/experimental_full_per_case.json   cnn2d/cfdac_mag baseline
  results/p1_1/experimental_full_per_case.json       cnn2d/cfdac_mag after P1.1
  results/per_case_vision/type_convnext_tiny_cfdac_all.json  vision synth-only
  results/trenchcoat_eval.json                       trenchcoat best aggregator

Outputs:
  results/figures/severity_stratified/per_model_curves.png
  results/figures/severity_stratified/per_type_breakdown.png
  results/severity_stratified.json
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

_REPO = Path(__file__).resolve().parent.parent
OUT_DIR = _REPO / "results" / "figures" / "severity_stratified"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Per-type physical severity bounds (must match case_design.SEVERITY_BOUNDS).
SEVERITY_BOUNDS = {
    1: (5.0, 95.0),   # Bolt   (percent loosening)
    2: (1.0,  8.0),   # Crack  (mm)
    3: (1.0,  6.0),   # Hole   (mm)
    4: (0.1,  2.5),   # Mass   (kg)
}
TYPE_NAMES = ["Pristine", "Bolt", "Crack", "Hole", "Mass"]


def _normalise_severity(tc, sev):
    """Map raw physical severity to [0,1] per type.  Pristine returns 0."""
    out = np.zeros_like(sev, dtype=np.float32)
    for k, (lo, hi) in SEVERITY_BOUNDS.items():
        sel = tc == k
        out[sel] = (sev[sel] - lo) / (hi - lo)
    return out


def _load_per_case(path: Path, by_case_name: dict, model: str = None,
                     feature: str = None):
    """Yield (idx, y_true, y_pred) tuples aligned with the experimental
    dataset.  ``by_case_name`` maps experimental case_name → row index."""
    rows = json.loads(path.read_text())
    out = []
    if isinstance(rows, list):
        for r in rows:
            if "task" in r and r.get("task") != "type":
                continue
            if model is not None and r.get("model") != model:
                continue
            if feature is not None and r.get("feature") != feature:
                continue
            name = r.get("case")
            if name not in by_case_name:
                continue
            out.append((by_case_name[name], int(r["y_true"]), int(r["y_pred"])))
    elif isinstance(rows, dict) and "rows" in rows:
        # per_case_final / per_case_vision format
        for r in rows["rows"]:
            name = r.get("case")
            if name not in by_case_name:
                continue
            out.append((by_case_name[name], int(r["y_true"]), int(r["y_pred"])))
    return out


def _load_trenchcoat():
    """Pull per-sample (true, pred) from trenchcoat_eval.json.

    The trenchcoat JSON stores the FULL 2638-case predictions in
    ``per_case``, ordered the same as the experimental file (idx 0..2637).
    """
    d = json.loads((_REPO / "results" / "trenchcoat_eval.json").read_text())
    rows = d["per_case"]
    return [(i, int(r["type_code"]), int(r["pred"]))
              for i, r in enumerate(rows)]


def compute_curves():
    # Load experimental case metadata
    with h5py.File(_REPO / "dataset" / "experimental_features.h5", "r") as f:
        names = [str(s) for s in f["names"][:]]
        tc_all = f["type_code"][:]
        sev_phys = f["severity"][:]
    by_name = {n: i for i, n in enumerate(names)}
    sev_norm = _normalise_severity(tc_all, sev_phys)

    # Model list: each entry → (label, per-case rows)
    models = []

    # Baseline cnn2d (pre-fix, original report) — different filename
    p = _REPO / "results" / "baseline" / "experimental_per_case.json"
    if p.exists():
        rs = _load_per_case(p, by_name, model="cnn2d", feature="cfdac_mag")
        if rs: models.append(("cnn2d/cfdac_mag (baseline)", rs))

    # P0.1 snapshot (after the reference-FRF fix)
    p = _REPO / "results" / "p0_1" / "experimental_full_per_case.json"
    if p.exists():
        rs = _load_per_case(p, by_name, model="cnn2d", feature="cfdac_mag")
        if rs: models.append(("cnn2d/cfdac_mag (P0.1 ref-FRF)", rs))

    # Latest cnn2d (current state of results/)
    p = _REPO / "results" / "experimental_full_per_case.json"
    if p.exists():
        rs = _load_per_case(p, by_name, model="cnn2d", feature="cfdac_mag")
        if rs: models.append(("cnn2d/cfdac_mag (current)", rs))

    # Vision sweep — synth-only
    p = (_REPO / "results" / "per_case_vision"
            / "type_convnext_tiny_cfdac_all.json")
    if p.exists():
        rs = _load_per_case(p, by_name)
        if rs: models.append(("ConvNeXt-T/cfdac_all (vision v1)", rs))

    rs = _load_trenchcoat()
    if rs: models.append(("trenchcoat (dataset_zscore)", rs))

    # Sweep severity thresholds; report:
    #   accuracy on damage subset (severity >= τ AND type_code != Pristine)
    #   macro-F1 on damage subset (computed over 4 damage classes)
    #   sample count remaining at τ
    thresholds = np.concatenate([np.linspace(0.0, 0.9, 19),
                                          [0.95, 0.99]])
    out = {"thresholds": thresholds.tolist(), "models": {}}

    for label, rs in models:
        ys = np.array([r[1] for r in rs])
        yhats = np.array([r[2] for r in rs])
        idxs = np.array([r[0] for r in rs])
        # Use the severity associated with the actual case
        sev_for_rs = sev_norm[idxs]
        tc_for_rs = tc_all[idxs]

        accs, f1s, ns = [], [], []
        for t in thresholds:
            mask = (tc_for_rs != 0) & (sev_for_rs >= t)
            if mask.sum() < 5:
                accs.append(float("nan")); f1s.append(float("nan"))
                ns.append(int(mask.sum())); continue
            acc = accuracy_score(ys[mask], yhats[mask])
            # macro over the 4 damage classes (label 1..4)
            f1m = f1_score(ys[mask], yhats[mask], labels=[1,2,3,4],
                              average="macro", zero_division=0)
            accs.append(float(acc))
            f1s.append(float(f1m))
            ns.append(int(mask.sum()))
        out["models"][label] = {"accuracy": accs, "macro_f1": f1s,
                                       "n_remaining": ns}

    (_REPO / "results" / "severity_stratified.json").write_text(
        json.dumps(out, indent=2))
    return out, by_name, sev_norm, tc_all


def plot_per_model_curves(out):
    thresholds = np.array(out["thresholds"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    colors = plt.cm.tab10.colors
    for i, (label, d) in enumerate(out["models"].items()):
        accs = np.array(d["accuracy"])
        f1s = np.array(d["macro_f1"])
        ns  = np.array(d["n_remaining"])
        axes[0].plot(thresholds, accs, "-o", color=colors[i],
                          label=label, markersize=5)
        axes[1].plot(thresholds, f1s, "-o", color=colors[i],
                          label=label, markersize=5)
    axes[0].set_ylabel("accuracy (damage cases only, severity ≥ τ)")
    axes[1].set_ylabel("macro-F1 (4 damage classes, severity ≥ τ)")
    for ax, ttl in zip(axes, ["accuracy vs severity threshold",
                                       "macro-F1 vs severity threshold"]):
        ax.set_xlabel("severity threshold τ (normalised per type)")
        ax.set_title(ttl)
        ax.grid(linestyle=":", alpha=0.4)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.25, color="gray", linestyle="--", linewidth=0.8,
                      label=None)  # random for 4-class
    # Secondary axis for sample count
    # Reference: chance for 4-class restricted-to-damage = 0.25
    fig.suptitle("Accuracy and macro-F1 as a function of severity threshold\n"
                  "(synth-only models on the full 2638-case experimental set, "
                  "damage cases only)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path = OUT_DIR / "per_model_curves.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_n_remaining(out):
    thresholds = np.array(out["thresholds"])
    fig, ax = plt.subplots(figsize=(7, 4))
    # All curves share the same n_remaining (same exp set, same thresholds);
    # take the first model's counts.
    first = next(iter(out["models"].values()))
    ns = np.array(first["n_remaining"])
    ax.plot(thresholds, ns, "-o", color="#888888", markersize=5)
    ax.set_xlabel("severity threshold τ"); ax.set_ylabel("damage cases retained")
    ax.set_title("How many damage cases survive each severity threshold?")
    ax.grid(linestyle=":", alpha=0.4)
    for x, n in zip(thresholds[::3], ns[::3]):
        ax.text(x, n + 30, str(n), ha="center", fontsize=7, color="black")
    fig.tight_layout()
    out_path = OUT_DIR / "n_remaining.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_per_type_breakdown(by_name, sev_norm, tc_all):
    """For the trenchcoat aggregator and the cnn2d baseline, plot
    per-true-type accuracy vs severity threshold."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, src, title in zip(axes,
                                          ["baseline_cnn2d", "trenchcoat"],
                                          ["cnn2d/cfdac_mag baseline",
                                            "trenchcoat (dataset_zscore)"]):
        if src == "baseline_cnn2d":
            p = _REPO / "results" / "experimental_full_per_case.json"
            rs = _load_per_case(p, by_name, model="cnn2d",
                                    feature="cfdac_mag")
        else:
            rs = _load_trenchcoat()
        if not rs:
            ax.set_title(f"{title}: no data"); continue
        ys = np.array([r[1] for r in rs])
        yhats = np.array([r[2] for r in rs])
        idxs = np.array([r[0] for r in rs])
        sev_for_rs = sev_norm[idxs]
        tc_for_rs = tc_all[idxs]

        thresholds = np.linspace(0.0, 0.9, 10)
        for k, name in enumerate(TYPE_NAMES):
            if k == 0: continue   # skip Pristine
            cls_acc = []
            for t in thresholds:
                mask = (tc_for_rs == k) & (sev_for_rs >= t)
                if mask.sum() < 5:
                    cls_acc.append(float("nan")); continue
                cls_acc.append(accuracy_score(ys[mask], yhats[mask]))
            ax.plot(thresholds, cls_acc, "-o", label=name, markersize=5)
        ax.set_title(title)
        ax.set_xlabel("severity threshold τ")
        ax.grid(linestyle=":", alpha=0.4)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("per-class accuracy")
    fig.suptitle("Per-true-type accuracy as a function of severity threshold",
                  fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path = OUT_DIR / "per_type_breakdown.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_confidence_stratified(by_name, tc_all):
    """Accuracy as a function of model confidence (= max softmax prob).

    Only the vision per-case JSONs include proba, so this plot is
    restricted to those.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    pc_dir = _REPO / "results" / "per_case_vision"
    curves = []
    for p in sorted(pc_dir.glob("type_*.json")):
        d = json.loads(p.read_text())
        rows_pc = d["rows"]
        if not rows_pc or "proba" not in rows_pc[0]:
            continue
        ys, yhats, conf, idxs = [], [], [], []
        for r in rows_pc:
            name = r.get("case")
            if name not in by_name:
                continue
            ys.append(int(r["y_true"])); yhats.append(int(r["y_pred"]))
            conf.append(float(max(r["proba"])))
            idxs.append(by_name[name])
        ys = np.array(ys); yhats = np.array(yhats)
        conf = np.array(conf); idxs = np.array(idxs)
        tc_for_rs = tc_all[idxs]
        damage_mask = tc_for_rs != 0
        # Severity isn't needed here -- we stratify by confidence
        thresholds = np.linspace(0.0, max(0.99, conf.max() * 0.99), 20)
        accs, ns = [], []
        for t in thresholds:
            mask = damage_mask & (conf >= t)
            if mask.sum() < 5:
                accs.append(float("nan")); ns.append(int(mask.sum())); continue
            accs.append(float(accuracy_score(ys[mask], yhats[mask])))
            ns.append(int(mask.sum()))
        backbone = d["meta"]["backbone"]
        feature  = d["meta"]["feature"]
        # Only plot the top few for readability
        curves.append({"label": f"{backbone}/{feature}",
                          "thresholds": thresholds, "accs": accs, "ns": ns,
                          "max_conf": conf.max()})

    # Plot top-5 by maximum accuracy reached
    curves.sort(key=lambda c: -np.nanmax(c["accs"]))
    for i, c in enumerate(curves[:5]):
        axes[0].plot(c["thresholds"], c["accs"], "-o",
                          label=c["label"], markersize=4)
        axes[1].plot(c["thresholds"], c["ns"], "-o",
                          label=c["label"], markersize=4)
    axes[0].set_xlabel("confidence threshold τ (= max softmax prob)")
    axes[0].set_ylabel("accuracy on damage cases with conf ≥ τ")
    axes[0].set_title("Accuracy vs model-confidence threshold")
    axes[0].grid(linestyle=":", alpha=0.4)
    axes[0].axhline(0.25, color="gray", linestyle="--", linewidth=0.8)
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].set_ylim(0, 1.05)

    axes[1].set_xlabel("confidence threshold τ")
    axes[1].set_ylabel("damage cases retained (log scale)")
    axes[1].set_title("Sample retention")
    axes[1].set_yscale("log")
    axes[1].grid(linestyle=":", alpha=0.4)
    axes[1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Confidence stratification — does the model "
                  "do better when it commits more strongly?", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path = OUT_DIR / "confidence_stratified.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    out, by_name, sev_norm, tc_all = compute_curves()
    plot_per_model_curves(out)
    plot_n_remaining(out)
    plot_per_type_breakdown(by_name, sev_norm, tc_all)
    plot_confidence_stratified(by_name, tc_all)
    print(f"\nfigures in {OUT_DIR}")

    # Print headline table
    print("\nAccuracy by severity threshold (4-class damage subset):")
    thresholds = np.array(out["thresholds"])
    print(f"{'model':<40s} τ=0.0  τ=0.3  τ=0.5  τ=0.7  τ=0.9  τ=0.99")
    for label, d in out["models"].items():
        accs = d["accuracy"]
        # interpolate at chosen thresholds
        vals = []
        for t in (0.0, 0.3, 0.5, 0.7, 0.9, 0.99):
            i = np.argmin(np.abs(thresholds - t))
            vals.append(accs[i])
        ns_at_99 = d["n_remaining"][np.argmin(np.abs(thresholds - 0.99))]
        print(f"{label:<40s} " + "  ".join(
            f"{v:.2f}" if not np.isnan(v) else " -- " for v in vals)
            + f"  (n@τ=0.99: {ns_at_99})")


if __name__ == "__main__":
    main()
