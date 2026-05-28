"""Modal-feature synth→real gap diagnostic (training-free).

Why: the v2a ablation (REPORT_v2a_chunk_regen.md) showed the asymmetric
damage geometry does not fix the col_location / Crack / Hole synth-real
gap on the modal-MLP cells, and the report flagged the modal feature
pathway itself as the suspect. This script quantifies *where* the gap
lives, without training the full pipeline.

Three probes, all on the 81-dim ``modal`` feature
(9 channels × {3 peak freq/amp pairs, mean/std log-amp, band energy}):

1. Covariate shift: per-dim standardized mean shift of experimental
   features on the synth z-scale. Grouped by the 9 within-channel stats.

2. Discriminant transfer: fit logistic regression on synth z-features
   for a binary task (hole-vs-rest, crack-vs-rest, col-pair), report
   in-domain balanced accuracy vs experimental transfer.

3. Discriminant alignment: per-dim class-mean-difference vectors in
   synth vs experimental, and their cosine alignment. A low cosine means
   the *direction* that separates the classes differs between domains —
   i.e. a synth-trained linear model points the wrong way on real data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_score

_REPO = Path(__file__).resolve().parent.parent
SYN = _REPO / "dataset" / "features.h5"
EXP = _REPO / "dataset" / "experimental_features.h5"

TYPE_PRISTINE, TYPE_BOLT, TYPE_CRACK, TYPE_HOLE, TYPE_MASS = 0, 1, 2, 3, 4

# 9 within-channel stat names, repeated per channel to label all 81 dims.
STAT_NAMES = ["pk1_f", "pk1_a", "pk2_f", "pk2_a", "pk3_f", "pk3_a",
              "mean_logA", "std_logA", "band_E"]


def _load(path: Path):
    with h5py.File(path, "r") as h:
        modal = h["modal"][:].astype(np.float64)
        if "type_code" in h:
            tc = h["type_code"][:]
            end = h["end"][:]
        else:  # synth nests under labels/
            tc = h["labels"]["type_code"][:]
            end = h["labels"]["end"][:]
    return modal, tc.astype(int), end.astype(int)


def _zscale(syn, exp):
    """Standardize both on synth statistics (matches eval normalize)."""
    mu = syn.mean(0)
    sd = syn.std(0)
    sd[sd < 1e-9] = 1.0
    return (syn - mu) / sd, (exp - mu) / sd


def _discriminant_transfer(name, syn_z, ys, exp_z, ye):
    """Synth in-domain CV BA, exp transfer BA, and class-mean-diff cosine."""
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    cv = cross_val_score(clf, syn_z, ys, cv=5,
                         scoring="balanced_accuracy")
    clf.fit(syn_z, ys)
    exp_ba = balanced_accuracy_score(ye, clf.predict(exp_z))

    # Class-mean-difference vectors per domain (pos - neg), unit-normed.
    def mdiff(X, y):
        d = X[y == 1].mean(0) - X[y == 0].mean(0)
        n = np.linalg.norm(d)
        return d / n if n > 0 else d, d
    ws, ds = mdiff(syn_z, ys)
    we, de = mdiff(exp_z, ye)
    cos = float(ws @ we)

    print(f"\n[{name}]  n_syn={len(ys)} (pos {int(ys.sum())})  "
          f"n_exp={len(ye)} (pos {int(ye.sum())})")
    print(f"  synth 5-fold CV balanced-acc : {cv.mean():.3f} ± {cv.std():.3f}")
    print(f"  EXPERIMENTAL transfer BA     : {exp_ba:.3f}  "
          f"(chance 0.500)")
    print(f"  class-mean-diff cosine(syn,exp): {cos:+.3f}  "
          f"(1=aligned, 0=orthogonal, <0=inverted)")

    # Which stats carry the synth discriminant, and do they invert on exp?
    per_stat = {}
    for i in range(81):
        s = STAT_NAMES[i % 9]
        per_stat.setdefault(s, []).append(i)
    print("  per-stat synth weight vs syn/exp sign agreement:")
    for s, idx in per_stat.items():
        idx = np.array(idx)
        contrib = float(np.abs(ds[idx]).sum())
        # fraction of dims in this stat where syn & exp mean-diff agree in sign
        agree = float(np.mean(np.sign(ds[idx]) == np.sign(de[idx])))
        print(f"    {s:<10s} |Δsyn|sum={contrib:6.2f}   sign-agree={agree:.2f}")
    return cv.mean(), exp_ba, cos


def main():
    syn, syn_tc, syn_end = _load(SYN)
    exp, exp_tc, exp_end = _load(EXP)
    print(f"synth modal {syn.shape}, exp modal {exp.shape}")
    syn_z, exp_z = _zscale(syn, exp)

    # ── Probe 1: covariate shift ─────────────────────────────────────────
    shift = exp_z.mean(0)  # synth mean is 0, std 1 on this scale
    print("\n=== PROBE 1: covariate shift (exp mean on synth z-scale) ===")
    print(f"  |shift| mean={np.abs(shift).mean():.2f}  "
          f"max={np.abs(shift).max():.2f}  "
          f"dims>2σ: {int((np.abs(shift) > 2).sum())}/81")
    by_stat = {}
    for i in range(81):
        by_stat.setdefault(STAT_NAMES[i % 9], []).append(abs(shift[i]))
    print("  mean |shift| by within-channel stat:")
    for s, v in by_stat.items():
        print(f"    {s:<10s} {np.mean(v):.2f}")

    # ── Probe 2/3: discriminant transfer + alignment ─────────────────────
    print("\n=== PROBE 2/3: discriminant transfer (synth→experimental) ===")

    # is_hole: hole vs everything else
    _discriminant_transfer(
        "is_hole",
        syn_z, (syn_tc == TYPE_HOLE).astype(int),
        exp_z, (exp_tc == TYPE_HOLE).astype(int))

    # is_crack
    _discriminant_transfer(
        "is_crack",
        syn_z, (syn_tc == TYPE_CRACK).astype(int),
        exp_z, (exp_tc == TYPE_CRACK).astype(int))

    # col_location proxy: among crack/hole damage, "which end".
    # Synth v1 records end as BD(0)/AD(1); exp records corner 0..3.
    # Map both to a binary "low side {0,1} vs high side {2,3}" for exp,
    # and BD(0) vs AD(1) for synth (the only two synth values).
    def col_mask_label(tc, end, hi_threshold):
        m = np.isin(tc, [TYPE_CRACK, TYPE_HOLE]) & (end >= 0)
        lab = (end[m] >= hi_threshold).astype(int)
        return m, lab
    sm, sl = col_mask_label(syn_tc, syn_end, hi_threshold=1)  # AD vs BD
    em, el = col_mask_label(exp_tc, exp_end, hi_threshold=2)  # {2,3} vs {0,1}
    print(f"\n[col_pair] synth end values: "
          f"{np.unique(syn_end[sm], return_counts=True)}")
    print(f"[col_pair] exp end values:   "
          f"{np.unique(exp_end[em], return_counts=True)}")
    if sl.sum() > 5 and (sl == 0).sum() > 5 and el.sum() > 5 and (el == 0).sum() > 5:
        _discriminant_transfer("col_pair (crack+hole)",
                                syn_z[sm], sl, exp_z[em], el)
    else:
        print("[col_pair] degenerate in v1 synth (symmetric damage encodes "
              "no column side) — transfer undefined; this IS the col_location "
              f"finding. syn pos {int(sl.sum())}/{len(sl)}, "
              f"exp pos {int(el.sum())}/{len(el)}")


if __name__ == "__main__":
    main()
