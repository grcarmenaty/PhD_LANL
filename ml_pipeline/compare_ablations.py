"""Side-by-side comparison of `experimental_full_evaluation.json` from
multiple ablation snapshots (results/baseline, results/p0_1, ...).

Usage:
    python -m ml_pipeline.compare_ablations \\
        results/baseline results/p0_1 results/p0_2 results/p0_3 results/p1_1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(p: Path):
    rows = json.loads((p / "experimental_full_evaluation.json").read_text())
    return {(r["task"], r["model"], r["feature"]): r for r in rows}


def _best_per_task(d):
    out = {}
    for (t, m, f), r in d.items():
        v = r.get("value")
        if v is None:
            continue
        if t not in out or v > out[t][0]:
            out[t] = (v, m, f)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("snapshots", nargs="+", type=Path)
    args = p.parse_args()

    tables = {p.name: _load(p) for p in args.snapshots}
    names = list(tables.keys())

    print(f"\nBest per task across {len(names)} snapshots:\n")
    print(f"{'task':<14s} " + "  ".join(f"{n:<28s}" for n in names))
    print("-" * (14 + 30 * len(names)))
    all_bests = {n: _best_per_task(tables[n]) for n in names}
    for task in ("binary", "type", "severity", "col_location",
                    "mass_location"):
        row = [f"{task:<14s} "]
        for n in names:
            b = all_bests[n].get(task)
            if b is None:
                row.append(f"{'-':<28s}")
            else:
                row.append(f"{b[0]:>+7.3f} ({b[1]}/{b[2]})".ljust(30))
        print("  ".join(row))

    # Per-cell deltas between first and last snapshot.
    if len(names) >= 2:
        a, b = names[0], names[-1]
        print(f"\nPer-cell deltas {a} -> {b}:\n")
        keys = sorted(set(tables[a].keys()) | set(tables[b].keys()))
        print(f"{'task':<14s} {'model':<14s} {'feature':<18s} "
                  f"{a[:14]:>12s} {b[:14]:>12s} {'delta':>8s}")
        print("-" * 80)
        for k in keys:
            t, m, f = k
            va = tables[a].get(k, {}).get("value")
            vb = tables[b].get(k, {}).get("value")
            if va is None or vb is None:
                continue
            if not isinstance(va, (int, float)) or abs(va) > 1e10:
                va_str = f"{va:>12}"
            else:
                va_str = f"{va:>+12.3f}"
            if not isinstance(vb, (int, float)) or abs(vb) > 1e10:
                vb_str = f"{vb:>12}"
            else:
                vb_str = f"{vb:>+12.3f}"
            if (isinstance(va, (int, float)) and isinstance(vb, (int, float))
                    and abs(va) < 1e10 and abs(vb) < 1e10):
                d = vb - va
                d_str = f"{d:>+8.3f}"
            else:
                d_str = f"{'-':>8s}"
            print(f"{t:<14s} {m:<14s} {f:<18s} {va_str} {vb_str} {d_str}")


if __name__ == "__main__":
    main()
