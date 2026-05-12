"""Emit Markdown fragments for the transfer-learning (§10) and the
feature-resolution-sweep (§11) sections of REPORT.md.

Inputs:
    results/transfer_learning.json   (transfer_learn.py output)
    results/resolution_sweep.json    (resolution_sweep.py output)

Outputs:
    /tmp/section_transfer.md
    /tmp/section_resolution.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

FRACTIONS = (0.10, 0.20, 0.30, 0.40, 0.50)
TASKS = ("binary", "type", "severity", "col_location", "mass_location")


def _fmt(v, fmt="{:+.3f}"):
    return "—" if v is None else fmt.format(v)


def build_transfer_section(rows: list[dict]) -> str:
    if not rows:
        return "_transfer-learning sweep not yet run_"
    out: list[str] = []
    out.append("Synth-trained Torch models are fine-tuned on a fraction "
                  "of the **balanced experimental data** (40-per-cell IQS "
                  "subset, see § 2.6) and evaluated on the held-out "
                  "remainder.  Two unfreeze depths are reported:\n")
    out.append("* `head` — only the model's final `Linear` layer is "
                  "trainable;")
    out.append("* `head_proj` — the entire `head` sub-module is "
                  "trainable (final Linear plus the projection block "
                  "that lives between global-pool and output).")
    out.append("")
    out.append("Five fractions of experimental data are used for fine-"
                  "tuning: `k ∈ {10, 20, 30, 40, 50} %`, drawn with a "
                  "stratified split (seed `20260511 + 100·k`).  The "
                  "remaining `100 − k` % is the held-out test slice.")
    out.append("")
    out.append("### 10.1 Best transfer-learning cell per task, per k")
    out.append("")
    out.append("| task | k=10 % | k=20 % | k=30 % | k=40 % | k=50 % |")
    out.append("|---|---|---|---|---|---|")
    for task in TASKS:
        cells = [f"| `{task}` "]
        for k in FRACTIONS:
            rk = [r for r in rows if r["task"] == task and r["fraction"] == k]
            if not rk:
                cells.append("| — ")
                continue
            best = max(rk, key=lambda r: r["value"])
            cells.append(
                f"| **{best['value']:+.3f}** "
                f"(`{best['model']}`/`{best['feature']}`/"
                f"`{best['unfreeze']}`) "
            )
        cells.append("|")
        out.append("".join(cells))
    out.append("")
    out.append("### 10.2 Headline lifts vs zero-shot synth model")
    out.append("")
    out.append("Compared with the zero-shot best (§ 9, balanced):")
    out.append("")
    out.append("| task | zero-shot exp | best fine-tuned at k=50 % | Δ |")
    out.append("|---|---|---|---|")
    zero_shot = {
        "binary":         (0.941, "MLP/modal"),
        "type":           (0.403, "2-D CNN/cfdac_mag"),
        "severity":       (0.022, "XGB/modal"),
        "col_location":   (0.287, "1-D CNN/frf_mag"),
        "mass_location":  (0.338, "2-D CNN/cfdac_all"),
    }
    for task in TASKS:
        rk = [r for r in rows if r["task"] == task and r["fraction"] == 0.5]
        if not rk:
            continue
        best = max(rk, key=lambda r: r["value"])
        zs_val, zs_cell = zero_shot[task]
        delta = best["value"] - zs_val
        out.append(
            f"| `{task}` | {zs_val:+.3f} ({zs_cell}) | "
            f"{best['value']:+.3f} (`{best['model']}`/`{best['feature']}`/"
            f"`{best['unfreeze']}`) | "
            f"**{delta:+.3f}** |"
        )
    out.append("")
    out.append("### 10.3 Per-cell trend with k")
    out.append("")
    out.append("Average metric per `(task, model, feature)` cell across "
                  "the 5 fractions (mean of `head` and `head_proj`); the "
                  "best cell per task by `k=50 %` is bolded.")
    out.append("")
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[(r["task"], r["model"], r["feature"])].append(r)
    for task in TASKS:
        cells = sorted(
            [(t, m, f) for (t, m, f) in by_cell if t == task],
            key=lambda k: -max(r["value"]
                                       for r in by_cell[k]
                                       if r["fraction"] == 0.5)
        )
        if not cells:
            continue
        out.append(f"#### {task}")
        out.append("")
        out.append("| model | feature | k=10 % | k=20 % | k=30 % | "
                      "k=40 % | k=50 % |")
        out.append("|---|---|---|---|---|---|---|")
        for (_, model, feat) in cells:
            row = ["", model, f"`{feat}`"]
            for k in FRACTIONS:
                vals = [r["value"] for r in by_cell[(task, model, feat)]
                          if r["fraction"] == k]
                if vals:
                    v = sum(vals) / len(vals)
                    row.append(f"{v:+.3f}")
                else:
                    row.append("—")
            out.append("| " + " | ".join(row[1:]) + " |")
        out.append("")
    return "\n".join(out)


def build_resolution_section(rows: list[dict]) -> str:
    if not rows:
        return "_resolution sweep still running; this section will be filled in after `python ml_pipeline/build_sweep_sections.py` is re-run._"
    out: list[str] = []
    out.append("Every `(task, model, feature)` cell is retrained on the "
                  "balanced synthetic data at five resolution ratios "
                  "(applied via `scipy.signal.resample`, which is a "
                  "Fourier-domain decimation + low-pass that preserves "
                  "the spectrum at the new Nyquist).  Ratios: "
                  "**1.000, 0.875, 0.750, 0.625, 0.500**.  After "
                  "resampling, the feature axes carry the following "
                  "lengths:")
    out.append("")
    out.append("| feature    | full | 0.875 | 0.750 | 0.625 | 0.500 |")
    out.append("|---|---|---|---|---|---|")
    out.append("| `modal`    | 81  | 71  | 61  | 51  | 41  |")
    out.append("| `frf_mag`  | 381 | 333 | 286 | 238 | 191 |")
    out.append("| `timeseries` | 1024 | 896 | 768 | 640 | 512 |")
    out.append("| `cfdac_*`  | 128² | 112² | 96² | 80² | 64² |")
    out.append("")
    out.append("### 11.1 Best cell per task, per ratio")
    out.append("")
    out.append("| task | r=1.000 | r=0.875 | r=0.750 | r=0.625 | r=0.500 |")
    out.append("|---|---|---|---|---|---|")
    for task in TASKS:
        cells = [f"| `{task}` "]
        for r_ratio in (1.000, 0.875, 0.750, 0.625, 0.500):
            rk = [r for r in rows
                       if r["task"] == task and abs(r["ratio"] - r_ratio) < 1e-3]
            if not rk:
                cells.append("| — ")
                continue
            best = max(rk, key=lambda r: r["value"])
            cells.append(f"| **{best['value']:+.3f}** "
                              f"(`{best['model']}`/`{best['feature']}`) ")
        cells.append("|")
        out.append("".join(cells))
    out.append("")
    out.append("### 11.2 Per-cell metric vs ratio")
    out.append("")
    by_cell = defaultdict(dict)
    for r in rows:
        by_cell[(r["task"], r["model"], r["feature"])][r["ratio"]] = r["value"]
    for task in TASKS:
        cells = sorted(
            [(t, m, f) for (t, m, f) in by_cell if t == task],
            key=lambda k: -by_cell[k].get(1.000, -1e30),
        )
        if not cells:
            continue
        out.append(f"#### {task}")
        out.append("")
        out.append("| model | feature | r=1.000 | r=0.875 | r=0.750 | "
                      "r=0.625 | r=0.500 |")
        out.append("|---|---|---|---|---|---|---|")
        for (_, model, feat) in cells:
            row = [model, f"`{feat}`"]
            for ratio in (1.000, 0.875, 0.750, 0.625, 0.500):
                v = by_cell[(task, model, feat)].get(ratio)
                row.append(_fmt(v, "{:+.3f}"))
            out.append("| " + " | ".join(row) + " |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=_REPO / "results")
    p.add_argument("--out", type=Path, default=Path("/tmp"))
    args = p.parse_args()

    tl_path = args.results / "transfer_learning.json"
    rs_path = args.results / "resolution_sweep.json"
    tl_rows = json.loads(tl_path.read_text()) if tl_path.exists() else []
    rs_rows = json.loads(rs_path.read_text()) if rs_path.exists() else []

    (args.out / "section_transfer.md").write_text(
        build_transfer_section(tl_rows))
    (args.out / "section_resolution.md").write_text(
        build_resolution_section(rs_rows))
    print(f"wrote /tmp/section_transfer.md  ({len(tl_rows)} rows)")
    print(f"wrote /tmp/section_resolution.md ({len(rs_rows)} rows)")


if __name__ == "__main__":
    main()
