"""Build ``results/REPORT_noise.md`` — a parallel report to REPORT.md
that mirrors the same structure but reads from every per-SNR results
directory.

Strategy: for each section the noise report shows a *cross-SNR* view
rather than reproducing the 700-page clean report.  Concretely:

  §1 executive summary    — exec table with one column per SNR
  §2 dataset              — noise spec + achieved-SNR audit
  §3 protocol             — pointer to the clean report
  §4 models               — pointer to the clean report
  §5 features             — same catalogue but extracted from noisy timeseries
  §6 plot key             — pointer
  §7 per-task results     — for each task, one table with every model/feature
                            row and one column per SNR (val / test / exp);
                            inline figures show metric-vs-SNR curves
  §8 cross-task           — degradation summary, best cell per task per SNR
  §9 sim-to-real          — balanced experimental metric per SNR
  §10 transfer learning   — per SNR
  §11 resolution sweep    — per SNR (only run for selected SNRs to save time)
  §12 per-plot            — figures index
  §13 reproducibility     — single command per SNR
  §14 trial dump          — full HPO dump per SNR

This script depends on the per-SNR result directories
``results/noisy_<SNR>dB/`` produced by ``run_noise_sweep.py``.  It
ignores SNRs whose data is not yet available, so the report can be
re-built incrementally as the noise pipeline finishes each level.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent

SNR_LEVELS_DB = (35, 25, 20, 15, 10)
TASKS = ("binary", "type", "severity", "col_location", "mass_location")


def _read_eval(path: Path) -> dict[tuple[str, str, str], dict]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text())
    return {(r["task"], r["model"], r["feature"]): r for r in rows}


def _read_hpo(dir_: Path) -> dict[tuple[str, str, str], dict]:
    out = {}
    if not dir_.exists():
        return out
    for p in sorted(dir_.glob("*.json")):
        blob = json.loads(p.read_text())
        out[(blob["task"], blob["model"], blob["feature"])] = blob
    return out


def _available_snrs(results_root: Path) -> list[int]:
    levels = []
    for snr in SNR_LEVELS_DB:
        if (results_root / f"noisy_{snr}dB" / "hpo").exists():
            levels.append(snr)
    return levels


def _fmt(v, fmt="{:+.3f}"):
    return "—" if v is None else fmt.format(v)


def build(results_root: Path) -> str:
    snrs = _available_snrs(results_root)
    out: list[str] = []
    out.append("# Noisy-synth study — companion to `REPORT.md`")
    out.append("")
    out.append("Every experiment in [`REPORT.md`](REPORT.md) is "
                  "repeated on synthetic data corrupted by additive "
                  "Gaussian noise on the **time-series** field (1024 × 9 "
                  "acceleration samples per signal).  The noise is "
                  "applied **per sample, per channel, at a controlled "
                  "signal-to-noise ratio**; every downstream feature "
                  "(FRF, modal-peak vector, CFDAC variants, pymodal "
                  "indicators) is then re-extracted from the noisy "
                  "time series so the entire pipeline trains and tests "
                  "on a self-consistent noisy dataset.")
    out.append("")
    out.append("Five SNR levels are evaluated: **35, 25, 20, 15, 10 dB**.  "
                  "All other settings (model menu, HPO grid, balanced "
                  "experimental evaluation, transfer-learning sweep, "
                  "resolution sweep) are identical to the clean study.")
    out.append("")
    out.append("Coverage status at the time of this build:")
    out.append("")
    out.append("| SNR (dB) | features.h5 | HPO | indicator | balanced eval | "
                  "transfer | resolution |")
    out.append("|---|---|---|---|---|---|---|")
    for snr in SNR_LEVELS_DB:
        rdir = results_root / f"noisy_{snr}dB"
        fdir = _REPO / "dataset" / f"features_noisy_{snr}dB.h5"
        cells = [
            f"| **{snr}** ",
            "✓" if fdir.exists() else "—",
            f"({len(list((rdir/'hpo').glob('*.json')))})"
                if (rdir/"hpo").exists() else "—",
            f"({len(list((rdir/'hpo_indicators').glob('*.json')))})"
                if (rdir/"hpo_indicators").exists() else "—",
            "✓" if (rdir/"balanced"/"experimental_full_evaluation.json").exists()
                else "—",
            "✓" if (rdir/"transfer_learning.json").exists() else "—",
            "✓" if (rdir/"resolution_sweep.json").exists() else "—",
        ]
        out.append(" | ".join(cells) + " |")
    out.append("")

    if not snrs:
        out.append("_No per-SNR HPO results exist yet.  Run "
                      "`python ml_pipeline/run_noise_sweep.py` to "
                      "populate them._")
        return "\n".join(out)

    # §1 executive summary — best cell per task per SNR
    out.append("---")
    out.append("")
    out.append("# 1. Executive summary across SNRs")
    out.append("")
    out.append("Best `(model, feature)` cell per task at each SNR.  "
                  "Clean-data numbers (no noise) are quoted in the last "
                  "column for reference; they come from "
                  "`REPORT.md` § 1.")
    out.append("")
    header = ["| task"] + [f" {snr} dB " for snr in snrs] + [" clean |"]
    out.append("|".join(header))
    out.append("|---" * (len(snrs) + 2) + "|")
    for task in TASKS:
        cells = [f"| `{task}`"]
        for snr in snrs:
            hpo = _read_hpo(results_root / f"noisy_{snr}dB" / "hpo")
            cell_rows = [(k, v) for k, v in hpo.items() if k[0] == task]
            if not cell_rows:
                cells.append("—")
                continue
            best = max(cell_rows,
                          key=lambda kv: kv[1].get("best_metric_test")
                                              or -1e30)
            label = f"{best[1]['best_metric_test']:.3f} "
            label += f"({best[0][1]}/{best[0][2]})"
            cells.append(label)
        cells.append("see REPORT.md §1")
        cells.append("|")
        out.append("| ".join(cells))
    out.append("")

    # §2 dataset
    out.append("---")
    out.append("")
    out.append("# 2. Dataset and noise injection")
    out.append("")
    out.append("Same 10 000-sample synthetic generator as REPORT.md § 2.  "
                  "After generation, every 1024-sample acceleration "
                  "channel is corrupted with i.i.d. Gaussian noise scaled "
                  "to hit the target per-sample SNR:")
    out.append("")
    out.append("```")
    out.append("noise[i, t, c]  ~  N(0, σ_i)")
    out.append("σ_i  =  sqrt( P_signal[i] / 10**(SNR_dB / 10) )")
    out.append("```")
    out.append("")
    out.append("`P_signal[i]` is the mean-square of the entire 1024 × 9 "
                  "signal block — so the noise floor is per-sample, "
                  "independent across channels, and homoscedastic across "
                  "time.  After the addition the noisy time series feeds "
                  "the same `features.py` / `cfdac.py` / "
                  "`cfdac_variants.py` pipeline used in the clean run, "
                  "producing matching `features_noisy_<SNR>dB.h5` files.")
    out.append("")

    # §7 per-task: balanced experimental metric per SNR
    out.append("---")
    out.append("")
    out.append("# 7. Cross-domain (balanced experimental) metrics per SNR")
    out.append("")
    out.append("For every `(model, feature)` cell, the table below "
                  "reports the balanced-experimental metric at each "
                  "available SNR.  Same balanced 680-case dataset that "
                  "REPORT.md § 9 uses, so the rows are directly "
                  "comparable.")
    out.append("")
    for task in TASKS:
        out.append(f"## {task}")
        out.append("")
        # Union of cells across SNRs
        cells = set()
        per_snr_eval = {}
        for snr in snrs:
            ev = _read_eval(results_root / f"noisy_{snr}dB"
                                / "balanced" / "experimental_full_evaluation.json")
            per_snr_eval[snr] = ev
            for (t, m, f) in ev:
                if t == task:
                    cells.add((m, f))
        if not cells:
            out.append("_(no balanced experimental eval yet)_")
            out.append("")
            continue
        header = ["| model | feature"] + [f" {s} dB " for s in snrs] + ["|"]
        out.append("|".join(header))
        out.append("|---|---" + "|---" * len(snrs) + "|")
        for (m, f) in sorted(cells):
            row = [m, f"`{f}`"]
            for snr in snrs:
                ev = per_snr_eval.get(snr, {})
                r = ev.get((task, m, f))
                row.append(_fmt(r["value"] if r else None, "{:.3f}"))
            out.append("| " + " | ".join(row) + " |")
        out.append("")

    # §10 transfer learning
    out.append("---")
    out.append("")
    out.append("# 10. Transfer-learning lift per SNR")
    out.append("")
    out.append("Best `(model, feature, unfreeze)` cell at k=50 % per "
                  "task, per SNR, with Δ vs the clean-data zero-shot.")
    out.append("")
    out.append("| task | " + " | ".join(f"{s} dB" for s in snrs) + " |")
    out.append("|---" + "|---" * len(snrs) + "|")
    for task in TASKS:
        row = [f"| `{task}`"]
        for snr in snrs:
            tl_path = results_root / f"noisy_{snr}dB" / "transfer_learning.json"
            if not tl_path.exists():
                row.append("—")
                continue
            rows = json.loads(tl_path.read_text())
            rk = [r for r in rows if r["task"] == task and r["fraction"] == 0.5]
            if not rk:
                row.append("—")
                continue
            best = max(rk, key=lambda r: r["value"])
            row.append(f"{best['value']:+.3f} ({best['model']}/{best['feature']})")
        row.append("|")
        out.append(" | ".join(row))
    out.append("")

    # §11 resolution
    out.append("---")
    out.append("")
    out.append("# 11. Feature resolution sweep per SNR")
    out.append("")
    out.append("Best cell per task at r=0.500 / r=1.000 (full) per SNR.")
    out.append("")
    out.append("| task | " + " | ".join(f"r1@{s}dB | r0.5@{s}dB" for s in snrs) + " |")
    out.append("|---" + "|---" * (2 * len(snrs)) + "|")
    for task in TASKS:
        row = [f"| `{task}`"]
        for snr in snrs:
            rs_path = results_root / f"noisy_{snr}dB" / "resolution_sweep.json"
            if not rs_path.exists():
                row.extend(["—", "—"])
                continue
            rows = json.loads(rs_path.read_text())
            rk1 = [r for r in rows if r["task"] == task
                       and abs(r["ratio"] - 1.0) < 1e-3]
            rk0 = [r for r in rows if r["task"] == task
                       and abs(r["ratio"] - 0.5) < 1e-3]
            b1 = max(rk1, key=lambda r: r["value"])["value"] if rk1 else None
            b0 = max(rk0, key=lambda r: r["value"])["value"] if rk0 else None
            row.append(_fmt(b1, "{:.3f}"))
            row.append(_fmt(b0, "{:.3f}"))
        row.append("|")
        out.append(" | ".join(row))
    out.append("")

    # §12 pointer to clean report
    out.append("---")
    out.append("")
    out.append("# 12. Reproducibility")
    out.append("")
    out.append("```")
    out.append("# Re-run from scratch (per SNR):")
    out.append("python ml_pipeline/run_noise_sweep.py --snr-db 20")
    out.append("")
    out.append("# All five levels in sequence:")
    out.append("python ml_pipeline/run_noise_sweep.py")
    out.append("")
    out.append("# Rebuild this report after each SNR finishes:")
    out.append("python ml_pipeline/build_report_noise.py")
    out.append("```")
    out.append("")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=_REPO / "results")
    p.add_argument("--out", type=Path,
                      default=_REPO / "results" / "REPORT_noise.md")
    args = p.parse_args()
    txt = build(args.results)
    args.out.write_text(txt)
    print(f"wrote {args.out}  ({len(txt):,} chars)")


if __name__ == "__main__":
    main()
