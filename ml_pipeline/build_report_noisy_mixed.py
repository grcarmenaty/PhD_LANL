"""Build ``results/REPORT_noisy_mixed.md`` by cloning ``REPORT.md`` and
re-running ``integrate_report.py`` against the noisy-mixed pipeline
outputs in ``results/noisy_mixed/``.

The handwritten sections (§§ 2-6 dataset / models / features / plot
key) are inherited verbatim from ``REPORT.md``; only the cross-model
comparison tables (§§ 7.1.19, 7.2.19, 7.3.19, 7.4.20, 7.5.19) and the
22 per-indicator subsections (§ 7.6.7-7.6.28) are regenerated from
the noisy_mixed HPO and balanced-experimental evaluation artefacts.

This is the step-8 entry point for ``run_noisy_mixed_pipeline.sh``;
``build_report_noise.py`` is reserved for the separate per-SNR sweep
launched by ``run_noise_sweep.py``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ml_pipeline.build_report_sections import (  # noqa: E402
    INDICATOR_DESC,
    INDICATOR_ORDER,
    _fmt,
)


_BANNER = (
    "> **NOTE - noisy-mixed variant.**  This report has the same "
    "structure as [`REPORT.md`](REPORT.md), but every cross-model "
    "table and per-indicator section below was rebuilt from the "
    "**noisy_mixed** pipeline outputs in `results/noisy_mixed/` "
    "(mixed-SNR additive Gaussian noise on the time series).  All "
    "hand-written sections (dataset, models, features, plot key) "
    "are inherited verbatim from `REPORT.md`; only the auto-generated "
    "results tables reflect the noisy run.\n\n"
)


def _write_banner(report_path: Path) -> None:
    text = report_path.read_text()
    anchor = "A single document"
    i = text.find(anchor)
    if i < 0 or _BANNER.strip().splitlines()[0][:30] in text:
        return
    report_path.write_text(text[:i] + _BANNER + text[i:])


def _build_indicator_subs(results_root: Path, out_path: Path) -> None:
    hpo_ind_dir = results_root / "hpo_indicators"
    full_path = results_root / "indicator_predictions_full.json"
    full_rows = json.loads(full_path.read_text()) if full_path.exists() else []
    exp_lookup = {(r["indicator"], r["model"], r["feature"]): r
                       for r in full_rows}

    pieces: list[str] = []
    for k, ind in enumerate(INDICATOR_ORDER, start=7):
        rows = []
        for model in ("rf", "xgb", "mlp"):
            p = hpo_ind_dir / f"{ind}__{model}__modal.json"
            if not p.exists():
                continue
            blob = json.loads(p.read_text())
            erow = exp_lookup.get((ind, model, "modal"))
            rows.append({
                "model": model,
                "val":  blob.get("best_metric_val"),
                "test": blob.get("best_metric_test"),
                "hp":   blob.get("best_hyperparams"),
                "exp":  erow["exp_R2"] if erow else None,
                "mae":  erow["exp_MAE"] if erow else None,
            })

        sane = [r for r in rows if r["exp"] is not None and r["exp"] > -1e3]
        if sane:
            best = max(sane, key=lambda r: r["exp"])
        elif rows:
            best = max(rows, key=lambda r: r["test"] or -1e30)
        else:
            best = None

        if best is None:
            transfer = "No cross-domain metric recorded for this indicator."
        elif best["exp"] is None:
            transfer = (f"Best in-domain cell is {best['model']}, "
                              f"synth test {_fmt(best['test'])}.")
        elif best["exp"] >= 0.49:
            transfer = (f"**Transfers cleanly** (exp R2 "
                              f"{_fmt(best['exp'], '{:+.3f}')} for "
                              f"{best['model']}/modal); usable as a "
                              "cross-domain damage proxy.")
        elif best["exp"] >= 0:
            transfer = (f"Marginal cross-domain transfer "
                              f"(exp R2 {_fmt(best['exp'], '{:+.3f}')}); "
                              "no better than the experimental mean.")
        else:
            transfer = (f"**Does not transfer** - exp R2 "
                              f"{_fmt(best['exp'], '{:+.3f}')} for "
                              f"{best['model']}/modal; synth-only signal.")

        pieces.append(f"### 7.6.{k} `{ind}`")
        pieces.append("")
        pieces.append(INDICATOR_DESC[ind])
        pieces.append("")
        pieces.append("| model | val R2 | test R2 | exp R2 | exp MAE | best HPs |")
        pieces.append("|---|---|---|---|---|---|")
        for r in rows:
            pieces.append(
                f"| {r['model']:<3} | {_fmt(r['val'])} | "
                f"{_fmt(r['test'])} | {_fmt(r['exp'], '{:+.3f}')} | "
                f"{_fmt(r['mae'], '{:.4g}')} | `{r['hp']}` |"
            )
        pieces.append("")
        if best is not None:
            scatter = f"figures/indicators/scatter/{ind}_{best['model']}_modal.png"
            hpo_img = f"figures/indicators/hpo/{ind}__{best['model']}__modal.png"
            pieces.append(f"![{ind} scatter - {best['model']}/modal]({scatter})")
            pieces.append(f"![{ind} HPO - {best['model']}/modal]({hpo_img})")
            pieces.append("")
        pieces.append(transfer)
        pieces.append("")

    out_path.write_text("\n".join(pieces))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path,
                      default=_REPO / "results" / "noisy_mixed")
    p.add_argument("--source-report", type=Path,
                      default=_REPO / "results" / "REPORT.md")
    p.add_argument("--out", type=Path,
                      default=_REPO / "results" / "REPORT_noisy_mixed.md")
    args = p.parse_args()

    shutil.copyfile(args.source_report, args.out)
    _write_banner(args.out)

    tmp_subs = _REPO / "results" / ".noisy_mixed_indicator_subs.md"
    _build_indicator_subs(args.results, tmp_subs)

    # The noisy-mixed exp figures live under results/noisy_mixed/figures_exp/
    # but the report itself lives in results/ - the relative path from the
    # report has to walk into the results/noisy_mixed/ subdirectory.
    exp_fig_rel = str(args.results.relative_to(args.out.parent) / "figures_exp")

    cmd = [
        sys.executable, "-m", "ml_pipeline.integrate_report",
        "--report", str(args.out),
        "--hpo", str(args.results / "hpo"),
        "--exp", str(args.results / "experimental_full_evaluation.json"),
        "--ind-subs", str(tmp_subs),
        "--ind-vs-dmg", str(_REPO / "results" / "indicator_vs_damage_table.md"),
        "--exp-figures-dir", exp_fig_rel,
    ]
    subprocess.check_call(cmd, cwd=_REPO)

    tmp_subs.unlink(missing_ok=True)
    n_lines = sum(1 for _ in args.out.open())
    print(f"wrote {args.out}  ({n_lines:,} lines)")


if __name__ == "__main__":
    main()
