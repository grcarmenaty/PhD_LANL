"""Generate the in-domain (synthetic-test) companion report from zoo_summary.json,
for a given resolution. Mirrors the consolidated report's data so both are
reproducible. Writes results/REPORT_synth{sfx}.md (sfx='' for 1601, '_128' else).

Run: python ml_pipeline/build_hires_synth_report.py --res {1601,128}
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import defaultdict

_REPO = Path(__file__).resolve().parent.parent
TASKS = ["mass_location", "is_mass", "binary", "is_pristine", "is_bolt", "type",
         "is_hole", "is_crack", "col_location", "severity"]
CHANCE = {"binary":0.5,"is_pristine":0.5,"is_bolt":0.5,"is_crack":0.5,"is_hole":0.5,
          "is_mass":0.5,"type":0.2,"col_location":1/6,"mass_location":0.25,"severity":None}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--res", type=int, default=1601); a = ap.parse_args()
    res = a.res; sfx = "" if res == 1601 else f"_{res}"
    figrel = "hires" if res == 1601 else f"hires{res}"
    S = json.loads((_REPO/"results_hires"/"zoo_summary.json").read_text())
    cells = defaultdict(list)
    for v in S.values():
        if v.get("res") == res:
            cells[v["task"]].append(v)

    def best_indomain(recs):
        return max(recs, key=lambda r: (r.get("synth") or -9))
    def best_exp(recs):
        reg = recs[0]["kind"] == "reg"
        key = (lambda r: r.get("exp_r2", -9)) if reg else (lambda r: r.get("exp_bal_acc", 0))
        return max(recs, key=key), key

    out = []; A = out.append
    A(f"# LANL 3SBB — Synthetic-domain (in-domain) results @{res}")
    A(f"**Companion to** [`REPORT_CONSOLIDATED{sfx}.md`](REPORT_CONSOLIDATED{sfx}.md) (the full"
      " cross-domain / experimental study). **Date:** 2026-06-09.")
    A(f"**Scope.** How well the **{res}-bin CFDAC model zoo** learns each diagnosis task *within the"
      " synthetic domain* — trained and tested on held-out synthetic data (the upper bound, before any"
      " sim-to-real transfer).\n")
    A("---\n## Why this report")
    A(f"The consolidated report measures **zero-shot transfer to real data**. This one isolates the"
      " **in-domain ceiling**: with synth-only training and a held-out synth test fold, *can the models learn"
      " the task at all, and how well?* The gap between this report (in-domain) and the consolidated report"
      " (experimental) **is** the sim-to-real problem, quantified per task. Same zoo, same"
      f" {res}-bin features/models, same 70/15/15 split and train-to-convergence protocol. Metric: held-out"
      " **synthetic macro-F1** (classification) or **R²** (severity).\n")
    A(f"---\n## In-domain results (held-out synthetic test, {res})\n")
    A("| task | chance | best in-domain (macro-F1 / R²) | best in-domain cell |")
    A("|---|---|---|---|")
    for t in TASKS:
        if t not in cells: continue
        b = best_indomain(cells[t]); ch = CHANCE[t]
        A(f"| **{t}** | {('%.2f'%ch) if ch else '—'} | **{(b.get('synth') or 0):.2f}** | "
          f"`{b['model']} / {b['feature']}` |")
    A("\n### Observations")
    A("- **Most tasks are learned well synthetically** — the models and features are not the bottleneck; the"
      " synthetic task is solvable. The differences only emerge under transfer (consolidated report).")
    A("- **`col_location` is the in-domain hard case**: symmetric crack/hole damage makes the two column ends"
      " nearly degenerate in the linear reduced model — an intrinsic ceiling, not a learning failure.")
    A("- **Severity is learnable but only moderate** in-domain — a real but imperfect regression even before"
      " transfer.\n")
    A("---\n## The sim-to-real gap (in-domain → experiment)\n")
    A("Best cell per task; in-domain macro-F1/R² vs zero-shot balanced-acc / R²:\n")
    A("| task | in-domain | experiment (zero-shot) | metric |")
    A("|---|---|---|---|")
    for t in TASKS:
        if t not in cells: continue
        b, key = best_exp(cells[t]); reg = b["kind"] == "reg"
        ind = b.get("synth") or 0.0
        A(f"| {t} | {ind:.2f} | {key(b):+.2f} | {'R²' if reg else 'bal-acc'} |")
    A(f"\n![in-domain vs zero-shot, best cell per task]({figrel}/zoo_synth_vs_exp.png)\n")
    A("**Interpretation.** The drop from this report to the experimental one is the sim-to-real gap. It is"
      " **largest where the synthetic model is most confident** (a hallmark of covariate shift: the model"
      " locks onto synthetic spectral structure that does not match reality). The smaller drops are the"
      " severe-damage detectors that survive transfer (and improve further at high severity — see the DT"
      f" sweep in `REPORT_CONSOLIDATED{sfx}.md`).\n")
    A("---\n## Takeaways")
    A("1. **In-domain is essentially solved** for detection/typing; the challenge is transfer.")
    A("2. **The most over-confident synthetic tasks transfer worst** — chase domain adaptation, not in-domain"
      " accuracy.")
    A("3. **`col_location` and `severity`** are limited *in-domain too*, so their poor transfer is partly an"
      " intrinsic ceiling, not only covariate shift.\n")
    A(f"*Experimental transfer, the DT severity sweep, and the per-representation analysis are in"
      f" [`REPORT_CONSOLIDATED{sfx}.md`](REPORT_CONSOLIDATED{sfx}.md).*")

    (_REPO/"results"/f"REPORT_synth{sfx}.md").write_text("\n".join(out)+"\n")
    print(f"wrote results/REPORT_synth{sfx}.md (res={res})")


if __name__ == "__main__":
    main()
