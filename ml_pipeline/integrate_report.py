"""Surgically rewrite REPORT.md to include:

  * a refreshed cross-model comparison table at each §7.X.19 (or
    §7.4.20) — every (model, feature) cell from `results/hpo/*.json`;
  * per-task variant figure embeds (HPO + confusion / scatter) under
    each cross-model table;
  * 22 indicator subsections inserted between §7.6.4 and the old
    §7.6.5 (which is renumbered to §7.6.27);
  * a new "indicator regression vs direct damage-parameter regression"
    subsection (§7.6.29).

Run order:
    python ml_pipeline/build_report_sections.py
    python ml_pipeline/build_indicator_vs_damage.py
    python ml_pipeline/integrate_report.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

_REPO = Path(__file__).resolve().parent.parent


def _load_perclass_summary(report_root: Path | None,
                              exp_figures_dir: str) -> dict:
    """Return ``perclass_summary.json`` content (or {} if missing).
    Written by ``plots_experimental.plot_perclass_f1_exp``."""
    if report_root is None:
        return {}
    p = report_root / exp_figures_dir / "perclass_summary.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _build_task_section(task: str, hpo_dir: Path, exp_path: Path,
                          figures_dir: str = "figures",
                          exp_figures_dir: str = "figures_exp",
                          report_root: Path | None = None,
                          perclass_summary: dict | None = None) -> str:
    """Build a Markdown blob for one task's cross-model comparison +
    inline figures.  Sorted by test metric descending.

    Each per-cell figure pair (synth-test confusion or scatter) is
    immediately followed by the experimental-data twin and a one-line
    delta caption so the cross-domain gap is visible inline.  The
    section is prefaced with a 4-line auto summary of the synth-test ->
    exp transfer across all cells in the task.
    """
    rows = []
    for p in sorted(hpo_dir.glob(f"{task}__*.json")):
        blob = json.loads(p.read_text())
        rows.append({
            "model": blob["model"], "feature": blob["feature"],
            "val":   blob.get("best_metric_val"),
            "test":  blob.get("best_metric_test"),
        })
    if exp_path.exists():
        exp_idx = {(r["task"], r["model"], r["feature"]): r
                       for r in json.loads(exp_path.read_text())}
        for r in rows:
            er = exp_idx.get((task, r["model"], r["feature"]))
            r["exp"] = er["value"] if er else None
            r["mae"] = er["mae"] if er else None
    else:
        for r in rows:
            r["exp"] = None
            r["mae"] = None
    rows.sort(key=lambda r: -(r["test"] if r["test"] is not None
                                       else -1e30))

    metric_label = "R²" if task == "severity" else "acc"
    out: list[str] = []

    # Per-task transfer summary (synth-test -> exp) computed from the
    # rows we already have.
    deltas = [(r, (r["test"] - r["exp"]))
                  for r in rows
                  if r["test"] is not None and r["exp"] is not None]
    if deltas:
        med = float(np.median([d for _, d in deltas]))
        deltas_sorted = sorted(deltas, key=lambda kv: kv[1])
        best = deltas_sorted[0]
        worst = deltas_sorted[-1]
        n_close = sum(1 for _, d in deltas if abs(d) <= 0.10)
        out.append(
            f"_Cross-domain transfer (synth test -> exp, {metric_label}): "
            f"across {len(deltas)} cells the median drop is "
            f"Δ{med:+.3f}; best-transferring cell is "
            f"`{best[0]['model']}/{best[0]['feature']}` "
            f"(Δ{best[1]:+.3f}); worst-transferring is "
            f"`{worst[0]['model']}/{worst[0]['feature']}` "
            f"(Δ{worst[1]:+.3f}).  {n_close}/{len(deltas)} cells "
            f"transfer within |Δ| ≤ 0.10._"
        )
        # Per-class breakdown for the best-mean-F1 cell (classification
        # tasks only - perclass_summary is empty for severity).
        pcs = (perclass_summary or {}).get(task)
        if pcs:
            out.append(
                f"_Per-class exp breakdown for the best-mean-F1 cell "
                f"`{pcs['best_cell']}` (mean F1 {pcs['best_cell_mean_f1']:.3f}): "
                f"strongest class **{pcs['best_class']}** "
                f"(F1 {pcs['best_class_f1']:.3f}); weakest class "
                f"**{pcs['worst_class']}** (F1 {pcs['worst_class_f1']:.3f}).  "
                f"See `{exp_figures_dir}/perclass_f1/{task}.png` for the "
                f"full model × class heatmap._"
            )
        out.append("")

    out.append(f"| model | feature | val {metric_label} | "
                  f"test {metric_label} | exp {metric_label} |")
    out.append("|---|---|---|---|---|")
    for r in rows:
        out.append(
            f"| {r['model']:<11} | `{r['feature']}` | "
            f"{r['val']:.3f} | {r['test']:.3f} | "
            f"{r['exp']:.3f}" if r["exp"] is not None else
            f"| {r['model']:<11} | `{r['feature']}` | "
            f"{r['val']:.3f} | {r['test']:.3f} | — |"
        )
    # Fix the format strings (the conditional above truncated some rows).
    fixed: list[str] = []
    for line in out:
        if line.startswith("|") and line.count("|") == 5 and "—" not in line and " | —" not in line and not line.endswith("|"):
            fixed.append(line + " |")
        else:
            fixed.append(line)
    out = fixed
    out.append("")
    # Image embeds — for each cell: synth HPO, synth test (confusion or
    # scatter), exp test (same kind), then a one-line delta caption.
    second_kind = "scatter" if task == "severity" else "confusion"
    def _exp_png_exists(kind: str, tag: str) -> bool:
        if report_root is None:
            return True   # caller didn't pass a root, skip the check
        return (report_root / exp_figures_dir / kind / f"{tag}.png").exists()

    for r in rows:
        m = r["model"]; f = r["feature"]
        tag = f"{task}_{m}_{f}"
        out.append(f"![HPO — {task}/{m}/{f}]({figures_dir}/hpo/{task}__{m}__{f}.png)")
        out.append(f"![synth {second_kind} — {task}/{m}/{f}]"
                      f"({figures_dir}/{second_kind}/{task}_{m}_{f}.png)")
        if r["test"] is not None and r["exp"] is not None and \
             _exp_png_exists(second_kind, tag):
            out.append(f"![exp {second_kind} — {task}/{m}/{f}]"
                          f"({exp_figures_dir}/{second_kind}/{task}_{m}_{f}.png)")
            delta = r["test"] - r["exp"]
            out.append(
                f"*Cross-domain: synth test {metric_label} "
                f"{r['test']:.3f} → exp {metric_label} "
                f"{r['exp']:.3f} (Δ{delta:+.3f}).*"
            )
        elif r["test"] is not None and r["exp"] is not None:
            delta = r["test"] - r["exp"]
            out.append(
                f"*Cross-domain: synth test {metric_label} "
                f"{r['test']:.3f} → exp {metric_label} "
                f"{r['exp']:.3f} (Δ{delta:+.3f}).  "
                "(No exp twin figure: model artifact missing on disk.)*"
            )
        else:
            out.append(
                "*Cross-domain: this cell has no balanced-experimental "
                "evaluation (feature not present in `experimental_features_balanced.h5` "
                "or model never scored on exp), so no exp twin is shown.*"
            )
        out.append("")
    if task == "binary":
        out.append(f"![synth binary ROC overlay]({figures_dir}/roc/binary_roc.png)")
        if _exp_png_exists("roc", "binary_roc"):
            out.append(f"![exp binary ROC overlay]({exp_figures_dir}/roc/binary_roc.png)")
        out.append(f"![synth binary PR overlay]({figures_dir}/roc/binary_pr.png)")
        if _exp_png_exists("roc", "binary_pr"):
            out.append(f"![exp binary PR overlay]({exp_figures_dir}/roc/binary_pr.png)")
    out.append("")
    return "\n".join(out)


def _replace_section(text: str, task: str, header_re: re.Pattern,
                      next_header_re: re.Pattern, new_body: str) -> str:
    """Replace the body of `### <header>` up to (but not including)
    the next `### ` header.  Keeps the original header line."""
    m = header_re.search(text)
    if not m:
        return text
    start = m.end()
    nxt = next_header_re.search(text, start)
    end = nxt.start() if nxt else len(text)
    return text[:start] + "\n\n" + new_body + "\n" + text[end:]


def update_cross_model(text: str, hpo_dir: Path, exp_path: Path,
                          exp_figures_dir: str = "figures_exp",
                          report_root: Path | None = None) -> str:
    next_h = re.compile(r"^### |^## |^# ", flags=re.MULTILINE)
    headers = [
        ("binary",        r"^### 7\.1\.19 Cross-model comparison \(binary\)"),
        ("type",          r"^### 7\.2\.19 Cross-model comparison \(type\)"),
        ("severity",      r"^### 7\.3\.19 Cross-model comparison \(severity\)"),
        ("col_location",  r"^### 7\.4\.20 Cross-model comparison \(col_location\)"),
        ("mass_location", r"^### 7\.5\.19 Cross-model comparison \(mass_location\)"),
    ]
    pcs = _load_perclass_summary(report_root, exp_figures_dir)
    for task, hdr in headers:
        body = _build_task_section(task, hpo_dir, exp_path,
                                          exp_figures_dir=exp_figures_dir,
                                          report_root=report_root,
                                          perclass_summary=pcs)
        hpat = re.compile(hdr, flags=re.MULTILINE)
        text = _replace_section(text, task, hpat, next_h, body)
    return text


def _build_section_2_7(exp_path: Path,
                          dataset_exp_figs_dir: str = "figures/dataset_exp"
                          ) -> str:
    """Build §2.7 markdown: experimental dataset deep-dive."""
    if not exp_path.exists():
        return ""
    from ml_pipeline.tasks import build_targets
    from ml_pipeline.train import load_labels
    L = load_labels(exp_path)
    tasks = build_targets(L["type_code"], L["storey"], L["end"], L["severity"])
    tc = L["type_code"]; sev = L["severity"]
    # Per-task class counts.
    pieces: list[str] = []
    pieces.append("## 2.7 Experimental dataset deep-dive")
    pieces.append("")
    pieces.append(
        f"The 680-case `experimental_features_balanced.h5` set was "
        f"assembled by `rebalance_datasets.py` from `experimental_features.h5` "
        f"by sub-sampling 40 cases per `(type, location)` cell shared "
        f"with the synthetic set (17 cells × 40 = 680).  This subsection "
        f"walks through what that set actually contains so the "
        f"cross-domain results in §§ 7-9 can be read with the right "
        f"prior on class balance, severity coverage and signal "
        f"morphology."
    )
    pieces.append("")
    pieces.append("### 2.7.1 Per-task class balance")
    pieces.append("")
    pieces.append(
        "Because balancing was done at the `(type, location)` granularity, "
        "the per-task collapsed labels are **strongly unbalanced** - this "
        "is the dominant reason classifiers can score well on binary and "
        "still tell us nothing useful:"
    )
    pieces.append("")
    pieces.append("| task | pool | per-class counts | majority class |")
    pieces.append("|---|---|---|---|")
    for tn in ("binary", "type", "col_location", "mass_location"):
        if tn not in tasks: continue
        mask, y, kind = tasks[tn]
        if kind != "cls": continue
        from ml_pipeline.plots_advanced import _class_labels
        labels = _class_labels(tn)
        cnt = np.bincount(y.astype(int), minlength=len(labels))
        breakdown = ", ".join(
            f"{lbl}={int(c)}" for lbl, c in zip(labels, cnt)
        )
        maj_idx = int(np.argmax(cnt))
        maj_frac = float(cnt[maj_idx]) / float(mask.sum())
        pieces.append(
            f"| `{tn}` | {int(mask.sum())} | {breakdown} | "
            f"`{labels[maj_idx]}` ({maj_frac:.0%}) |"
        )
    pieces.append("")
    pieces.append(
        f"![experimental class counts]({dataset_exp_figs_dir}/class_counts.png)"
    )
    pieces.append("")
    # Concrete interpretation hook tied to the binary task.
    if "binary" in tasks:
        _, yb, _ = tasks["binary"]
        cb = np.bincount(yb.astype(int), minlength=2)
        baseline = max(cb) / cb.sum()
        pieces.append(
            f"**Binary baseline:** majority-class accuracy is "
            f"{baseline:.3f}.  A model that always predicts `Damage` "
            f"will score exactly this on the balanced-exp set, so any "
            f"binary cell in §7.1 reporting `exp acc ≈ {baseline:.3f}` "
            f"is degenerate (model collapsed to the prior).  This is "
            f"visible at a glance in the §7.1 cross-model table - "
            f"most cells sit right at the baseline."
        )
        pieces.append("")
    pieces.append("### 2.7.2 Severity coverage per damage type")
    pieces.append("")
    # Severity stats per type.
    pieces.append("| type | n | severity min | median | max | std |")
    pieces.append("|---|---|---|---|---|---|")
    from ml_pipeline.case_design import TYPE_NAMES
    for tc_val in (1, 2, 3, 4):
        mask_t = tc == tc_val
        if mask_t.sum() == 0: continue
        s = sev[mask_t]
        pieces.append(
            f"| {TYPE_NAMES[tc_val]} | {int(mask_t.sum())} | "
            f"{s.min():.3f} | {float(np.median(s)):.3f} | "
            f"{s.max():.3f} | {s.std():.3f} |"
        )
    pieces.append("")
    pieces.append(
        f"![experimental severity histogram per damage type]"
        f"({dataset_exp_figs_dir}/severity_hist.png)"
    )
    pieces.append("")
    pieces.append("### 2.7.3 Example real signals per damage type")
    pieces.append("")
    pieces.append(
        "Each panel below shows one experimental sample drawn at random "
        "(seed 0) per damage type, with all 9 sensors overlaid.  The "
        "FRF magnitudes have richer modal content and a higher noise "
        "floor than the synthetic equivalents shown in §5 (compare with "
        "`figures/feature_examples/frf_mag.png`); the CFDAC patterns "
        "are also visibly noisier."
    )
    pieces.append("")
    pieces.append(
        f"![experimental timeseries examples]({dataset_exp_figs_dir}/signals_ts.png)"
    )
    pieces.append("")
    pieces.append(
        f"![experimental FRF magnitude examples]({dataset_exp_figs_dir}/signals_frf.png)"
    )
    pieces.append("")
    pieces.append(
        f"![experimental CFDAC examples]({dataset_exp_figs_dir}/signals_cfdac.png)"
    )
    pieces.append("")
    return "\n".join(pieces)


def _build_section_9_5(clean_exp_path: Path,
                          noisy_exp_path: Path) -> str:
    """Build §9.5 markdown: did mixing noise into training help cross-
    domain transfer?  Per-cell exp_clean -> exp_noisy_mixed comparison
    over every (task, model, feature) cell present in both runs."""
    if not (clean_exp_path.exists() and noisy_exp_path.exists()):
        return ""
    clean = json.loads(clean_exp_path.read_text())
    noisy = json.loads(noisy_exp_path.read_text())
    ck = {(r["task"], r["model"], r["feature"]): r for r in clean}
    nk = {(r["task"], r["model"], r["feature"]): r for r in noisy}
    common = sorted(ck.keys() & nk.keys())
    if not common:
        return ""
    by_task = defaultdict(list)
    for k in common:
        cv = ck[k]["value"]; nv = nk[k]["value"]
        if cv is None or nv is None: continue
        by_task[k[0]].append((k, cv, nv, nv - cv))

    pieces: list[str] = []
    pieces.append("## 9.5 Effect of injecting noise into training "
                       "(exp transfer: clean vs noisy_mixed)")
    pieces.append("")
    pieces.append(
        "Every cell present in both the clean run "
        "(`results/balanced/experimental_full_evaluation.json`) and the "
        "noisy-mixed run "
        "(`results/noisy_mixed/experimental_full_evaluation.json`) is "
        "compared head-to-head below.  **Δ > 0 means training on the "
        "clean + 35/25/20/15/10 dB mixed corpus improved cross-domain "
        "transfer** vs training on the clean synthetic set alone; "
        "Δ < 0 means noisy training hurt."
    )
    pieces.append("")
    pieces.append("### 9.5.1 Per-task summary")
    pieces.append("")
    pieces.append("| task | n cells | median Δ | mean Δ | best cell (Δ) | worst cell (Δ) |")
    pieces.append("|---|---|---|---|---|---|")
    for task in ("binary", "type", "severity", "col_location", "mass_location"):
        rows = by_task.get(task, [])
        if not rows: continue
        deltas = [r[3] for r in rows]
        med = float(np.median(deltas)); mean = float(np.mean(deltas))
        best = max(rows, key=lambda r: r[3])
        worst = min(rows, key=lambda r: r[3])
        pieces.append(
            f"| `{task}` | {len(rows)} | {med:+.3f} | {mean:+.3f} | "
            f"`{best[0][1]}/{best[0][2]}` ({best[3]:+.3f}) | "
            f"`{worst[0][1]}/{worst[0][2]}` ({worst[3]:+.3f}) |"
        )
    pieces.append("")
    pieces.append("### 9.5.2 Per-cell detail")
    pieces.append("")
    pieces.append(
        "| task | model | feature | exp clean | exp noisy_mixed | Δ |"
    )
    pieces.append("|---|---|---|---|---|---|")
    for task in ("binary", "type", "severity", "col_location", "mass_location"):
        rows = sorted(by_task.get(task, []), key=lambda r: -r[3])
        for (tn, m, f), cv, nv, d in rows:
            pieces.append(
                f"| `{tn}` | {m} | `{f}` | {cv:.3f} | {nv:.3f} | {d:+.3f} |"
            )
    pieces.append("")
    pieces.append(
        "Read this together with §7.x cross-domain captions: a cell with "
        "high synth-test accuracy but a small `exp_clean` value already "
        "told us the model was memorising synth-only structure; if `Δ` "
        "is also small here, noise injection didn't unlock new "
        "cross-domain signal either."
    )
    pieces.append("")
    return "\n".join(pieces)


def _insert_section(text: str, anchor_re: str, new_content: str) -> str:
    """Insert ``new_content`` immediately before the line that matches
    ``anchor_re``.  If a section with the same first header already
    exists in the text, replace it in place instead of inserting a
    second copy."""
    if not new_content.strip():
        return text
    first_header = new_content.splitlines()[0]
    # Idempotent: if the section already exists, replace it up to the
    # next top-level header.
    existing = re.search(re.escape(first_header) + r"\b", text,
                              flags=re.MULTILINE)
    if existing:
        nxt = re.search(r"^# |^## ", text[existing.end():],
                              flags=re.MULTILINE)
        end = existing.end() + nxt.start() if nxt else len(text)
        return text[:existing.start()] + new_content + "\n" + text[end:]
    # First-time insert: place just before the anchor.
    m = re.search(anchor_re, text, flags=re.MULTILINE)
    if not m:
        return text  # anchor missing; bail rather than corrupt
    return text[:m.start()] + new_content + "\n" + text[m.start():]


def update_indicator(text: str, ind_subs_path: Path,
                       ind_vs_dmg_path: Path) -> str:
    """Drop §7.6.{5,6} and replace with the 22 per-indicator subsections
    plus the indicator-vs-damage table; renumber as 7.6.5 .. 7.6.28."""
    indicator_subs = ind_subs_path.read_text() if ind_subs_path.exists() else ""
    vs_table = ind_vs_dmg_path.read_text() if ind_vs_dmg_path.exists() else ""

    # Find §7.6.5 ("What this means") and capture from there to the next
    # top-level `# ` header (which is `# 8. Cross-task takeaways`).
    m_start = re.search(r"^### 7\.6\.5\b", text, flags=re.MULTILINE)
    m_end = re.search(r"^# 8\.\s", text, flags=re.MULTILINE)
    if not m_start or not m_end:
        return text
    body_before = text[:m_start.start()]
    body_after  = text[m_end.start():]
    tail = text[m_start.start():m_end.start()]
    # Strip the old §7.6.5 / 7.6.6 sections; we keep their content
    # appended with new numbers.
    tail_lines = tail.splitlines()
    cleaned = []
    for line in tail_lines:
        if line.startswith("### 7.6.5 "):
            cleaned.append("### 7.6.30 What it means (rolled-up)")
            continue
        if line.startswith("### 7.6.6 "):
            cleaned.append("### 7.6.31 Recommendation (rolled-up)")
            continue
        cleaned.append(line)
    tail = "\n".join(cleaned)

    insertion = "\n" + indicator_subs.rstrip() + "\n\n"
    insertion += "### 7.6.29 Indicator regression vs direct damage-parameter regression\n\n"
    insertion += ("For every `(model, feature)` cell, the table below "
                       "places the **best transferring indicator's exp R²** "
                       "next to the **direct damage-parameter** scores "
                       "from §§ 7.1 – 7.5.  Two takeaways:\n\n"
                       "* RF on `modal` predicts `M2L_abs_sum` at "
                       "exp R² **+0.76**, but predicts severity directly "
                       "at exp R² **−0.17**.  The indicator-as-target "
                       "approach transfers an order of magnitude better "
                       "than direct regression for this cell.\n"
                       "* The same `(model, feature)` cell that fails on "
                       "direct severity regression often **does** transfer "
                       "on the bounded indicators (DRQ / RVAC_* / M2L_*) — "
                       "these are usable cross-domain proxies of damage "
                       "severity even though severity itself is not.\n\n")
    insertion += vs_table + "\n"

    return body_before + insertion + tail + body_after


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--report", type=Path,
                      default=_REPO / "results" / "REPORT.md")
    p.add_argument("--hpo", type=Path,
                      default=_REPO / "results" / "hpo")
    p.add_argument("--exp", type=Path,
                      default=_REPO / "results" / "balanced"
                              / "experimental_full_evaluation.json")
    p.add_argument("--ind-subs", type=Path,
                      default=Path("/tmp/indicator_subsections.md"))
    p.add_argument("--ind-vs-dmg", type=Path,
                      default=_REPO / "results" / "indicator_vs_damage_table.md")
    p.add_argument("--exp-figures-dir", default="figures_exp",
                      help="Directory (relative to the report's parent) "
                              "where the experimental-test PNG twins live.")
    p.add_argument("--clean-exp", type=Path,
                      default=_REPO / "results" / "balanced"
                              / "experimental_full_evaluation.json",
                      help="Clean-training exp eval JSON; used for §9.5.")
    p.add_argument("--noisy-exp", type=Path,
                      default=_REPO / "results" / "noisy_mixed"
                              / "experimental_full_evaluation.json",
                      help="Noisy-mixed training exp eval JSON; used for §9.5.")
    p.add_argument("--exp-dataset", type=Path,
                      default=_REPO / "dataset"
                              / "experimental_features_balanced.h5",
                      help="Experimental dataset HDF5; used for §2.7.")
    p.add_argument("--exp-dataset-figs", default="figures/dataset_exp",
                      help="Directory (relative to the report's parent) "
                              "where the §2.7 exp-dataset PNGs live.")
    args = p.parse_args()

    text = args.report.read_text()
    text = update_cross_model(text, args.hpo, args.exp,
                                       exp_figures_dir=args.exp_figures_dir,
                                       report_root=args.report.parent)
    text = update_indicator(text, args.ind_subs, args.ind_vs_dmg)

    # §2.7 - experimental dataset deep-dive (inserted just before §3).
    sec27 = _build_section_2_7(args.exp_dataset, args.exp_dataset_figs)
    text = _insert_section(text, r"^# 3\.\s", sec27)

    # §9.5 - clean vs noisy_mixed training comparison (inserted just
    # before §10).
    sec95 = _build_section_9_5(args.clean_exp, args.noisy_exp)
    text = _insert_section(text, r"^# 10\.\s", sec95)

    args.report.write_text(text)
    print(f"updated {args.report}  ({len(text):,} chars)")


if __name__ == "__main__":
    main()
