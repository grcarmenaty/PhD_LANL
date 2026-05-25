"""V1 vs V2 (variation v1 vs variation_v2 chunk regeneration) comparison.

Judges the v2 result against the pre-registered criteria in
`results/chunk_regen_preregistered.md`. Reads:

    results/multiseed_summary.json                       (v1 3-seed)
    results/experimental_full_evaluation_v2_seed{42,101,202}.json (v2 seeds)

Aggregates v2 across the seeds that have landed (1, 2, or all 3) and
emits PASS / FAIL / INCONCLUSIVE for each criterion, plus a top-line
ADOPT / REJECT / INCONCLUSIVE decision. Output:
`results/chunk_regen_v2_decision.json` + `REPORT_v2_chunk_regen.md`
stub.

Pre-registered criteria (paraphrased from chunk_regen_preregistered.md):
    C1 (primary) is_crack/mlp/modal BA must beat 0.787
    C2 (primary) col_location/mlp/modal macro-F1 must beat 0.329
    C3 (secondary) is_hole/mlp/modal BA must beat 0.833
    C4 (floor) is_hole/mlp/modal BA must stay >= 0.653
    C5 (floor) binary best macro-F1 must stay >= v1_best - 2*0.086

Decision rule: ADOPT iff (C1 or C2 pass) AND C4 AND C5 pass; REJECT iff
C4 or C5 fail; INCONCLUSIVE otherwise.
"""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path
from typing import Optional

_REPO = Path(__file__).resolve().parents[1]


def _load_v2_seed(out_dir: Path, seed: int, label: str = "v2") -> Optional[list[dict]]:
    p = out_dir / f"experimental_full_evaluation_{label}_seed{seed}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _find_cell(rows: list[dict], task: str, model: str, feature: str) -> Optional[dict]:
    for r in rows:
        if r.get("task") == task and r.get("model") == model and r.get("feature") == feature:
            return r
    return None


def _agg_v2_cell(seeds_rows: dict[int, list[dict]], task: str, model: str,
                 feature: str, metric_key: str) -> Optional[dict]:
    vals = []
    seeds_used = []
    for sd, rows in seeds_rows.items():
        cell = _find_cell(rows, task, model, feature)
        if cell is None: continue
        v = cell.get(metric_key)
        if v is None: continue
        vals.append(v); seeds_used.append(sd)
    if not vals: return None
    return {
        "task": task, "model": model, "feature": feature,
        "metric": metric_key, "values": vals, "seeds": seeds_used,
        "mean": statistics.fmean(vals),
        "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
    }


def _v1_cell(v1_rows: list[dict], task: str, model: str, feature: str,
             metric_key: str) -> Optional[dict]:
    for r in v1_rows:
        if r["task"] == task and r["model"] == model and r["feature"] == feature:
            if metric_key.startswith("balanced_acc"):
                return {"mean": r.get("balanced_acc_mean"),
                        "sd":   r.get("balanced_acc_sd")}
            if metric_key.startswith("macro_f1"):
                return {"mean": r.get("macro_f1_mean"),
                        "sd":   r.get("macro_f1_sd")}
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--v1", type=Path,
                   default=_REPO / "results" / "multiseed_summary.json")
    p.add_argument("--v2-dir", type=Path, default=_REPO / "results")
    p.add_argument("--label", default="v2",
                   help="Label in the eval JSON filename — 'v2' looks for "
                        "experimental_full_evaluation_v2_seedN.json; 'v2a' "
                        "for the v2a ablation; etc.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output JSON path; defaults to "
                        "results/chunk_regen_<label>_decision.json.")
    args = p.parse_args()
    if args.out is None:
        args.out = _REPO / "results" / f"chunk_regen_{args.label}_decision.json"

    v1_rows = json.loads(args.v1.read_text())
    seeds_rows = {sd: r for sd in (42, 101, 202)
                  if (r := _load_v2_seed(args.v2_dir, sd, args.label)) is not None}
    if not seeds_rows:
        print("No v2 seed evals found yet — nothing to compare.")
        return
    print(f"v2 seeds available: {sorted(seeds_rows)}")

    P90_TORCH = 0.086
    BAND = 2 * P90_TORCH  # 0.172

    criteria = []

    # C1 primary — is_crack/mlp/modal BA improves
    v1 = _v1_cell(v1_rows, "is_crack", "mlp", "modal", "balanced_acc")
    v2 = _agg_v2_cell(seeds_rows, "is_crack", "mlp", "modal", "balanced_acc")
    if v1 and v2:
        thr = v1["mean"] + BAND
        criteria.append({"id": "C1", "kind": "primary",
                         "name": "is_crack/mlp/modal BA",
                         "v1_mean": v1["mean"], "v1_sd": v1["sd"],
                         "v2_mean": v2["mean"], "v2_sd": v2["sd"],
                         "v2_seeds": v2["seeds"],
                         "threshold": thr,
                         "pass": v2["mean"] >= thr})

    # C2 primary — col_location/mlp/modal macro-F1 improves
    v1 = _v1_cell(v1_rows, "col_location", "mlp", "modal", "macro_f1")
    v2 = _agg_v2_cell(seeds_rows, "col_location", "mlp", "modal", "macro_f1")
    if v1 and v2:
        thr = v1["mean"] + BAND
        criteria.append({"id": "C2", "kind": "primary",
                         "name": "col_location/mlp/modal macro-F1",
                         "v1_mean": v1["mean"], "v1_sd": v1["sd"],
                         "v2_mean": v2["mean"], "v2_sd": v2["sd"],
                         "v2_seeds": v2["seeds"],
                         "threshold": thr,
                         "pass": v2["mean"] >= thr})

    # C3 secondary — is_hole/mlp/modal BA improves
    v1 = _v1_cell(v1_rows, "is_hole", "mlp", "modal", "balanced_acc")
    v2 = _agg_v2_cell(seeds_rows, "is_hole", "mlp", "modal", "balanced_acc")
    if v1 and v2:
        thr = v1["mean"] + BAND
        criteria.append({"id": "C3", "kind": "secondary",
                         "name": "is_hole/mlp/modal BA improves",
                         "v1_mean": v1["mean"], "v1_sd": v1["sd"],
                         "v2_mean": v2["mean"], "v2_sd": v2["sd"],
                         "v2_seeds": v2["seeds"],
                         "threshold": thr,
                         "pass": v2["mean"] >= thr})

        # C4 floor — same cell must NOT regress beyond 2*sd
        thr_floor = v1["mean"] - 2 * v1["sd"]
        criteria.append({"id": "C4", "kind": "floor",
                         "name": "is_hole/mlp/modal BA no regression",
                         "v1_mean": v1["mean"], "v1_sd": v1["sd"],
                         "v2_mean": v2["mean"], "v2_sd": v2["sd"],
                         "v2_seeds": v2["seeds"],
                         "threshold": thr_floor,
                         "pass": v2["mean"] >= thr_floor})

    # C5 floor — binary best macro-F1 does not regress beyond noise band
    # First find v1 best binary cell
    binary_v1 = [r for r in v1_rows if r["task"] == "binary"]
    if binary_v1:
        best_v1 = max(binary_v1, key=lambda r: r.get("macro_f1_mean") or 0)
        # Best v2 binary across all cells in seed-aggregated form
        v2_binary_means = {}
        for sd, rows in seeds_rows.items():
            for r in rows:
                if r.get("task") != "binary": continue
                key = (r["model"], r["feature"])
                v2_binary_means.setdefault(key, []).append(r.get("macro_f1") or 0)
        if v2_binary_means:
            best_v2_key = max(v2_binary_means,
                              key=lambda k: statistics.fmean(v2_binary_means[k]))
            best_v2_vals = v2_binary_means[best_v2_key]
            best_v2_mean = statistics.fmean(best_v2_vals)
            thr_floor = (best_v1.get("macro_f1_mean") or 0) - BAND
            criteria.append({"id": "C5", "kind": "floor",
                             "name": f"binary best macro-F1 (v1 best: {best_v1['model']}/{best_v1['feature']}; v2 best: {best_v2_key[0]}/{best_v2_key[1]})",
                             "v1_mean": best_v1.get("macro_f1_mean"),
                             "v1_sd": best_v1.get("macro_f1_sd"),
                             "v2_mean": best_v2_mean,
                             "v2_sd": statistics.stdev(best_v2_vals) if len(best_v2_vals) > 1 else 0.0,
                             "v2_seeds": sorted(seeds_rows.keys()),
                             "threshold": thr_floor,
                             "pass": best_v2_mean >= thr_floor})

    # Decision
    by_id = {c["id"]: c for c in criteria}
    floors_ok = by_id.get("C4", {}).get("pass", False) and by_id.get("C5", {}).get("pass", False)
    primary_ok = by_id.get("C1", {}).get("pass", False) or by_id.get("C2", {}).get("pass", False)
    any_floor_fail = (by_id.get("C4") and not by_id["C4"]["pass"]) or \
                      (by_id.get("C5") and not by_id["C5"]["pass"])
    if any_floor_fail:
        decision = "REJECT"
    elif floors_ok and primary_ok:
        decision = "ADOPT"
    else:
        decision = "INCONCLUSIVE"

    out = {
        "v2_seeds_evaluated": sorted(seeds_rows),
        "n_v2_seeds": len(seeds_rows),
        "noise_band_2x_p90_torch": BAND,
        "criteria": criteria,
        "decision": decision,
    }

    args.out.write_text(json.dumps(out, indent=2))

    print(f"\n=== V2 chunk-regen decision ===")
    print(f"V2 seeds evaluated: {out['v2_seeds_evaluated']} ({out['n_v2_seeds']} of 3)")
    print(f"Noise band 2 × p90(torch) = {BAND:.3f}")
    print()
    print(f'{"id":3s} {"kind":10s} {"v1":>8s} {"v2":>8s} {"thr":>8s} {"pass":>6s}  name')
    for c in criteria:
        print(f'{c["id"]:3s} {c["kind"]:10s} {c["v1_mean"]:8.3f} {c["v2_mean"]:8.3f} {c["threshold"]:8.3f} {"YES" if c["pass"] else "NO":>6s}  {c["name"]}')
    print(f"\nDECISION: **{decision}**")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
