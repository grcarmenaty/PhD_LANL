#!/usr/bin/env python3
"""ETA for the currently-running noisy_mixed pipeline step.

Reads completed and partial trial JSONs from results/noisy_mixed/hpo/,
infers the active step from the running python process, and prints
``step=<name> trials=<done>/<total> avg=<s>s remaining~<H>h<MM>m``
to stdout.  Stays silent if nothing is running.
"""
from __future__ import annotations

import glob
import json
import subprocess
import sys
from pathlib import Path

REPO    = Path("/home/user/PhD_LANL")
HPO_DIR = REPO / "results" / "noisy_mixed" / "hpo"

# Trials planned per step where we know the number.  hpo and
# hpo_cfdac_variants come directly from their log banners.
STEP_TOTAL_TRIALS = {
    "hpo":                  255,
    "hpo_cfdac_variants":   100,
}


def _active_step() -> str | None:
    try:
        out = subprocess.check_output(
            ["pgrep", "-af", r"python.*ml_pipeline\."],
            stderr=subprocess.DEVNULL,
        ).decode()
    except subprocess.CalledProcessError:
        return None
    candidates = [
        "build_report_noise", "resolution_sweep", "transfer_learn",
        "evaluate_full_experimental", "train_indicator_predictors",
        "hpo_cfdac_allmodels", "hpo_cfdac_variants", "hpo",
    ]
    for line in out.splitlines():
        for s in candidates:
            if f"ml_pipeline.{s} " in line + " ":
                return s
    return None


def _belongs_to_variants(name: str) -> bool:
    """True iff this hpo cell was produced by hpo_cfdac_variants
    (cnn2d on cfdac_<variant>, or cnn3d on cfdac3d_<variant>)."""
    return ("__cnn2d__cfdac_" in name) or ("__cnn3d__cfdac3d_" in name)


def main() -> int:
    step = _active_step()
    if step is None or step not in STEP_TOTAL_TRIALS:
        return 0

    total = STEP_TOTAL_TRIALS[step]
    done, runtimes = 0, []
    for path in glob.glob(str(HPO_DIR / "*.json")):
        name = Path(path).name
        if step == "hpo_cfdac_variants":
            if not _belongs_to_variants(name):
                continue
        elif step == "hpo":
            if _belongs_to_variants(name):
                continue
        try:
            data = json.loads(Path(path).read_text())
        except Exception:
            continue
        for t in data.get("trials", []) or []:
            done += 1
            rt = t.get("runtime_s")
            if isinstance(rt, (int, float)) and rt > 0:
                runtimes.append(float(rt))

    if not runtimes:
        return 0
    avg = sum(runtimes) / len(runtimes)
    remaining = max(0, total - done)
    eta_s = remaining * avg
    h = int(eta_s // 3600)
    m = int((eta_s % 3600) // 60)
    print(f"step={step} trials={done}/{total} avg={avg:.0f}s remaining~{h}h{m:02d}m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
