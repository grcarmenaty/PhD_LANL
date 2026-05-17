#!/usr/bin/env bash
# Orchestrator for the noisy-mixed pipeline.
#
# Runs each step sequentially; per-step `.done` marker files allow the
# orchestrator to be killed and re-run any time (e.g. after a VM reboot)
# and pick up where it left off.  Each step logs to logs/mixed_<step>_*.log.
#
# Steps:
#   1. hpo.py                          → results/noisy_mixed/hpo/*
#   2. hpo_cfdac_variants.py           → results/noisy_mixed/hpo_cfdac_variants/*
#   3. hpo_cfdac_allmodels.py          → results/noisy_mixed/hpo_cfdac_allmodels/*
#   4. train_indicator_predictors.py   → results/noisy_mixed/indicator_predictors/*
#   5. evaluate_full_experimental.py   → results/noisy_mixed/full_experimental/*
#   6. transfer_learn.py               → results/noisy_mixed/transfer/*
#   7. resolution_sweep.py             → results/noisy_mixed/resolution_sweep/*
#   7.5 plots_experimental.py        → results/noisy_mixed/figures_exp/*
#   7.6 plot_exp_dataset.py          → results/figures/dataset_exp/* (shared)
#   8. build_report_noisy_mixed.py    → results/REPORT_noisy_mixed.md
set -u

REPO="${REPO:-/home/user/PhD_LANL}"
cd "$REPO"

FEATS="$REPO/dataset/features_mixed.h5"
EXP="$REPO/dataset/experimental_features_balanced.h5"
OUT="$REPO/results/noisy_mixed"
STAMP_DIR="$OUT/.pipeline_stamps"
LOG_DIR="$REPO/logs"
mkdir -p "$STAMP_DIR" "$LOG_DIR"

step() {
  local name="$1"; shift
  local stamp="$STAMP_DIR/$name.done"
  if [ -f "$stamp" ]; then
    echo "[orchestrator] $name: already done"
    return 0
  fi
  local log="$LOG_DIR/mixed_${name}_$(date -u +%Y%m%d_%H%M%S).log"
  echo "[orchestrator] $name: running -> $log"
  if "$@" > "$log" 2>&1; then
    touch "$stamp"
    echo "[orchestrator] $name: OK"
    return 0
  else
    echo "[orchestrator] $name: FAILED (exit $?); see $log"
    return 1
  fi
}

step hpo  python -m ml_pipeline.hpo \
  --features "$FEATS" --out "$OUT"                                      || exit 1
step cfdac_variants  python -m ml_pipeline.hpo_cfdac_variants \
  --features "$FEATS" --out "$OUT"                                      || exit 1
step cfdac_allmodels  python -m ml_pipeline.hpo_cfdac_allmodels \
  --features "$FEATS" --out "$OUT"                                      || exit 1
step indicators  python -m ml_pipeline.train_indicator_predictors \
  --features "$FEATS" --out "$OUT"                                      || exit 1
step eval_experimental  python -m ml_pipeline.evaluate_full_experimental \
  --syn "$FEATS" --exp "$EXP" --out "$OUT"                              || exit 1
step transfer  python -m ml_pipeline.transfer_learn \
  --syn "$FEATS" --exp "$EXP" --results "$OUT"                          || exit 1
step resolution_sweep  python -m ml_pipeline.resolution_sweep \
  --features "$FEATS" --out "$OUT"                                      || exit 1
step exp_plots  python -m ml_pipeline.plots_experimental \
  --syn-features "$FEATS" --exp "$EXP" \
  --results "$OUT" --out "$OUT/figures_exp"                             || exit 1
step report  python -m ml_pipeline.build_report_noisy_mixed \
  --results "$OUT" --out "$REPO/results/REPORT_noisy_mixed.md"          || exit 1

echo "[orchestrator] ALL DONE"
