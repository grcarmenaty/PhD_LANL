#!/usr/bin/env bash
# Value-ordered, resumable top-3 vision sweep.
#
#   backbones : convnext_tiny, resnet50, vit_b_16   (top-3 by macro-F1)
#   features  : cfdac_all (4ch), cfdac_mag (1ch), cfdac_realimag (2ch)
#   tasks     : all 10
#   variants  : v1 -> v2 -> v2a   (most-informative first)
#   seeds     : 42 -> 101 -> 202
#
# = 3 x 3 x 10 x 3 x 3 = 810 cells. Disk-light streaming mode: models are
# never written to disk (--no-save-model); each cell's per-case JSON is the
# resume unit, so re-running this script picks up exactly where it left off.
#
# Weights: timm ImageNet-1k variants whose .pth live on reachable GitHub
# releases (download.pytorch.org / HuggingFace are blocked here).
set -u
cd /home/user/PhD_LANL
export HF_HUB_OFFLINE=1
export PYTHONPATH="/home/user/pymodal:${PYTHONPATH:-}"

BACKBONES="convnext_tiny resnet50 vit_b_16"
FEATURES="cfdac_all cfdac_mag cfdac_realimag"
# Ordered so the most interpretable tasks land first.
TASKS="type is_bolt is_hole severity col_location mass_location is_mass is_crack is_pristine binary"
SEEDS="42 101 202"
SUB=1500; EP=4; PROBE=2; BATCH=64

declare -A SYN=( [v1]="dataset/features.h5" \
                 [v2]="dataset/features_v2.h5" \
                 [v2a]="dataset/features_v2a.h5" )

for variant in v1 v2 v2a; do
  syn="${SYN[$variant]}"
  if [ ! -f "$syn" ]; then
    echo "[skip $variant] missing $syn (build it, then re-run this script)"
    continue
  fi
  for seed in $SEEDS; do
    out="results_vision/${variant}_seed${seed}"
    pc="$out/per_case_vision"
    expected=$((${#BACKBONES} * 0 + 90))   # 3 backbones × 3 features × 10 tasks
    # Retry until this (variant, seed) actually has 90 per-case JSONs, so a
    # killed/OOM'd python doesn't skip the rest of the (variant, seed).
    attempt=0
    while true; do
      done_n=$(find "$pc" -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
      [ "$done_n" -ge 90 ] && break
      attempt=$((attempt + 1))
      echo "===== VARIANT=$variant SEED=$seed attempt=$attempt done=$done_n/90  @ $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
      python -m ml_pipeline.train_vision \
        --syn "$syn" --exp dataset/experimental_features.h5 \
        --out "$out" --per-case-out "$pc" --no-save-model \
        --backbones $BACKBONES --features $FEATURES --tasks $TASKS \
        --seed "$seed" --epochs $EP --probe-epochs $PROBE \
        --subsample $SUB --batch $BATCH || true
      # If no new cells landed this attempt, sleep before retrying so we
      # don't tight-loop on a persistent failure.
      new_n=$(find "$pc" -name '*.json' 2>/dev/null | wc -l | tr -d ' ')
      [ "$new_n" -le "$done_n" ] && sleep 15
    done
    echo "===== VARIANT=$variant SEED=$seed COMPLETE (90/90)  @ $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
  done
done
echo "ALL DONE @ $(date -u +%Y-%m-%dT%H:%M:%SZ)"
