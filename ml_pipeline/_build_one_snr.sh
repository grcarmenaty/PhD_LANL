#!/usr/bin/env bash
# Build the full features stack (chunks + features + cfdac + cfdac_variants)
# for one SNR level, then delete the per-SNR chunk dir to save disk.
# Usage: _build_one_snr.sh <SNR_DB>
set -euo pipefail

SNR="${1:?usage: $0 <SNR_DB>}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"
TS="$(date -u +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/build_snr${SNR}_${TS}.log"

log() { printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG" >&2; }

log "=== build SNR ${SNR} dB ==="

CHUNK_DIR="$REPO/dataset/noisy_${SNR}dB"
FEAT_FILE="$REPO/dataset/features_noisy_${SNR}dB.h5"

if [ -f "$FEAT_FILE" ]; then
  log "features_noisy_${SNR}dB.h5 already exists, skipping"
  exit 0
fi

log "1/4 build_noisy_chunks --snr-db ${SNR}"
python "$REPO/ml_pipeline/build_noisy_chunks.py" --snr-db "$SNR" >>"$LOG" 2>&1

log "2/4 features.py --dataset $CHUNK_DIR --out $FEAT_FILE"
python "$REPO/ml_pipeline/features.py" \
    --dataset "$CHUNK_DIR" --out "$FEAT_FILE" >>"$LOG" 2>&1

log "3/4 cfdac.py --features $FEAT_FILE"
python "$REPO/ml_pipeline/cfdac.py" --features "$FEAT_FILE" >>"$LOG" 2>&1

log "4/4 cfdac_variants.py --features $FEAT_FILE"
python "$REPO/ml_pipeline/cfdac_variants.py" --features "$FEAT_FILE" >>"$LOG" 2>&1

log "deleting per-SNR chunk dir to reclaim disk"
rm -rf "$CHUNK_DIR"

log "=== SNR ${SNR} dB DONE: $FEAT_FILE ($(stat -c%s "$FEAT_FILE") bytes) ==="
