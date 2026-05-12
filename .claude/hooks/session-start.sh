#!/usr/bin/env bash
# SessionStart hook: bootstrap env for the ml_pipeline and auto-resume the
# noise sweep if one is in progress.
#
# This is synchronous (no `{"async": true}` line) so deps and pymodal are
# guaranteed ready before Claude takes over.  The sweep itself is launched
# detached so it survives the hook's exit (and dies on session disconnect,
# which is fine - the next session-start picks up where it left off).
set -euo pipefail

# Only run inside Claude Code on the web (where the VM is ephemeral).
[ "${CLAUDE_CODE_REMOTE:-}" = "true" ] || exit 0

REPO="${CLAUDE_PROJECT_DIR:-$(pwd)}"
cd "$REPO"

log() { printf '[session-start %s] %s\n' "$(date -u +%FT%TZ)" "$*" >&2; }

# ---------- 1.  pymodal sibling clone -------------------------------------
PYMODAL_DIR="$(dirname "$REPO")/pymodal"
if [ ! -d "$PYMODAL_DIR/pymodal" ]; then
  log "cloning grcarmenaty/pymodal to $PYMODAL_DIR"
  git clone --depth 1 https://github.com/grcarmenaty/pymodal "$PYMODAL_DIR" >&2 \
    || { log "pymodal clone failed"; exit 1; }
else
  log "pymodal already present at $PYMODAL_DIR"
fi

# ---------- 2.  python deps ----------------------------------------------
if ! python -c 'import torch, sklearn, xgboost, h5py, scipy, matplotlib, pint' 2>/dev/null; then
  log "installing ML deps (torch + sklearn + xgboost + matplotlib + pint + ...)"
  pip install --quiet --no-input \
      numpy scipy h5py matplotlib scikit-learn xgboost pandas pint torch \
      >&2 || { log "pip install failed"; exit 1; }
else
  log "ML deps already importable"
fi

# Make pymodal importable from any subprocess started by Claude.
echo "export PYTHONPATH=\"$PYMODAL_DIR:\${PYTHONPATH:-}\"" >> "${CLAUDE_ENV_FILE:-/dev/null}"

# ---------- 3.  resume noise sweep (background, detached) ----------------
PAUSE_FLAG="$REPO/.claude/PAUSE_SWEEP"
SWEEP_PIDFILE="/tmp/noise_sweep.pid"
WATCHDOG_PIDFILE="/tmp/noise_sweep_watchdog.pid"

if [ -f "$PAUSE_FLAG" ]; then
  log "PAUSE_SWEEP flag present; not launching sweep"
  exit 0
fi

# If a previous sweep is still alive (shouldn't happen on web - VM is fresh
# - but matters for local testing), don't launch a duplicate.
if [ -f "$SWEEP_PIDFILE" ] && kill -0 "$(cat "$SWEEP_PIDFILE")" 2>/dev/null; then
  log "sweep already running (pid $(cat "$SWEEP_PIDFILE")); not relaunching"
  exit 0
fi

# Has the sweep already completed for every target SNR?  If so, nothing to do.
ALL_DONE=true
for snr in 35 25 15 10; do
  if [ ! -f "$REPO/results/noisy_${snr}dB/transfer_learning.json" ]; then
    ALL_DONE=false
    break
  fi
done
if $ALL_DONE; then
  log "all four SNR levels complete; nothing to resume"
  exit 0
fi

mkdir -p "$REPO/logs"
SWEEP_LOG="$REPO/logs/sweep_$(date -u +%Y%m%d_%H%M%S).log"
WATCHDOG_LOG="$REPO/logs/watchdog_$(date -u +%Y%m%d_%H%M%S).log"

log "launching sweep + watchdog (logs in $REPO/logs/)"
setsid nohup python "$REPO/ml_pipeline/run_noise_sweep.py" \
    --snr-db 35 25 15 10 \
    > "$SWEEP_LOG" 2>&1 < /dev/null &
SWEEP_PID=$!
echo "$SWEEP_PID" > "$SWEEP_PIDFILE"
disown $SWEEP_PID 2>/dev/null || true

setsid nohup bash "$REPO/ml_pipeline/_sweep_watchdog.sh" "$SWEEP_PID" 600 \
    > "$WATCHDOG_LOG" 2>&1 < /dev/null &
WATCH_PID=$!
echo "$WATCH_PID" > "$WATCHDOG_PIDFILE"
disown $WATCH_PID 2>/dev/null || true

log "sweep pid=$SWEEP_PID watchdog pid=$WATCH_PID"
exit 0
