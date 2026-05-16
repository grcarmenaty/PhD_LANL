#!/usr/bin/env bash
# SessionStart hook: bootstrap env for the ml_pipeline and auto-resume the
# mixed-training noise pipeline if one is in progress.
#
# This is synchronous (no `{"async": true}` line) so deps and pymodal are
# guaranteed ready before Claude takes over.  The pipeline itself is launched
# detached so it survives the hook's exit.
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

# ---------- 3.  resume mixed-training pipeline ----------------------------
PAUSE_FLAG="$REPO/.claude/PAUSE_SWEEP"
MIXED_FEATURES="$REPO/dataset/features_mixed.h5"
MIXED_OUT="$REPO/results/noisy_mixed"

if [ -f "$PAUSE_FLAG" ]; then
  log "PAUSE_SWEEP flag present; not launching mixed pipeline"
  exit 0
fi

if [ ! -f "$MIXED_FEATURES" ]; then
  log "mixed features VDS missing ($MIXED_FEATURES); skipping auto-launch"
  exit 0
fi

# If the orchestrator script is running, don't relaunch (it owns hpo.py).
if pgrep -af "run_noisy_mixed_pipeline.sh" >/dev/null; then
  log "noisy-mixed orchestrator already running; not relaunching"
elif pgrep -af "hpo.py --features $MIXED_FEATURES" >/dev/null; then
  log "hpo.py on mixed already running; not relaunching orchestrator"
else
  mkdir -p "$REPO/logs"
  LOG="$REPO/logs/orchestrator_$(date -u +%Y%m%d_%H%M%S).log"
  log "starting noisy-mixed orchestrator -> $LOG"
  setsid nohup bash "$REPO/.claude/run_noisy_mixed_pipeline.sh" \
      > "$LOG" 2>&1 < /dev/null &
  PID=$!
  echo "$PID" > /tmp/mixed_orchestrator.pid
  disown $PID 2>/dev/null || true
  log "orchestrator pid=$PID"
fi

# ---------- 4.  remind Claude to re-arm the 10-min heartbeat --------------
# The Monitor process dies on every VM reboot.  Only Claude can call Monitor,
# so we emit a system-reminder telling it to re-arm if no ping script is alive.
if ! pgrep -f "$REPO/.claude/hpo_ping.sh" >/dev/null; then
  cat <<EOF
<system-reminder>
The 10-minute hpo.py status pings are NOT running.  Re-arm them by calling
the Monitor tool with persistent=true and command=
'bash $REPO/.claude/hpo_ping.sh'
This should be done in the very first response of this session so the user
keeps getting 10-min pings in chat.  Do this BEFORE any other work.
</system-reminder>
EOF
fi

exit 0
