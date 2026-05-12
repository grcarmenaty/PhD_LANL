#!/usr/bin/env bash
set -u
SWEEP_PID="${1:?usage: $0 SWEEP_PID [INTERVAL_SEC]}"
INTERVAL="${2:-600}"
BR=$(git rev-parse --abbrev-ref HEAD)

cd "$(git rev-parse --show-toplevel)"

log() { printf '[watchdog %s] %s\n' "$(date -u +%FT%TZ)" "$*"; }

commit_round() {
  git add -A results/ 2>/dev/null
  if git diff --cached --quiet; then
    log "no changes"
    return
  fi
  git -c user.email=watchdog@local -c user.name=watchdog \
      commit -m "watchdog auto-checkpoint $(date -u +%FT%TZ) (sweep pid $SWEEP_PID)" >/dev/null \
    && log "committed"
  for attempt in 1 2 3 4; do
    if git push origin "$BR" 2>&1 | tail -3; then
      log "pushed"
      return
    fi
    sleep $((2 ** attempt))
  done
  log "push failed after 4 attempts"
}

log "started, interval=${INTERVAL}s, watching pid $SWEEP_PID on branch $BR"
while kill -0 "$SWEEP_PID" 2>/dev/null; do
  sleep "$INTERVAL"
  commit_round
done

log "sweep pid $SWEEP_PID exited; final commit round"
commit_round
log "done"
