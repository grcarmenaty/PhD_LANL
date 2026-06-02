#!/usr/bin/env bash
# Supervises the long vision sweep so multi-day, reap-prone CPU compute
# is durable:
#   * (re)installs timm/torchvision if the ephemeral container lost them
#   * runs the resumable sweep; relaunches it if it dies before ALL DONE
#   * every COMMIT_EVERY seconds, commits + pushes new per-case JSONs to
#     the rescue branch so a container reap loses at most a few minutes
set -u
cd /home/user/PhD_LANL
export HF_HUB_OFFLINE=1
export PYTHONPATH="/home/user/pymodal:${PYTHONPATH:-}"

BRANCH="claude/rescue-failing-session-xjHZb"
COMMIT_EVERY=600
LOG=/tmp/vsweep.log

ensure_deps() {
  python -c "import timm, torchvision" 2>/dev/null || \
    pip install -q timm torchvision 2>>/tmp/pipinstall.log
}

commit_progress() {
  git add results_vision 2>/dev/null || return 0
  if git diff --cached --quiet 2>/dev/null; then return 0; fi
  local n
  n=$(find results_vision -name '*.json' -path '*per_case_vision*' 2>/dev/null | wc -l | tr -d ' ')
  git commit -q -m "vision sweep progress: ${n} per-case files ($(date -u +%Y-%m-%dT%H:%M:%SZ))" 2>/dev/null || return 0
  local i
  for i in 1 2 3 4; do
    git push -q origin "$BRANCH" 2>/dev/null && break || sleep $((2**i))
  done
  echo "[supervisor] committed+pushed ${n} per-case files @ $(date -u +%H:%M:%SZ)"
}

ensure_deps
echo "[supervisor] start @ $(date -u +%Y-%m-%dT%H:%M:%SZ)"

while true; do
  ensure_deps
  bash ml_pipeline/run_vision_sweep.sh > "$LOG" 2>&1 &
  SWEEP=$!
  # Periodic commit while the sweep runs.
  while kill -0 "$SWEEP" 2>/dev/null; do
    sleep "$COMMIT_EVERY"
    commit_progress
  done
  wait "$SWEEP" 2>/dev/null
  commit_progress
  if grep -q "ALL DONE" "$LOG" 2>/dev/null; then
    echo "[supervisor] sweep reported ALL DONE @ $(date -u +%H:%M:%SZ)"
    break
  fi
  echo "[supervisor] sweep exited early; relaunching in 10s @ $(date -u +%H:%M:%SZ)"
  sleep 10
done

commit_progress
echo "[supervisor] finished @ $(date -u +%Y-%m-%dT%H:%M:%SZ)"
