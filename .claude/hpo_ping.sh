#!/usr/bin/env bash
# Two cadences in one loop:
#   * every 60s  -> auto-commit any new noisy_mixed/ artefacts (keeps tree
#                   clean so the stop-hook stops complaining).
#   * every 600s -> emit a verbose multi-line status block + relaunch
#                   hpo.py if it has died.
cd /home/user/PhD_LANL

auto_commit() {
  local new_files
  new_files=$(git ls-files --others --exclude-standard results/noisy_mixed/ 2>/dev/null)
  [ -z "$new_files" ] && return 0
  local n_new
  n_new=$(wc -l <<<"$new_files")
  git add results/noisy_mixed/ >/dev/null 2>&1
  git -c user.email=claude@local -c user.name=claude \
      commit -m "noisy_mixed: auto-commit +${n_new} cell artefacts" \
      >/dev/null 2>&1 || return 0
  git push origin HEAD >/dev/null 2>&1
}

emit_ping() {
  local ts proc relaunch=""
  ts=$(TZ=Europe/Madrid date +'%H:%M:%S %Z')
  proc=$(pgrep -af 'hpo\.py.*features_mixed' | grep -v 'hpo_ping' | head -1)
  if [ -z "$proc" ]; then
    local LOG="logs/mixed_hpo_resume_$(date -u +%Y%m%d_%H%M%S).log"
    setsid nohup python ml_pipeline/hpo.py \
        --features dataset/features_mixed.h5 \
        --out results/noisy_mixed > "$LOG" 2>&1 < /dev/null &
    disown
    sleep 2
    proc=$(pgrep -af 'hpo\.py.*features_mixed' | grep -v 'hpo_ping' | head -1)
    relaunch=" [RELAUNCHED]"
  fi
  local pid ps_line elapsed rss_kb pcpu rss_gb latest last_log
  local done_cells last_cell free_mb used_mb disk_free
  pid=$(awk '{print $1}' <<<"$proc")
  ps_line=$(ps -o etime=,rss=,pcpu= -p "$pid" 2>/dev/null)
  elapsed=$(awk '{print $1}' <<<"$ps_line")
  rss_kb=$(awk '{print $2}' <<<"$ps_line")
  pcpu=$(awk '{print $3}' <<<"$ps_line")
  rss_gb=$(awk -v r="${rss_kb:-0}" 'BEGIN{printf "%.1f", r/1024/1024}')
  latest=$(ls -t logs/mixed_hpo_*.log 2>/dev/null | head -1)
  last_log=$(tail -1 "$latest" 2>/dev/null | head -c 140)
  done_cells=$(ls results/noisy_mixed/hpo/*.json 2>/dev/null | wc -l)
  last_cell=$(ls -t results/noisy_mixed/hpo/*.json 2>/dev/null | head -1 | xargs -I{} basename {} .json)
  free_mb=$(free -m | awk '/^Mem:/ {print $7}')
  used_mb=$(free -m | awk '/^Mem:/ {print $3}')
  disk_free=$(df -BG /home/user 2>/dev/null | awk 'NR==2 {print $4}')
  echo "============================================================"
  echo "PING $ts$relaunch"
  echo "  hpo.py pid=$pid etime=$elapsed rss=${rss_gb}GB cpu=${pcpu}%"
  echo "  cells: $done_cells/40 done   last cell: ${last_cell:-none}"
  echo "  vm: ram free=${free_mb}MB used=${used_mb}MB   disk free=${disk_free:-?}"
  echo "  log: ${last_log:-(no log lines)}"
  echo "============================================================"
}

while true; do
  emit_ping
  auto_commit
  # 9 inner ticks of 60s, each running auto_commit, then one more sleep 60
  # to round out 600s before the next verbose ping.
  for _ in 1 2 3 4 5 6 7 8 9; do
    sleep 60
    auto_commit
  done
  sleep 60
done
