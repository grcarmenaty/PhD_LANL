#!/usr/bin/env bash
# Emits one multi-line status block every 600s; restarts hpo.py if dead.
cd /home/user/PhD_LANL
while true; do
  ts=$(date -u +%H:%M:%SZ)
  proc=$(pgrep -af 'hpo\.py.*features_mixed' | grep -v 'hpo_ping' | head -1)
  if [ -z "$proc" ]; then
    LOG="logs/mixed_hpo_resume_$(date -u +%Y%m%d_%H%M%S).log"
    setsid nohup python ml_pipeline/hpo.py \
        --features dataset/features_mixed.h5 \
        --out results/noisy_mixed > "$LOG" 2>&1 < /dev/null &
    disown
    sleep 2
    proc=$(pgrep -af 'hpo\.py.*features_mixed' | grep -v 'hpo_ping' | head -1)
    relaunch=" [RELAUNCHED]"
  else
    relaunch=""
  fi
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
  cur_feat=$(grep -oE 'loading feature: [a-z_]+' "$latest" 2>/dev/null | tail -1 | awk '{print $3}')
  free_mb=$(free -m | awk '/^Mem:/ {print $7}')
  used_mb=$(free -m | awk '/^Mem:/ {print $3}')
  disk_free=$(df -BG /home/user 2>/dev/null | awk 'NR==2 {print $4}')
  echo "============================================================"
  echo "PING $ts$relaunch"
  echo "  hpo.py pid=$pid etime=$elapsed rss=${rss_gb}GB cpu=${pcpu}%"
  echo "  cells: $done_cells/40 done   feature loaded: ${cur_feat:-none}   last cell: ${last_cell:-none}"
  echo "  vm: ram free=${free_mb}MB used=${used_mb}MB   disk free=${disk_free:-?}"
  echo "  log: ${last_log:-(no log lines)}"
  # Auto-commit any new HPO artefacts so they're durable through VM reboots.
  new_files=$(git ls-files --others --exclude-standard results/noisy_mixed/ 2>/dev/null)
  if [ -n "$new_files" ]; then
    n_new=$(wc -l <<<"$new_files")
    git add results/noisy_mixed/ >/dev/null 2>&1
    git -c user.email=claude@local -c user.name=claude \
        commit -m "noisy_mixed: auto-commit +${n_new} cell artefacts ($ts)" \
        >/dev/null 2>&1
    if git push origin HEAD >/dev/null 2>&1; then
      echo "  auto-commit: +${n_new} files pushed"
    else
      echo "  auto-commit: +${n_new} files committed locally (push failed)"
    fi
  fi
  echo "============================================================"
  sleep 600
done
