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
    relaunch=" [relaunched]"
  else
    relaunch=""
  fi
  pid=$(awk '{print $1}' <<<"$proc")
  elapsed=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')
  latest=$(ls -t logs/mixed_hpo_*.log 2>/dev/null | head -1)
  tail_lines=$(tail -3 "$latest" 2>/dev/null | tr '\n' '|')
  done_cells=$(find results/noisy_mixed -name best.json 2>/dev/null | wc -l)
  mem=$(free -m | awk '/^Mem:/ {printf "free=%dMB used=%dMB", $7, $3}')
  printf "PING %s%s pid=%s etime=%s done=%s/40 %s | log: %s\n" \
    "$ts" "$relaunch" "$pid" "$elapsed" "$done_cells" "$mem" "$tail_lines"
  sleep 600
done
