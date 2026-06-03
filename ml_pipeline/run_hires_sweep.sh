#!/usr/bin/env bash
# Autonomous per-cell hi-res 1601 CFDAC sweep driver.
# Runs the top-CFDAC cell per task one at a time (skip-if-exists), and
# commits + pushes results to main after EACH cell so a container reclaim
# loses at most one in-flight cell. Order: col_location first (explicit
# instruction), then the cheaper cnn2d cells, then the vision backbones.
set -u
cd /home/user/PhD_LANL

TASKS=(col_location mass_location severity type is_bolt is_crack is_mass is_hole is_pristine)
LOG=results_hires/sweep.log

echo "=== hires sweep start $(date -u +%H:%M:%S)UTC ===" | tee -a "$LOG"
for task in "${TASKS[@]}"; do
  echo "--- [$task] start $(date -u +%H:%M:%S)UTC ---" | tee -a "$LOG"
  python3 ml_pipeline/train_hires_top_cells.py --tasks "$task" >> "$LOG" 2>&1
  rc=$?
  echo "--- [$task] python rc=$rc $(date -u +%H:%M:%S)UTC ---" | tee -a "$LOG"
  # Commit whatever landed (result JSONs only; sweep.log is gitignored so
  # the working tree stays clean between cells).
  git add results_hires/per_case/*.json results_hires/synth_test.json 2>/dev/null
  if ! git diff --cached --quiet; then
    git commit -q -m "hires sweep: ${task} cell @1601 (synth + exp)" \
      && echo "[$task] committed" | tee -a "$LOG"
    n=0
    until git push -u origin main 2>>"$LOG"; do
      n=$((n+1)); [ "$n" -ge 4 ] && { echo "[$task] push failed x4" | tee -a "$LOG"; break; }
      sleep $((2**n))
    done
  else
    echo "[$task] no changes (skipped)" | tee -a "$LOG"
  fi
done
echo "=== hires sweep done $(date -u +%H:%M:%S)UTC ===" | tee -a "$LOG"
touch results_hires/.sweep_complete
