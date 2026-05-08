@echo off
title 3SBB git sync
cd /d "C:\Users\user\Documents\Claude\Projects\PhD\3SBB_data"

echo === Removing stale lock files ===
if exist ".git\index.lock"      del /f ".git\index.lock"
if exist ".git\HEAD.lock"       del /f ".git\HEAD.lock"
if exist ".git\config.lock"     del /f ".git\config.lock"
if exist ".git\refs\heads\main.lock" del /f ".git\refs\heads\main.lock"
echo Done.

echo.
echo === git pull ===
git pull origin main
if %ERRORLEVEL% NEQ 0 (
    echo PULL FAILED - check above
    pause & exit 1
)

echo.
echo === Updating .gitignore (only block large raw files) ===
(
echo # ── Large raw data files ^(too big for GitHub^) ──────────────────────
echo experimental_frfs.h5
echo raw_frfs.npz
echo.
echo # ── Python ───────────────────────────────────────────────────────────
echo __pycache__/
echo *.py[cod]
echo *.egg-info/
echo.
echo # ── Virtual environment ──────────────────────────────────────────────
echo .venv/
echo.
echo # ── Jupyter ──────────────────────────────────────────────────────────
echo .ipynb_checkpoints/
echo.
echo # ── OS ────────────────────────────────────────────────────────────────
echo .DS_Store
echo Thumbs.db
echo test_write.tmp
echo *.swp
echo *~
) > .gitignore

echo.
echo === Staging changes ===
git add -A
echo.
echo === Diff summary ===
git status --short

echo.
echo === Committing ===
git commit -m "Add median_frfs.h5 (61 scenarios x 9 channels), update .gitignore, recalibration results"

echo.
echo === Pushing ===
git push origin main
if %ERRORLEVEL% NEQ 0 (
    echo PUSH FAILED
    pause & exit 1
)

echo.
echo === DONE ===
echo https://github.com/grcarmenaty/PhD_LANL
pause
