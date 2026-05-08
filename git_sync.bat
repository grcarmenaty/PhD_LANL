@echo off
title 3SBB git push
cd /d "C:\Users\user\Documents\Claude\Projects\PhD\3SBB_data"

echo === Removing stale lock files ===
if exist ".git\index.lock"           del /f ".git\index.lock"
if exist ".git\HEAD.lock"            del /f ".git\HEAD.lock"
if exist ".git\refs\heads\main.lock" del /f ".git\refs\heads\main.lock"

echo.
echo === Staging all changes ===
git add -A

echo.
echo === What will be committed ===
git status --short

echo.
echo === Committing ===
git commit -m "Add median_frfs.h5: complex median FRF per scenario per sensor (61 scenarios x 9 channels x 1601 freq)"

echo.
echo === Pushing ===
git push origin main
if %ERRORLEVEL% NEQ 0 (
    echo PUSH FAILED
    pause & exit 1
)

echo.
echo ============================================================
echo  DONE  ^>  https://github.com/grcarmenaty/PhD_LANL
echo ============================================================
pause
