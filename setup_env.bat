@echo off
cd /d "%~dp0"

echo Running setup_env.py ...
echo.

py setup_env.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo [FAILED] Something went wrong. See error above.
)

echo.
pause
