@echo off
:: ============================================================
:: run_notebook.bat  —  launches Jupyter Lab for 3SBB_exploration.ipynb
:: ============================================================
setlocal

set "VENV_DIR=%~dp0.venv"
set "NB=%~dp03SBB_exploration.ipynb"

:: Check setup has been run
if not exist "%VENV_DIR%\Scripts\jupyter.exe" (
    echo [ERROR] Environment not found. Run setup_env.bat first.
    pause & exit /b 1
)

echo Opening 3SBB_exploration.ipynb in Jupyter Lab ...
echo (Press Ctrl+C in this window to stop the server)
echo.

:: Launch from the 3SBB_data directory so DATA_DIR = Path('.') resolves correctly
cd /d "%~dp0"
"%VENV_DIR%\Scripts\jupyter.exe" lab "%NB%"

endlocal
