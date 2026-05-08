@echo off
:: ============================================================
:: serve_lan.bat  —  Jupyter Lab accessible over the LAN
:: ============================================================
setlocal
set "VENV_DIR=%~dp0.venv"
set "NB_DIR=%~dp0"

if not exist "%VENV_DIR%\Scripts\jupyter.exe" (
    echo [ERROR] .venv not found. Run setup_env.bat first.
    pause & exit /b 1
)

:: Print LAN address hint
echo.
echo  ================================================================
echo   Starting Jupyter Lab  --  accessible on your LAN
echo  ================================================================
echo.
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /C:"IPv4"') do (
    for /f "tokens=1" %%b in ("%%a") do (
        echo   Open on this machine :  http://127.0.0.1:8888/lab
        echo   Open on LAN devices  :  http://%%b:8888/lab
        echo.
    )
)
echo  (No token required — lab runs with --IdentityProvider.token="" )
echo  Press Ctrl+C in this window to stop the server.
echo.

cd /d "%NB_DIR%"
"%VENV_DIR%\Scripts\jupyter.exe" lab ^
    --ip=0.0.0.0 ^
    --port=8888 ^
    --no-browser ^
    --IdentityProvider.token="" ^
    --ServerApp.password="" ^
    --notebook-dir="%NB_DIR%"

endlocal
