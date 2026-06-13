@echo off
:: ─────────────────────────────────────────────────────────────────────────────
:: CryptoTracker Pro — Daily Prediction Runner
:: Runs prediction_tracker.py and logs output to prediction_log.txt
:: ─────────────────────────────────────────────────────────────────────────────

set SCRIPT_DIR=%~dp0
set LOG_FILE=%SCRIPT_DIR%prediction_log.txt

echo [%DATE% %TIME%] Starting daily prediction run >> "%LOG_FILE%"

:: Try 'python' first (standard Windows install), fall back to 'python3'
where python >nul 2>&1
if %ERRORLEVEL% == 0 (
    python "%SCRIPT_DIR%prediction_tracker.py" >> "%LOG_FILE%" 2>&1
) else (
    python3 "%SCRIPT_DIR%prediction_tracker.py" >> "%LOG_FILE%" 2>&1
)

if %ERRORLEVEL% == 0 (
    echo [%DATE% %TIME%] Run completed successfully >> "%LOG_FILE%"
) else (
    echo [%DATE% %TIME%] Run FAILED with exit code %ERRORLEVEL% >> "%LOG_FILE%"
)
