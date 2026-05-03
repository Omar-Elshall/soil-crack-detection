@echo off
:: install_laptop_autostart.bat — RUN ONCE.
::
:: 1. Self-elevates (one UAC prompt — accept it)
:: 2. Registers a Scheduled Task that runs the elevated PowerShell watcher
::    on every Windows logon. No further prompts ever.
:: 3. Starts the watcher immediately so you don't have to log out.
::
:: Watcher script: %~dp0laptop_autoconnect.ps1

setlocal

set "SCRIPT_DIR=%~dp0"
set "PS1=%SCRIPT_DIR%laptop_autoconnect.ps1"
set "TASK_NAME=SoilCrackAutoConnect"

:: Self-elevate
net session >nul 2>&1
if %errorlevel% NEQ 0 (
  echo Requesting admin (UAC prompt^)...
  powershell -Command "Start-Process -Verb RunAs -FilePath '%~f0'"
  exit /b 0
)

if not exist "%PS1%" (
  echo ERROR: %PS1% not found.
  pause & exit /b 1
)

echo Installing scheduled task '%TASK_NAME%' to run the watcher elevated at every logon...

:: Delete any existing task with the same name
schtasks /Delete /TN "%TASK_NAME%" /F >nul 2>&1

:: Create scheduled task: at logon, highest privileges, run hidden
schtasks /Create ^
  /TN "%TASK_NAME%" ^
  /SC ONLOGON ^
  /RL HIGHEST ^
  /TR "powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File \"%PS1%\"" ^
  /F >nul

if %errorlevel% NEQ 0 (
  echo ERROR: Failed to register scheduled task.
  pause & exit /b 1
)

echo OK.
echo.
echo Starting the watcher now (no need to log out)...
schtasks /Run /TN "%TASK_NAME%" >nul 2>&1

echo.
echo Done. From now on, the watcher launches at every Windows logon.
echo It runs in the background — no terminal window opens.
echo.
echo To check it's running:
echo   tasklist ^| findstr powershell
echo To remove the autostart:
echo   schtasks /Delete /TN "%TASK_NAME%" /F
echo.
pause
