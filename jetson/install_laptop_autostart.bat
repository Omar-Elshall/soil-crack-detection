@echo off
:: install_laptop_autostart.bat — Run ONCE on the demo laptop.
:: Installs a Windows autostart entry that launches laptop_autoconnect.sh
:: invisibly inside WSL on every login. No terminal window opens.
::
:: Usage:
::   1. Open this file in File Explorer and double-click, OR
::   2. From cmd: install_laptop_autostart.bat
::
:: To uninstall, delete the .vbs file from the Startup folder shown below.

setlocal

:: WSL path to the watcher script (relative to wherever the user cloned the repo)
:: Edit REPO_PATH below if your clone lives somewhere else.
set "REPO_PATH=~/soil-crack-detection"
set "WATCHER=%REPO_PATH%/jetson/laptop_autoconnect.sh"

:: Windows Startup folder
set "STARTUP_DIR=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "VBS_FILE=%STARTUP_DIR%\soil-crack-autoconnect.vbs"

echo Installing autostart entry...
echo   Repo (in WSL): %REPO_PATH%
echo   Startup file:  %VBS_FILE%
echo.

:: Write a tiny VBS that runs WSL invisibly. The 0 in WshShell.Run hides the window.
> "%VBS_FILE%" echo Set WshShell = CreateObject("WScript.Shell")
>>"%VBS_FILE%" echo WshShell.Run "wsl.exe -e bash -c ""bash %WATCHER% ^>^> /tmp/laptop_autoconnect.log 2^>^&1""", 0, False

if exist "%VBS_FILE%" (
  echo.
  echo Done. The watcher will start automatically on every Windows login.
  echo Logs: in WSL run 'tail -f /tmp/laptop_autoconnect.log'
  echo.
  echo Starting it now so you don't have to log out and back in...
  cscript //nologo "%VBS_FILE%"
  echo.
  echo Watcher is running in the background. Plug the USB-C cable into the
  echo Jetson and the browser will open + WiFi creds will be shared.
) else (
  echo ERROR: Could not write %VBS_FILE%. Check permissions.
  exit /b 1
)

pause
