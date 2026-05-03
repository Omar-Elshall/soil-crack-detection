@echo off
:: uninstall_laptop_autostart.bat — Remove the autostart entry created by
:: install_laptop_autostart.bat.

setlocal
set "STARTUP_DIR=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "VBS_FILE=%STARTUP_DIR%\soil-crack-autoconnect.vbs"

if exist "%VBS_FILE%" (
  del "%VBS_FILE%"
  echo Removed: %VBS_FILE%
) else (
  echo No autostart entry found at %VBS_FILE%
)

echo.
echo Note: this only removes the autostart trigger. Any currently-running
echo watcher process keeps going until you reboot or kill it manually:
echo   wsl pkill -f laptop_autoconnect.sh

pause
