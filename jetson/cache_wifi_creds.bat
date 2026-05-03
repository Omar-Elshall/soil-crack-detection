@echo off
:: cache_wifi_creds.bat — Run ONCE per WiFi network you want the watcher to
:: silently bootstrap with. Right-click → "Run as administrator" so netsh can
:: read the saved keys.
::
:: Dumps every saved WiFi profile's password into the watcher's cache so
:: laptop_autoconnect.sh never has to ask.

setlocal EnableDelayedExpansion

:: Self-elevate if not running as admin
net session >nul 2>&1
if %errorlevel% NEQ 0 (
  echo Requesting admin (UAC prompt^)...
  powershell -Command "Start-Process -Verb RunAs -FilePath '%~f0'"
  exit /b 0
)

echo Caching WiFi credentials for the watcher...
echo.

:: Find WSL home and the cache dir we want to write into
for /f "delims=" %%H in ('wsl.exe -e bash -c "echo $HOME"') do set "WSL_HOME=%%H"
set "CACHE_DIR=%WSL_HOME%/.cache/laptop_autoconnect"
wsl.exe -e bash -c "mkdir -p '%CACHE_DIR%' && chmod 700 '%CACHE_DIR%'"

:: Iterate every saved profile, extract key, write to cache
for /f "tokens=2 delims=:" %%P in ('netsh wlan show profiles ^| findstr /C:"All User Profile"') do (
  set "SSID=%%P"
  set "SSID=!SSID:~1!"
  for /f "tokens=2 delims=:" %%K in ('netsh wlan show profile name^="!SSID!" key^=clear ^| findstr /C:"Key Content"') do (
    set "KEY=%%K"
    set "KEY=!KEY:~1!"
    if not "!KEY!" == "" (
      :: Hash SSID with sha256 → first 16 chars (matches watcher convention)
      for /f "delims=" %%S in ('wsl.exe -e bash -c "echo -n \"!SSID!\" | sha256sum | cut -c1-16"') do set "SSID_HASH=%%S"
      :: Write the password into the cache file
      wsl.exe -e bash -c "echo -n '!KEY!' > '%CACHE_DIR%/!SSID_HASH!' && chmod 600 '%CACHE_DIR%/!SSID_HASH!'"
      echo   cached: !SSID!
    )
  )
)

echo.
echo Done. Watcher is now no-touch — no password prompts.
echo.
pause
