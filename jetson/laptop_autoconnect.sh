#!/usr/bin/env bash
# laptop_autoconnect.sh — Run this on the laptop's WSL.
# Watches for the Jetson over USB ethernet (192.168.55.1). When the cable is
# plugged in:
#   1. (best-effort) reads the laptop's current WiFi creds from Windows and
#      tells the Jetson to join that WiFi
#   2. opens the browser to http://soilcrack.local:5173
# When the cable is unplugged, returns to watching.
#
# Usage:
#   bash jetson/laptop_autoconnect.sh
#
# Leave it running in a terminal. Ctrl+C to stop.
#
# Prereqs on the laptop:
#   - WSL with `ssh` configured (key auth to Jetson at 192.168.55.1)
#   - Windows PowerShell + WSL interop (the script calls powershell.exe)
#   - cmd.exe accessible from PATH (for opening browser)
#
# Note on WiFi sharing: reading the WiFi password from Windows requires
# admin privileges (`netsh wlan show profile name=X key=clear`). If the
# script can't read the password, it just opens the browser and leaves the
# Jetson on whichever WiFi it auto-joined.

JETSON_IP="${JETSON_IP:-192.168.55.1}"
URL="${URL:-http://soilcrack.local:5173}"
SSH_USER="${SSH_USER:-sdp-w-nano}"

c_g() { printf "\033[32m%s\033[0m" "$*"; }
c_y() { printf "\033[33m%s\033[0m" "$*"; }
c_d() { printf "\033[2m%s\033[0m" "$*"; }

echo "==> Watching for Jetson at $JETSON_IP (USB-C cable)"
echo "    On cable plug-in: pull latest repo + share WiFi creds + open browser"
echo "    Ctrl+C to stop"
echo

# Pull latest on startup so the laptop's clone of demo scripts stays current.
# Soft-fails so a missing internet connection doesn't kill the watcher.
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
if [ -d "$REPO_DIR/.git" ]; then
  echo "[$(date +%H:%M:%S)] $(c_d 'pulling latest scripts...')"
  ( cd "$REPO_DIR" && git pull --ff-only 2>&1 | tail -3 ) || true
fi
echo

was_up=0
while :; do
  if ssh -o ConnectTimeout=2 -o BatchMode=yes "$SSH_USER@$JETSON_IP" 'echo up' >/dev/null 2>&1; then
    if [ "$was_up" = "0" ]; then
      echo "[$(date +%H:%M:%S)] $(c_g 'Jetson detected') over USB"

      # ── Best-effort WiFi credential sharing ──────────────────────────────
      ssid=""
      key=""
      if command -v powershell.exe >/dev/null 2>&1; then
        ssid=$(powershell.exe -NoProfile -Command "(Get-NetConnectionProfile | Where-Object {\$_.IPv4Connectivity -eq 'Internet'} | Select-Object -First 1).Name" 2>/dev/null | tr -d '\r\n ')
        if [ -n "$ssid" ]; then
          # netsh wlan show profile reveals the key for known networks; admin needed.
          key=$(powershell.exe -NoProfile -Command "(netsh wlan show profile name=\\\"$ssid\\\" key=clear | Select-String 'Key Content').Line.Split(':')[1].Trim()" 2>/dev/null | tr -d '\r\n')
        fi
      fi

      if [ -n "$ssid" ] && [ -n "$key" ]; then
        echo "    Sharing WiFi creds: SSID=$(c_d "$ssid")"
        ssh "$SSH_USER@$JETSON_IP" "
          sudo nmcli connection modify Hotspot connection.autoconnect no 2>/dev/null
          sudo nmcli connection down Hotspot 2>/dev/null
          # Use & + nohup so SSH session returns even if WiFi switch drops other links
          nohup sudo bash -c 'nmcli device wifi connect \"$ssid\" password \"$key\"' > /tmp/wifi-autoconnect.log 2>&1 &
          disown
          exit 0
        " 2>/dev/null
        echo "    $(c_g 'WiFi join attempted') — Jetson now also reachable via $URL"
      else
        if [ -z "$ssid" ]; then
          echo "    $(c_y 'WiFi share skipped'): no internet-connected WiFi found"
        else
          echo "    $(c_y 'WiFi share skipped'): could not read password (run WSL/PowerShell as admin to enable)"
          echo "    SSID=$(c_d "$ssid") found, but key required admin"
        fi
      fi

      # ── Open browser ─────────────────────────────────────────────────────
      if command -v cmd.exe >/dev/null 2>&1; then
        cmd.exe /c start "" "$URL" >/dev/null 2>&1
        echo "    $(c_g 'Browser opened') → $URL"
      else
        echo "    Open this URL manually: $(c_y "$URL")"
      fi

      was_up=1
    fi
  else
    if [ "$was_up" = "1" ]; then
      echo "[$(date +%H:%M:%S)] $(c_y 'Jetson cable unplugged')"
      was_up=0
    fi
  fi
  sleep 3
done
