#!/usr/bin/env bash
# enable_hotspot.sh — Switch the Jetson from regular WiFi to its own hotspot.
#
# Run this from the laptop while the laptop is still on the SAME WiFi as the
# Jetson (so SSH still works). After running, the Jetson drops the regular WiFi
# and starts broadcasting "soil-crack-demo".
#
# Usage:
#   bash jetson/enable_hotspot.sh
#
# Then:
#   1. Switch the laptop's WiFi to "soil-crack-demo" (password: cracksoil2026)
#   2. Run: SSH_HOST=10.42.0.1 bash jetson/demo_start.sh
#
# To switch back (Jetson rejoins regular WiFi):
#   bash jetson/disable_hotspot.sh

SSH_HOST="${SSH_HOST:-jetson}"

set -e

echo "==> Switching Jetson to hotspot mode"
echo

# Set hotspot autoconnect, then activate it via a backgrounded sudo so the
# SSH command can return cleanly before the WiFi switches and our connection
# drops. The 'sleep 3' lets the SSH session close first.
ssh "$SSH_HOST" '
  sudo nmcli connection modify Hotspot connection.autoconnect yes
  sudo nmcli connection modify Hotspot connection.autoconnect-priority 100
  echo "Scheduling hotspot switch in 3s (SSH will drop)..."
  nohup sudo bash -c "sleep 3 && nmcli connection up Hotspot" > /tmp/hotspot-switch.log 2>&1 &
  disown
  exit 0
'

echo
echo "Hotspot will activate in a few seconds."
echo "On the laptop:"
echo "  1. Switch WiFi to 'soil-crack-demo'  (password: cracksoil2026)"
echo "  2. Run: SSH_HOST=10.42.0.1 bash jetson/demo_start.sh"
echo
echo "If you ever need to verify from the Jetson directly:"
echo "  cat /tmp/hotspot-switch.log"
