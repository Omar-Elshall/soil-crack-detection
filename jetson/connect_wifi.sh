#!/usr/bin/env bash
# connect_wifi.sh — Add a new WiFi network to the Jetson and connect to it.
#
# Use case: at a demo location with an unfamiliar WiFi. The Jetson's hotspot
# is the fallback (auto-starts when no known WiFi is reachable), so:
#   1. Power on Jetson at demo location
#   2. Jetson's known WiFi networks fail to connect → Hotspot takes over
#   3. Laptop joins SSID "soil-crack-demo" (password cracksoil2026)
#   4. Run this script via the hotspot to add the demo location's WiFi
#   5. Jetson reconnects to the new WiFi; laptop also joins that WiFi
#   6. Browse to http://soilcrack.local:5173
#
# Usage:
#   bash jetson/connect_wifi.sh "DemoSSID" "demopassword"
#   SSH_HOST=10.42.0.1 bash jetson/connect_wifi.sh "DemoSSID" "demopassword"

SSH_HOST="${SSH_HOST:-jetson}"
SSID="${1:?Usage: $0 SSID PASSWORD}"
PASSWORD="${2:?Usage: $0 SSID PASSWORD}"

set -e

echo "==> Adding WiFi '$SSID' on the Jetson"
ssh "$SSH_HOST" "
  # Stop fighting: disable hotspot autoconnect so it stays out of the way.
  sudo nmcli connection modify Hotspot connection.autoconnect no
  sudo nmcli connection down Hotspot 2>/dev/null || true

  # Add the new WiFi connection. nmcli will save the profile so it auto-
  # reconnects on future boots even after Jetson is rebooted.
  echo 'Scheduling WiFi switch in 3s (SSH may drop)...'
  nohup sudo bash -c 'sleep 3 && nmcli device wifi connect \"$SSID\" password \"$PASSWORD\"' > /tmp/wifi-connect.log 2>&1 &
  disown
  exit 0
"

echo
echo "Switching now. Reconnect your laptop to '$SSID'."
echo "Then test:"
echo "  ssh sdp-w-nano@soilcrack.local 'hostname -I'"
echo
echo "If the Jetson failed to join, check /tmp/wifi-connect.log on the Jetson."
echo "To re-enable the hotspot fallback later:"
echo "  ssh jetson 'sudo nmcli connection modify Hotspot connection.autoconnect yes'"
