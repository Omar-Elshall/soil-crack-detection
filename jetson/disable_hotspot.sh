#!/usr/bin/env bash
# disable_hotspot.sh — Switch the Jetson back from hotspot to regular WiFi.
#
# Use this when you need the Jetson to have internet again (e.g. for git pull,
# apt install, etc.).
#
# Run from the laptop while connected to the Jetson's hotspot (so SSH works
# at 10.42.0.1). After running, the Jetson rejoins your regular WiFi and the
# laptop will need to do the same.
#
# Usage:
#   WIFI=YourHomeNetwork bash jetson/disable_hotspot.sh
#
# Then switch the laptop's WiFi back to that same network. The Jetson will be
# at whatever IP your router assigned (run `ssh jetson 'hostname -I'` to find).

SSH_HOST="${SSH_HOST:-10.42.0.1}"
WIFI="${WIFI:?set WIFI=YourHomeNetworkName before running}"

set -e

ssh "$SSH_HOST" "
  sudo nmcli connection modify Hotspot connection.autoconnect no
  sudo nmcli connection down Hotspot
  nohup sudo bash -c 'sleep 3 && nmcli connection up \"$WIFI\"' > /tmp/wifi-switch.log 2>&1 &
  disown
  exit 0
"

echo
echo "Jetson is rejoining '$WIFI' in a few seconds."
echo "Switch your laptop's WiFi back to '$WIFI' to reconnect."
