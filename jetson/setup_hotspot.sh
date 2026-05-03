#!/usr/bin/env bash
# setup_hotspot.sh — Configure the Jetson to broadcast its own WiFi network
# so the laptop can reach it without any infrastructure WiFi.
#
# Run this ON THE JETSON, once:
#   bash jetson/setup_hotspot.sh
#
# After running:
#   - Jetson broadcasts SSID "soil-crack-demo" (password set below)
#   - Jetson's address on that network is 10.42.0.1
#   - Hotspot is set to auto-start on boot
#   - mDNS is installed so the Jetson is also reachable as <hostname>.local
#
# On the laptop:
#   1. Disconnect from any other WiFi
#   2. Connect to the SSID "soil-crack-demo" with the password
#   3. SSH via 10.42.0.1 (or <hostname>.local if mDNS works)
#   4. Browser → http://10.42.0.1:5173
#
# To revert (use a regular WiFi instead):
#   sudo nmcli connection down Hotspot
#   sudo nmcli connection modify Hotspot connection.autoconnect no
#   sudo nmcli connection up <name-of-your-home-wifi>
#
# To delete entirely:
#   sudo nmcli connection delete Hotspot

SSID="${SSID:-soil-crack-demo}"
PASSWORD="${PASSWORD:-cracksoil2026}"
WIFI_IFACE="${WIFI_IFACE:-wlan0}"

set -e

echo "==> Setting up hotspot on $WIFI_IFACE"
echo "    SSID:     $SSID"
echo "    Password: $PASSWORD"
echo

# 1. Create the hotspot. NetworkManager auto-assigns 10.42.0.1/24 + DHCP/DNS.
echo "[1/4] Creating hotspot..."
sudo nmcli device wifi hotspot ifname "$WIFI_IFACE" ssid "$SSID" password "$PASSWORD"

# 2. Make it auto-start on boot AS A FALLBACK (low priority so a known
# regular WiFi wins when reachable; hotspot only kicks in when nothing
# else is available).
echo "[2/4] Setting auto-connect on boot (low priority — fallback only)..."
sudo nmcli connection modify Hotspot connection.autoconnect yes
sudo nmcli connection modify Hotspot connection.autoconnect-priority 1

# 3. Install avahi (mDNS) so the laptop can also reach us by hostname.
echo "[3/4] Ensuring mDNS (avahi-daemon) is installed and running..."
if ! dpkg -s avahi-daemon >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y avahi-daemon
fi
sudo systemctl enable --now avahi-daemon

# 4. Show summary.
HOSTNAME=$(hostname)
echo
echo "[4/4] Done."
echo
echo "    Jetson SSID:        $SSID"
echo "    Password:           $PASSWORD"
echo "    Jetson IP:          10.42.0.1"
echo "    Hostname:           $HOSTNAME      (try $HOSTNAME.local from laptop)"
echo
echo "On the laptop:"
echo "  1. Connect to WiFi: $SSID  (password: $PASSWORD)"
echo "  2. Test SSH:        ssh sdp-w-nano@10.42.0.1"
echo "                  or: ssh sdp-w-nano@$HOSTNAME.local"
echo "  3. Open browser:    http://10.42.0.1:5173"
echo "                  or: http://$HOSTNAME.local:5173"
