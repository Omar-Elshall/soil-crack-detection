#!/usr/bin/env bash
# share_wifi_with_jetson.sh — One-button "give the drone my WiFi creds"
#
# Run this on the demo laptop (in WSL) when the Jetson is in hotspot fallback
# (its known WiFi profiles couldn't reach). The script:
#
#   1. Reads laptop's current WiFi SSID + password (Windows: netsh)
#   2. Switches laptop to the Jetson's hotspot ("soil-crack-demo")
#   3. SSHs to the Jetson at 10.42.0.1, runs nmcli to add + switch to the
#      laptop's WiFi credentials
#   4. Switches laptop back to its original WiFi
#   5. Verifies the Jetson is reachable on the new WiFi
#
# Usage:
#   bash jetson/share_wifi_with_jetson.sh                 # full run
#   bash jetson/share_wifi_with_jetson.sh --dry-run       # show what would happen, don't change WiFi
#
# Requirements on the laptop (in addition to the standard demo setup):
#   - Elevated WSL/PowerShell (so netsh wlan show profile key=clear can
#     return the WiFi password). Without admin, the script falls back to
#     prompting you to type the password.
#   - SSH key authorised on the Jetson (we use sdp-w-nano@10.42.0.1)
#   - Windows-side: jetson's "soil-crack-demo" WiFi profile must already
#     exist (Windows remembers it once you've joined it before). If the
#     laptop has never joined it, the script imports a profile XML.

set -u

DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

JETSON_HOTSPOT_SSID="soil-crack-demo"
JETSON_HOTSPOT_PASS="cracksoil2026"
JETSON_HOTSPOT_IP="10.42.0.1"
SSH_USER="sdp-w-nano"

c_g() { printf "\033[32m%s\033[0m" "$*"; }
c_y() { printf "\033[33m%s\033[0m" "$*"; }
c_r() { printf "\033[31m%s\033[0m" "$*"; }
c_d() { printf "\033[2m%s\033[0m"  "$*"; }

run() {
  if [ "$DRY_RUN" = "1" ]; then
    echo "    $(c_d "[dry-run] $*")"
    return 0
  fi
  eval "$@"
}

# ── 1. Read laptop's current WiFi credentials ────────────────────────────────
echo "==> $(c_g 'Step 1') reading current laptop WiFi"

if ! command -v powershell.exe >/dev/null 2>&1; then
  echo "    $(c_r 'ERROR') powershell.exe not in PATH — WSL interop broken."
  echo "    Try: sudo bash -c 'echo :WSLInterop:M::MZ::/init:PF > /proc/sys/fs/binfmt_misc/register'"
  exit 1
fi

CURRENT_SSID=$(powershell.exe -NoProfile -Command "(Get-NetConnectionProfile | Where-Object {\$_.IPv4Connectivity -eq 'Internet'} | Select-Object -First 1).Name" 2>/dev/null | tr -d '\r\n ')

if [ -z "$CURRENT_SSID" ]; then
  echo "    $(c_r 'ERROR') no internet-connected WiFi found. Connect to your demo WiFi first."
  exit 1
fi
echo "    SSID: $(c_g "$CURRENT_SSID")"

# Try to read password (admin needed). Fall back to prompt.
CURRENT_PASS=$(powershell.exe -NoProfile -Command "(netsh wlan show profile name=\\\"$CURRENT_SSID\\\" key=clear | Select-String 'Key Content').Line.Split(':')[1].Trim()" 2>/dev/null | tr -d '\r\n')
if [ -z "$CURRENT_PASS" ]; then
  echo "    $(c_y 'No admin') — could not read password from netsh."
  echo -n "    Enter the WiFi password for '$CURRENT_SSID': "
  read -s CURRENT_PASS
  echo
fi
if [ -z "$CURRENT_PASS" ]; then
  echo "    $(c_r 'ERROR') no password available. Aborting."
  exit 1
fi
echo "    Password: $(c_d '<got it>')"
echo

# ── 2. Ensure Jetson hotspot WiFi profile exists on laptop ───────────────────
echo "==> $(c_g 'Step 2') ensuring '$JETSON_HOTSPOT_SSID' profile exists on laptop"
PROFILES=$(powershell.exe -NoProfile -Command "netsh wlan show profiles | Select-String 'All User Profile'" 2>/dev/null | tr -d '\r')
if echo "$PROFILES" | grep -q "$JETSON_HOTSPOT_SSID"; then
  echo "    $(c_g 'profile exists')"
else
  echo "    $(c_y 'profile not found — adding')"
  # Build a temp WiFi profile XML and import
  TMPXML=$(mktemp --suffix=.xml)
  cat > "$TMPXML" <<XML
<?xml version="1.0"?>
<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">
  <name>$JETSON_HOTSPOT_SSID</name>
  <SSIDConfig><SSID><name>$JETSON_HOTSPOT_SSID</name></SSID></SSIDConfig>
  <connectionType>ESS</connectionType>
  <connectionMode>auto</connectionMode>
  <MSM><security>
    <authEncryption><authentication>WPA2PSK</authentication><encryption>AES</encryption><useOneX>false</useOneX></authEncryption>
    <sharedKey><keyType>passPhrase</keyType><protected>false</protected><keyMaterial>$JETSON_HOTSPOT_PASS</keyMaterial></sharedKey>
  </security></MSM>
</WLANProfile>
XML
  WIN_TMP=$(wslpath -w "$TMPXML")
  run "powershell.exe -NoProfile -Command \"netsh wlan add profile filename='$WIN_TMP'\""
  rm -f "$TMPXML"
fi
echo

# ── 3. Switch laptop to Jetson hotspot ───────────────────────────────────────
echo "==> $(c_g 'Step 3') switching laptop WiFi to '$JETSON_HOTSPOT_SSID'"
run "powershell.exe -NoProfile -Command \"netsh wlan connect name='$JETSON_HOTSPOT_SSID'\""

# Wait for connection, then for SSH to Jetson to come up
echo -n "    waiting for Jetson at $JETSON_HOTSPOT_IP..."
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  sleep 2
  if [ "$DRY_RUN" = "1" ]; then echo " $(c_d '[dry-run] skipped')"; break; fi
  if ssh -o ConnectTimeout=2 -o BatchMode=yes "$SSH_USER@$JETSON_HOTSPOT_IP" 'echo up' >/dev/null 2>&1; then
    echo " $(c_g 'reached')"
    break
  fi
  printf "."
  if [ $i = 15 ]; then echo " $(c_r 'TIMED OUT')"; exit 2; fi
done
echo

# ── 4. Push WiFi creds to Jetson ─────────────────────────────────────────────
echo "==> $(c_g 'Step 4') pushing creds to Jetson — joining '$CURRENT_SSID'"
run "ssh '$SSH_USER@$JETSON_HOTSPOT_IP' \"
  sudo nmcli connection modify Hotspot connection.autoconnect-priority 1 2>/dev/null
  echo 'Adding WiFi profile + switching in 4s (SSH will drop)...'
  nohup sudo bash -c 'sleep 4 && nmcli device wifi connect \\\"$CURRENT_SSID\\\" password \\\"$CURRENT_PASS\\\"' > /tmp/wifi-share.log 2>&1 &
  disown
  exit 0
\""
echo

# ── 5. Switch laptop back to its original WiFi ───────────────────────────────
echo "==> $(c_g 'Step 5') switching laptop back to '$CURRENT_SSID'"
sleep 4   # give Jetson time to start switching
run "powershell.exe -NoProfile -Command \"netsh wlan connect name='$CURRENT_SSID'\""

echo -n "    waiting for laptop to be back on '$CURRENT_SSID'..."
for i in 1 2 3 4 5 6 7 8 9 10; do
  sleep 2
  if [ "$DRY_RUN" = "1" ]; then echo " $(c_d '[dry-run] skipped')"; break; fi
  CUR=$(powershell.exe -NoProfile -Command "(Get-NetConnectionProfile | Where-Object {\$_.IPv4Connectivity -eq 'Internet'} | Select-Object -First 1).Name" 2>/dev/null | tr -d '\r\n ')
  if [ "$CUR" = "$CURRENT_SSID" ]; then echo " $(c_g 'ok')"; break; fi
  printf "."
done
echo

# ── 6. Verify Jetson is now reachable on the new WiFi ────────────────────────
echo "==> $(c_g 'Step 6') verifying Jetson reachable on '$CURRENT_SSID'"
if [ "$DRY_RUN" = "1" ]; then
  echo "    $(c_d '[dry-run] would ssh jetson')"
else
  for i in 1 2 3 4 5 6 7 8 9 10; do
    sleep 3
    if ssh -o ConnectTimeout=3 -o BatchMode=yes jetson 'echo up' >/dev/null 2>&1 || \
       ssh -o ConnectTimeout=3 -o BatchMode=yes "$SSH_USER@soilcrack.local" 'echo up' >/dev/null 2>&1; then
      echo "    $(c_g 'Jetson reachable on shared WiFi.')"
      echo "    Browser: $(c_g 'http://soilcrack.local:5173')"
      exit 0
    fi
    printf "."
  done
  echo " $(c_y 'still propagating — may take another 15s for Jetson to fully join.')"
fi

echo
echo "Done. If anything's wrong, check Jetson directly:  ssh jetson 'hostname -I'"
