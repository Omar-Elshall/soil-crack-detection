#!/usr/bin/env bash
# laptop_autoconnect.sh — One script. Run it once. Leave it running.
#
# It does ALL of these automatically:
#   1. Probes the Jetson on every shared network (USB, jetson alias, mDNS)
#   2. If unreachable: assumes the Jetson is in hotspot fallback at the
#      venue. Reads laptop's current WiFi creds, briefly switches laptop to
#      the Jetson's hotspot, SSHs to the Jetson, runs nmcli to add the
#      laptop's WiFi, switches laptop back. Updates ~/.ssh/config so 'ssh
#      jetson' tracks the new IP.
#   3. Once Jetson is reachable, opens browser to the live UI.
#
# Usage:
#   bash jetson/laptop_autoconnect.sh
#
# Leave running in a WSL terminal. Reading WiFi password requires admin
# WSL/PowerShell; without it, the script prompts interactively once per
# bootstrap attempt.

set -u

SSH_USER="${SSH_USER:-sdp-w-nano}"
JETSON_HOTSPOT_SSID="${JETSON_HOTSPOT_SSID:-soil-crack-demo}"
JETSON_HOTSPOT_PASS="${JETSON_HOTSPOT_PASS:-cracksoil2026}"
JETSON_HOTSPOT_IP="${JETSON_HOTSPOT_IP:-10.42.0.1}"
URL_DEFAULT="${URL_DEFAULT:-http://soilcrack.local:5173}"
BOOTSTRAP_COOLDOWN_SECS="${BOOTSTRAP_COOLDOWN_SECS:-90}"

c_g() { printf "\033[32m%s\033[0m" "$*"; }
c_y() { printf "\033[33m%s\033[0m" "$*"; }
c_r() { printf "\033[31m%s\033[0m" "$*"; }
c_d() { printf "\033[2m%s\033[0m"  "$*"; }

# Probe order. First reachable target wins.
TARGETS=("${JETSON_HOST:-}" "192.168.55.1" "jetson" "soilcrack.local")

probe() {
  for h in "${TARGETS[@]}"; do
    [ -z "$h" ] && continue
    if [ "$h" = "jetson" ]; then
      ssh -o ConnectTimeout=2 -o BatchMode=yes "$h" 'echo up' >/dev/null 2>&1 && { echo "$h"; return 0; }
    else
      ssh -o ConnectTimeout=2 -o BatchMode=yes "$SSH_USER@$h" 'echo up' >/dev/null 2>&1 && { echo "$h"; return 0; }
    fi
  done
  return 1
}

open_browser() {
  local url="$1"
  if command -v cmd.exe >/dev/null 2>&1; then
    cmd.exe /c start "" "$url" 2>/dev/null && return 0
  fi
  if command -v powershell.exe >/dev/null 2>&1; then
    powershell.exe -NoProfile -Command "Start-Process '$url'" 2>/dev/null && return 0
  fi
  return 1
}

# Pull latest scripts. Soft-fail (no internet, key issue, whatever).
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo "==> Auto-onboard watcher"
echo "    Detect Jetson, bootstrap WiFi if needed, open browser. Ctrl+C to stop."
echo
if [ -d "$REPO_DIR/.git" ]; then
  echo "[$(date +%H:%M:%S)] $(c_d 'pulling latest scripts (best-effort)...')"
  ( cd "$REPO_DIR" && git pull --ff-only --quiet 2>&1 ) || echo "    $(c_d '(pull skipped/failed — keeping current)')"
fi
echo

bootstrap_via_hotspot() {
  echo "[$(date +%H:%M:%S)] $(c_y 'Jetson unreachable on shared WiFi') — bootstrapping via hotspot"

  if ! command -v powershell.exe >/dev/null 2>&1; then
    echo "    $(c_r 'WSL interop broken') — cannot drive netsh."
    echo "    Manual: bash jetson/share_wifi_with_jetson.sh"
    return 1
  fi

  local current_ssid current_pass
  current_ssid=$(powershell.exe -NoProfile -Command "(Get-NetConnectionProfile | Where-Object {\$_.IPv4Connectivity -eq 'Internet'} | Select-Object -First 1).Name" 2>/dev/null | tr -d '\r\n ')
  if [ -z "$current_ssid" ]; then
    echo "    $(c_y 'No internet WiFi found on laptop. Skipping bootstrap.')"
    return 1
  fi
  echo "    Laptop is on: $(c_g "$current_ssid")"

  # Cache file: store {SSID, password} so we never have to ask twice.
  local cache_dir="$HOME/.cache/laptop_autoconnect"
  local cache_file="$cache_dir/$(echo -n "$current_ssid" | sha256sum | cut -c1-16)"
  mkdir -p "$cache_dir" && chmod 700 "$cache_dir"

  # Try cache first (this avoids the prompt on every run)
  if [ -f "$cache_file" ]; then
    current_pass=$(cat "$cache_file" 2>/dev/null)
  fi
  # Then try netsh (works for personal profiles or with admin)
  if [ -z "$current_pass" ]; then
    current_pass=$(powershell.exe -NoProfile -Command "(netsh wlan show profile name=\\\"$current_ssid\\\" key=clear | Select-String 'Key Content').Line.Split(':')[1].Trim()" 2>/dev/null | tr -d '\r\n')
  fi
  # Last resort: prompt the user. We cache after success so this is a one-time ask.
  if [ -z "$current_pass" ]; then
    echo -n "    First-time only — WiFi password for '$current_ssid': "
    read -s current_pass
    echo
    if [ -z "$current_pass" ]; then
      echo "    $(c_r 'No password — abort bootstrap')"
      return 1
    fi
  fi
  # Cache it for next time (chmod 600 so only this user can read)
  echo -n "$current_pass" > "$cache_file" && chmod 600 "$cache_file"

  # Force-delete any existing Windows profile so we always re-create it as
  # connectionMode=manual (older versions of this script created it as auto,
  # which caused Windows to keep flipping back to the Jetson hotspot).
  powershell.exe -NoProfile -Command "netsh wlan delete profile name='$JETSON_HOTSPOT_SSID'" >/dev/null 2>&1
  echo "    $(c_d "Adding '$JETSON_HOTSPOT_SSID' profile (manual mode)...")"
  if true; then
    local tmpxml win_tmp
    tmpxml=$(mktemp --suffix=.xml)
    cat > "$tmpxml" <<XML
<?xml version="1.0"?>
<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">
<name>$JETSON_HOTSPOT_SSID</name>
<SSIDConfig><SSID><name>$JETSON_HOTSPOT_SSID</name></SSID></SSIDConfig>
<connectionType>ESS</connectionType><connectionMode>manual</connectionMode>
<MSM><security>
<authEncryption><authentication>WPA2PSK</authentication><encryption>AES</encryption><useOneX>false</useOneX></authEncryption>
<sharedKey><keyType>passPhrase</keyType><protected>false</protected><keyMaterial>$JETSON_HOTSPOT_PASS</keyMaterial></sharedKey>
</security></MSM>
</WLANProfile>
XML
    win_tmp=$(wslpath -w "$tmpxml")
    powershell.exe -NoProfile -Command "netsh wlan add profile filename='$win_tmp'" >/dev/null 2>&1
    rm -f "$tmpxml"
  fi

  echo "    Switching laptop → $(c_d "$JETSON_HOTSPOT_SSID")"
  powershell.exe -NoProfile -Command "netsh wlan connect name='$JETSON_HOTSPOT_SSID'" >/dev/null 2>&1

  # Wait for Jetson to be reachable on hotspot
  local i
  for i in $(seq 1 15); do
    sleep 2
    ssh -o ConnectTimeout=2 -o BatchMode=yes "$SSH_USER@$JETSON_HOTSPOT_IP" 'echo up' >/dev/null 2>&1 && break
    if [ "$i" = "15" ]; then
      echo "    $(c_r 'Could not reach Jetson on hotspot') — restoring laptop WiFi"
      powershell.exe -NoProfile -Command "netsh wlan connect name='$current_ssid'" >/dev/null 2>&1
      return 1
    fi
  done
  echo "    Reached Jetson on hotspot. Pushing creds..."

  ssh "$SSH_USER@$JETSON_HOTSPOT_IP" "
    sudo nmcli connection modify Hotspot connection.autoconnect-priority 1 2>/dev/null
    nohup sudo bash -c 'sleep 4 && nmcli device wifi connect \"$current_ssid\" password \"$current_pass\"' > /tmp/wifi-share.log 2>&1 &
    disown
    exit 0
  " 2>/dev/null

  echo "    Switching laptop back → $(c_d "$current_ssid")"
  sleep 4
  powershell.exe -NoProfile -Command "netsh wlan connect name='$current_ssid'" >/dev/null 2>&1

  # Resolve Jetson's new IP via Bonjour and update ~/.ssh/config
  local new_ip=""
  for i in $(seq 1 15); do
    sleep 3
    new_ip=$(powershell.exe -NoProfile -Command "(Resolve-DnsName -Name 'soilcrack.local' -ErrorAction SilentlyContinue | Where-Object {\$_.Type -eq 'A'} | Select-Object -First 1).IPAddress" 2>/dev/null | tr -d '\r\n ')
    if [ -n "$new_ip" ] && ssh -o ConnectTimeout=3 -o BatchMode=yes "$SSH_USER@$new_ip" 'echo up' >/dev/null 2>&1; then
      break
    fi
    new_ip=""
  done

  if [ -n "$new_ip" ]; then
    echo "    Jetson is at $(c_g "$new_ip")"
    local cfg=~/.ssh/config
    if grep -q "^Host jetson$" "$cfg" 2>/dev/null; then
      sed -i "/^Host jetson$/,/^Host /{s/^[[:space:]]*HostName .*/\tHostName $new_ip/}" "$cfg"
    else
      mkdir -p ~/.ssh && chmod 700 ~/.ssh
      cat >> "$cfg" <<EOF

Host jetson
	HostName $new_ip
	User $SSH_USER
	IdentityFile ~/.ssh/jetson_nano
EOF
      chmod 600 "$cfg"
    fi
    echo "    Updated ~/.ssh/config — $(c_g "'ssh jetson' now points at $new_ip")"
    return 0
  else
    echo "    $(c_y 'Bootstrap incomplete — Jetson not visible on shared WiFi yet')"
    return 1
  fi
}

# Main loop
last_bootstrap=0
was_up=0
URL=""
while :; do
  detected=$(probe || true)
  if [ -n "$detected" ]; then
    if [ "$was_up" = "0" ]; then
      echo "[$(date +%H:%M:%S)] $(c_g 'Jetson detected') via $detected"
      if [ "$detected" = "192.168.55.1" ]; then
        URL="http://192.168.55.1:5173"
      else
        URL="$URL_DEFAULT"
      fi
      if open_browser "$URL"; then
        echo "    $(c_g 'Browser opened') → $URL"
      else
        echo "    $(c_y 'Browser open failed') — open manually: $URL"
      fi
      was_up=1
    fi
  else
    if [ "$was_up" = "1" ]; then
      echo "[$(date +%H:%M:%S)] $(c_y 'Jetson unreachable')"
      was_up=0
    fi
    # Bootstrap if cooldown elapsed
    now=$(date +%s)
    if [ $((now - last_bootstrap)) -ge "$BOOTSTRAP_COOLDOWN_SECS" ]; then
      last_bootstrap=$now
      bootstrap_via_hotspot || true
    fi
  fi
  sleep 3
done
