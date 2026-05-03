#!/usr/bin/env bash
# demo_start.sh — bring up the whole demo from a fresh laptop session.
#
# Run from anywhere on a laptop that has:
#   - WSL / Linux shell
#   - SSH key configured to `ssh jetson` (works without password prompt)
#   - This repo cloned (script reads no local files; only needed for path)
#
# What it does on the Jetson:
#   1. Verifies SSH reachable
#   2. Kills any leftover services
#   3. Restarts nvargus-daemon (clears any stale camera capture session)
#   4. Launches start.sh in a detached session
#   5. Polls /status on all three services until ready (or 90s timeout)
#   6. Prints the Jetson IP + browser URL + summary of each service
#
# Defaults baked into the source code (so no env vars needed at launch):
#   - real_4 model, threshold 0.5
#   - sensor mode 1 (3840x2160 @ 30 FPS), preview 1440x1440
#   - 8 ms shutter cap, WB auto, no TNR/EE/sharpening
#
# Usage:
#   bash jetson/demo_start.sh                       # uses 'ssh jetson' from your config
#   SSH_HOST=10.42.0.1 bash jetson/demo_start.sh    # if the Jetson is in hotspot mode
#   SSH_HOST=ubuntu.local bash jetson/demo_start.sh # via mDNS (works on either network)
#
# When the Jetson is acting as a WiFi hotspot (see setup_hotspot.sh), the laptop
# joins SSID "soil-crack-demo" and the Jetson is at 10.42.0.1.

set -u

SSH_HOST="${SSH_HOST:-jetson}"
TIMEOUT="${TIMEOUT:-90}"

c_red()    { printf "\033[31m%s\033[0m" "$*"; }
c_green()  { printf "\033[32m%s\033[0m" "$*"; }
c_yellow() { printf "\033[33m%s\033[0m" "$*"; }
c_dim()    { printf "\033[2m%s\033[0m"  "$*"; }

echo "==> Soil Crack Detection — Demo Startup"
echo

# 1. SSH reachability
echo -n "[1/6] SSH to $SSH_HOST: "
if ssh -o ConnectTimeout=5 -o BatchMode=yes "$SSH_HOST" 'echo up' >/dev/null 2>&1; then
  c_green "ok"; echo
else
  c_red "FAILED"; echo
  echo "    Cannot reach the Jetson. Check that:"
  echo "    - The Jetson is powered on and on WiFi"
  echo "    - 'ssh $SSH_HOST' works without a password (key auth configured)"
  echo "    - Both devices are on the same network"
  exit 1
fi

# 2. Kill leftovers
echo -n "[2/6] Killing any leftover services: "
ssh "$SSH_HOST" "sudo fuser -k 8001/tcp 8002/tcp 8003/tcp 5173/tcp 2>/dev/null
                 pkill -9 -f uvicorn 2>/dev/null
                 pkill -9 -f 'http.server' 2>/dev/null
                 sleep 2" >/dev/null 2>&1
c_green "done"; echo

# 3. Restart nvargus-daemon (clears stuck camera state from prior runs)
echo -n "[3/6] Restarting nvargus-daemon (camera): "
ssh "$SSH_HOST" 'sudo systemctl restart nvargus-daemon; sleep 2' >/dev/null 2>&1
c_green "done"; echo

# 4. Launch start.sh detached
echo -n "[4/6] Launching all services via start.sh: "
ssh "$SSH_HOST" 'cd ~/soil-crack-detection && setsid bash jetson/start.sh > /tmp/services.log 2>&1 < /dev/null &
                 disown' >/dev/null 2>&1
c_green "launched"; echo

# 5. Poll for readiness
echo -n "[5/6] Waiting for all 4 endpoints (timeout ${TIMEOUT}s): "
deadline=$(( $(date +%s) + TIMEOUT ))
while :; do
  if ssh "$SSH_HOST" 'curl -sf -m 2 http://127.0.0.1:8001/status >/dev/null \
                   && curl -sf -m 2 http://127.0.0.1:8002/status >/dev/null \
                   && curl -sf -m 2 http://127.0.0.1:8003/status >/dev/null \
                   && curl -sf -m 2 -o /dev/null http://127.0.0.1:5173/index.html' 2>/dev/null; then
    c_green "ready"; echo
    break
  fi
  if [ $(date +%s) -ge $deadline ]; then
    c_red "TIMED OUT"; echo
    echo "    Last 25 lines of services log on the Jetson:"
    ssh "$SSH_HOST" 'tail -25 /tmp/services.log' | sed 's/^/      /'
    exit 2
  fi
  sleep 3
done

# 6. Pretty summary
echo "[6/6] Status summary:"
JETSON_IP=$(ssh "$SSH_HOST" "hostname -I | awk '{print \$1}'" 2>/dev/null)
INFERENCE=$(ssh "$SSH_HOST" 'curl -s http://127.0.0.1:8001/status' 2>/dev/null)
MAVLINK=$(ssh "$SSH_HOST" 'curl -s http://127.0.0.1:8002/status' 2>/dev/null)
DATA=$(ssh "$SSH_HOST" 'curl -s http://127.0.0.1:8003/status' 2>/dev/null)

INF_FPS=$(echo "$INFERENCE" | grep -oE '"fps":[0-9.]+' | cut -d: -f2)
INF_MODEL=$(echo "$INFERENCE" | grep -oE '"model":"[^"]+"' | cut -d'"' -f4)
MAV_CONNECTED=$(echo "$MAVLINK" | grep -oE '"connected":[a-z]+' | cut -d: -f2)
MAV_BAT=$(echo "$MAVLINK" | grep -oE '"battery_pct":[0-9]+' | cut -d: -f2)
MAV_MODE=$(echo "$MAVLINK" | grep -oE '"flight_mode":"[^"]+"' | cut -d'"' -f4)

printf "      %-12s %s @ %s FPS\n" "inference" "$INF_MODEL" "$INF_FPS"
printf "      %-12s connected=%s mode=%s battery=%s%%\n" "mavlink" "$MAV_CONNECTED" "$MAV_MODE" "$MAV_BAT"
printf "      %-12s %s\n" "data" "$DATA"
printf "      %-12s OK\n" "ui"
echo

echo "==> "
c_green "READY."; echo
echo
echo "    Open in your browser:"
c_yellow "      http://$JETSON_IP:5173"; echo
echo
echo "    Notes:"
echo "      - Hard-refresh on first load (Ctrl+Shift+R)"
echo "      - Demo runbook: jetson/DEMO_RUNBOOK.md"
echo "      - To stop everything later:"
echo "          ssh $SSH_HOST 'pkill -f uvicorn; pkill -f \"http.server\"'"
