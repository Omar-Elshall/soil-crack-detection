#!/usr/bin/env bash
# start.sh — Launch all three microservices + serve the built UI
# Run from ~/soil-crack-detection on the Jetson.
#
# Usage:
#   bash jetson/start.sh          # normal (connects to Pixhawk + camera)
#   DRY_RUN=1 bash jetson/start.sh  # dry run (no hardware required)

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export DRY_RUN="${DRY_RUN:-0}"
export PYTHONPATH="$ROOT"

echo "==> Starting Soil Crack Detection System"
echo "    Root:    $ROOT"
echo "    DryRun:  $DRY_RUN"
echo ""

# ── Inference service (port 8001) ────────────────────────────────────────────
echo "[1/3] Inference service → http://0.0.0.0:8001"
DRY_RUN=$DRY_RUN python3 -m uvicorn jetson.services.inference.main:app \
  --host 0.0.0.0 --port 8001 --log-level warning &
PID_INFERENCE=$!

# ── MAVLink service (port 8002) ──────────────────────────────────────────────
echo "[2/3] MAVLink service  → http://0.0.0.0:8002"
DRY_RUN=$DRY_RUN python3 -m uvicorn jetson.services.mavlink.main:app \
  --host 0.0.0.0 --port 8002 --log-level warning &
PID_MAVLINK=$!

# ── Data service (port 8003) ─────────────────────────────────────────────────
echo "[3/3] Data service     → http://0.0.0.0:8003"
python3 -m uvicorn jetson.services.data.main:app \
  --host 0.0.0.0 --port 8003 --log-level warning &
PID_DATA=$!

# ── Serve built UI (port 5173) ───────────────────────────────────────────────
UI_DIR="$ROOT/jetson/ui/dist"
if [ -d "$UI_DIR" ]; then
  echo "[UI]  Static UI        → http://0.0.0.0:5173"
  python3 -m http.server 5173 --directory "$UI_DIR" &
  PID_UI=$!
else
  echo "[UI]  dist/ not found — run 'npm run build' in jetson/ui first"
  PID_UI=""
fi

echo ""
echo "All services started. Press Ctrl+C to stop."
echo ""
echo "  Live UI:   http://$(hostname -I | awk '{print $1}'):5173"
echo "  Inference: http://$(hostname -I | awk '{print $1}'):8001"
echo "  MAVLink:   http://$(hostname -I | awk '{print $1}'):8002"
echo "  Data:      http://$(hostname -I | awk '{print $1}'):8003"

# ── Shutdown handler ─────────────────────────────────────────────────────────
cleanup() {
  echo ""
  echo "Shutting down..."
  kill $PID_INFERENCE $PID_MAVLINK $PID_DATA 2>/dev/null
  [ -n "$PID_UI" ] && kill $PID_UI 2>/dev/null
  wait 2>/dev/null
  echo "Done."
}
trap cleanup INT TERM

wait
