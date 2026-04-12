#!/usr/bin/env bash
# mavbridge.sh — Forward Pixhawk MAVLink to Mission Planner over UDP
#
# The Pixhawk is connected via USB to the Jetson (/dev/ttyACM0).
# Mission Planner runs on your PC and can't see the USB port directly.
# This script uses MAVProxy to bridge the serial connection and broadcast
# MAVLink packets over UDP so Mission Planner can connect.
#
# Usage:
#   bash jetson/mavbridge.sh                     # broadcasts on subnet
#   PC_IP=192.168.1.100 bash jetson/mavbridge.sh # forward only to your PC
#
# In Mission Planner: connect → UDP → port 14550 → Connect

SERIAL_PORT="${SERIAL_PORT:-/dev/ttyACM0}"
BAUD="${BAUD:-115200}"
UDP_PORT="${UDP_PORT:-14550}"
PC_IP="${PC_IP:-}"

echo "==> MAVProxy Bridge"
echo "    Serial:  $SERIAL_PORT @ $BAUD baud"
JETSON_IP=$(hostname -I | awk '{print $1}')
echo "    Jetson:  $JETSON_IP"
echo ""

if [ -n "$PC_IP" ]; then
  OUT="--out udp:${PC_IP}:${UDP_PORT}"
  echo "    Forwarding → UDP $PC_IP:$UDP_PORT"
else
  OUT="--out udpbcast:0.0.0.0:${UDP_PORT}"
  echo "    Broadcasting → UDP broadcast:$UDP_PORT (any PC on the subnet picks it up)"
fi

echo ""
echo "  In Mission Planner:"
echo "    1. Top-left dropdown → UDP"
echo "    2. Port: $UDP_PORT"
echo "    3. Click CONNECT"
echo ""
echo "  Press Ctrl+C to stop the bridge."
echo ""

# Run from /tmp so MAVProxy can write its log file
cd /tmp
mavproxy.py \
  --master "$SERIAL_PORT" \
  --baudrate "$BAUD" \
  $OUT \
  --out udp:127.0.0.1:14551 \
  --non-interactive
