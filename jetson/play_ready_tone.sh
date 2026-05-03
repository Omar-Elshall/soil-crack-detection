#!/usr/bin/env bash
# play_ready_tone.sh — Poll all 4 endpoints; once all healthy, send a tune
# to the Pixhawk's onboard buzzer. This is the audible "everything is up"
# signal — fires both on boot (via start.sh) and from demo_start.sh.
#
# Runs locally on the Jetson — no SSH needed.

TIMEOUT="${TIMEOUT:-120}"
deadline=$(( $(date +%s) + TIMEOUT ))

while :; do
  ok=1
  for url in \
    http://127.0.0.1:8001/status \
    http://127.0.0.1:8002/status \
    http://127.0.0.1:8003/status \
    http://127.0.0.1:5173/index.html
  do
    curl -sf -m 2 "$url" > /dev/null 2>&1 || { ok=0; break; }
  done
  if [ "$ok" = "1" ]; then
    # All 4 endpoints up. Beep the Pixhawk.
    curl -sf -m 4 -X POST 'http://127.0.0.1:8002/command/play-tone' > /dev/null 2>&1
    echo "[ready-tone] all services up; tone sent."
    exit 0
  fi
  if [ $(date +%s) -ge $deadline ]; then
    echo "[ready-tone] timed out after ${TIMEOUT}s."
    exit 1
  fi
  sleep 2
done
