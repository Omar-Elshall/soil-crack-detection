"""
laptop_mavlink_relay.py — runs on the demo laptop (Windows), reads MAVLink
from a SiK telemetry radio on a COM port, and exposes the same
WebSocket + REST contract as the Jetson MAVLink service so the UI can
target it identically.

Why: when the Jetson goes out of WiFi range mid-flight, the UI loses
telemetry. The SiK radio reaches hundreds of meters and is bidirectional,
so we use it as the *primary* MAVLink path for what the UI shows + sends.
The Jetson keeps its own /dev/ttyACM0 connection for inference annotation
(detection log entries get tagged with telemetry); this relay is independent.

Usage (Windows or Linux):
    python laptop_mavlink_relay.py
    RELAY_COM=COM5 RELAY_BAUD=57600 RELAY_PORT=18002 python laptop_mavlink_relay.py

If RELAY_COM isn't set, auto-detects the first SiK-like USB serial port.

Listens on http://127.0.0.1:18002 by default. Endpoints mirror
jetson/services/mavlink — see routes.py.
"""

import os
import sys
import time
from pathlib import Path

# Make the repo root importable so we can reuse the Jetson service modules.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from jetson.services.mavlink.connection import MAVLinkConnection
from jetson.services.mavlink.telemetry import TelemetryPoller
from jetson.services.mavlink.flight import FlightController
from jetson.services.mavlink import routes as _routes


def auto_detect_com_port() -> str | None:
    """Find the first USB-serial device that looks like a SiK telem radio.
    SiK clones use SiLabs CP210x, FTDI, or generic USB-serial chips.
    Returns the device path (Windows: 'COM5'; Linux: '/dev/ttyUSB0') or None.
    """
    try:
        from serial.tools import list_ports
    except ImportError:
        return None
    candidates = []
    for p in list_ports.comports():
        desc = (p.description or "").lower()
        manuf = (p.manufacturer or "").lower()
        # Common identifiers for SiK-radio USB adapters
        if any(k in desc + manuf for k in ("silicon", "cp210", "ftdi", "prolific", "usb serial", "usb-serial", "telem")):
            candidates.append(p.device)
    return candidates[0] if candidates else None


def main():
    # SiK telem radio at 57600 baud can't sustain 10 Hz across all streams.
    # Drop to a rate that fits the link so SYS_STATUS (battery voltage) lands.
    os.environ.setdefault("MAVLINK_STREAM_HZ", "3")

    port = os.environ.get("RELAY_COM")
    if not port:
        port = auto_detect_com_port()
    if not port:
        print("[relay] No COM port found. Set RELAY_COM=COM5 (or wherever the radio is) and retry.", flush=True)
        sys.exit(2)

    baud = int(os.environ.get("RELAY_BAUD", "57600"))
    listen_port = int(os.environ.get("RELAY_PORT", "18002"))

    print(f"[relay] connecting to MAVLink at {port} @ {baud}...", flush=True)
    conn = MAVLinkConnection(port=port, baud=baud)
    if not conn.connect(timeout=20):
        print(f"[relay] FAILED to connect on {port}. Check the radio + Pixhawk are powered and TELEM1 is configured (SERIAL1_PROTOCOL=2, SERIAL1_BAUD=57).", flush=True)
        sys.exit(3)

    poller = TelemetryPoller(conn)
    poller.start()
    flight = FlightController(conn)

    # Watchdog: when the radio is unplugged + replugged, pymavlink keeps
    # the old serial handle and silently stops receiving. Detect that and
    # re-open the connection in-place — no need to bounce the whole process.
    # 15 s of silence is the "link is really dead" threshold.
    import threading as _t
    def _watchdog():
        time.sleep(20)  # grace period for first messages
        while True:
            time.sleep(5)
            silence = time.time() - poller.last_msg_ts
            if silence < 15:
                continue
            print(f"[relay] WATCHDOG: no MAVLink for {silence:.1f}s — reconnecting...", flush=True)
            poller.pause()  # so its recv() doesn't race the serial close
            try:
                conn.disconnect()
            except Exception as e:
                print(f"[relay] disconnect error: {e}", flush=True)
            time.sleep(1)
            # Find the COM port again — radio may have been re-enumerated.
            new_port = os.environ.get("RELAY_COM") or auto_detect_com_port() or port
            conn.port = new_port
            ok = conn.connect(timeout=10)
            poller.resume()
            if ok:
                poller.last_msg_ts = time.time()  # reset the freshness clock
                print(f"[relay] WATCHDOG: reconnected on {new_port}", flush=True)
            else:
                print(f"[relay] WATCHDOG: reconnect on {new_port} failed; will retry in 5s", flush=True)
    _t.Thread(target=_watchdog, daemon=True).start()

    # Inject into the shared routes module — same pattern as Jetson main.py
    _routes.poller = poller
    _routes.flight = flight
    _routes.conn = conn

    app = FastAPI(title="MAVLink Relay (laptop)", version="1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.include_router(_routes.router)

    print(f"[relay] ready. http://127.0.0.1:{listen_port}/status", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=listen_port, log_level="warning")


if __name__ == "__main__":
    main()
