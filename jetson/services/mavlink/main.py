"""
MAVLink Service — port 8002
Connects to Pixhawk 6C via USB (/dev/ttyACM0), polls telemetry,
exposes WebSocket telemetry feed and REST flight commands.

Usage:
    cd ~/soil-crack-detection
    python3 -m uvicorn jetson.services.mavlink.main:app --host 0.0.0.0 --port 8002
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .connection import MAVLinkConnection
from .flight import FlightController
from .telemetry import TelemetryPoller
from . import routes as _routes

app = FastAPI(title="MAVLink Service", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(_routes.router)

_conn: MAVLinkConnection = None
_poller: TelemetryPoller = None
_flight: FlightController = None


@app.on_event("startup")
async def startup():
    global _conn, _poller, _flight

    dry_run = os.environ.get("DRY_RUN", "0") == "1"

    _conn = MAVLinkConnection(
        port=os.environ.get("MAVLINK_PORT", "/dev/ttyACM0"),
        baud=int(os.environ.get("MAVLINK_BAUD", "921600")),
    )

    if not dry_run:
        _conn.connect(timeout=15)
    else:
        print("DRY_RUN mode — Pixhawk connection skipped")

    _poller = TelemetryPoller(_conn)
    _poller.start()

    _flight = FlightController(_conn)

    # Inject into routes module
    _routes.poller = _poller
    _routes.flight = _flight
    _routes.conn = _conn

    print("MAVLink service ready.")


@app.on_event("shutdown")
async def shutdown():
    if _poller:
        _poller.stop()
    if _conn:
        _conn.disconnect()
