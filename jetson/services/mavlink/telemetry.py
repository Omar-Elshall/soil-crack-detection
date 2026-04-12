"""
telemetry.py — Polls MAVLink messages and maintains a shared telemetry snapshot.
Broadcasts to all connected WebSocket clients at 5 Hz.
"""

import threading
import time
from dataclasses import dataclass, asdict
from typing import Set

from fastapi import WebSocket


@dataclass
class TelemetrySnapshot:
    # GPS
    lat: float = 0.0
    lon: float = 0.0
    alt_m: float = 0.0
    heading_deg: float = 0.0
    gps_fix: int = 0          # 0=no fix, 2=2D, 3=3D
    # Attitude
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    # Velocity
    vx_ms: float = 0.0
    vy_ms: float = 0.0
    vz_ms: float = 0.0
    # NED local position (fallback when no GPS)
    north_m: float = 0.0
    east_m: float = 0.0
    # Flight state
    armed: bool = False
    flight_mode: str = "UNKNOWN"
    # Battery
    battery_v: float = 0.0
    battery_pct: float = 0.0
    # Connection
    connected: bool = False

    def to_dict(self):
        return asdict(self)


# Severity levels from MAVLink STATUSTEXT
_SEVERITY = {0: "EMERGENCY", 1: "ALERT", 2: "CRITICAL", 3: "ERROR",
             4: "WARNING", 5: "NOTICE", 6: "INFO", 7: "DEBUG"}


class TelemetryPoller:
    def __init__(self, connection):
        self.conn = connection
        self.snapshot = TelemetrySnapshot()
        self._lock = threading.Lock()
        self._ws_clients: Set[WebSocket] = set()
        self._running = False
        self._thread = None
        self.status_messages: list[dict] = []  # last 30 ArduPilot messages

    def start(self):
        self._running = True
        self._paused = False
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def pause(self):
        """Pause message consumption so blocking MAVLink exchanges (upload, etc.) can read ACKs."""
        self._paused = True

    def resume(self):
        self._paused = False

    def _poll_loop(self):
        """Continuously reads MAVLink messages and updates snapshot."""
        # ArduPilot flight mode mapping (subset)
        COPTER_MODES = {
            0: "STABILIZE", 2: "ALT_HOLD", 3: "AUTO", 4: "GUIDED",
            5: "LOITER", 6: "RTL", 9: "LAND", 16: "POSHOLD",
        }
        while self._running:
            if self._paused or not self.conn.connected:
                time.sleep(0.05)
                continue
            msg = self.conn.recv(blocking=False, timeout=0.05)
            if msg is None:
                continue
            mt = msg.get_type()
            with self._lock:
                if mt == "GLOBAL_POSITION_INT":
                    self.snapshot.lat = msg.lat / 1e7
                    self.snapshot.lon = msg.lon / 1e7
                    self.snapshot.alt_m = round(msg.relative_alt / 1000.0, 2)
                    self.snapshot.heading_deg = round(msg.hdg / 100.0, 1) if msg.hdg != 65535 else 0.0
                    self.snapshot.vx_ms = round(msg.vx / 100.0, 2)
                    self.snapshot.vy_ms = round(msg.vy / 100.0, 2)
                    self.snapshot.vz_ms = round(msg.vz / 100.0, 2)
                elif mt == "LOCAL_POSITION_NED":
                    self.snapshot.north_m = round(msg.x, 3)
                    self.snapshot.east_m = round(msg.y, 3)
                elif mt == "ATTITUDE":
                    import math
                    self.snapshot.roll_deg = round(math.degrees(msg.roll), 1)
                    self.snapshot.pitch_deg = round(math.degrees(msg.pitch), 1)
                    self.snapshot.yaw_deg = round(math.degrees(msg.yaw), 1)
                elif mt == "SYS_STATUS":
                    self.snapshot.battery_v = round(msg.voltage_battery / 1000.0, 2)
                    self.snapshot.battery_pct = round(msg.battery_remaining, 0) if msg.battery_remaining >= 0 else 0.0
                elif mt == "HEARTBEAT":
                    self.snapshot.armed = bool(msg.base_mode & 0x80)
                    self.snapshot.flight_mode = COPTER_MODES.get(msg.custom_mode, f"MODE_{msg.custom_mode}")
                    self.snapshot.connected = True
                elif mt == "GPS_RAW_INT":
                    self.snapshot.gps_fix = msg.fix_type
                elif mt == "STATUSTEXT":
                    text = msg.text.strip('\x00').strip()
                    if text:
                        self.status_messages.append({
                            "text": text,
                            "severity": _SEVERITY.get(msg.severity, "INFO"),
                            "severity_level": msg.severity,
                            "ts": time.time(),
                        })
                        self.status_messages = self.status_messages[-30:]

    def get(self) -> dict:
        with self._lock:
            d = self.snapshot.to_dict()
            d["connected"] = self.conn.connected
            d["status_messages"] = list(self.status_messages)
            return d

    def add_ws_client(self, ws: WebSocket):
        self._ws_clients.add(ws)

    def remove_ws_client(self, ws: WebSocket):
        self._ws_clients.discard(ws)

    def stop(self):
        self._running = False
